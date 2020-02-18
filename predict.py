import tensorflow as tf
import cv2
from ResNet import ResNet
import numpy as np
from ops import *
from attention import Attention
from tensorflow.contrib import slim
from metrics import char_accuracy, sequence_accuracy
import os
import time
import numpy as np
import tensorflow as tf

from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Predict(object):
    def __init__(self, batch_size, model_path, examples_path, vocab_size, test_file, restore):
        self.step = 0
        self.__model_path = model_path
        self.__save_path = os.path.join(model_path, 'ckp')

        self.__restore = restore
        self.vocab_size = vocab_size
        self.__training_name = str(int(time.time()))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.__session = tf.Session(config=config)
        self.__session = tf.Session()
        self.__bs = batch_size
        self.__sequence_length = 25
        self.__ex = examples_path
        self.__test_file = test_file
        # Building graph
        with self.__session.as_default():
            (
                self.__inputs,
                self.__output,
                self.__length,
                self.__scores,
                self.__ids,
                self.__log_probs,
                self.__init
            ) = self.ocr_test(vocab_size, self.__sequence_length, batch_size)

            self.__init.run()

        with self.__session.as_default():
            self.__saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
            # Loading last save if needed
            if self.__restore:
                print('Restoring')
                ckpt = tf.train.latest_checkpoint(self.__model_path)
                if ckpt:
                    print('Checkpoint is valid')
                    self.step = int(ckpt.split('-')[1])
                    self.__saver.restore(self.__session, ckpt)

    def ocr_test(self, vocab_size, max_seq = 25, batch_size = 8):
        """
            Builds the graph, returns the "important" ops
        """
        inputs = tf.placeholder(tf.float32, [None, 128, 400, 3])
        output = tf.placeholder(tf.int32, [None, max_seq])
        length = tf.placeholder(tf.int32, [None])
        resnet_34 = ResNet(34, 10)
        def resnet_34_backbone(x):
            out = resnet_34.network(x, is_training=False)
            print(out)
            return out

        feature_map_resnet = resnet_34_backbone(inputs)     #feature map of resnet 34
        feature_map = transform_dimension(feature_map_resnet, 1024)
        # print("feature map: ", feature_map)
        for i in range(6):
            global_representation = bottle_resblock(feature_map_resnet if i== 0 else global_representation, 512, is_training=False, scope='bottle_resblock_' + str(i))
        global_representation = global_avg_pooling(global_representation)
        global_representation = fully_conneted(global_representation, 512)
        ##########################################################DECODER########################################
        def decoder_embedding(y, vocab_size, embed_size=512, shifted=True):
            embeddings = tf.random_normal(shape=(vocab_size, embed_size))
            embedded = tf.nn.embedding_lookup(embeddings, y)
            return embedded
        def positional_encoding(x):
            """
                Not as described in paper since it lacked proper description of this step.
                This function is based on the "Attention is all you need" paper.
            """
            seq_len, dim = x.get_shape().as_list()[-2:]
            encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(seq_len) for i in range(dim)])
            encoded_vec[::2] = np.sin(encoded_vec[::2])
            encoded_vec[1::2] = np.cos(encoded_vec[1::2])
            encoded_vec_tensor = tf.convert_to_tensor(encoded_vec.reshape([seq_len, dim]), dtype=tf.float32)
            return tf.add(x, encoded_vec_tensor)
        def layer_norm(x):
            """
                Layer normalization as described in paper (p.4)
            """
            return tf.contrib.layers.layer_norm(x)

        y = decoder_embedding(output, vocab_size)

        y = tf.pad(
            y, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]            #shift right from official transformer
        # print("embedding: ", y)
        y = positional_encoding(y)
        # print("PE: ", y)               #(bs, seq_len, 512)

        #concatenate with global representation
        decoder_input = []
        for i in range(y.get_shape().as_list()[1]):
            decoder_input.append(tf.concat([global_representation, y[:,i,:]], 1))                  #(bs, 1, 512)
        decoder_input = tf.stack(decoder_input, 1)

        ####MASKED SELF ATTENTION###
        masked_self_attention = Attention(dropout=0)
        decoder_output = masked_self_attention.multi_head(decoder_input, decoder_input, decoder_input)
        norm_1 = layer_norm(decoder_output)
        decoder_output = decoder_input + norm_1
        
        ###2D self attention###
        two_D_attention = Attention(masked=False, dropout=0)
        rrr = feature_map.get_shape().as_list()[1] * feature_map.get_shape().as_list()[2]
        enc_reshape = tf.reshape(feature_map, [-1, rrr, decoder_output.get_shape().as_list()[-1]])
        decoder_output_2 = two_D_attention.multi_head(decoder_output, enc_reshape, enc_reshape)
        norm_2 = layer_norm(decoder_output_2)
        decoder_output = decoder_output + norm_2

        def position_wise_feed_forward_network(x):
            """
                Position-wise Feed-Forward Network as described in paper (p.4)
            """

            # First linear
            #linear_1 = tf.layers.dense(x, x.get_shape().as_list()[-1])
            linear_1 = tf.layers.conv1d(x, 2048, 1)

            # ReLU operation
            relu_1 = tf.nn.relu(linear_1)

            # Second linear
            linear_2 = tf.layers.conv1d(relu_1, x.get_shape().as_list()[-1], 1)

            return tf.nn.dropout(linear_2, 1)
        pwff = position_wise_feed_forward_network(decoder_output)
        norm_3 = layer_norm(pwff)
        decoder_output = decoder_output + norm_3

        # output_probabilities = tf.layers.dense(decoder_output, vocab_size, activation=tf.contrib.layers.softmax)
        output_probabilities = tf.layers.dense(decoder_output, vocab_size)

        ids, log_probs, scores = self.char_predictions(output_probabilities, vocab_size, max_seq)
        probs = tf.nn.softmax(output_probabilities)
        init = tf.global_variables_initializer()
        return inputs, output, length, scores, ids, probs, init

    def pred(self):
        with self.__session.as_default():
            print('Testing')
            list_file_test = ["IC13_1015.txt", "IC15_1811.txt", "IIIT5K.txt", "SVT.txt", "SVTP.txt", "CUTE80.txt"]
            for item in list_file_test:
                with open(os.path.join(self.__ex, item), 'r') as file:
                    lines = [line.strip('\n') for line in file.readlines()]
                num_batch = len(lines) // self.__bs
                word_accuracy = 0
                character_accuracy = 0
                for i in range(num_batch):
                    batch_x, batch_y, img_name = self.get_batch(self.__bs, i, lines, self.__ex)
                    y_feed = np.zeros([batch_x.shape[0], self.__sequence_length], dtype=np.int32)
                    z_feed = np.zeros(batch_x.shape[0], dtype=np.int32)
                    t1 = time.time()
                    for time_step in range(self.__sequence_length):
                        scores, ids = self.__session.run(
                            [self.__scores, self.__ids],
                            feed_dict={
                                self.__inputs: batch_x,
                                self.__output: y_feed,
                                self.__length: z_feed
                            }
                        )
                        y_feed = ids
                    if i == int(num_batch/3):
                        print(time.time()-t1)
                    w_a, _ = cal_accuracy(batch_y, ids, img_name, item + "_wrong_test.txt")
                    word_accuracy += w_a

                print("word accuracy: ", word_accuracy/num_batch)
        return None

    def get_batch(self, bs, step, annotations_row_files, example_path):
        batch_imgs = []
        batch_anno = []
        batch_names = []
        for i in range(3 * bs):
            try:
                idx = (bs * step + i) % len(annotations_row_files)
                tmp = annotations_row_files[idx]
                splits = tmp.split(' ')
                image_name = splits[0]
                gt_text = splits[1].strip()
                if len(gt_text) >= self.__sequence_length:
                    continue
                anno, flag = label_to_array_2(gt_text.upper(), self.__sequence_length)
                img = resize_image(os.path.join(example_path, image_name))
                if flag:     
                    batch_anno.append(anno)
                    batch_imgs.append(img)
                    batch_names.append(image_name)
                else: 
                    continue
                if len(batch_anno) == bs:
                    break
            except Exception as e:
                continue
        batch_imgs = np.array(batch_imgs)
        batch_anno = np.array(batch_anno)
        return batch_imgs, batch_anno, batch_names

    def char_predictions(self, chars_logit, num_class, seq_len):
        """Returns confidence scores (softmax values) for predicted characters.
        Args:
            chars_logit: chars logits, a tensor with shape
            [batch_size x seq_length x num_char_classes]
        Returns:
            A tuple (ids, log_prob, scores), where:
            ids - predicted characters, a int32 tensor with shape
                [batch_size x seq_length];
            log_prob - a log probability of all characters, a float tensor with
                shape [batch_size, seq_length, num_char_classes];
            scores - corresponding confidence scores for characters, a float
            tensor
                with shape [batch_size x seq_length].
        """
        log_prob = self.logits_to_log_prob(chars_logit)
        ids = tf.to_int32(tf.argmax(log_prob, axis=2), name='predicted_chars')
        mask = tf.cast(
            slim.one_hot_encoding(ids, num_class), tf.bool)
        all_scores = tf.nn.softmax(chars_logit)
        selected_scores = tf.boolean_mask(all_scores, mask, name='char_scores')
        scores = tf.reshape(selected_scores, shape=(-1, seq_len))
        return ids, log_prob, scores

    def logits_to_log_prob(self, logits):
        """Computes log probabilities using numerically stable trick.
        This uses two numerical stability tricks:
        1) softmax(x) = softmax(x - c) where c is a constant applied to all
        arguments. If we set c = max(x) then the softmax is more numerically
        stable.
        2) log softmax(x) is not numerically stable, but we can stabilize it
        by using the identity log softmax(x) = x - log sum exp(x)
        Args:
            logits: Tensor of arbitrary shape whose last dimension contains logits.
        Returns:
            A tensor of the same shape as the input, but with corresponding log
            probabilities.
        """

        with tf.variable_scope('log_probabilities'):
            reduction_indices = len(logits.shape.as_list()) - 1
            max_logits = tf.reduce_max(
                logits, reduction_indices=reduction_indices, keep_dims=True)
            safe_logits = tf.subtract(logits, max_logits)
            sum_exp = tf.reduce_sum(
                tf.exp(safe_logits),
                reduction_indices=reduction_indices,
                keep_dims=True)
            log_probs = tf.subtract(safe_logits, tf.log(sum_exp))
        return log_probs
    
