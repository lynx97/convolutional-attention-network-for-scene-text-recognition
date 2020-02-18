import tensorflow as tf
from ResNet import ResNet
import numpy as np
from ops import *
from attention import Attention
from tensorflow.contrib import slim
from metrics import char_accuracy, sequence_accuracy
import random

import os
import time
import numpy as np
import tensorflow as tf

from utils import resize_image, label_to_array_2, ground_truth_to_word

class SSCAN(object):
    def __init__(self, batch_size, model_path, examples_path, vocab_size, train_file, restore):
        self.step = 0
        self.__model_path = model_path
        self.__save_path = os.path.join(model_path, 'ckp')

        self.__restore = restore

        self.__training_name = str(int(time.time()))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.__session = tf.Session(config=config)
        self.__session = tf.Session()
        self.__bs = batch_size
        self.__ex = examples_path
        self.__train_file = train_file
        self.__sequence_length = 25
        # Building graph
        with self.__session.as_default():
            (
                self.__inputs,
                self.__output,
                self.__length,
                self.__loss,
                self.__optimizer,
                self.__prob,
                self.__loss_summary,
                self.__init,
                self.__word_acc
            ) = self.convolutional_attention_network(vocab_size, self.__sequence_length, batch_size)

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

    def convolutional_attention_network(self, vocab_size, max_seq = 25, batch_size = 8):
        """
            Builds the graph
        """
        inputs = tf.placeholder(tf.float32, [batch_size, 128, 400, 3])
        output = tf.placeholder(tf.int32, [batch_size, max_seq])
        length = tf.placeholder(tf.int32, [batch_size])
        resnet_34 = ResNet(34, 10)
        def resnet_34_backbone(x):
            out = resnet_34.network(x)
            print(out)
            return out

        feature_map_resnet = resnet_34_backbone(inputs)     #feature map of resnet 34
        feature_map = transform_dimension(feature_map_resnet, 1024)
        for i in range(6):
            global_representation = bottle_resblock(feature_map_resnet if i== 0 else global_representation, 512, scope='bottle_resblock_' + str(i))
        global_representation = global_avg_pooling(global_representation)
        global_representation = fully_conneted(global_representation, 512)

        ##########################################################DECODER########################################
        def decoder_embedding(y, vocab_size, embed_size=512, shifted=True):
            embeddings = tf.random_normal(shape=(vocab_size, embed_size))
            embedded = tf.nn.embedding_lookup(embeddings, y)
            return embedded
        def positional_encoding(x):
            seq_len, dim = x.get_shape().as_list()[-2:]
            encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(seq_len) for i in range(dim)])
            encoded_vec[::2] = np.sin(encoded_vec[::2])
            encoded_vec[1::2] = np.cos(encoded_vec[1::2])
            encoded_vec_tensor = tf.convert_to_tensor(encoded_vec.reshape([seq_len, dim]), dtype=tf.float32)
            return tf.add(x, encoded_vec_tensor)
        def layer_norm(x):
            return tf.contrib.layers.layer_norm(x)

        y = decoder_embedding(output, vocab_size)

        y = tf.pad(
            y, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]            #shift right from official transformer
        y = positional_encoding(y) #(bs, seq_len, 512)

        #concatenate with global representation
        decoder_input = []
        for i in range(y.get_shape().as_list()[1]):
            decoder_input.append(tf.concat([global_representation, y[:,i,:]], 1))                  #(bs, 1, 512)
        decoder_input = tf.stack(decoder_input, 1) #(bs, seq_len, 1024)

        ####MASKED SELF ATTENTION###
        masked_self_attention = Attention(dropout=0)
        decoder_output = masked_self_attention.multi_head(decoder_input, decoder_input, decoder_input)
        norm_1 = layer_norm(decoder_output)
        decoder_output = decoder_input + norm_1
        
        ###2D self attention###
        two_D_attention = Attention(masked=False, dropout=0)
        enc_reshape = tf.reshape(feature_map, [decoder_output.get_shape().as_list()[0], -1, decoder_output.get_shape().as_list()[-1]])
        decoder_output_2 = two_D_attention.multi_head(decoder_output, enc_reshape, enc_reshape)
        norm_2 = layer_norm(decoder_output_2)
        decoder_output = decoder_output + norm_2

        def position_wise_feed_forward_network(x):  #using conv1D
            # First linear
            linear_1 = tf.layers.conv1d(x, 2048, 1)
            # ReLU operation
            relu_1 = tf.nn.relu(linear_1)
            # Second linear
            linear_2 = tf.layers.conv1d(relu_1, x.get_shape().as_list()[-1], 1)
            return tf.nn.dropout(linear_2, 1)
        
        pwff = position_wise_feed_forward_network(decoder_output)
        norm_3 = layer_norm(pwff)
        decoder_output = decoder_output + norm_3

        output_probabilities = tf.layers.dense(decoder_output, vocab_size)

        loss = self._compute_loss(output_probabilities, output, length, batch_size)
        ids, log_probs, scores = self.char_predictions(output_probabilities, vocab_size, max_seq)
        char_acc = char_accuracy(ids, output, 0)
        word_acc = sequence_accuracy(ids, output, 0)

        with tf.name_scope('summaries'):
            tf.summary.scalar("loss", loss, collections=["train_summary"])
            tf.summary.scalar("character accuracy", char_acc, collections=["train_summary"])
            tf.summary.scalar("word accuracy", word_acc, collections=["train_summary"])

        summary_op = tf.summary.merge_all(key='train_summary')
        

        optimizer = tf.train.AdadeltaOptimizer(learning_rate=1).minimize(loss)

        init = tf.global_variables_initializer()
        return inputs, output, length, loss, optimizer, output_probabilities, summary_op, init, word_acc

    def train(self, iteration_count):
        self.step += 1
        print("Step: ", self.step)
        with self.__session.as_default():
            train_writer = tf.summary.FileWriter('./logs', self.__session.graph)
            step_summary = self.step
            print('Training')
            for epoch in range(self.step, iteration_count + self.step):
                iter_loss = 0
                with open(self.__train_file, 'r') as file:
                    lines = [line.strip('\n') for line in file.readlines()]
                random.shuffle(lines)
                num_batch = len(lines) // self.__bs

                for i in range(num_batch):
                    try:
                        batch_x, batch_y, batch_z = self.get_batch(self.__bs, i, lines, self.__ex)
                        _, loss_value, probabilities, loss_sum, word_step_acc = self.__session.run(
                            [self.__optimizer, self.__loss, self.__prob, self.__loss_summary, self.__word_acc],
                            feed_dict={
                                self.__inputs: batch_x,
                                self.__output: batch_y,
                                self.__length: batch_z
                            }
                        )
                        if i % int((num_batch/3)) == 0:
                            for j in range(1):
                                print(ground_truth_to_word(batch_y[j]))
                                prob = np.argmax(probabilities[j], axis=-1)
                                print(ground_truth_to_word(prob))
                                print("loss: ", loss_value)
                                print("step: {}/{}".format(i, num_batch))
                        train_writer.add_summary(loss_sum, step_summary)
                        step_summary += 1
                        iter_loss += loss_value
                    except Exception as e:
                        print(str(e))
                        continue
                self.__saver.save(
                    self.__session,
                    self.__save_path,
                    global_step=epoch
                )
                print('[{}] Iteration loss: {}'.format(epoch, iter_loss))
                self.step += 1
        return None

    def get_batch(self, bs, step, annotations_row_files, example_path):
        batch_imgs = []
        batch_anno = []
        batch_len = []
        for i in range(3 * bs):
            try:
                idx = (bs * step + i) % len(annotations_row_files)
                anno, img, seq, flag = self.get_idx(idx, annotations_row_files, example_path)
                if flag:     
                    batch_anno.append(anno)
                    batch_imgs.append(img)
                    batch_len.append(seq)
                else: 
                    continue

                if len(batch_anno) == bs:
                    break
            except Exception as e:
                continue
        batch_imgs = np.array(batch_imgs)
        batch_anno = np.array(batch_anno)
        batch_len = np.array(batch_len)
        return batch_imgs, batch_anno, batch_len

    def get_idx(self, idx, row_annos, example_path):
        tmp = row_annos[idx]
        splits = tmp.split(' ')
        image_name = splits[0]
        gt_text = splits[1].strip()
        if len(gt_text) >= self.__sequence_length:
            return 0, 0, 0, False
        anno, flag = label_to_array_2(gt_text.upper(), self.__sequence_length)
        img = resize_image(os.path.join(example_path, image_name))
        if flag:     
            seq = len(gt_text) + 1
            return anno, img, seq, flag
        else: 
            return 0, 0, 0, False

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
    def _compute_loss(self, logits, labels, labels_length, batch_size):
        """Computes the loss for this model.
        """
        with tf.name_scope("compute_loss"):

            _losses = self._cross_entropy_sequence_loss(
                logits=tf.transpose(logits, [1, 0, 2]),
                targets=tf.transpose(labels, [1, 0]),
                sequence_length=labels_length)
            _losses = tf.reduce_mean(_losses)
        return _losses

    def _cross_entropy_sequence_loss(self, logits, targets, sequence_length):
        """Calculates the per-example cross-entropy loss for a sequence of logits
            and masks out all losses passed the sequence length.
            Args:
            logits: Logits of shape `[T, B, vocab_size]`
            targets: Target classes of shape `[T, B]`
            sequence_length: An int32 tensor of shape `[B]` corresponding
                to the length of each input
            Returns:
            A tensor of shape [T, B] that contains the loss per example,
            per time step.
        """
        with tf.name_scope("cross_entropy_sequence_loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=targets)

            # Mask out the losses we don't care about
            loss_mask = tf.sequence_mask(
                tf.to_int32(sequence_length), tf.to_int32(tf.shape(targets)[0]))
            losses = losses * tf.transpose(tf.to_float(loss_mask), [1, 0])
        return losses
