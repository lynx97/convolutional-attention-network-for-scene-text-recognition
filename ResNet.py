import time
from ops import *

class ResNet(object):
    def __init__(self, res_n, out_dim):
        self.res_n = res_n
        self.label_dim = out_dim

    ##################################################################################
    # Generator
    ##################################################################################

    def network(self, x, is_training=True, reuse=False):
        with tf.variable_scope("network", reuse=reuse):

            if self.res_n < 50 :
                residual_block = resblock
            else :
                residual_block = bottle_resblock

            residual_list = get_residual_layer(self.res_n)

            ch = 64 # paper is 64
            x = conv(x, channels=ch, kernel=7, stride=2, scope='conv')
            x = max_pooling(x)
            for i in range(residual_list[0]) :
                x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))
            ########################################################################################################

            x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')
            for i in range(1, residual_list[1]) :
                x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))
            ########################################################################################################

            x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')
            for i in range(1, residual_list[2]) :
                x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))
            ########################################################################################################

            x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')
            for i in range(1, residual_list[3]) :
                x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))
            ########################################################################################################


            x = batch_norm(x, is_training, scope='batch_norm')
            x = relu(x)

            # x = global_avg_pooling(x)
            # x = fully_conneted(x, units=self.label_dim, scope='logit')
            return x
