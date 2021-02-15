'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file MCClass.py

    \brief Definition of the network architecture MCClass for classification 
           tasks.

    \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import os
import math
import tensorflow as tf

#BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('.')))
PROJECT_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(PROJECT_DIR)
sys.path.append(os.path.join(os.path.join(ROOT_DIR, 'MCCNN'), 'tf_ops'))
sys.path.append(os.path.join(os.path.join(ROOT_DIR, 'MCCNN'), 'utils'))

from MCConvBuilder import PointHierarchy, ConvolutionBuilder
from MCNetworkUtils import MLP_2_hidden, batch_norm_RELU_drop_out, conv_1x1


def resblock_2d(x, filters, kernel_size, depth):
    
    for i in range(depth):
        x_r = x
        BN = tf.keras.layers.BatchNormalization()
        x = BN(x)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, filters, kernel_size=kernel_size, padding='same')
        BN = tf.keras.layers.BatchNormalization()
        x = BN(x)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, filters, kernel_size=kernel_size, padding='same')
        x = x+x_r
    return x

def create_network(points, features, batchIds, batchSize, numInputFeatures, k,  numOutCat ,numOutputs, isTraining, 
    keepProbConv, keepProbFull, useConvDropOut = False, useDropOutFull = True, useRenorm =False, BNMomentum = 0.99, activation = 'relu'):
    
    if activation == 'relu':
        from MCNetworkUtils import batch_norm_RELU_drop_out as batch_norm_activation_drop_out
    elif activation == 'leakyrelu':
        from MCNetworkUtils import batch_norm_leakyRELU_drop_out as batch_norm_activation_drop_out
    elif activation == 'mish':
        from MCNetworkUtils import batch_norm_mish_drop_out as batch_norm_activation_drop_out
    else:
        print('\nError: Unknown activation function: %s\n' %(activation))
    

    mPointHierarchy = PointHierarchy(points, features, batchIds, [0.025, 0.1, 0.4, math.sqrt(3.0)+0.1], "MCSphere", batchSize)

    ############################################ Convolutions
    mConvBuilder = ConvolutionBuilder(KDEWindow=0.25)
    
    # Zeroth Pooling
    convFeatures0 = conv_1x1("Reduce_Pool_0", features, numInputFeatures, k*2)
    convFeatures0 = batch_norm_activation_drop_out("Reduce_Pool_0_Out_BN", convFeatures0, isTraining, useConvDropOut, keepProbConv, useRenorm, BNMomentum)
    poolFeatures0 = mConvBuilder.create_convolution(
        convName="Pool_0", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=0, 
        outPointLevel=1, 
        inFeatures=convFeatures0,
        inNumFeatures=k*2, 
        convRadius=0.05,
        KDEWindow= 0.2)

    # First Pooling
    convFeatures1 = batch_norm_activation_drop_out("Reduce_Pool_1_In_BN", poolFeatures0, isTraining, useConvDropOut, keepProbConv, useRenorm, BNMomentum)
    convFeatures1 = conv_1x1("Reduce_Pool_1", convFeatures1, k*2, k*4)
    convFeatures1 = batch_norm_activation_drop_out("Reduce_Pool_1_Out_BN", convFeatures1, isTraining, useConvDropOut, keepProbConv, useRenorm, BNMomentum)
    poolFeatures1 = mConvBuilder.create_convolution(
        convName="Pool_1", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=1, 
        outPointLevel=2, 
        inFeatures=convFeatures1,
        inNumFeatures=k*4, 
        convRadius=0.2,
        KDEWindow= 0.2)

    # Second Pooling
    convFeatures2 = batch_norm_activation_drop_out("Reduce_Pool_2_In_BN", poolFeatures1, isTraining, useConvDropOut, keepProbConv, useRenorm, BNMomentum)
    convFeatures2 = conv_1x1("Reduce_Pool_2", convFeatures2, k*4, k*16)
    convFeatures2 = batch_norm_activation_drop_out("Reduce_Pool_2_Out_BN", convFeatures2, isTraining, useConvDropOut, keepProbConv, useRenorm, BNMomentum)
    poolFeatures2 = mConvBuilder.create_convolution(
        convName="Pool_2", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=2, 
        outPointLevel=3, 
        inFeatures=convFeatures2,
        inNumFeatures=k*16, 
        convRadius=0.3,
        KDEWindow= 0.2)

    # Third Pooling
    convFeatures3 = batch_norm_activation_drop_out("Reduce_Pool_3_In_BN", poolFeatures2, isTraining, useConvDropOut, keepProbConv, useRenorm, BNMomentum)
    convFeatures3 = conv_1x1("Reduce_Pool_3", convFeatures3, k*16, k*32)
    convFeatures3 = batch_norm_activation_drop_out("Reduce_Pool_3_Out_BN", convFeatures3, isTraining, useConvDropOut, keepProbConv, useRenorm, BNMomentum)
    poolFeatures3 = mConvBuilder.create_convolution(
        convName="Pool_3", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=3, 
        outPointLevel=4, 
        inFeatures=convFeatures3,
        inNumFeatures=k*32, 
        convRadius=math.sqrt(3.0)+0.1,
        KDEWindow= 0.2)
        
    #Image decoder - Global features.
    encoderOutput = batch_norm_activation_drop_out("BNRELUDROP_finalencoder", poolFeatures3, isTraining, useConvDropOut, keepProbConv, useRenorm, BNMomentum)
    encoderOutput = tf.reshape(encoderOutput, [-1,1,1,k*32])
    finalPredictions = []

    for vq_i in range(numOutputs):
        deconvLayer0 = tf.keras.layers.Conv2DTranspose(k*16, [4,4], strides=(1,1), data_format='channels_last', input_shape=[1, 1, k*32], name= 'deconvFeatures0'+str(vq_i))
        deconvFeatures0 = deconvLayer0(encoderOutput)
        deconvFeatures0 = resblock_2d(deconvFeatures0, k*16, [3,3],2)
        deconvFeatures0 = batch_norm_activation_drop_out("BNRELUDROP_deconv0"+str(vq_i), deconvFeatures0, isTraining, useConvDropOut, keepProbConv, useRenorm, BNMomentum)

        deconvLayer1 = tf.keras.layers.Conv2DTranspose(k*4, [4,4], strides=(4,4), data_format='channels_last', input_shape=[4, 4, k*16], name= 'deconvFeatures0'+str(vq_i))
        deconvFeatures1 = deconvLayer1(deconvFeatures0)
        deconvFeatures1 = resblock_2d(deconvFeatures1, k*4, [3,3],2)
        deconvFeatures1 = batch_norm_activation_drop_out("BNRELUDROP_deconv1"+str(vq_i), deconvFeatures1, isTraining, useConvDropOut, keepProbConv, useRenorm, BNMomentum)

        deconvLayer2 = tf.keras.layers.Conv2DTranspose(1, [2,2], strides=(2,2), data_format='channels_last', input_shape=[16, 16, k*4], name= 'deconvFeatures0'+str(vq_i))
        predicted_images = deconvLayer2(deconvFeatures1)
        # predicted_images = tf.squeeze(predicted_images)
        finalPredictions.append(tf.reshape(predicted_images,[-1,32,32]))
        
    

    return finalPredictions
