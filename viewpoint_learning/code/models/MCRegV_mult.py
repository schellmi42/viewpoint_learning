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
        
    #Fully connected MLP - Global features.
    finalInput = batch_norm_activation_drop_out("BNRELUDROP_final", poolFeatures3, isTraining, useConvDropOut, keepProbConv, useRenorm, BNMomentum)
    final = []
    for vq_i in range(numOutputs):
        Predictions = MLP_2_hidden(finalInput, k*32, k*16, k*8, numOutCat, "Final_Predictions" + str(vq_i), keepProbFull, isTraining, useDropOutFull, useRenorm = useRenorm, BNMomentum=BNMomentum)
        final.append(Predictions)
        
    finalPredictions = tf.reshape(tf.stack(final, axis = 1),[-1,3*numOutputs])
    

    return finalPredictions
