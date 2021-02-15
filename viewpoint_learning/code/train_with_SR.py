'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \brief Code to train using SR

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


import sys
import math
import time
import argparse
import importlib
import os
import numpy as np
import tensorflow as tf
import multiprocessing as mp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'helpers'))
sys.path.append(ROOT_DIR + '/MCCNN/utils')

from PyUtils import visualize_progress
from train_ops import create_mult_loss_approx, BN_decay, create_feed_dict_mult, fill_dummy_ptH
from DataSet_SR import VQDataSet
#from VQs import getAs, getPz, getProb, getFaceIds, getIds, vq4, vq5, vq7, vq8, vq12, vq14
#from Application import GLScene
from PointHierarchyDummy import PointHierarchyDummy

current_milli_time = lambda: time.time() * 1000.0

def fibonacci_sphere(samples=1000,randomize=False):
    # creates almost uniformly distributed grid of points over unit sphere
    # INPUT:
    #   samples: int - number of points to be generated
    #   randomize: bool - if True shuffles the points
    # OUTPUT:
    #   points: list nx3 - points on the ficonacci_sphere
    
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x,y,z])

    return points



unif_pts = fibonacci_sphere(1000)


def signed_views(pred_views_all, pred_signs, numOutputs):
    signed_views_list = []
    for i in range(numOutputs):
        views = tf.slice(pred_views_all,[0,3*i],[-1,3])
        sign_ids = pred_signs[i]
        signs = 1-2*tf.concat([tf.mod(sign_ids,2), tf.mod(sign_ids/2,2), tf.mod(sign_ids/4,2)], axis=1)
        signed_views_list.append(tf.to_float(signs)*views)
    return tf.concat(signed_views_list, axis = 1)

def create_loss_mult_tf(pred_views_all, numOutputs, labels, sizes, t, weightDecay, cosine):
    loss_list = []
    loss_n_list = []
    accuracy_list = []
    for i in range(numOutputs):
        pred_views = tf.slice(pred_views_all,[0,3*i],[-1,3])
        loss, accuracy = create_loss_tf(pred_views, labels[i], sizes[i], t, cosine)
        
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        
    loss = tf.stack(loss_list)
    accuracy = tf.stack(accuracy_list)
    
    ### reg
    regularizer = tf.contrib.layers.l2_regularizer(scale=weightDecay)
    regVariables = tf.get_collection('weight_decay_loss')
    regTerm = tf.contrib.layers.apply_regularization(regularizer, regVariables)
    
    lossGraph = tf.reduce_mean(loss) + regTerm
    
    return lossGraph, loss, accuracy
    
def create_loss_tf(pred_views, labels, sizes, t, cosine):
    if cosine:
        pred_views_exp = tf.tile(tf.expand_dims(pred_views,1), sizes)
        min_dist = tf.reduce_min(1-tf.reduce_sum(tf.multiply(pred_views_exp, labels), axis = 2), axis=1)
        threshold = tf.fill(tf.shape(min_dist),t)
        return tf.reduce_mean(min_dist), tf.divide(tf.count_nonzero(tf.math.greater(threshold, min_dist)),tf.shape(min_dist, out_type=tf.int64)[0])
    else:
        return tf.losses.mean_squared_error(labels, pred_views)
    
def create_classification_loss_mult_tf(predLogits, numOutputs, labels):
    pred_signs_list = []
    loss_list = []
    for i in range(numOutputs):
        logits = tf.slice(predLogits, [0,8*i],[-1,8])
        loss, pred_sign = create_classification_loss_tf(logits, labels[i])
        loss_list.append(loss)
        pred_signs_list.append(pred_sign)
    loss = tf.stack(loss_list)
    pred_signs = tf.stack(pred_signs_list)
    loss_class = tf.reduce_mean(loss)
    return loss_class, pred_signs

def create_classification_loss_tf(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    xentropyloss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    _, logitsIndexs = tf.nn.top_k(logits)
    return xentropyloss, logitsIndexs

def create_training(lossGraph, learningRate, minLearningRate, learningDecayFactor, learningDecayRate, global_step):
    learningRateExp = tf.train.exponential_decay(learningRate, global_step, learningDecayRate, learningDecayFactor, staircase=True)
    learningRateExp = tf.maximum(learningRateExp, minLearningRate)
    optimizer = tf.train.AdamOptimizer(learning_rate =learningRateExp)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(lossGraph, global_step=global_step)
    return train_op, learningRateExp

def create_loss_metrics_mult(loss, accuracy, VQLoss, numOutputs):
    loss_list = []
    acc_list = []
    vq_list = []
    train_list = []
    test_list = []
    vq_op_list = []
    for i in range(numOutputs):
        accumLoss, accumAccuracy, accumVQLoss, accumOpsTrain, accumOpsTest, accumVQLossOp = create_loss_metrics(loss[i], accuracy[i], VQLoss[i])
        loss_list.append(accumLoss)
        acc_list.append(accumAccuracy)
        vq_list.append(accumVQLoss)
        train_list.append(accumOpsTrain)
        test_list.append(accumOpsTest)
        vq_op_list.append(accumVQLossOp)
    
    accumTrainOp = tf.tuple(train_list)
    accumTestOp = tf.tuple(test_list)
    accumVQLossOp = tf.tuple(vq_op_list)
    return loss_list, acc_list, vq_list, accumTrainOp, accumTestOp, accumVQLossOp

def create_loss_metrics(loss, accuracy, VQLoss):
    accumLoss, accumLossOp = tf.metrics.mean(loss, name = 'metrics')
    accumAccuracy, accumAccuracyOp = tf.metrics.mean(accuracy, name = 'metrics')
    accumVQLoss, accumVQLossOp = tf.metrics.mean(VQLoss, name = 'metrics')
    accumOpsTrain = tf.group(accumLossOp, accumAccuracyOp)
    accumOpsTest = tf.group(accumLossOp, accumAccuracyOp)
    return accumLoss, accumAccuracy, accumVQLoss, accumOpsTrain, accumOpsTest, accumVQLossOp

def create_summaries(accumLoss, accumAccuracy, accumVQLoss, VQs, numOutputs):
    train_list = []
    test_list = []
    
    for i in range(numOutputs):
        # training
        train_list.append(tf.summary.scalar('Train VQ ' + VQs[i] + ' cosine distance', accumLoss[i]))
        train_list.append(tf.summary.scalar('Train VQ ' + VQs[i] + ' accuracy', accumAccuracy[i]))
        train_list.append(tf.summary.scalar('Train VQ ' + VQs[i] + ' loss', accumVQLoss[i]))
        # testing
        test_list.append(tf.summary.scalar('Val VQ ' + VQs[i] + ' cosine distance', accumLoss[i]))
        test_list.append(tf.summary.scalar('Val VQ ' + VQs[i] + ' accuracy', accumAccuracy[i]))
        test_list.append(tf.summary.scalar('Val VQ ' + VQs[i] + ' loss', accumVQLoss[i]))
        
    accumLoss = tf.reduce_mean(tf.stack(accumLoss))
    accumAccuracy = tf.reduce_mean(tf.stack(accumAccuracy))
    accumVQLoss = tf.reduce_mean(tf.stack(accumVQLoss))
    train_list.append(tf.summary.scalar('Train total cosine distance', accumLoss))
    train_list.append(tf.summary.scalar('Train total accuracy', accumAccuracy))
    test_list.append(tf.summary.scalar('Train total VQ loss', accumVQLoss))
    # testing
    test_list.append(tf.summary.scalar('Val total cosine distance', accumLoss))
    test_list.append(tf.summary.scalar('Val total accuracy', accumAccuracy))
    test_list.append(tf.summary.scalar('Val total VQ loss', accumVQLoss))
    train_list.append(tf.summary.scalar('learninRate', learningRateExp))
    train_list.append(tf.summary.scalar('BN_Momentum', BNMomentum))
        
    trainingSummary = tf.summary.merge(train_list)
    testSummary = tf.summary.merge(test_list)
    
    return trainingSummary, testSummary, accumLoss, accumAccuracy, accumVQLoss

def logfolder_name(args):
    
    normal_string = ''
    if args.useNormals:
        normal_string = 'wNormals_'
    
    if args.cates == None or ',' in args.cates:
        cate_string = ''
    else:
        cate_string = args.cates + '_'
    
    aug_string = 'aug'
    if not args.augment:
        aug_string = 'noaug'
    elif args.rotation_axis != '012':
        aug_string += 'rot'+args.rotation_axis
    aug_string += '_'
    if args.noise:
        aug_string += 'noise_'
    if args.symmetric_deformations:
        aug_string += 'symdef_'

    return args.logFolder + '/numPts'+str(args.maxNumPts)+'_LR' + str(args.initLearningRate) + '_LDR' + str(args.learningDecayRate) + '_LDF' + str(args.learningDecayFactor)+ '_grow' + str(args.grow)+'_bs'+str(args.batchSize) + '_' + cate_string + normal_string + aug_string + args.affix
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to train MCCNN for View Point Optimization of point clouds (SHREC15)')
    parser.add_argument('--logFolder', default='log', help='Folder of the output models (default: log)')
    parser.add_argument('--folders','--f', default='ModelNet40', help='data folders')
    parser.add_argument('--model', default='MCSphere_mult', help='model (default: MCRegV_small)')
    parser.add_argument('--activation', default = 'relu', help = 'activation function in the model')
    parser.add_argument('--grow', default=64, type=int, help='Grow rate (default: 64)')
    parser.add_argument('--batchSize', default=8, type=int, help='Batch size  (default: 8)')
    parser.add_argument('--maxEpoch', default=201, type=int, help='Max Epoch  (default: 201)')
    parser.add_argument('--initLearningRate', default=0.001, type=float, help='Init learning rate  (default: 0.005)')
    parser.add_argument('--learningDecayFactor', default=0.5, type=float, help='Learning decay factor (default: 0.5)')
    parser.add_argument('--learningDecayRate', default=20, type=int, help='Learning decay rate  (default: 20 Epochs)')
    parser.add_argument('--minLearningRate', default=0.00001, type=float, help='Minimum Learning rate (default: 0.00001)')
    parser.add_argument('--weightDecay', default=0.0, type=float, help='WeightDecay ( default:0.0(disabled))')
    parser.add_argument('--useDropOut', action='store_true', help='Use drop out (default: False)')
    parser.add_argument('--dropOutKeepProb', default=0.5, type=float, help='Keep neuron probabillity drop out  (default: 0.5)')
    parser.add_argument('--useDropOutConv', action='store_true', help='Use drop out in convolution layers (default: False)')
    parser.add_argument('--dropOutKeepProbConv', default=0.8, type=float, help='Keep neuron probabillity drop out in convolution layers (default: 0.8)')
    parser.add_argument('--augment', action='store_true', help='Augment data using rotations (default: False)')
    parser.add_argument('--noise', action='store_true', help='Augment data using noise (default: False)')
    parser.add_argument('--symmetric_deformations', action='store_true', help='Augment data using symmetric deformations (default: False)')
    parser.add_argument('--gpu', default='0', help='GPU (default: 0)')
    parser.add_argument('--restore', action='store_true', help='Restore previous model (default: False)')
    parser.add_argument('--no_eval', action='store_true', help='Do not run the evaluation at the end')
    parser.add_argument('--resolution', default=1024, type=int, help='Resolution used for rendering images(default: 1024)')
    parser.add_argument('--trackVQLoss', action='store_true', help='track loss in View Quality for the training set, slower.(default:False)')
    parser.add_argument('--cosineLoss', action='store_false', help='Use the cosine distance instead of MSE.(default: True)')
    parser.add_argument('--VQ', default='4.1,5.1,7.1,8.1', help='View Quality measure used')
    parser.add_argument('--label_threshold', default = 0.01, type=float, help = 'Relative threshold for labels created (only used for performance measure, not for loss (default: 0.01)')
    parser.add_argument('--useBRN', action = 'store_true', help='Use BatchRenormalisation (default: False)')
    parser.add_argument('--initBNMom', default = 0.5, type=float, help = 'initial Momentum for Batch (Re-) Normalization')
    parser.add_argument('--affix', default = '', help='string attached to the logfolder name')
    parser.add_argument('--no_categories',action='store_false', help='Ignore category filtering with respect to Pascel3D (default: False)')
    parser.add_argument('--cates', default = None, help=' string containing the categories to use, separated by commas')
    parser.add_argument('--maxNumPts', default = 1e6, type = int, help = ' maximum number of point per model (default: 2000)')
    parser.add_argument('--rotation_axis', default ='012', help = 'axis aroung which to augment, string (default: 2 (z axis))')
    parser.add_argument('--pts_source', default = 'pts_unif_own', help= 'name of folder containing the point clouds')
    parser.add_argument('--useNormals', action = 'store_true', help = 'use normals as input features')
    args = parser.parse_args()
    
    
    if not os.path.exists(args.logFolder): os.mkdir(args.logFolder)
    
    args.logFolder = logfolder_name(args)
    
    folders = args.folders.split(',')
    


    if args.cates != None:
        args.cates = args.cates.split(',')
    VQ = args.VQ.split(',')
    if 'UV' in VQ:
        uv_ind = VQ.index('UV')
    numVQs = len(VQ)
    
    

    train_to_val_ratio = 5

        
    #Create log folder.
    if not os.path.exists(args.logFolder): os.mkdir(args.logFolder)
    #os.system('cp models/%s.py %s' % (args.model, args.logFolder))
    #os.system('cp VQ_train_multMLP.py %s' % (args.logFolder))
    logFile = args.logFolder+"/log.txt"

    #Create Folder for results
    resultFolder = args.logFolder + '/results_val'
    if not os.path.exists(resultFolder): os.mkdir(resultFolder)
    with open(resultFolder + '/VQs.txt','w') as outFile:
        for vq in VQ:
            outFile.write(vq+'\n')
    #Write execution info.
    with open(logFile, "a") as myFile:
        myFile.write("Model: "+args.model+"\n")
        myFile.write("Grow: "+str(args.grow)+"\n")
        myFile.write("VQ: "+ str(VQ) +"\n")
        myFile.write("BatchSize: "+str(args.batchSize)+"\n")
        myFile.write("MaxEpoch: "+str(args.maxEpoch)+"\n")
        myFile.write("WeightDecay: "+str(args.weightDecay)+"\n")
        myFile.write("InitLearningRate: "+str(args.initLearningRate)+"\n")
        myFile.write("LearningDecayFactor: "+str(args.learningDecayFactor)+"\n")
        myFile.write("LearningDecayRate: "+str(args.learningDecayRate)+"\n")
        myFile.write("MinLearningRate: "+str(args.minLearningRate)+"\n")
        myFile.write("UseDropOut: "+str(args.useDropOut)+"\n")
        myFile.write("DropOutKeepProb: "+str(args.dropOutKeepProb)+"\n")
        myFile.write("UseDropOutConv: "+str(args.useDropOutConv)+"\n")
        myFile.write("DropOutKeepProbConv: "+str(args.dropOutKeepProbConv)+"\n")
        myFile.write('Resolution: '+str(args.resolution)+'\n')
        myFile.write("Augment: "+str(args.augment)+"\n")
        myFile.write("Noise: "+str(args.noise)+"\n")
        myFile.write("Symmetric Deformations: "+str(args.symmetric_deformations)+"\n")

    print("Model: "+args.model)
    print("Grow: "+str(args.grow))
    print("VQs: "+ str(VQ))
    print("BatchSize: "+str(args.batchSize))
    print("MaxEpoch: "+str(args.maxEpoch))
    print("WeightDecay: "+str(args.weightDecay))
    print("InitLearningRate: "+str(args.initLearningRate))
    print("LearningDecayFactor: "+str(args.learningDecayFactor))
    print("LearningDecayRate: "+str(args.learningDecayRate))
    print("minLearningRate: "+str(args.minLearningRate))
    print("UseDropOut: "+str(args.useDropOut))
    print("DropOutKeepProb: "+str(args.dropOutKeepProb))
    print("UseDropOutConv: "+str(args.useDropOutConv))
    print("DropOutKeepProbConv: "+str(args.dropOutKeepProbConv))
    print('Resolution: '+str(args.resolution))
    print("Augment: "+str(args.augment))
    print("Noise: "+str(args.noise))
    print("Symmetric Deformations: "+str(args.symmetric_deformations))

    #Load the model
    model = importlib.import_module(args.model)

    #Get train and test datasets
    maxStoredPoints = args.maxNumPts

    mTrainDataSet = VQDataSet('train', maxStoredPoints, args.batchSize, args.augment, args.noise, args.symmetric_deformations, VQs = VQ, folders =  folders, label_threshold = args.label_threshold, filter_categories=args.no_categories, categories = args.cates, pts_source = args.pts_source, useNormalsAsFeatures = args.useNormals, rotation_axis = args.rotation_axis)
    mValDataSet = VQDataSet('val', maxStoredPoints, args.batchSize, args.augment, VQs=VQ, folders =  folders,filter_categories=args.no_categories, categories = args.cates, pts_source = args.pts_source, useNormalsAsFeatures = args.useNormals, rotation_axis = args.rotation_axis)
    mTestDataSet = VQDataSet('test', maxStoredPoints, args.batchSize, args.augment, VQs=VQ, folders =  folders,filter_categories=args.no_categories, categories = args.cates, pts_source = args.pts_source, useNormalsAsFeatures = args.useNormals, rotation_axis = args.rotation_axis)
    
    numTrainModels = mTrainDataSet.get_num_models()
    numTestModels = mTestDataSet.get_num_models()
    numBatchesXEpoch = numTrainModels/args.batchSize
    if numTrainModels%args.batchSize != 0:
        numBatchesXEpoch = numBatchesXEpoch + 1
    numValModels = mValDataSet.get_num_models()
    numTestModels = mTestDataSet.get_num_models()
    print("Train models: " + str(numTrainModels))
    print("Val models: " + str(numValModels))

    #Create variable and place holders
    if args.useNormals:
        numInputFeatures = 3
    else:
        numInputFeatures = 1
    
    global_step = tf.Variable(0, name='global_step', trainable=False)
    inPts = tf.placeholder(tf.float32,[None,3], name='Points')
    inFeatures = tf.placeholder(tf.float32,[None,numInputFeatures], name = 'Features')        
    inBatchIds = tf.placeholder(tf.int32, [None,1], name = 'Batchids')
    inPtHier_tf = [inPts, inFeatures, inBatchIds]
    isTraining = tf.placeholder(tf.bool, shape=(), name = 'isTraining')
    inLabels = [tf.placeholder(tf.float32, [None,None,3], name = 'Labels_' + str(i)) for i in range(numVQs)]
    inLabels_mult = [tf.placeholder(tf.float32, [None,None,3], name = 'Labels_mult_' + str(i)) for i in range(numVQs)]
    inSizes = tf.placeholder(tf.int32, [numVQs,3], name = 'inSizes')
    inSizes_mult = tf.placeholder(tf.int32, [numVQs,3], name = 'inSizes_mult')
    keepProbConv = tf.placeholder(tf.float32, name= 'keepProbConv') 
    keepProbFull = tf.placeholder(tf.float32, name = 'keepProbFull')
    inVQLoss = tf.placeholder(tf.float32, [numVQs], name = 'VQLoss') 
    inSigns = tf.placeholder(tf.int32, [numVQs, None], name = 'SignLabels')
    t = tf.constant(0.1)


    #Create the network
    useRenorm = args.useBRN
    BNMomentum = 0.99 #BN_decay(args.initBNMom, global_step, numBatchesXEpoch, 0.9)
    
    pred_views_abs, logits = model.create_network(inPts, inFeatures, inBatchIds, args.batchSize, numInputFeatures, args.grow, 3, numVQs, 
        isTraining, keepProbConv, keepProbFull, args.useDropOutConv, args.useDropOut, activation = args.activation)
    #Create Loss
    lossClass, pred_signs = create_classification_loss_mult_tf(logits, numVQs, inSigns)
    pred_views = signed_views(pred_views_abs, pred_signs, numVQs)
    lossPred, _, _ = create_loss_mult_tf(pred_views_abs, numVQs, tf.abs(inLabels), inSizes, t, args.weightDecay, args.cosineLoss)
    _, loss, accuracy = create_loss_mult_tf(pred_views, numVQs, inLabels_mult, inSizes_mult, t, args.weightDecay, args.cosineLoss)
    
    lossGraph = lossPred+lossClass
    
    
    #Create training
    trainning, learningRateExp = create_training(lossGraph, 
        args.initLearningRate, args.minLearningRate, args.learningDecayFactor, 
        args.learningDecayRate*numBatchesXEpoch, global_step)

    
    #Create loss metrics
    accumLoss, accumAccuracy, accumVQLoss, accumTrain, accumTest, accumVQLossOp = create_loss_metrics_mult(loss, accuracy, inVQLoss, numVQs)
    metricsVars = tf.contrib.framework.get_variables('metrics', collection=tf.GraphKeys.LOCAL_VARIABLES)
    resetMetrics = tf.variables_initializer(metricsVars)

    #Create Summaries
    trainingSummary, testSummary, accumLoss, accumAccuracy, accumVQLoss = create_summaries(accumLoss, accumAccuracy, accumVQLoss, VQ, numVQs)

    
    #Create init variables 
    init = tf.global_variables_initializer()
    initLocal = tf.local_variables_initializer()

    #create the saver
    saver = tf.train.Saver()
    
    #Create session
    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=args.gpu)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    #Create the summary writer
    summary_writer = tf.summary.FileWriter(args.logFolder, sess.graph)
    summary_writer.add_graph(sess.graph)
    
    #Init variables
    sess.run(init, {isTraining: True})
    sess.run(initLocal, {isTraining: True})
    np.random.seed(int(time.time()))
    

    TestLossList = []

    
    if args.restore == True:
        loader = tf.train.import_meta_graph(args.logFolder+"/model.ckpt.meta")
        loader.restore(sess, args.logFolder+"/model.ckpt")
        #TestLossList = list(np.genfromtxt(resultFolder+'/TestLoss.txt', delimiter = ',').reshape(-1,numVQs))
        #mTrainDataSet.ptHIndex_ = int(np.genfromtxt(args.logFolder+'/ptH_log', delimiter = ','))
    
    #MyGL = GLScene(args.resolution, args.resolution)

    #Train
    print(''); print('########## Training'); print('')
    for epoch in range(args.maxEpoch):
        startEpochTime = current_milli_time()
        startTrainTime = current_milli_time()
        
        PtHier = []
        epochStep = 0
        lossInfoCounter = 0
        visStep = 0
        
        lossAccumValue = 0.0
        lossTotal = 0.0
        
        distance = 0.0
        distanceAccumValue = 0.0
        distanceTotal = 0.0
        accuracyTotal = 0.0
        processedModels = 0

        #Iterate over all the train files
        mTrainDataSet.start_iteration()
            
        while mTrainDataSet.has_more_batches():
            currbatchSize, pts, features, batchIds, labels, labels_mult, signs, modelList, modelParamsList, areas, referenceValuesList, invRotationMatrix,_, vqs3D = mTrainDataSet.get_next_batch()
            processedModels += currbatchSize
            
            feed_dict = create_feed_dict_mult(inLabels, labels, inLabels_mult, labels_mult, inSizes, inSizes_mult, inPtHier_tf, [pts, features, batchIds], keepProbConv, args.dropOutKeepProbConv, keepProbFull, args.dropOutKeepProb, numVQs)
            feed_dict[isTraining] = True
            feed_dict[inSigns] = signs
                
            Pred_views, _, _, step = sess.run([pred_views_abs, trainning, accumTrain, global_step], feed_dict)
            #print('Pred_views:', Pred_views)
            if args.trackVQLoss:
                if args.augment:
                    for i in range(len(Pred_views)):
                        for vq_i in range(numVQs):
                            ind = range(vq_i*3, (vq_i+1)*3)
                            Pred_views[i,ind] = np.dot(Pred_views[i,ind].reshape(1,3),invRotationMatrix[i])
                Loss = create_mult_loss_approx(Pred_views, modelList, vqs3D, currbatchSize, VQ, unif_pts)
                lossAccumValue += np.mean(Loss)
                sess.run(accumVQLossOp, {inVQLoss : Loss})
                lossTotal += np.mean(Loss)           
            
            lossInfoCounter += 1

            if lossInfoCounter == 1000/args.batchSize or not mTrainDataSet.has_more_batches():
                endTrainTime = current_milli_time()  
                trainingSumm, distanceAccumValue, accuracyAccumValue = sess.run([trainingSummary, accumLoss, accumAccuracy])
                summary_writer.add_summary(trainingSumm, step)
                summary_writer.flush()
                sess.run(resetMetrics)                 
                if args.trackVQLoss:
                    visualize_progress(min(epochStep, numBatchesXEpoch), numBatchesXEpoch, "Distance: %.6f | VQLoss: %.6f | Time: %4ds " 
                                       %(distanceAccumValue, lossAccumValue/float(lossInfoCounter), (endTrainTime-startTrainTime)/1000.0)) 
                else:
                    visualize_progress(min(epochStep, numBatchesXEpoch), numBatchesXEpoch, "Dist Loss: %.6f | Acc: %.2f | Time: %2ds " 
                                       %(distanceAccumValue, accuracyAccumValue*100, (endTrainTime-startTrainTime)/1000.0))

                with open(logFile, "a") as myfile:
                    if args.trackVQLoss:
                        myfile.write("Step: %6d (%4d) | Distance: %.6f | VQLoss: %.6f\n" % (step, epochStep, distanceAccumValue, lossAccumValue/float(lossInfoCounter)))
                    else:
                        myfile.write("Step: %6d (%4d) | Distance: %.6f | Accuracy: %.2f \n" % (step, epochStep, distanceAccumValue, accuracyAccumValue*100))
                
                distanceTotal += distanceAccumValue * processedModels
                accuracyTotal += accuracyAccumValue * processedModels
                processedModels = 0
                lossInfoCounter = 0
                lossAccumValue = 0.0
                startTrainTime = current_milli_time()
                visStep += 1
            epochStep += 1
            
        distanceTotal = distanceTotal/numTrainModels
        accuracyTotal = accuracyTotal/numTrainModels
        lossTotal = lossTotal/epochStep
        
        endEpochTime = current_milli_time()   
        
        
        if args.trackVQLoss:
            print("Epoch %3d  Train Time: %.2fs | Train Distance: %.6f | Train VQ-Loss: %.6f" %(epoch, (endEpochTime-startEpochTime)/1000.0, distanceTotal, lossTotal))
        else:
            print("Epoch %3d  Train Time: %.2fs |  Dist: %.6f | Acc: %.2f " %(epoch, (endEpochTime-startEpochTime)/1000.0, distanceTotal, accuracyTotal*100))
            
        with open(logFile, "a") as myfile:
            if args.trackVQLoss:
                myfile.write("Epoch %3d  Train Time: %.2fs  |  Train Distance: %.6f | Train VQ-Loss: %.6f\n" %(epoch, (endEpochTime-startEpochTime)/1000.0, distanceTotal, lossTotal))
            else:
                myfile.write("Epoch %3d  Train Time: %.2fs  |Distance: %.6f | Accuracy: %.2f\n" %(epoch, (endEpochTime-startEpochTime)/1000.0, distanceTotal, accuracyTotal*100))

        if epoch%train_to_val_ratio==0:
            saver.save(sess, args.logFolder+"/model.ckpt")
        
            startTestTime = current_milli_time()  
            #Test data
            accumTestLoss = 0.0
            TestModels = []
            it = 0
            mValDataSet.start_iteration()
            
            while mValDataSet.has_more_batches():

                currbatchSize, pts, features, batchIds, labels, labels_mult, signs, modelList, modelParamsList, areas, referenceValuesList, invRotationMatrix ,_, vqs3D= mValDataSet.get_next_batch()
                feed_dict = create_feed_dict_mult(inLabels, labels, inLabels_mult, labels_mult, inSizes, inSizes_mult, inPtHier_tf, [pts, features, batchIds], keepProbConv, 1.0, keepProbFull, 1.0, numVQs)
                feed_dict[isTraining] = False
                feed_dict[inSigns] = signs
                Pred_views, _= sess.run([pred_views, accumTest], feed_dict)
                
                if args.augment:
                    for i in range(len(Pred_views)):
                        for vq_i in range(numVQs):
                            ind = range(vq_i*3, (vq_i+1)*3)
                            Pred_views[i,ind] = np.dot(Pred_views[i,ind].reshape(1,3),invRotationMatrix[i])
                
                Loss = create_mult_loss_approx(Pred_views, modelList, vqs3D, currbatchSize, VQ, unif_pts)
                #Loss = 0
                sess.run(accumVQLossOp, {inVQLoss : Loss})
                accumTestLoss += np.mean(Loss)*currbatchSize
                
                if it%(400/args.batchSize) == 0 or not mValDataSet.has_more_batches():
                    visualize_progress(it, numValModels/currbatchSize)
                
                it += 1
            
            TestSumm, accumTestDistance, accumTestAccuracy = sess.run([testSummary, accumLoss, accumAccuracy])
            summary_writer.add_summary(TestSumm, step)
            summary_writer.flush()
            sess.run(resetMetrics)
            
            accumTestLoss = accumTestLoss/numValModels
            TestLossList.append(accumTestLoss)
            endTestTime = current_milli_time()  
            
            print("Val Time: %.2fs | VQ-Loss: %.6f | Dist: %.6f | Acc: %.2f" % ((endTestTime-startTestTime)/1000.0, np.mean(accumTestLoss), accumTestDistance, accumTestAccuracy*100))

    
    # after training save Val results
    if not args.no_eval:
        
        sess.run(resetMetrics)
        print(''); print('########## Evaluation'); print('')
        
        startTestTime = current_milli_time()  
        #Test data
        accumTestLoss = 0.0
        TestModels = []
        TestLoss = [[] for vqi in range(numVQs)]
        TestViews = [[] for vqi in range(numVQs)]
        it = 0
        mValDataSet.start_iteration()
        mValDataSet.batchSize_ = 1
        
        while mValDataSet.has_more_batches():

            currbatchSize, pts, features, batchIds, labels, labels_mult, signs, modelList, modelParamsList, areas, referenceValuesList, invRotationMatrix ,_, vqs3D= mValDataSet.get_next_batch()
            feed_dict = create_feed_dict_mult(inLabels, labels, inLabels_mult, labels_mult, inSizes, inSizes_mult, inPtHier_tf, [pts, features, batchIds], keepProbConv, 1.0, keepProbFull, 1.0, numVQs)
            feed_dict[isTraining] = False
            feed_dict[inSigns] = signs
            Pred_views, _= sess.run([pred_views, accumTest], feed_dict)
            
            
            TestModels.append(modelList[0])
            
            if args.augment:
                for i in range(len(Pred_views)):
                    for vq_i in range(numVQs):
                        ind = range(vq_i*3, (vq_i+1)*3)
                        Pred_views[i,ind] = np.dot(Pred_views[i,ind].reshape(1,3),invRotationMatrix[i])
                        TestViews[vq_i].append(Pred_views[:,ind][i])
            Loss = create_mult_loss_approx(Pred_views, modelList, vqs3D, currbatchSize, VQ, unif_pts)
            #Loss = 0
            sess.run(accumVQLossOp, {inVQLoss : Loss})
            accumTestLoss += np.mean(Loss)*currbatchSize
            
            for vq_i in range(numVQs):
                TestLoss[vq_i].append(Loss[vq_i])
            
            if it%(400/args.batchSize) == 0 or not mValDataSet.has_more_batches():
                visualize_progress(it, numValModels/currbatchSize)
            
            it += 1
        
        TestSumm, accumTestDistance, accumTestAccuracy = sess.run([testSummary, accumLoss, accumAccuracy])
        
        accumTestLoss = accumTestLoss/numValModels
        TestLossList.append(accumTestLoss)
        endTestTime = current_milli_time()  
        
        print("Val Time: %.2fs | VQ-Loss: %.6f | Dist: %.6f | Acc: %.2f" % ((endTestTime-startTestTime)/1000.0, np.mean(accumTestLoss), accumTestDistance, accumTestAccuracy*100))

        with open(resultFolder+'/ValDistRes.txt', "w") as myfile:
            myfile.write(str(accumTestDistance) + '\n')
        with open(resultFolder+'/TestLoss.txt', "w") as myfile:
            myfile.write(np.array2string(np.mean(TestLoss,axis=1),precision=2)[1:-1] + '\n')
        with open(resultFolder+'/ValAccuracy.txt', "w") as myfile:
            myfile.write(str(accumTestAccuracy) + '\n')

        for vq_i, vq in enumerate(VQ):
            np.savetxt(resultFolder+'/Loss_VQ'+vq +'.txt', TestLoss[vq_i], delimiter = ',', fmt = '%s')
            #np.savetxt(resultFolder+'/VQ'+vq +'.txt', TestVQ[vq_i], delimiter = ',', fmt = '%s')
            #np.savetxt(resultFolder+'/Distance.txt', TestDistance, delimiter = ',', fmt = '%s')
            np.savetxt(resultFolder+'/Models.txt', TestModels, delimiter = ',', fmt = '%s')
            #np.savetxt(resultFolder+'/Result_VQ'+vq +'.txt', ResultTest[vq_i], delimiter = ',', fmt = '%s')
            np.savetxt(resultFolder+'/Views_VQ'+vq +'.txt', TestViews[vq_i], delimiter = ',', fmt = '%s')
            
        sess.run(resetMetrics)
        print(''); print('########## Testing'); print('')
        resultFolder = args.logFolder + '/results_test'
        if not os.path.exists(resultFolder): os.mkdir(resultFolder)
        with open(resultFolder + '/VQs.txt','w') as outFile:
            for vq in VQ:
                outFile.write(vq+'\n')
        startTestTime = current_milli_time()  
        #Test data
        accumTestLoss = 0.0
        TestModels = []
        TestLoss = [[] for vqi in range(numVQs)]
        TestViews = [[] for vqi in range(numVQs)]
        it = 0
        mTestDataSet.start_iteration()
        mTestDataSet.batchSize_ = 1
        
        while mTestDataSet.has_more_batches():

            currbatchSize, pts, features, batchIds, labels, labels_mult, signs, modelList, modelParamsList, areas, referenceValuesList, invRotationMatrix ,_, vqs3D= mTestDataSet.get_next_batch()
            feed_dict = create_feed_dict_mult(inLabels, labels, inLabels_mult, labels_mult, inSizes, inSizes_mult, inPtHier_tf, [pts, features, batchIds], keepProbConv, 1.0, keepProbFull, 1.0, numVQs)
            feed_dict[isTraining] = False
            feed_dict[inSigns] = signs
            Pred_views, _= sess.run([pred_views, accumTest], feed_dict)
            
            
            TestModels.append(modelList[0])
            
            if args.augment:
                for i in range(len(Pred_views)):
                    for vq_i in range(numVQs):
                        ind = range(vq_i*3, (vq_i+1)*3)
                        Pred_views[i,ind] = np.dot(Pred_views[i,ind].reshape(1,3),invRotationMatrix[i])
                        TestViews[vq_i].append(Pred_views[:,ind][i])
            Loss = create_mult_loss_approx(Pred_views, modelList, vqs3D, currbatchSize, VQ, unif_pts)
            #Loss = 0
            sess.run(accumVQLossOp, {inVQLoss : Loss})
            accumTestLoss += np.mean(Loss)*currbatchSize
            
            for vq_i in range(numVQs):
                TestLoss[vq_i].append(Loss[vq_i])
            
            if it%(400/args.batchSize) == 0 or not mTestDataSet.has_more_batches():
                visualize_progress(it, numTestModels/currbatchSize)
            
            it += 1
        
        TestSumm, accumTestDistance, accumTestAccuracy = sess.run([testSummary, accumLoss, accumAccuracy])
        
        accumTestLoss = accumTestLoss/numTestModels
        TestLossList.append(accumTestLoss)
        endTestTime = current_milli_time()  
        
        print("Test Time: %.2fs | VQ-Loss: %.6f | Dist: %.6f | Acc: %.2f" % ((endTestTime-startTestTime)/1000.0, np.mean(accumTestLoss), accumTestDistance, accumTestAccuracy*100))

        with open(resultFolder+'/TestDistRes.txt', "w") as myfile:
            myfile.write(str(accumTestDistance) + '\n')
        with open(resultFolder+'/TestLoss.txt', "w") as myfile:
            myfile.write(np.array2string(np.mean(TestLoss,axis=1),precision=2)[1:-1] + '\n')
        with open(resultFolder+'/TestAccuracy.txt', "w") as myfile:
            myfile.write(str(accumTestAccuracy) + '\n')

        for vq_i, vq in enumerate(VQ):
            np.savetxt(resultFolder+'/Loss_VQ'+vq +'.txt', TestLoss[vq_i], delimiter = ',', fmt = '%s')
            #np.savetxt(resultFolder+'/VQ'+vq +'.txt', TestVQ[vq_i], delimiter = ',', fmt = '%s')
            #np.savetxt(resultFolder+'/Distance.txt', TestDistance, delimiter = ',', fmt = '%s')
            np.savetxt(resultFolder+'/Models.txt', TestModels, delimiter = ',', fmt = '%s')
            #np.savetxt(resultFolder+'/Result_VQ'+vq +'.txt', ResultTest[vq_i], delimiter = ',', fmt = '%s')
            np.savetxt(resultFolder+'/Views_VQ'+vq +'.txt', TestViews[vq_i], delimiter = ',', fmt = '%s')
        
    print('[done]')
    with open(logFile, "a") as myfile:
        myfile.write('[done]')
