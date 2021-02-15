'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    \brief Code to train viewpoint prediction using GL

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
from train_ops import create_mult_loss_approx, BN_decay, create_feed_dict_gauss
from VQDataSet_all_VQs_GL import VQDataSet
#from VQs import getAs, getPz, getProb, getFaceIds, getIds, vq4, vq5, vq7, vq8, vq12, vq14
#from Application import GLScene

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


def create_loss_mult_tf(pred_views_all, numOutputs, inSpheres, inVQs3d, std, offset, weightDecay, cosine):
    pred_views_norm_list = []
    loss_list = []
    loss_n_list = []
    for i in range(numOutputs):
        pred_views = tf.slice(pred_views_all,[0,3*i],[-1,3])
        pred_views_norm = tf.nn.l2_normalize(pred_views, axis = 1)
        loss= create_loss_tf(pred_views_norm, inSpheres, inVQs3d[i], std, offset, cosine)
        loss_normalizing = tf.losses.mean_squared_error(pred_views_norm, pred_views)
        
        pred_views_norm_list.append(pred_views_norm)
        loss_list.append(loss)
        loss_n_list.append(loss_normalizing)
        
    pred_views_norm = tf.concat(pred_views_norm_list, axis = 1)
    loss = tf.stack(loss_list)
    loss_normalizing = tf.stack(loss_n_list)
    
    ### reg
    #regularizer = tf.contrib.layers.l2_regularizer(scale=weightDecay)
    #regVariables = tf.get_collection('weight_decay_loss')
    #regTerm = tf.contrib.layers.apply_regularization(regularizer, regVariables)
    
    lossGraph = tf.reduce_mean(loss)  + args.normFactor*tf.reduce_mean(loss_normalizing)# + regTerm
    
    return lossGraph, pred_views_norm, loss
    
def create_loss_tf(pred_views, inSpheres, inVQs3d, std, offset, cosine):
    
    pred_views_formatted = tf.reshape(pred_views, [-1,1,3])
    gauss = gauss3d_tf(inSpheres,pred_views_formatted,std,offset)
    Z3d = gauss*inVQs3d
    
    max_inds_cols = tf.argmax(Z3d,axis=1)
    rows = tf.range(tf.cast(tf.shape(pred_views)[0], dtype = tf.int64))
    max_inds = tf.stack([rows, max_inds_cols],axis=1)
    labels = tf.gather_nd(inSpheres,max_inds)
    
    if cosine:
        return tf.losses.cosine_distance(labels, pred_views, axis=1)
    else:
        return tf.losses.mean_squared_error(labels, pred_views)

def gauss3d_tf(pts,mean,std,offset):
    return tf.exp(-tf.norm(pts-mean,axis=2)/(2*std**2))+offset

def create_training(lossGraph, learningRate, minLearningRate, learningDecayFactor, learningDecayRate, global_step):
    learningRateExp = tf.train.exponential_decay(learningRate, global_step, learningDecayRate, learningDecayFactor, staircase=True)
    learningRateExp = tf.maximum(learningRateExp, minLearningRate)
    optimizer = tf.train.AdamOptimizer(learning_rate =learningRateExp)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(lossGraph, global_step=global_step)
    return train_op, learningRateExp

def create_loss_metrics_mult(loss, VQLoss, numOutputs):
    loss_list = []
    acc_list = []
    vq_list = []
    train_list = []
    test_list = []
    vq_op_list = []
    for i in range(numOutputs):
        accumLoss, accumVQLoss, accumOpsTrain, accumOpsTest, accumVQLossOp = create_loss_metrics(loss[i], VQLoss[i])
        loss_list.append(accumLoss)
        vq_list.append(accumVQLoss)
        train_list.append(accumOpsTrain)
        test_list.append(accumOpsTest)
        vq_op_list.append(accumVQLossOp)
    
    accumTrainOp = tf.tuple(train_list)
    accumTestOp = tf.tuple(test_list)
    accumVQLossOp = tf.tuple(vq_op_list)
    return loss_list, vq_list, accumTrainOp, accumTestOp, accumVQLossOp

def create_loss_metrics(loss, VQLoss):
    accumLoss, accumLossOp = tf.metrics.mean(loss, name = 'metrics')
    accumVQLoss, accumVQLossOp = tf.metrics.mean(VQLoss, name = 'metrics')
    accumOpsTrain = tf.group(accumLossOp)
    accumOpsTest = tf.group(accumLossOp )
    return accumLoss, accumVQLoss, accumOpsTrain, accumOpsTest, accumVQLossOp

def create_summaries(accumLoss, accumVQLoss, VQs, numOutputs):
    train_list = []
    test_list = []
    
    for i in range(numOutputs):
        # training
        train_list.append(tf.summary.scalar('Train VQ ' + VQs[i] + ' cosine distance', accumLoss[i]))
        train_list.append(tf.summary.scalar('Train VQ ' + VQs[i] + ' loss', accumVQLoss[i]))
        # testing
        test_list.append(tf.summary.scalar('Val VQ ' + VQs[i] + ' cosine distance', accumLoss[i]))
        test_list.append(tf.summary.scalar('Val VQ ' + VQs[i] + ' loss', accumVQLoss[i]))
        
    accumLoss = tf.reduce_mean(tf.stack(accumLoss))
    accumVQLoss = tf.reduce_mean(tf.stack(accumVQLoss))
    train_list.append(tf.summary.scalar('Train total cosine distance', accumLoss))
    train_list.append(tf.summary.scalar('Train total VQ loss', accumVQLoss))
    # testing
    test_list.append(tf.summary.scalar('Val total cosine distance', accumLoss))
    test_list.append(tf.summary.scalar('Val total VQ loss', accumVQLoss))
    train_list.append(tf.summary.scalar('learninRate', learningRateExp))
    train_list.append(tf.summary.scalar('BN_Momentum', BNMomentum))
        
    trainingSummary = tf.summary.merge(train_list)
    testSummary = tf.summary.merge(test_list)
    
    return trainingSummary, testSummary, accumLoss, accumVQLoss


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
    parser.add_argument('--model', default='MCRegV_mult', help='model (default: MCRegV_small)')
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
    parser.add_argument('--fix_path', action='store_true', help='dont generate a subfolder for log')
    parser.add_argument('--resolution', default=1024, type=int, help='Resolution used for rendering images(default: 1024)')
    parser.add_argument('--trackVQLoss', action='store_true', help='track loss in View Quality for the training set, slower.(default:False)')
    parser.add_argument('--cosineLoss', action='store_false', help='Use the cosine distance instead of MSE.(default: True)')
    parser.add_argument('--VQ', default='4,5,7,8', help='View Quality measure used')
    parser.add_argument('--label_threshold', default = 0.01, type=float)
    parser.add_argument('--normFactor', default = 0.0, type = float)
    parser.add_argument('--useBRN', action = 'store_true')
    parser.add_argument('--initBNMom', default = 0.5, type=float)
    parser.add_argument('--affix', default = '')
    parser.add_argument('--no_categories',action='store_false')
    parser.add_argument('--maxNumPts', default = 1024, type = int)
    parser.add_argument('--cates', default = None)
    parser.add_argument('--rotation_axis', default ='012', help = 'axis aroung which to augment, string (default: 2 (z axis))')
    parser.add_argument('--smallrotations', action = 'store_true', help = 'use small rotations')
    parser.add_argument('--pts_source', default = 'pts_unif_own', help= 'name of folder containing the point clouds')
    parser.add_argument('--useNormals', action = 'store_true', help = 'use normals as input features')
    parser.add_argument('--std', default = 2, help = 'standard deviation of the gaussian filter')
    parser.add_argument('--offset', default = 1, help = 'offset of the gaussian filter')
    parser.add_argument('--tta', action = 'store_true', help = 'use test time augmentation')
    args = parser.parse_args()
    
    #if args.VQ == '4,5,7,8': args.no_categories = False
    
    folders = args.folders.split(',')
    if not os.path.exists(args.logFolder): os.mkdir(args.logFolder)
    if not args.fix_path:
        args.logFolder = logfolder_name(args)
    
    VQ = args.VQ.split(',')
    if args.cates != None:
        args.cates = args.cates.split(',')
    if 'UV' in VQ:
        uv_ind = VQ.index('UV')
    numVQs = len(VQ)
    
    
    if not args.augment:
        augments_per_epoch = 2
    else:
        augments_per_epoch = 10
        
    #Create log folder.
    if not os.path.exists(args.logFolder): os.mkdir(args.logFolder)
    #os.system('cp models/%s.py %s' % (args.model, args.logFolder))
    #os.system('cp train_with_sigmoid.py %s' % (args.logFolder))
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

    #Load the model
    model = importlib.import_module(args.model)

    #Get train and test datasets
    maxStoredPoints = args.maxNumPts

    mTrainDataSet = VQDataSet('train', maxStoredPoints, args.batchSize, args.augment, args.noise, VQs = VQ, folders =  folders, label_threshold = args.label_threshold, filter_categories=args.no_categories, categories = args.cates, pts_source= args.pts_source, aug_sym = args.symmetric_deformations, rotation_axis=args.rotation_axis, smallrotations=args.smallrotations)
    mValDataSet = VQDataSet('val', maxStoredPoints, args.batchSize, args.augment, VQs=VQ, folders =  folders,filter_categories=args.no_categories, categories = args.cates, pts_source = args.pts_source, rotation_axis=args.rotation_axis, smallrotations=args.smallrotations)
    mTestDataSet = VQDataSet('test', maxStoredPoints, args.batchSize, args.augment, VQs=VQ, folders =  folders,filter_categories=args.no_categories, categories = args.cates, pts_source = args.pts_source, rotation_axis=args.rotation_axis, smallrotations=args.smallrotations)
    
    numTrainModels = mTrainDataSet.get_num_models()
    numBatchesXEpoch = numTrainModels/args.batchSize
    if numTrainModels%args.batchSize != 0:
        numBatchesXEpoch = numBatchesXEpoch + 1
    numValModels = mValDataSet.get_num_models()
    numTestModels = mTestDataSet.get_num_models()
    print("Train models: " + str(numTrainModels))
    print("Val models: " + str(numValModels))

    #Create variable and place holders
    global_step = tf.Variable(0, name='global_step', trainable=False)
    inPts = tf.placeholder(tf.float32,[None,3], name='Points')
    inFeatures = tf.placeholder(tf.float32,[None,1], name = 'Features')
    inBatchIds = tf.placeholder(tf.int32, [None,1], name = 'Batchids')
    inPtHier_tf = [inPts, inFeatures, inBatchIds]
    isTraining = tf.placeholder(tf.bool, shape=(), name = 'isTraining')
    inSpheres = tf.placeholder(tf.float32, [None, 1000, 3], name = 'PointSpheres')
    inVQs3d = tf.placeholder(tf.float32, [numVQs, None, 1000], name = 'VQSpheres')
    keepProbConv = tf.placeholder(tf.float32, name= 'keepProbConv') 
    keepProbFull = tf.placeholder(tf.float32, name = 'keepProbFull')
    inVQLoss = tf.placeholder(tf.float32, [numVQs], name = 'VQLoss') 
    
    std = tf.constant(args.std, dtype = tf.float32)
    offset = tf.constant(args.offset, dtype = tf.float32)


    #Create the network
    useRenorm = args.useBRN
    BNMomentum = BN_decay(args.initBNMom, global_step, numBatchesXEpoch, 0.9)
    
    pred_views = model.create_network(inPts, inFeatures, inBatchIds, args.batchSize, 1, args.grow, 3, numVQs, 
        isTraining, keepProbConv, keepProbFull, args.useDropOutConv, args.useDropOut, useRenorm, BNMomentum)
    
    #pred_views = tf.tanh(pred_views)
    #Create Loss
    lossGraph, pred_views_norm, loss = create_loss_mult_tf(pred_views, numVQs, inSpheres, inVQs3d, std, offset, args.weightDecay, args.cosineLoss)
    
    #Create training
    trainning, learningRateExp = create_training(lossGraph, 
        args.initLearningRate, args.minLearningRate, args.learningDecayFactor, 
        args.learningDecayRate*numBatchesXEpoch, global_step)

    
    #Create loss metrics
    accumLoss, accumVQLoss, accumTrain, accumTest, accumVQLossOp = create_loss_metrics_mult(loss, inVQLoss, numVQs)
    metricsVars = tf.contrib.framework.get_variables('metrics', collection=tf.GraphKeys.LOCAL_VARIABLES)
    resetMetrics = tf.variables_initializer(metricsVars)

    #Create Summaries
    trainingSummary, testSummary, accumLoss, accumVQLoss = create_summaries(accumLoss, accumVQLoss, VQ, numVQs)

    
    #Create init variables 
    init = tf.global_variables_initializer()
    initLocal = tf.local_variables_initializer()

    #create the saver
    saver = tf.train.Saver()
    
    #Create session
    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=args.gpu)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    #from tensorflow.python import debug as tf_debug
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    
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
        processedModels = 0

        #Iterate over all the train files
        mTrainDataSet.start_iteration()
            
        while mTrainDataSet.has_more_batches():
            currbatchSize, pts, features, batchIds, modelList, referenceValuesList, invRotationMatrix, _, vqs3D, ptSpheres = mTrainDataSet.get_next_batch()
            processedModels += currbatchSize
            
            feed_dict = create_feed_dict_gauss(inPtHier_tf, [pts, features, batchIds], keepProbConv, args.dropOutKeepProbConv, keepProbFull, args.dropOutKeepProb)
            feed_dict[inSpheres] = ptSpheres
            feed_dict[inVQs3d] = vqs3D
            feed_dict[isTraining] = True
                
            Pred_views, _, _, step = sess.run([pred_views_norm, trainning, accumTrain, global_step], feed_dict)
            #print(Pred_views[0,0:3])
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
                trainingSumm, distanceAccumValue = sess.run([trainingSummary, accumLoss])
                summary_writer.add_summary(trainingSumm, step)
                summary_writer.flush()
                sess.run(resetMetrics)                 
                if args.trackVQLoss:
                    visualize_progress(min(epochStep, numBatchesXEpoch), numBatchesXEpoch, "Distance: %.6f | VQLoss: %.6f | Time: %4ds " 
                                       %(distanceAccumValue, lossAccumValue/float(lossInfoCounter), (endTrainTime-startTrainTime)/1000.0)) 
                else:
                    visualize_progress(min(epochStep, numBatchesXEpoch), numBatchesXEpoch, "Dist Loss: %.6f | Time: %2ds " 
                                       %(distanceAccumValue, (endTrainTime-startTrainTime)/1000.0))

                with open(logFile, "a") as myfile:
                    if args.trackVQLoss:
                        myfile.write("Step: %6d (%4d) | Distance: %.6f | VQLoss: %.6f\n" % (step, epochStep, distanceAccumValue, lossAccumValue/float(lossInfoCounter)))
                    else:
                        myfile.write("Step: %6d (%4d) | Distance: %.6f \n" % (step, epochStep, distanceAccumValue))
                
                distanceTotal += distanceAccumValue * processedModels
                processedModels = 0
                lossInfoCounter = 0
                lossAccumValue = 0.0
                startTrainTime = current_milli_time()
                visStep += 1
            epochStep += 1
            
        distanceTotal = distanceTotal/numTrainModels
        lossTotal = lossTotal/epochStep
        
        endEpochTime = current_milli_time()   
        
        
        if args.trackVQLoss:
            print("Epoch %3d  Train Time: %.2fs | Train Distance: %.6f | Train VQ-Loss: %.6f" %(epoch, (endEpochTime-startEpochTime)/1000.0, distanceTotal, lossTotal))
        else:
            print("Epoch %3d  Train Time: %.2fs |  Dist: %.6f" %(epoch, (endEpochTime-startEpochTime)/1000.0, distanceTotal))
            
        with open(logFile, "a") as myfile:
            if args.trackVQLoss:
                myfile.write("Epoch %3d  Train Time: %.2fs  |  Train Distance: %.6f | Train VQ-Loss: %.6f\n" %(epoch, (endEpochTime-startEpochTime)/1000.0, distanceTotal, lossTotal))
            else:
                myfile.write("Epoch %3d  Train Time: %.2fs  |Distance: %.6f\n" %(epoch, (endEpochTime-startEpochTime)/1000.0, distanceTotal))

        if epoch%augments_per_epoch==0:
            if epoch %50==0:
                saver.save(sess, args.logFolder+"/model.ckpt")
        
            startTestTime = current_milli_time()  
            #Test data
            accumTestLoss = 0.0
            TestModels = []
            it = 0
            mValDataSet.start_iteration()
            
            while mValDataSet.has_more_batches():

                currbatchSize, pts, features, batchIds, modelList, referenceValuesList, invRotationMatrix,_, vqs3D, ptSpheres = mValDataSet.get_next_batch()
                feed_dict = create_feed_dict_gauss(inPtHier_tf, [pts, features, batchIds], keepProbConv, 1.0, keepProbFull, 1.0)
                feed_dict[isTraining] = False
                feed_dict[inSpheres] = ptSpheres
                feed_dict[inVQs3d] = vqs3D
                Pred_views, _= sess.run([pred_views_norm, accumTest], feed_dict)
                
                if args.augment:
                    for i in range(len(Pred_views)):
                        for vq_i in range(numVQs):
                            ind = range(vq_i*3, (vq_i+1)*3)
                            Pred_views[i,ind] = np.dot(Pred_views[i,ind].reshape(1,3),invRotationMatrix[i])

                Loss = create_mult_loss_approx(Pred_views, modelList, vqs3D, currbatchSize, VQ, unif_pts)
                sess.run(accumVQLossOp, {inVQLoss : Loss})
                accumTestLoss += Loss*currbatchSize
                
                if it%(400/args.batchSize) == 0 or not mValDataSet.has_more_batches():
                    visualize_progress(it, numValModels/currbatchSize)
                
                it += 1
            
            TestSumm, accumTestDistance = sess.run([testSummary, accumLoss])
            summary_writer.add_summary(TestSumm, step)
            summary_writer.flush()
            sess.run(resetMetrics)
            
            accumTestLoss = accumTestLoss/numValModels
            TestLossList.append(accumTestLoss)
            endTestTime = current_milli_time()  
            
            print("Val Time: %.2fs | VQ-Loss: %.6f | Dist: %.6f" % ((endTestTime-startTestTime)/1000.0, np.mean(accumTestLoss), accumTestDistance))
    
    # after training save Val results
    
    if not args.no_eval:
            
        
        #print(''); print('########## Evaluation'); print('')
        
        #startTestTime = current_milli_time()  
        ##Test data
        #accumTestLoss = 0.0
        #TestModels = []
        #TestLoss = [[] for vqi in range(numVQs)]
        #TestViews = [[] for vqi in range(numVQs)]
        #TestRotations=[]
        #it = 0
        #mValDataSet.start_iteration()
        #mValDataSet.batchSize_ = 1
        
        #while mValDataSet.has_more_batches():

            #currbatchSize, pts, features, batchIds, modelList, referenceValuesList, invRotationMatrix, RotationMatrix, vqs3D, ptSpheres = mValDataSet.get_next_batch()
            #feed_dict = create_feed_dict_gauss(inPtHier_tf, [pts, features, batchIds], keepProbConv, 1.0, keepProbFull, 1.0)
            #feed_dict[isTraining] = False
            #feed_dict[inSpheres] = ptSpheres
            #feed_dict[inVQs3d] = vqs3D
            #Pred_views, _= sess.run([pred_views_norm, accumTest], feed_dict)
            #Pred_views, _= sess.run([pred_views_norm, accumTest], feed_dict)
            
            
            #TestModels.append(modelList[0])
            #TestRotations.append(RotationMatrix[0])
            
            #if args.augment:
                #for i in range(len(Pred_views)):
                    #for vq_i in range(numVQs):
                        #ind = range(vq_i*3, (vq_i+1)*3)
                        #Pred_views[i,ind] = np.dot(Pred_views[i,ind].reshape(1,3),invRotationMatrix[i])
                        #TestViews[vq_i].append(Pred_views[:,ind][i])
            #Loss = create_mult_loss_approx(Pred_views, modelList, vqs3D, currbatchSize, VQ, unif_pts)
            ##Loss = 0
            #sess.run(accumVQLossOp, {inVQLoss : Loss})
            #accumTestLoss += np.mean(Loss)*currbatchSize
            
            #for vq_i in range(numVQs):
                #TestLoss[vq_i].append(Loss[vq_i])
            
            #if it%(400/args.batchSize) == 0 or not mValDataSet.has_more_batches():
                #visualize_progress(it, numValModels/currbatchSize)
            
            #it += 1
        
        #TestSumm, accumTestDistance = sess.run([testSummary, accumLoss])
        
        #accumTestLoss = accumTestLoss/numValModels
        #TestLossList.append(accumTestLoss)
        #endTestTime = current_milli_time()  
        
        #print("Val Time: %.2fs | VQ-Loss: %.6f | Dist: %.6f" % ((endTestTime-startTestTime)/1000.0, np.mean(accumTestLoss), accumTestDistance))

        #with open(resultFolder+'/TestDistRes.txt', "w") as myfile:
            #myfile.write(str(accumTestDistance) + '\n')
        #with open(resultFolder+'/TestLoss.txt', "w") as myfile:
            #myfile.write(np.array2string(np.mean(TestLoss,axis=1),precision=2)[1:-1] + '\n')

        #for vq_i, vq in enumerate(VQ):
            #np.savetxt(resultFolder+'/Loss_VQ'+vq +'.txt', TestLoss[vq_i], delimiter = ',', fmt = '%s')
            ##np.savetxt(resultFolder+'/VQ'+vq +'.txt', TestVQ[vq_i], delimiter = ',', fmt = '%s')
            ##np.savetxt(resultFolder+'/Distance.txt', TestDistance, delimiter = ',', fmt = '%s')
            #np.savetxt(resultFolder+'/Models.txt', TestModels, delimiter = ',', fmt = '%s')
            ##np.savetxt(resultFolder+'/Result_VQ'+vq +'.txt', ResultTest[vq_i], delimiter = ',', fmt = '%s')
            #np.savetxt(resultFolder+'/Views_VQ'+vq +'.txt', TestViews[vq_i], delimiter = ',', fmt = '%s')
        #np.savetxt(resultFolder+'/Rotations.txt',np.array(TestRotations).reshape(-1,9), delimiter=',', fmt='%s')
            
            
        sess.run(resetMetrics)
        print(''); print('########## Testing'); print('')
        resultFolder = args.logFolder + '/results_test' + args.tta*'_tta'        
        if args.pts_source != 'pts_unif_own':
            resultFolder += '_' + args.pts_source
        if args.folders != 'ModelNet40_simplified':
            resultFolder += '_' + args.folders
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
        TestRotations=[]
        it = 0
        mTestDataSet.start_iteration()
        if args.tta:
            mTestDataSet.batchSize_ = args.batchSize
        else:
            mTestDataSet.batchSize_ = 1
        
        while mTestDataSet.has_more_batches():

            currbatchSize, pts, features, batchIds, modelList, referenceValuesList, invRotationMatrix, RotationMatrix, vqs3D, ptSpheres = mTestDataSet.get_next_batch(repeatModelInBatch=args.tta)
            feed_dict = create_feed_dict_gauss(inPtHier_tf, [pts, features, batchIds], keepProbConv, 1.0, keepProbFull, 1.0)
            feed_dict[isTraining] = False
            feed_dict[inSpheres] = ptSpheres
            feed_dict[inVQs3d] = vqs3D
            Pred_views, _= sess.run([pred_views_norm, accumTest], feed_dict)
            Pred_views, _= sess.run([pred_views_norm, accumTest], feed_dict)
            
            
            TestModels.append(modelList[0])
            TestRotations += RotationMatrix
            
            
            for vq_i in range(numVQs):
                for i in range(len(Pred_views)):
                    ind = range(vq_i*3, (vq_i+1)*3)
                    if args.augment:
                        Pred_views[i,ind] = np.dot(Pred_views[i,ind].reshape(1,3),invRotationMatrix[i])
                    TestViews[vq_i].append(Pred_views[:,ind][i])
            Loss = create_mult_loss_approx(Pred_views, modelList, vqs3D, currbatchSize, VQ, unif_pts)
            #Loss = 0
            sess.run(accumVQLossOp, {inVQLoss : Loss})
            accumTestLoss += np.mean(Loss)
            
            for vq_i in range(numVQs):
                TestLoss[vq_i].append(Loss[vq_i])
            
            if it%(400/args.batchSize) == 0 or not mTestDataSet.has_more_batches():
                visualize_progress(it, numTestModels)
            
            it += 1
        
        TestSumm, accumTestDistance = sess.run([testSummary, accumLoss])
        
        accumTestLoss = accumTestLoss/numTestModels
        TestLossList.append(accumTestLoss)
        endTestTime = current_milli_time()  
        
        print("Test Time: %.2fs | VQ-Loss: %.6f | Dist: %.6f" % ((endTestTime-startTestTime)/1000.0, np.mean(accumTestLoss), accumTestDistance))

        with open(resultFolder+'/TestDistRes.txt', "w") as myfile:
            myfile.write(str(accumTestDistance) + '\n')
        with open(resultFolder+'/TestLoss.txt', "w") as myfile:
            myfile.write(np.array2string(np.mean(TestLoss,axis=1),precision=2)[1:-1] + '\n')

        for vq_i, vq in enumerate(VQ):
            np.savetxt(resultFolder+'/Loss_VQ'+vq +'.txt', TestLoss[vq_i], delimiter = ',', fmt = '%s')
            #np.savetxt(resultFolder+'/VQ'+vq +'.txt', TestVQ[vq_i], delimiter = ',', fmt = '%s')
            #np.savetxt(resultFolder+'/Distance.txt', TestDistance, delimiter = ',', fmt = '%s')
            np.savetxt(resultFolder+'/Models.txt', TestModels, delimiter = ',', fmt = '%s')
            #np.savetxt(resultFolder+'/Result_VQ'+vq +'.txt', ResultTest[vq_i], delimiter = ',', fmt = '%s')
            np.savetxt(resultFolder+'/Views_VQ'+vq +'.txt', TestViews[vq_i], delimiter = ',', fmt = '%s')
        np.savetxt(resultFolder+'/Rotations.txt',np.array(TestRotations).reshape(-1,9), delimiter=',', fmt='%s')
    print('[done]')
    with open(logFile, "a") as myfile:
        myfile.write('[done]')

