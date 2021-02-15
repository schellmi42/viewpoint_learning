'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    \brief Code with tf trianing operations

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import sys, os
import tensorflow as tf
import numpy as np
#from VQs import getProb, getFaceIds, getIds, getPz, vq4, vq5, vq7, vq8, vq12, vq14
from MCConvBuilder import PointHierarchy

BASE_DIR = os.path.abspath('.')
PROJECT_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

sys.path.append(os.path.join(BASE_DIR, 'helpers'))
sys.path.append(ROOT_DIR + '/MCCNN/utils')

def eps_steps(pred_views, eps):
    # adds +/- epsilon steps for all coordinates
    # INPUT:
    #   pred_views: (nx3 array) - batch of 3D vectors
    #   eps: (float) - step size
    # OUTPUT:
    #   (7xnx3 array) - the original 3D vectors and the vectors with steps in all directions in order:
    #   (0,1+,1-,2+,,2-,3+,3-)
    return np.concatenate((pred_views, 
                           pred_views + [eps,0,0], pred_views - [eps,0,0], 
                           pred_views + [0,eps,0], pred_views - [0,eps,0], 
                           pred_views +[0,0,eps], pred_views - [0,0,eps])) 

def create_feed_dict_gauss(PtHier_tf, PtHier, keepProbConv, dropOutKeepProbConv, keepProbFull, dropOutKeepProb):

    feed_dict ={keepProbConv: dropOutKeepProbConv, keepProbFull: dropOutKeepProb}
    
    for i in range(3):
        feed_dict[PtHier_tf[i]] = PtHier[i]
    return feed_dict

def create_feed_dict_mult(tf_labellist, labels, tf_labellist_mult, labels_mult, inSizes, inSizes_mult, PtHier_tf, PtHier, keepProbConv, dropOutKeepProbConv, keepProbFull, dropOutKeepProb, numVQs):

    feed_dict ={keepProbConv: dropOutKeepProbConv, keepProbFull: dropOutKeepProb, inSizes: [[1,len(labels[i][0]),1] for i in range(numVQs)], inSizes_mult: [[1,len(labels_mult[i][0]),1] for i in range(numVQs)]}
    
    for i in range(numVQs):
        feed_dict[tf_labellist[i]] = labels[i]
        feed_dict[tf_labellist_mult[i]] = labels_mult[i]
    for i in range(3):
        feed_dict[PtHier_tf[i]] = PtHier[i]
    return feed_dict
    
def fill_dummy_ptH(inPointHierarchy, numFeatures=3):
    inPtHierPoints = [tf.placeholder(tf.float32, [None,3], name = 'dummy_ptH_points_' + str(i)) for i in range(4)]
    inPtHierSampledPts = [tf.placeholder(tf.int32, [None,], name = 'dummy_ptH_sampledInd_' + str(i)) for i in range(4)] # 6
    inPtHierFeatures = tf.placeholder(tf.float32, [None,numFeatures], name = 'dummy_ptH_features_1') # 16
    inPtHierBatchIds = [tf.placeholder(tf.int32, [None,1], name = 'dummy_ptH_batchIds_' + str(i)) for i in range(4)] # 11
    inPtHierAabbMin = tf.placeholder(tf.float32, [None,3], name = 'dummy_ptH_min') # 20
    inPtHierAabbMax = tf.placeholder(tf.float32, [None,3], name  = 'dummy_ptH_max') # 21
    
    inPtHier_tf = [inPtHierPoints, inPtHierSampledPts, inPtHierFeatures, inPtHierBatchIds, inPtHierAabbMin, inPtHierAabbMax]
    
    inPointHierarchy.points_[1:5] = inPtHierPoints
    inPointHierarchy.sampledIndexs_[0:4] = inPtHierSampledPts
    inPointHierarchy.features_[1] = inPtHierFeatures
    inPointHierarchy.batchIds_[1:5] = inPtHierBatchIds
    inPointHierarchy.aabbMin_ = inPtHierAabbMin
    inPointHierarchy.aabbMax_ = inPtHierAabbMax
    
    return inPointHierarchy, inPtHier_tf
    
def ptH(radii, scope, batchSize, numFeatures = 3):
    inPts = tf.placeholder(tf.float32,[None,3], name = 'ptH_points')
    inFeatures = tf.placeholder(tf.float32,[None,numFeatures], name = 'ptH_features')
    inBatchIds = tf.placeholder(tf.int32, [None,1], name = 'ptH_batchIds')
    
    feed_ptH = [inPts, inFeatures, inBatchIds]
    
    outPointHierarchy = PointHierarchy(inPts, inFeatures, inBatchIds, radii, scope, batchSize)
    
    PtHierPoints = [[]] + outPointHierarchy.points_[1:5]
    PtHierSampledPts = outPointHierarchy.sampledIndexs_[0:4]
    PtHierFeatures = [[]] + [outPointHierarchy.features_[1]]
    PtHierBatchIds = [[]] + outPointHierarchy.batchIds_[1:5]
    PtHierAabbMin = outPointHierarchy.aabbMin_
    PtHierAabbMax = outPointHierarchy.aabbMax_ 
    
    outPtH = [PtHierPoints, PtHierSampledPts, PtHierFeatures, PtHierBatchIds, PtHierAabbMin, PtHierAabbMax]
    
    return outPtH, feed_ptH
    
def create_mult_loss(pred_views, modelList, modelParamsList, areas, referenceValuesList, batchSize, MyGL, resolution, eps, gradient_accuracy, VQs, do_print = False, evaluation = False):
    numVQs = len(VQs)
    vq_mean = np.zeros(numVQs)
    vq = np.zeros([numVQs, batchSize])
    vq_norm = np.zeros([numVQs, batchSize])
    gradient_norm = np.zeros([batchSize,3*numVQs])
    for vq_i, VQ in enumerate(VQs):
        if VQ == 'UV':
            continue
        ind = range(vq_i*3, (vq_i+1)*3)
        out_1, out_2 = create_loss(pred_views[:,ind], modelList, modelParamsList, areas, referenceValuesList[vq_i], 
                                        batchSize, MyGL, resolution, eps, gradient_accuracy, VQ = VQ, do_print = do_print, evaluation = evaluation)
        if not evaluation:
            vq_mean[vq_i] = out_1
            gradient_norm[:,ind] = out_2
        else:
            vq_norm[vq_i] = out_1
            vq[vq_i] = out_2
    if not evaluation:
        return vq_mean, gradient_norm
    else:
        return vq_norm, vq

def create_loss(pred_views, modelList, modelParamsList, areas, referenceValuesList, batchSize, MyGL, resolution, eps, gradient_accuracy, VQ, do_print = False, evaluation = False):
    # Initialize variables
    vq = np.zeros(batchSize)
    gradient = np.zeros([batchSize,3])
    # go through the batch
    for model in range(batchSize):
        if do_print:
            print('------------------')
        # get triangle identifiers per pixel
        texIds,_,_,numFaces = getIds(modelParamsList[model], viewDir = pred_views[model], MyGL=MyGL)
        if do_print:
            print('File: ' + modelList[model])
            print('Number of Triangles: ' + str(numFaces))
            print('best: ' + str(referenceValuesList[0][model]))
            print('worst: ' + str(referenceValuesList[1][model]))
        # extract triangle identifiers
        if VQ == '4':
            _, vis_z = getFaceIds(texIds, numFaces, resolution)
            # get polygon areas of the model
            A_z = areas[model]
            A_t =  np.sum(A_z)
            # view quality
            vq[model] = vq4(A_z, A_t, vis_z)
        elif VQ == '5':
            a_t, a_z =  getProb(texIds, numFaces)
            vq[model] = vq5(a_z[1:], a_t)
        elif VQ == '7':
            A_z = areas[model]
            A_t =  np.sum(A_z)
            a_t, a_z =  getProb(texIds, numFaces)
            vq[model] = vq7(a_z[1:], a_t, A_z, A_t)
        elif VQ == '8':
            p_z = getPz(modelList[model])
            a_t, a_z =  getProb(texIds, numFaces)
            vq[model] = vq8(a_z[1:], a_t, p_z)
        if do_print:
            print('VQ: ' + str(vq[model]))
        # gradient calculation
        if gradient_accuracy ==2: # higher accuracy (slower)
            # go through coordinates for vector derivatives
            for coord in range(3):
                if do_print:
                    print('Coordinate' + str(coord+1))
                # small step in coord direction and triangle identifier extraction
                view_plus = pred_views[model+(2*coord+1)*batchSize]
                texIds,_,_,numFaces = getIds(modelParamsList[model], viewDir = view_plus, MyGL=MyGL)
                if VQ == '4':
                    _, vis_z_plus = getFaceIds(texIds, numFaces, resolution)
                elif VQ == '5':
                    a_t_plus, a_z_plus = getProb(texIds, numFaces, resolution)
                elif VQ == '7':
                    a_t_plus, a_z_plus =  getProb(texIds, numFaces)
                elif VQ == '8':
                    a_t_plus, a_z_plus =  getProb(texIds, numFaces)
                # small step in negative coord dierction and triangle identifier extraction
                view_minus = pred_views[model+2*(coord+1)*batchSize]
                texIds,_,_,numFaces = getIds(modelParamsList[model], viewDir = view_minus, MyGL=MyGL)
                if VQ == '4':
                    _, vis_z_minus = getFaceIds(texIds, numFaces, resolution)
                    # view quality calculation
                    vq_plus = vq4(A_z, A_t, vis_z_plus)
                    vq_minus = vq4(A_z, A_t, vis_z_minus)
                    choose = 'max'
                    # gradient calculation
                elif VQ == '5':
                    a_t_minus, a_z_minus = getProb(texIds, numFaces, resolution)
                    # view quality calculation
                    vq_plus = vq5(a_z_plus[1:], a_t_plus)
                    vq_minus = vq5(a_z_minus[1:], a_t_minus)
                    choose = 'max'
                elif VQ == '7':
                    a_t_minus, a_z_minus =  getProb(texIds, numFaces)
                    vq_plus = vq7(a_z_plus[1:], a_t_plus, A_z, A_t)
                    vq_minus = vq7(a_z_minus[1:], a_t_minus, A_z, A_t)
                    choose = 'min'
                elif VQ == '8':
                    a_t_minus, a_z_minus =  getProb(texIds, numFaces)
                    vq_plus = vq8(a_z_plus[1:], a_t_plus, p_z)
                    vq_minus = vq8(a_z_minus[1:], a_t_pminus, p_z)
                    choose = 'min'
                    if choose == 'max':
                        gradient[model,coord] = (vq_minus - vq_plus)/(2*eps)
                    elif choose == 'min':
                        gradient[model,coord] = (vq_plus - vq_minus)/(2*eps)
                if do_print:
                    print('Loss: %.6f, Plus: %.6f, Minus %.6f' %(vq[model], vq_plus-vq[model], vq_minus-vq[model]))
        elif gradient_accuracy ==1: # lower accuracy (faster)
            # go through coordinates for vector derivatives
            for coord in range(3):
                # small step in coord direction and triangle identifier extraction
                view_plus = pred_views[model+(2*coord+1)*batchSize]
                texIds,_,_,numFaces = getIds(modelParamsList[model], viewDir = view_plus, MyGL=MyGL)
                if VQ == '4':
                    _, vis_z_plus = getFaceIds(texIds, numFaces, resolution)
                    # gradient calculation
                    gradient[model,coord] = (vq4(A_z, A_t, vis_z_plus) - vq[model])/(eps) 
                if VQ == '5':
                    a_t_plus, a_z_plus = getProb(texIds, numFaces, resolution)
                    gradient[model,coord] = (vq5(a_z_plus[1:], a_t_plus) - vq[model])/(eps) 
                elif VQ == '7':
                    a_t_plus, a_z_plus =  getProb(texIds, numFaces)
                    vq_plus = vq7(a_z_plus[1:], a_t_plus, A_z, A_t)
                    gradient[model,coord] = (vq7(a_z_plus[1:], a_t_plus, A_z, A_t) - vq[model])/(eps) 
                elif VQ == '8':
                    a_t_plus, a_z_plus =  getProb(texIds, numFaces)
                    vq_plus = vq8(a_z_plus[1:], a_t_plus, p_z)
                    gradient[model,coord] = (vq8(a_z_plus[1:], a_t_plus, p_z) - vq[model])/(eps) 
        else: # gradient_accuracy == 0 no gradient calculation (for validation set)
            gradient = np.zeros([batchSize,3])
    # normalize View Quality
    vq_norm = (np.array(referenceValuesList[0])- vq) / (np.array(referenceValuesList[0]) - np.array(referenceValuesList[1]))
    if do_print:
        print(vq)
    if not evaluation:
        scale= np.array(referenceValuesList[0]) - np.array(referenceValuesList[1])
        gradient_norm = gradient/scale[:,None]
        return np.mean(vq_norm), gradient_norm
    else:
        return vq_norm, vq

def create_mult_loss_approx(pred_views, modelList, vqs3D,  batchSize,  VQs, unif_pts):
    numVQs = len(VQs)
    loss_mean = np.zeros(numVQs)
    for vq_i, VQ in enumerate(VQs):
        if VQ == 'UV' or VQ == 'FV':
            continue
        ind = range(vq_i*3, (vq_i+1)*3)
        loss_mean[vq_i] = create_loss_approx(pred_views[:,ind], modelList, vqs3D[vq_i], batchSize, VQ, unif_pts)
    return loss_mean

def create_loss_approx(pred_views, modelList, vqs3D, batchSize, VQ, unif_pts):
    # Initialize variables
    loss = np.zeros(batchSize)
    # go through the batch
    for model in range(batchSize):
        dist = np.linalg.norm(pred_views[model]-unif_pts, axis = 1)
        loss[model] = 1-vqs3D[model][np.argmin(dist)]
    return np.mean(loss)


#### TENSORFLOW ####

def BN_decay(init_momentum, global_step, decay_rate, decay_factor = 0.9):
    BNMomentum = 1-tf.train.exponential_decay(init_momentum, global_step, decay_rate, decay_factor, staircase=True)
    return tf.minimum(BNMomentum, 0.99)
    

#### slow
def create_gradient_batchwise(grad_loss, pred_views, grads_and_vars_reg, batchSize):
    # Function for the custom gradient
    # extract variable values
    vals =  [v for g,v in grads_and_vars_reg]
    grad_sum = [0 for i in range(len(vals))]
    # extract gradient values
    for index in range(batchSize):
        grads_and_vars_x = optimizer.compute_gradients(var_list=tf.trainable_variables(),loss=pred_views[index][0])
        grads_and_vars_y = optimizer.compute_gradients(var_list=tf.trainable_variables(),loss=pred_views[index][1])
        grads_and_vars_z = optimizer.compute_gradients(var_list=tf.trainable_variables(),loss=pred_views[index][2])
        
        grad_x = [grad_loss[index][0]*g for g,v in grads_and_vars_x]
        grad_y = [grad_loss[index][1]*g for g,v in grads_and_vars_y]
        grad_z = [grad_loss[index][2]*g for g,v in grads_and_vars_z]
        
        grad_sum = [grad_sum[i] + grad_x[i] + grad_y[i] + grad_z[i] for i in range(len(grad_x))]
        
    grad_reg = [g for g,v in grads_and_vars_reg]
    # modify gradient values
    grad_sum = [grad_sum[i]/batchSize + grad_reg[i] for i in range(len(vals))]
    grad_sum,_ = tf.clip_by_global_norm(grad_sum, 0.5)
    # zip together for grads_and_vars format as input for tf's gradient computation
    gradient = zip(grad_sum, vals)
    return gradient

#### not correct
def create_gradient_wrong(grad_loss, grads_and_vars_x, grads_and_vars_y, grads_and_vars_z):
    # Function for the custom gradient
    # extract variable values
    vals =  [v for g,v in grads_and_vars_x]
    # extract gradient values
    grad_x = [grad_loss[0]*g for g,v in grads_and_vars_x]
    grad_y = [grad_loss[1]*g for g,v in grads_and_vars_y]
    grad_z = [grad_loss[2]*g for g,v in grads_and_vars_z]
    # modify gradient values
    grad_sum = [grad_x[i] + grad_y[i] + grad_z[i] for i in range(len(grad_x))]
    # zip together for grads_and_vars format as input for tf's gradient computation
    gradient = zip(grad_sum, vals)
    
    return gradient


#### Printing ####

def print_moments(Pred_views_norm, Pred_views, isTraining = True):
    if isTraining:
        print('Train: VN: %.3f | MN: [ %.3f %.3f %.3f ] | V: %.3f | M: [ %.3f %.3f %.3f ] | MinMax: %.2f %.2f' 
            %(np.mean((Pred_views-np.mean(Pred_views,0))**2), 
                np.mean(Pred_views_norm,0)[0], np.mean(Pred_views_norm,0)[1], np.mean(Pred_views_norm,0)[2],
                np.mean((Pred_views-np.mean(Pred_views,0))**2),np.mean(Pred_views,0)[0], np.mean(Pred_views,0)[1], np.mean(Pred_views,0)[2], np.min(Pred_views), np.max(Pred_views)))
    else:
        print('Test: VN: %.3f | MN: [ %.3f %.3f %.3f ] | V: %.3f | M: [ %.3f %.3f %.3f ] | MinMax: %.2f %.2f' 
            %(np.mean((Pred_views-np.mean(Pred_views,0))**2), 
                np.mean(Pred_views_norm,0)[0], np.mean(Pred_views_norm,0)[1], np.mean(Pred_views_norm,0)[2],
                np.mean((Pred_views-np.mean(Pred_views,0))**2),np.mean(Pred_views,0)[0], np.mean(Pred_views,0)[1], np.mean(Pred_views,0)[2], np.min(Pred_views), np.max(Pred_views)))
           
def print_ptH(PtHier, isTraining = True):
    PtHierPoints, PtHierSampledPts, PtHierFeatures, PtHierBatchIds, PtHierAabbMin, PtHierAabbMax = PtHier
    if isTraining:
        print('#### Train INPUT PtH')
    else:
        
        print('#### Test INPUT PtH')
    for i in range(1,5):
        print('level: '  + str(i))
        print(PtHierPoints[i].shape)
        print(PtHierSampledPts[i-1].shape)
        print(PtHierBatchIds[i].shape)
        
    print('features: ' + str(PtHierFeatures[1].shape))
    print(PtHierAabbMin.shape)
    print(PtHierAabbMax.shape)
    print('#### MODELS')
    print(modelList)
           
def print_BN_moments(mean, var, isTraining = True):
    means = [np.mean(mean[i]) for i in range(7)]
    varis = [np.mean(var[i]) for i in range(7)]
    if isTraining:
        print('Train BN:#############')
    else:
        print('Test BN:#############')
    print(means)
    print(varis)
    print('################')
