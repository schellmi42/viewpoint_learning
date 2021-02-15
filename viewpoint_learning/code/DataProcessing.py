'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    \brief Code to prepare data.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys, argparse, os, math, random, time
#import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from PIL import Image

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_DIR = os.path.abspath('.')
PROJECT_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(PROJECT_DIR)

#sys.path.append(ROOT_DIR + '/MCCNN/utils')
sys.path.append('../../code/helpers')
sys.path.append('../../code')

from VQs import *
#from Application import GLScene
from MeshHelpers2 import *
#from MCConvBuilder import PointHierarchy

current_milli_time = lambda: time.time() * 1000.0

#### small helper functions

def getPz(model):
    return np.load('./param/pz/' + model[11:-4] + '.npz.npy')


def mkdir(path):
    if not os.path.exists(path): os.makedirs(path)
    
def load_data_from_disk(modelPath, delimiter = ','):
    fileDataArray = []
    with open(modelPath, 'r') as modelFile:        
        for line in modelFile:
            line = line.replace("\n", "")
            currPoint = line.split(delimiter)
            fileDataArray.append(currPoint)

    fileData = np.array(fileDataArray, dtype = float)
    if fileData.shape[1] == 1:
        fileData = fileData.reshape(-1)
    return fileData

def create_dirs(path, folder_list, subfolders=['']):
    # creates paths for the results of generate_views
    mkdir(path)
    for s in subfolders:
        mkdir(path+s)
        for f in folder_list:
            mkdir(path+s+f)

def load_file_names(DATA_DIR, subset, obj = False):
    # reads files names from a .txt file
    file_list = []
    with open(DATA_DIR + subset + '_small.txt','r') as f:
        for line in f:
            if obj == True:
                file_list.append(line[:-4]+'obj')
            else:
                file_list.append(line[:-1])
    return file_list, folder_list

def list_files(in_direction):
    root_length = len(in_direction)
    file_list = []
    folder_list = []
    for root, folders, files in os.walk(in_direction):
        for f in files:
            file_list.append(os.path.join(root,f)[root_length:])
            if not root[root_length:] in folder_list:
                folder_list.append(root[root_length:])
    return file_list, folder_list

def acc_rej_sampling(pdf,n=1000):
    # acceptance rejection sampling
    # INPUT:
    #   pdf: (mx list) - probability density function (not necessarily normalized)
    #   n: (int) - number of samples
    # OUTPUT:
    #   ind: (nx list) - indices of sampled points
    
    ind = []
    while(len(ind)<n):
        for i,p in enumerate(pdf):
            x = np.random.rand()
            if x < p:
                ind.append(i)
    return ind

def cdf_sampling(pdf,n=1000):
    # random sampling using the cdf
    # INPUT:
    #   pdf: (mx list) - probability density function (not necessarily normalized)
    #   n: (int) - number of samples
    # OUTPUT:
    #   ind: (nx list) - indices of sampled points
    
    ind = []
    cdf = np.cumsum(pdf)/np.sum(pdf)
    while len(ind)<n:
        x = np.random.rand()
        i = np.argmax(cdf>x)
        ind.append(i)
    return ind

def cart2pol(xyz):
    # transforms cartesian coordinates into polar coordinates
    # INPUT: 
    #   xyz: 3x1 - 3 dimensional cartesian coordinates
    # OUTPUT:
    #   longitude: float -polar coordinate in range [0,2*pi]
    #   latitude: float - polar coordinate in range[-pi/2, pi/2]
    
    longitude = np.arctan2(xyz[1],xyz[0])
    if longitude <0:
        longitude += 2*np.pi
    latitude = np.arcsin(xyz[2])
    return longitude, latitude

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

    return np.array(points)

#def visualize_fibonacci_sphere(dim = '2D', samples=1000):
    ## plots a fibonacci sphere using a mercator projection
    #views = fibonacci_sphere(samples)
    #print(len(views))
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.set_xlabel('longitude')
    #ax.set_ylabel('latitude')
    #ax.set_xlim((0,2*np.pi))
    #ax.set_ylim((-0.5*np.pi,0.5*np.pi))
    #points_x = np.zeros(len(views))
    #points_y = np.zeros(len(views))
    #for index, view in enumerate(views):
        #points_x[index], points_y[index] = cart2pol(view)
    #ax.plot(points_x, points_y, 'ko', ms=3)
    #fig.show()

#### generative function

def generate_views(DATA_DIR, objfolder, dimension = '3D', resolution=1024, reverse =False, recalc = False, n_points = 1000, subset = 'full', skip = False):
    # Samples the view quality measures on a unifrom grid(2D) or a fibonacci sphere (3D).
    # Currently supports VQ4, VQ5, VQ7, VQ8 .
    # Saves results in 'DATA_DIR/best_views/vqx/dimension'
    # Also calculates the highest and lowest VQ-value for each model and saves them in 'DATA_DIR/best_views/vqx/best' .
    # INPUT:
    #   DATA_DIR: string - direction to the dataset
    #   objfolder: string - name of the folder in DATA_DIR containing the objfiles
    #   dimension: string - if '3D': evaluates the VQ-measures on points on a fibonacci sphere (for label generation)
    #                       if '2D' evaluates the VQ-measures on a uniform grid in polar coordinates (for contour plots)
    #   resolution: int - the VQ-measures will be evaluated on resolution x resolution images
    #   reverse: bool - reverses the order in which the objfiles are processed
    #   recalc: bool - if True existing results will be overwritter
    #   n_points: int - number of points to be sampled
    
    print('--Generating View Quality Meshes---')
    param_dir = DATA_DIR + 'param/'
    mkdir('best_views')
    # input direction with .obj files for image rendering
    input_dir = DATA_DIR + objfolder + '/'
    
    link_table = load_table('Pascal3D+')
    
    # get .obj files
    # all files from the directory
    fileList_1, folderList = list_files(input_dir)
    
    fileList_1=[]
    fileList = []
    # only files with a limited number of faces (stored in small.txt
    with open('small.txt','r') as myFile:
        for line in myFile:
            fileList_1.append(line[9:-1])
            
    # only files with a certain category
    for i,f in enumerate(fileList_1):
        curr_cat = f.split('/')[0]
        if curr_cat in link_table:
            fileList.append(f)
        
    if reverse:
        fileList = fileList[::-1]
    if skip:
        fileList = fileList[1::2]
    

    # output direction
    create_dirs(param_dir, folderList, ['area/','pv/','pz/'])
    output_dir4 = DATA_DIR + 'best_views/' + str(resolution) +'vq' + '4' + '/'
    output_dir5 = DATA_DIR + 'best_views/' + str(resolution) +'vq' + '5' + '/'
    output_dir7 = DATA_DIR + 'best_views/' + str(resolution) +'vq' + '7' + '/'
    output_dir8 = DATA_DIR + 'best_views/' + str(resolution) +'vq' + '8' + '/'
    if args.resolution == 1024:
        output_dir4 = DATA_DIR + 'best_views/' + 'vq' + '4' + '/'
        output_dir5 = DATA_DIR + 'best_views/' + 'vq' + '5' + '/'
        output_dir7 = DATA_DIR + 'best_views/' + 'vq' + '7' + '/'
        output_dir8 = DATA_DIR + 'best_views/' + 'vq' + '8' + '/'
        
    # if recalc is True existing files will be ignored and overwritten
    MyGL = GLScene(resolution, resolution)

    # create output path if not existent
    create_dirs(output_dir4, folderList, ['3D/','2D/','best/'])
    create_dirs(output_dir5, folderList, ['3D/','2D/','best/'])
    create_dirs(output_dir7, folderList, ['3D/','2D/','best/'])
    create_dirs(output_dir8, folderList, ['3D/','2D/','best/'])

    if dimension == '3D':
        
        numModels = len(fileList)
        # create grid on unit sphere
        views = np.array(fibonacci_sphere(n_points))
        np.savetxt(output_dir4 + '3D/_point_list.txt', views, delimiter=',', fmt = '%s')
        np.savetxt(output_dir5 + '3D/_point_list.txt', views, delimiter=',', fmt = '%s')
        np.savetxt(output_dir7 + '3D/_point_list.txt', views, delimiter=',', fmt = '%s')
        np.savetxt(output_dir8 + '3D/_point_list.txt', views, delimiter=',', fmt = '%s')
        
        
        # main part of the code for view quality calculation
        for index, file in enumerate(fileList):# get starting time
            print(file + '  |  %3d / %3d' %(index+1, numModels))
            start_time = current_milli_time() 
            model = input_dir + file
            print(model)
            # check if current model was already processed
            if os.path.exists(output_dir8  + '3D/' + file[:-4] + '_vq_list.txt') and recalc==False:
                continue
           
            # loads the model parameters ( True for train set, False for test set)
            model_params = read_and_generate_buffers(model)
            A_t, A_z = getAs(model)
            # reset variables
            vqs4 = np.zeros([n_points])
            vqs5 = np.zeros([n_points])
            vqs7 = np.zeros([n_points])
            vqs8 = np.zeros([n_points])
            min_vq4 = np.ones(1)
            max_vq4 = np.zeros(1)
            min_vq5 = np.ones(1)
            max_vq5 = np.zeros(1)
            min_vq7 = np.ones(1)
            max_vq7 = np.zeros(1)
            min_vq8 = np.ones(1)
            max_vq8 = np.zeros(1)
            # go through the grid
            
            numTriangles = len(model_params[2])/3
            
            a_t_list = np.zeros(n_points)
            a_z_list = np.zeros([numTriangles, n_points])
            pzv_list = np.zeros([numTriangles, n_points])
            pv_list = np.zeros(n_points)
            pz_list = np.zeros(numTriangles)
            
            
            for i, curr_view in enumerate(views):
                #if i%100 == 0:
                    #curr_time = (current_milli_time()-start_time)/1000
                    #print('Step: %3d' %(i))
                    #print('VQ4: %.6f   %.6f' %(max_vq4, min_vq4))
                    #print('VQ5: %.6f   %.6f' %(max_vq5, min_vq5))
                    #print('VQ7: %.6f   %.6f' %(max_vq7, min_vq7))
                # calculate the loss
                texIds,_,_,numFaces = getIds(model_params, viewDir = curr_view, MyGL = MyGL)
                # VQ4
                a_t, a_z, vis_z =  getProb_and_visz(texIds, numFaces)
                curr_vq4 = vq4(A_z, A_t, vis_z)
                #VQ5
                curr_vq5 = vq5(a_z[1:], a_t)
                #VQ7
                #A_t, A_z = getAs(model)
                #a_z = np.zeros(numFaces+1)
                #texIds2 = (texIds).reshape(resolution**2)
                #for ind in range(resolution**2):
                #    a_z[texIds2[ind]] += 1
                #a_t = np.sum(texIds!=0)
                
                a_z_list[:,i] = a_z[1:]
                a_t_list[i]= a_t.copy()
                if a_t != 0:
                    pzv_list[:,i] = a_z[1:]/a_t
                
                curr_vq7 = vq7(a_z[1:], a_t, A_z, A_t)

                vqs4[i] = curr_vq4.copy()
                vqs5[i] = curr_vq5.copy()
                vqs7[i] = curr_vq7.copy()
                
                if i ==  0:
                    min_vq4 = curr_vq4
                    max_vq4 = curr_vq4
                    best_view4 = curr_view.copy()
                    min_vq5 = curr_vq5
                    max_vq5 = curr_vq5
                    best_view5 = curr_view.copy()
                    min_vq7 = curr_vq7
                    max_vq7 = curr_vq7
                    best_view7 = curr_view.copy()
                    
                if curr_vq4 > max_vq4:
                    # save highest values
                    max_vq4 = curr_vq4
                    best_view4 = curr_view.copy()
                if curr_vq4 < min_vq4:
                    # save lowest value
                    min_vq4 = curr_vq4
                    
                if curr_vq5 > max_vq5:
                    # save highest values
                    max_vq5 = curr_vq5
                    best_view5 = curr_view.copy()
                if curr_vq5 < min_vq5:
                    # save lowest value
                    min_vq5 = curr_vq5
                    
                if curr_vq7 > max_vq7:
                    # save highest values
                    max_vq7 = curr_vq7
                if curr_vq7 < min_vq7:
                    # save lowest value
                    min_vq7 = curr_vq7
                    best_view7 = curr_view.copy()
                       
            pv_list = a_t_list/np.sum(a_t_list)
            pz_list = np.matmul(pzv_list, pv_list)
            
            # get ending time
            print('-----------------------------')
            # save results to file
            min_vq7, max_vq7 = max_vq7, min_vq7
            np.savetxt(output_dir4 + 'best/' + file[:-4] + '.txt', [max_vq4, min_vq4, best_view4[0], best_view4[1], best_view4[2]], delimiter=',', fmt='%s')
            np.savetxt(output_dir4 + '3D/' + file[:-4] + '_vq_list.txt', vqs4, delimiter=',', fmt = '%s')
            np.savetxt(output_dir5 + 'best/' + file[:-4] + '.txt', [max_vq5, min_vq5, best_view5[0], best_view5[1], best_view5[2]], delimiter=',', fmt='%s')
            np.savetxt(output_dir5 + '3D/' + file[:-4] + '_vq_list.txt', vqs5, delimiter=',', fmt = '%s')
            np.savetxt(output_dir7 + 'best/' + file[:-4] + '.txt', [max_vq7, min_vq7, best_view7[0], best_view7[1], best_view7[2]], delimiter=',', fmt='%s')
            np.savetxt(output_dir7 + '3D/' + file[:-4] + '_vq_list.txt', vqs7, delimiter=',', fmt = '%s')
            
            np.save(param_dir + 'pv/' + file[:-4] + '.npz', pv_list)
            np.save(param_dir + 'pz/' + file[:-4] + '.npz', pz_list)
            p_z = pz_list
            for i, curr_view in enumerate(views):
                #if i%100 == 0:
                    #curr_time = (current_milli_time()-start_time)/1000
                    #print('Step: %3d | VQ8: %.6f   %.6f' %(i, max_vq8, min_vq8))
                # calculate the loss
                texIds,_,_,numFaces = getIds(model_params, viewDir = curr_view, MyGL = MyGL)
                a_t, a_z =  getProb(texIds, numFaces)
                curr_vq8 = vq8(a_z[1:], a_t, p_z)
                
                vqs8[i] = curr_vq8.copy()
                
                if i ==  0:
                    min_vq8 = curr_vq8
                    max_vq8 = curr_vq8
                    best_view8 = curr_view.copy()

                if curr_vq8 > max_vq8:
                    # save highest values
                    max_vq8 = curr_vq8
                if curr_vq8 < min_vq8:
                    # save lowest value
                    min_vq8 = curr_vq8
                    best_view8 = curr_view.copy()
            
            # get ending time
            end_time = current_milli_time()
            print('Time: ' + str((end_time - start_time)/1000))
            print('-----------------------------')
            # save results to file
            min_vq8, max_vq8 = max_vq8, min_vq8
            np.savetxt(output_dir8 + 'best/' + file[:-4] + '.txt', [max_vq8, min_vq8, best_view8[0], best_view8[1], best_view8[2]], delimiter=',', fmt='%s')
            np.savetxt(output_dir8 + '3D/' + file[:-4] + '_vq_list.txt', vqs8, delimiter=',', fmt = '%s')

    if dimension == '2D':
        numModels = len(fileList)
        size = 32
        longitude = np.linspace(0,2*np.pi,size)
        latitude = np.linspace(-np.pi/2,np.pi/2,size)
        X, Y = np.meshgrid(longitude, latitude)
        np.savetxt(output_dir4 + '2D/_X.txt', X, delimiter=',', fmt = '%s')
        np.savetxt(output_dir4 + '2D/_Y.txt', Y, delimiter=',', fmt = '%s')
        np.savetxt(output_dir5 + '2D/_X.txt', X, delimiter=',', fmt = '%s')
        np.savetxt(output_dir5 + '2D/_Y.txt', Y, delimiter=',', fmt = '%s')
        np.savetxt(output_dir7 + '2D/_X.txt', X, delimiter=',', fmt = '%s')
        np.savetxt(output_dir7 + '2D/_Y.txt', Y, delimiter=',', fmt = '%s')
        np.savetxt(output_dir8 + '2D/_X.txt', X, delimiter=',', fmt = '%s')
        np.savetxt(output_dir8 + '2D/_Y.txt', Y, delimiter=',', fmt = '%s')
        Z4 = np.zeros(X.shape)
        Z5 = np.zeros(X.shape)
        Z7 = np.zeros(X.shape)
        Z8 = np.zeros(X.shape)
        for index, file in enumerate(fileList):
            print(file + '  |  %3d / %3d' %(index+1, numModels))
            # get starting time
            start_time = current_milli_time()
            # check if current model was already processed
            if os.path.exists(output_dir5 + '2D/' + file[:-4] + '_Z.txt') and recalc==False:
                continue
            model = input_dir + file
            print(model)
            # loads the model parameters ( True for train set, False for test set)
            model_params = read_and_generate_buffers(model)
            A_t, A_z = getAs(model)
            p_z = getPz(model)
            # go through the grid
            for i in range(size):
                for j in range(size):
                    x = np.cos(Y[i][j]) * np.cos(X[i][j])
                    y = np.cos(Y[i][j]) * np.sin(X[i][j])
                    z = np.sin(Y[i][j])
                    curr_view = [x,y,z]
                    # calculate the loss
                    texIds,_,_,numFaces = getIds(model_params, viewDir = curr_view, MyGL = MyGL)
                    a_t, a_z, vis_z =  getProb_and_visz(texIds, numFaces)
                    
                    curr_vq4 = vq4(A_z, A_t, vis_z)

                    curr_vq5 = vq5(a_z[1:], a_t)
                    
                    #A_t, A_z = getAs(model)
                    #a_z = np.zeros(numFaces+1)
                    #texIds2 = (texIds).reshape(resolution**2)
                    #for ind in range(resolution**2):
                    #    a_z[texIds2[ind]] += 1
                    #a_t = np.sum(texIds!=0)
                    curr_vq7 = vq7(a_z[1:], a_t, A_z, A_t)
                    
                    
                    #a_z = np.zeros(numFaces+1)
                    #texIds2 = (texIds).reshape(resolution**2)
                    #for ind in range(resolution**2):
                    #    a_z[texIds2[ind]] += 1
                    #a_t = np.sum(texIds!=0)
                    curr_vq8 = vq8(a_z[1:], a_t, p_z)
                    Z4[i][j] = curr_vq4.copy()
                    Z5[i][j] = curr_vq5.copy()
                    Z7[i][j] = curr_vq7.copy()
                    Z8[i][j] = curr_vq8.copy()
            # get ending time
            end_time = current_milli_time()
            print('Time: ' + str((end_time - start_time)/1000))
            print('-----------------------------')
            # save results to file
            np.savetxt(output_dir4 + '2D/' + file[:-4] + '_Z.txt', Z4, delimiter=',', fmt = '%s')
            np.savetxt(output_dir5 + '2D/' + file[:-4] + '_Z.txt', Z5, delimiter=',', fmt = '%s')
            np.savetxt(output_dir7 + '2D/' + file[:-4] + '_Z.txt', Z7, delimiter=',', fmt = '%s')
            np.savetxt(output_dir8 + '2D/' + file[:-4] + '_Z.txt', Z8, delimiter=',', fmt = '%s')

    
def generate_labels(DATA_DIR, threshold = 0.01, VQs=['4','5','7','8'], recalc = False):
    # creates a label file which contains all viewpoints with a VQ-value with a distance of
    #   d = (vq_max - vq_min)/threshold
    # to the best view quality.
    # DEPENDENCIES:     depends on the output of generate_views with dimension = '3D'
    # INPUT:
    #   DATA_DIR: string - direction of the dataset
    #   threshold- float - relative radius in which labels are accepted
    
    viewdir = DATA_DIR + 'best_views/'
    points = load_data_from_disk(viewdir + 'vq7/3D/_point_list.txt', delimiter = ',')
    for vqdir in VQs:
        vqdir = 'vq'+vqdir+'/'
        if os.path.exists(viewdir+vqdir+'t_label.txt'):
            threshold_old = load_data_from_disk(viewdir+vqdir+'t_label.txt', delimiter = ',')
            if threshold == threshold_old and not recalc:
                continue
        indir = viewdir + vqdir + '3D/'
        outdir = viewdir + vqdir + 'best/'
        labeldir = viewdir + vqdir + 'labels/'
        if not os.path.exists(labeldir): os.mkdir(labeldir)
        if not os.path.exists(outdir): os.mkdir(outdir)

        models, folderList = list_files(indir)
        create_dirs(labeldir, folderList,)
        numLabels = []
        sumLabels = 0
        for model in models:
            if model == '_point_list.txt': continue
            filename = model[:-12]+'.txt'
            vqs = load_data_from_disk(indir + model, delimiter = ',')
            if '7' in vqdir or '8' in vqdir:
                index_best = np.argmin(vqs)
                best_view = points[index_best]
                best_vq = np.min(vqs)
                worst_vq = np.max(vqs)
            else:
                index_best = np.argmax(vqs)
                best_view = points[index_best]
                best_vq = np.max(vqs)
                worst_vq = np.min(vqs)
            eps = abs(best_vq -worst_vq) * threshold
            labels = points[np.abs(best_vq-vqs)<eps]
            numLabels.append(len(labels))
            np.savetxt(outdir+filename, [best_vq, worst_vq, best_view[0], best_view[1], best_view[2]], delimiter = ',')
            np.savetxt(labeldir+filename, labels, delimiter = ',')
        with open(viewdir+vqdir+'t_label.txt', 'w') as f:
            print >>f, threshold
        print(vqdir, max(numLabels), np.mean(numLabels))

def generate_random_points_new(DATA_DIR, objfolder, minPoints = 20000, subset = 'full', recalc = False):
    # Creates csv files from obj files. the csv files contain at least minPoints points sampled onm the obj surface.
    # Sampled at least one point per face (except for degenerated faces). Sampling is done according to a random uniform distribution per face.
    # Also saves areas of the faces in 'DATA_DIR/param/areas'
    # INPUT:
    #   DATA_DIR: string - directory to the dataset
    #   objfolder: string - folder containing obj files in DATA_DIR
    #   minPoints: int - minimum number of points to be generated
    print('')
    print('BROKEN')
    print('')
    print('--Generating Random Points---')
    out_dir = DATA_DIR + 'csv_rand_2/'
    # create output_dir if not existent
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    obj_dir = DATA_DIR + objfolder
    objfileList = sorted(os.listdir(obj_dir))
    if subset == 'full': 
        objfileList = sorted(os.listdir(obj_dir))
    else:
        objfileList = load_file_names(DATA_DIR, subset, obj=True)
    for i, file in enumerate(objfileList):
        if i%10 == 0:
            print('%3d / %3d' %(i,len(objfileList)))
        filedir = obj_dir + '/' + file
        if os.path.exists(out_dir + file[:-4] + '.txt') and not recalc:
            continue
        file_out = out_dir + file[:-4] + '.txt'
        
        model = read_model2(filedir)
        param_dir = DATA_DIR+'param/area/' + file[:-4] + '.txt'
        if not os.path.exists(DATA_DIR+'param/'):
            os.mkdir(DATA_DIR+'param/')
            os.mkdir(DATA_DIR+'param/area/')
        # read vertex coordinates from files
        vertices = model[0]
        # read triangle index from files
        triangle_indices = model[2]
        triangle_indices = np.array(triangle_indices)
        triangle_indices = triangle_indices[:,:,0]
        numFaces = len(triangle_indices)
        # read bounding box
        coordMin = model[3]
        coordMax = model[4]
        # size of bounding box
        # initialize array for the edges of the triangles 
        # initialize array for the area of the triangles
        ABC = vertices[triangle_indices]
        A = ABC[:,0]
        B = ABC[:,1]
        C = ABC[:,2]
        a = np.linalg.norm(ABC[:,2] - ABC[:,1], axis=-1)
        b = np.linalg.norm(ABC[:,2] - ABC[:,0], axis=-1)
        c = np.linalg.norm(ABC[:,1] - ABC[:,0], axis=-1)
        s = (a+b+c)/2.0
        area = s*(s-a)*(s-b)*(s-c)
        area[area<0] = 0
        area = np.sqrt(area)
        total_area = np.sum(area)
        area_step_width = total_area/float(minPoints)
        if not os.path.exists(param_dir):
            np.save(param_dir[:-4], area)
        # normal calculation
        normals = np.cross(A-B,B-C)
        normals = normals/np.linalg.norm(normals,axis=1).reshape(-1,1)
        # sample one point per polygon
        rn1 = np.sqrt(np.random.rand(numFaces,1))
        rn2 = np.random.rand(numFaces,1)
        sampled_points = A*(1-rn1) + B*rn1*(1-rn2) + C*rn2*rn1
        
        sampled_points = sampled_points.astype(str)
        sampled_normals = normals.astype(str)
        with open(file_out, 'w') as f:
            for i in range(len(sampled_points)):
                f.write(sampled_points[i,0] +','+ sampled_points[i,1] +','+ sampled_points[i,2] +','+ sampled_normals[i,0] +','+ sampled_normals[i,1] +','+ sampled_normals[i,2] + '\n') 
                
        # sample points for bigger polygons
        step = 1
        indices = np.linspace(0,numFaces-1,numFaces, dtype = int)
        bool_indices = area>(i*area_step_width)
        while np.any(bool_indices):
            indices = indices[bool_indices]
            numPts = len(indices)
            rn1 = np.sqrt(np.random.rand(numPts,1))
            rn2 = np.random.rand(numPts,1)
            sampled_points = A[indices]*(1-rn1) + B[indices]*rn1*(1-rn2) + C[indices]*rn2*rn1
            sampled_normals = normals[indices]
            
            sampled_points = sampled_points.astype(str)
            sampled_normals = sampled_normals.astype(str)
            
            with open(file_out, 'a') as f:
                for i in range(numPts):
                    f.write(sampled_points[i,0] +','+ sampled_points[i,1] +','+ sampled_points[i,2] +','+ sampled_normals[i,0] +','+ sampled_normals[i,1] +','+ sampled_normals[i,2] + '\n') 
            # iterator
            step += 1
            bool_indices = area[indices]>=(step*area_step_width)
            
def generate_random_points(DATA_DIR, objfolder, minPoints = 20000, subset = 'full', recalc = False):
    out_dir = DATA_DIR + 'csv_rand/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    obj_dir = DATA_DIR + objfolder
    objfileList = sorted(os.listdir(obj_dir))
    if subset == 'full': 
        objfileList = sorted(os.listdir(obj_dir))
    else:
        objfileList = load_file_names(DATA_DIR, subset, obj=True)
    # create output_dir if not existent
    for i, file in enumerate(objfileList):
        filedir = obj_dir + '/' + file
        if os.path.exists(out_dir + file[:-4] + '.txt'):
            continue
        if i%10 == 0:
            print('%3d / %3d' %(i,len(objfileList)))
        model = read_model2(filedir)
        param_dir = os.path.dirname(os.path.dirname(filedir))+'/param/area/' + file[:-4] + '.txt'
        if not os.path.exists(os.path.dirname(os.path.dirname(filedir))+'/param/'):
            os.mkdir(os.path.dirname(os.path.dirname(filedir))+'/param/')
            os.mkdir(os.path.dirname(os.path.dirname(filedir))+'/param/area/')
        # read vertex coordinates from files
        vertices = model[0]
        # read triangle index from files
        triangle_indices = model[2]
        triangle_indices = np.array(triangle_indices)
        triangle_indices = triangle_indices[:,:,0]
        # read bounding box
        coordMin = model[3]
        coordMax = model[4]
        # size of bounding box
        # initialize array for the edges of the triangles 
        A = np.zeros(triangle_indices.shape)
        B = np.zeros(triangle_indices.shape)
        C = np.zeros(triangle_indices.shape)
        # initialize array for the area of the triangles
        area = np.zeros(triangle_indices.shape[0])
        # claculate the area of all triangles
        for triangle in range(triangle_indices.shape[0]):
            # get the corners of the triangle
            A[triangle] = vertices[triangle_indices[triangle,0]]
            B[triangle] = vertices[triangle_indices[triangle,1]]
            C[triangle] = vertices[triangle_indices[triangle,2]]
            # calculate the area of the triangle using Heron's formula
            a = np.linalg.norm(C[triangle]-B[triangle])
            b = np.linalg.norm(C[triangle]-A[triangle])
            c = np.linalg.norm(B[triangle]-A[triangle])
            s = (a + b + c) / 2.0
            if s * (s - a) * (s - b) * (s - c) <= 0:
                area[triangle] = 0
            else:
                area[triangle] = np.sqrt(s * (s - a) * (s - b) * (s - c))
        total_area = np.sum(area)
        area_step = total_area/float(minPoints)
        np.savetxt(param_dir, area, delimiter=',', fmt='%s')
        np.savez(param_dir[:-4], area)
        # list for the generated points and array for the current point
        points = []
        point = np.zeros(6)
        # 
        for triangle in range(triangle_indices.shape[0]):
            # normal vector of the triangle via cross product
            if area[triangle] == 0:
                continue
            n = np.cross(A[triangle]-B[triangle], B[triangle]-C[triangle])
            if np.count_nonzero((np.isnan(n)))>0:
                continue
            n = n / np.linalg.norm(n)
            if np.count_nonzero((np.isnan(n)))>0:
                continue
            point[3:6] = n
            # create random points depending on the area of the triangle
            for index in range(int(np.ceil(area[triangle]/area_step))):
                # create two random number (uniformly distibuted)
                rn = np.random.rand(2)
                # random point inside the triangle
                point[0:3] = (1 - np.sqrt(rn[0])) * A[triangle] + (np.sqrt(rn[0])*(1 - rn[1])) * B[triangle] + (rn[1]*np.sqrt(rn[0])) * C[triangle]
                if np.count_nonzero((np.isnan(point)))>0:
                    print('WARNING')
                    continue
                points.append(point.copy())
                
        out_points = np.array(points)
        # ouput file direction
        file_out = out_dir + file[:-4] + '.txt'
        # save file as .csv
        np.savetxt(file_out, out_points, delimiter=',', fmt='%s')
            
            
def generate_uniform_points(DATA_DIR, objfolder, numPoints = 20000, subset = 'full'):
    # Creates csv files from obj files. the csv files contain at least numPoints points sampled onm the obj surface.
    # Sampling is done according to a random uniform distribution using acceptance rejection sampling with polygon areas as pdf..
    # Also saves areas of the faces in 'DATA_DIR/param/areas'
    # INPUT:
    #   DATA_DIR: string - directory to the dataset
    #   objfolder: string - folder containing obj files in DATA_DIR
    #   minPoints: int - minimum number of points to be generated
    
    print('--Generating Uniform Points---')
    out_dir = DATA_DIR + 'csv_unif/'
    # create output_dir if not existent
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    obj_dir = DATA_DIR + objfolder
    objfileList = sorted(os.listdir(obj_dir))
    if subset == 'full': 
        objfileList = sorted(os.listdir(obj_dir))
    else:
        objfileList = load_file_names(DATA_DIR, subset, obj=True)
    for i, file in enumerate(objfileList):
        if i%10 == 0:
            print('%3d / %3d' %(i,len(objfileList)))
        filedir = obj_dir + '/' + file
        if os.path.exists(out_dir + file[:-4] + '.txt'):
            continue
        model = read_model2(filedir)
        param_dir = DATA_DIR+'param/area/' + file[:-4] + '.txt'
        if not os.path.exists(DATA_DIR+'param/'):
            os.mkdir(DATA_DIR+'param/')
            os.mkdir(DATA_DIR+'param/area/')
        # read vertex coordinates from files
        vertices = model[0]
        # read triangle index from files
        triangle_indices = model[2]
        triangle_indices = np.array(triangle_indices)
        triangle_indices = triangle_indices[:,:,0]
        # read bounding box
        coordMin = model[3]
        coordMax = model[4]
        # read triangle corners
        ABC = vertices[triangle_indices]
        A = ABC[:,0]
        B = ABC[:,1]
        C = ABC[:,2]
        # calculate triangle edge lengths
        a = np.linalg.norm(C - B, axis=-1)
        b = np.linalg.norm(C - A, axis=-1)
        c = np.linalg.norm(B - A, axis=-1)
        # calculate triangle areas using Heron's formula
        s = (a+b+c)/2.0
        area = s*(s-a)*(s-b)*(s-c)
        # catch degenerate faces
        area[area<0] = 0
        area = np.sqrt(area)
        total_area = np.sum(area)
        # save areas
        if not os.path.exists(param_dir):
            np.save(param_dir[:-4], area)
        
        # sample triangle indices uniformly according to their area
        rand_ind = cdf_sampling(area, numPoints)
        # normal calculation
        normals = np.cross(A-B,B-C)
        normals = normals/np.linalg.norm(normals,axis=1).reshape(-1,1)
        # generate points
        rn1 = np.sqrt(np.random.rand(numPoints,1))
        rn2 = np.random.rand(numPoints,1)
        sampled_points = A[rand_ind]*(1-rn1) + B[rand_ind]*rn1*(1-rn2) + C[rand_ind]*rn2*rn1
        sampled_normals = normals[rand_ind]
        
        file_out = out_dir + file[:-4] + '.txt'
        sampled_points = sampled_points.astype(str)
        sampled_normals = sampled_normals.astype(str)
        with open(file_out, 'w') as f:
            for i in range(numPoints):
                f.write(sampled_points[i,0] +','+ sampled_points[i,1] +','+ sampled_points[i,2] +','+ sampled_normals[i,0] +','+ sampled_normals[i,1] +','+ sampled_normals[i,2] + '\n') 
        
def generate_pointHierarchies(DATA_DIR, csv_folder = 'csv_rand', radii = [0.01, 0.05, 0.3, math.sqrt(3.0)+0.1], numPtH = 10, subset = 'full', recalc = 'False'):
    # generates csv files containing point hierarchies. 
    # Does not override existing files.
    # The results are saved in 'DATA_DIR/csv_ptH/radii'.
    # Results consist of point levels 1 to 5, feature level 1, sampled indices level 1 to 5
    # DEPENDECIES:  depends on the output of generate_random_points
    # INPUT:
    #   DATA_DIR: string - directory of the data set_title
    #   csv_folder: string - folder in DATA_DIR containing the point clouds
    #   radii: list of floats - radii used for the poisson disk sampling
    #   numPtH: int - number of point hierarchies to be generated per model
    #
    # CONTAINS TENSORFLOW 1.11.0 OPERATIONS 
    
    print('--Generating Point Hierarchies---')
    out_dir = DATA_DIR + 'csv_ptH/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir = out_dir + '[0.01,0.05,0.3,math.sqrt(3.0)+0.1]/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)   
    inPts = tf.placeholder(tf.float32, [None, 3])
    inFeatures = tf.placeholder(tf.float32, [None, 3])
    inBatchIds = tf.placeholder(tf.int32, [None,1])
    # create point hierarchy
    ptHier = PointHierarchy(inPts, inFeatures, inBatchIds, radii, relativeRadius = True)
    #Create init variables 
    init = tf.global_variables_initializer()
    initLocal = tf.local_variables_initializer()
    #Create session
    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list='0')
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # Initialize variables
    sess.run(init)
    sess.run(initLocal)
    
    #
    if os.path.exists(out_dir + 'ind.txt') and not recalc:
        start_index = int(load_data_from_disk(out_dir + 'ind.txt', delimiter =','))
    else:
        start_index = 0
    csv_dir = DATA_DIR + csv_folder + '/'
    if subset == 'full':
        fileList = os.listdir(csv_dir)
    else:
        fileList = load_file_names(DATA_DIR, subset)
    for index, file in enumerate(fileList):
        filedir = csv_dir + file
        out_dir_file = out_dir + os.path.basename(filedir)[:-4]
        if not os.path.exists(out_dir_file):
            os.makedirs(out_dir_file)
        if index%10 == 0:
            print('%3d / %3d' %(index, len(fileList)))
        # extract point cloud
        input_data = load_data_from_disk(filedir, delimiter =',')
        # get x y z coordinates
        points = input_data[:,:3]
        # get normal vector
        features = input_data[:,3:]
        batchIds = np.zeros([points.shape[0],1])

        for ptH_index in range(start_index, numPtH):
            if os.path.exists(out_dir_file + '/points' + str(ptH_index)):
                continue
            # Use Poisson Disk sampling to create point hierarchy
            PtHierPoints, PtHierSampledInd, PtHierFeatures = sess.run(
                        [ptHier.points_, ptHier.sampledIndexs_, ptHier.features_],
                        {inPts: points, inBatchIds: batchIds, inFeatures: features})   
            for level in range(1,len(radii)+1):
                np.save(out_dir_file + '/points'+ str(ptH_index)+'_'+str(level), PtHierPoints[level])
                np.save(out_dir_file + '/sampledInd'+ str(ptH_index)+'_'+str(level-1), PtHierSampledInd[level-1])
            np.save(out_dir_file + '/features'+ str(ptH_index)+'_'+str(1), PtHierFeatures[1])
        with open(out_dir + 'ind.txt', 'w') as f:
            print >>f, numPtH

def generate_images(DATA_DIR, objfolder = 'objfiles', resolution = 1024, VQs = ['4','5','7', '8'], subset = 'test'):
    # Saves images rendered with the best viewpoints.
    # INPUT:
    #   DATA_DIR: string - direction of the data_set
    #   objfolder: string - folder in DATA_DIR containing obj files
    #   resolution: int - images will be with with resolution x resolution pixels
    #   VQs: list of strings - View quality measures to use.
    
    print('--Generating Images---')
    #Render
    MyGL = GLScene(resolution, resolution)
    objdir = DATA_DIR + objfolder + '/' 
    
    refFolder = []
    outFolder = []
    for VQ in VQs:
        refFolder.append(DATA_DIR +  'best_views/vq' + VQ + '/best/')
        outFolder.append(DATA_DIR +  'best_views/vq' + VQ + '/images/')
        if not os.path.exists(outFolder[-1]): os.mkdir(outFolder[-1])
    rotate = False
    if 'ShapeNet' in DATA_DIR or 'TetWild' in DATA_DIR or 'Thingi' in DATA_DIR:
        R = np.array([[1,0,0],[0,0,-1],[0,1,0]])
        rotate = True
    if subset == 'full':
        view_files = os.listdir(objdir)
    else:
        view_files = load_file_names(DATA_DIR, subset)
    for index, file in enumerate(view_files):
        if not os.path.exists(objdir + file[:-4] +'.obj'):
            continue
        file = file[:-4] + '.txt'
        vertexs, normals, faces, coordMin, coordMax =read_model2(objdir + file[:-4] +'.obj')
        if rotate:
            vertexs = np.dot(vertexs,R)
            normals = np.dot(np.array(normals),R)
            coordMin = np.dot(coordMin,R)
            coordMax = np.dot(coordMax,R)
        rendVert, rendVertTrianIds, rendFaces = generate_rendering_buffers(vertexs, np.array(normals), np.array(faces,dtype=int))
        if index%10 ==0:
            print('%3d / %3d' %(index, len(view_files)))
        for i in range(len(VQs)):
            viewDir = load_data_from_disk(refFolder[i] + file, delimiter=',')
            viewDir = viewDir[2:]
            if rotate:
                viewDir = np.dot(viewDir,R)
            texIds, texNormals, texColors = MyGL.generate_images(coordMin, coordMax, rendVert, rendVertTrianIds, rendFaces, np.array(viewDir))
            texColorsExp = (texColors*255.0)[:,:,:3]
            img = Image.fromarray(texColorsExp.astype('uint8'), 'RGB')
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.save(outFolder[i] + file[:-4] + '.png')

#def generate_contour_plots(DATA_DIR, objfolder = 'objfiles', VQs = ['4','5','7', '8'], with_labels = False):
    ## generates contour plots in polar coordinates of the view quality measures
    ## DEPENDENCIES:  output files of generate_views with dimension = '2D'
    ##               if with_labels: output files of generate_labels
    ## INPUT:
    ##   DATA_DIR: string - directory of the data set
    ##   objfolder: string - folder in DATA_DIR containing obj files
    ##   VQs: list of strings - view quality measures to use
    ##   with_labels: bool - if True includes the labels in the plot
    
    #print('--Generating Contour Plots---')
    #for VQ in VQs:
        #viewdir = DATA_DIR + 'best_views/vq' + VQ + '/2D/'
        ## output direction
        #output_dir = DATA_DIR + 'best_views/vq' + VQ + '/contour/'
        #labeldir = DATA_DIR + 'best_views/vq' + VQ + '/labels/'
        ## create output path if not existent
        #if not os.path.exists(output_dir):
            #os.makedirs(output_dir)

        ## get .obj files
        #fileList = sorted(os.listdir(DATA_DIR + objfolder))
        
        #if not os.path.exists(viewdir + '_X.txt'): 
            #print('skip  VQ' + VQ)
            #continue
        #X = load_data_from_disk(viewdir + '_X.txt', delimiter = ',')
        #Y = load_data_from_disk(viewdir + '_Y.txt', delimiter = ',')
        #print('### VQ ' + VQ + ' ###') 
        #for i, file in enumerate(fileList):
            #if not os.path.exists(viewdir + file[:-4] +'_Z.txt'):
                #print('skip: ' + file)
                #continue
            #if (i+1)%10 ==0:
                #print('processed models: %3d / %3d' %(i+1, len(fileList)))
            #Z = load_data_from_disk(viewdir + file[:-4] +'_Z.txt', delimiter = ',')
            #fig = plt.figure()
            #ax = fig.add_subplot(111)
            #levels = np.linspace(np.min(Z), np.max(Z), 20)
            #CS = ax.contour(X, Y, Z, levels=levels)
            #cntr1 = ax.contourf(X, Y, Z, levels=levels)
            #fig.colorbar(cntr1, ax=ax)
            #ax.set_title(file + ': View Quality')
            #ax.set_xlabel('longitude')
            #ax.set_ylabel('latitude')
            #if with_labels == True:
                #views = load_data_from_disk(labeldir + file[:-4] + '.txt', delimiter = ',').reshape(-1,3)
                #points_x = np.zeros(len(views))
                #points_y = np.zeros(len(views))
                #for index, view in enumerate(views):
                    #points_x[index], points_y[index] = cart2pol(view)
                #ax.plot(points_x, points_y, 'ko', ms=3)
                #ax.set_title(file + ': View Quality')
                #ax.legend(['Labels'])
            #plt.savefig(output_dir + '/' + file[:-4] + '.png')
            #plt.close()


#def check_resolutions(model, resolutions, views, VQs = ['4','5','7','8']):
    #for resolution in resolutions:
        #MyGL = GLScene(resolution, resolution)

        #views = np.array(fibonacci_sphere(n_points))
        
        #model_params = read_and_generate_buffers(model)
        #A_t, A_z = getAs(model)
        #p_z = getPz(model)
        ## reset variables
        #for rep in range(reps):
            #print(rep)
            #t1 = current_milli_time()
            
            #vqs4 = np.zeros([n_points])
            #vqs5 = np.zeros([n_points])
            #vqs7 = np.zeros([n_points])
            #vqs8 = np.zeros([n_points])
            #min_vq4 = np.ones(1)
            #max_vq4 = np.zeros(1)
            #min_vq5 = np.ones(1)
            #max_vq5 = np.zeros(1)
            #min_vq7 = np.ones(1)
            #max_vq7 = np.zeros(1)
            #min_vq8 = np.ones(1)
            #max_vq8 = np.zeros(1)
            ## go through the grid
            
            #numTriangles = len(model_params[2])/3
            
            #for i, curr_view in enumerate(views):
                ## calculate the loss
                #texIds,_,_,numFaces = getIds(model_params, viewDir = curr_view, MyGL = MyGL)
                ## VQ4
                #a_t, a_z, vis_z =  getProb_and_visz(texIds, numFaces)
                #curr_vq4 = vq4(A_z, A_t, vis_z)
                ##VQ5
                #curr_vq5 = vq5(a_z[1:], a_t)
                
                #curr_vq7 = vq7(a_z[1:], a_t, A_z, A_t)

                #curr_vq8 = vq8(a_z[1:], a_t, p_z)
        
                #vqs4[i] = curr_vq4.copy()
                #vqs5[i] = curr_vq5.copy()
                #vqs7[i] = curr_vq7.copy()
                #vqs8[i] = curr_vq8.copy()
                
                #if i ==  0:
                    #min_vq4 = curr_vq4
                    #max_vq4 = curr_vq4
                    #best_view4 = curr_view.copy()
                    #min_vq5 = curr_vq5
                    #max_vq5 = curr_vq5
                    #best_view5 = curr_view.copy()
                    #min_vq7 = curr_vq7
                    #max_vq7 = curr_vq7
                    #best_view7 = curr_view.copy()
                    #min_vq8 = curr_vq8
                    #max_vq8 = curr_vq8
                    #best_view8 = curr_view.copy()
                    
                #if curr_vq4 > max_vq4:
                    ## save highest values
                    #max_vq4 = curr_vq4
                    #best_view4 = curr_view.copy()
                #if curr_vq4 < min_vq4:
                    ## save lowest value
                    #min_vq4 = curr_vq4
                    
                #if curr_vq5 > max_vq5:
                    ## save highest values
                    #max_vq5 = curr_vq5
                    #best_view5 = curr_view.copy()
                #if curr_vq5 < min_vq5:
                    ## save lowest value
                    #min_vq5 = curr_vq5
                    
                #if curr_vq7 > max_vq7:
                    ## save highest values
                    #max_vq7 = curr_vq7
                #if curr_vq7 < min_vq7:
                    ## save lowest value
                    #min_vq7 = curr_vq7
                    #best_view7 = curr_view.copy()
                
                #if curr_vq8 > max_vq8:
                    ## save highest values
                    #max_vq8 = curr_vq8
                #if curr_vq8 < min_vq8:
                    ## save lowest value
                    #min_vq8 = curr_vq8
                    #best_view8 = curr_view.copy()

            #min_vq7, max_vq7 = max_vq7, min_vq7
            
            #t2 = current_milli_time()
            #times.append(t2-t1)

#def generate_tet_meshing(DATA_DIR, objfolder = 'objfiles'):
    ## generates tetrahedral meshes using the TetWild algorithm
    ## adds normal information using ctmconv
    ## results are saved in a folder with appendix '_tet' in the same directory as DATA_DIR
    ## INPUT:
    ##   DATA_DIR: string - directory of the data set
    ##   objfolder: string - folder in DATA_DIR containing obj files
    
    #TetWild_dir = '~/git/TetWild/build/TetWild'
    
    #print('--Generating Tetrahedral Meshes---')
    #objdir = DATA_DIR + objfolder
    #output = DATA_DIR[:-1] + '_tet/' + objfolder
    #if not os.path.exists(DATA_DIR[:-1] + '_tet/'): os.mkdir(DATA_DIR[:-1] + '_tet/')
    #if not os.path.exists(output): os.mkdir(output)
    #file_list = os.listdir(objdir)
    #t1 = time.time()
    #for i, model in enumerate(file_list):
        #out_file = output + '/' + model[:-4]
        #command = TetWild_dir + ' --input ' + objdir + '/' + model + ' --output ' + out_file+ ' --is-quiet'
        #os.system(command)
        #os.system('rm ' + out_file)
        #os.system('ctmconv %s %s' %(out_file + '__sf.obj', out_file + '__sf.obj --calc-normals'))
        #os.system('mv ' + out_file + '__sf.obj ' + out_file + '_sf.obj')
        #if i%1 ==0:
            #t2 = time.time()
            #print('%3d/%3d | time: %.2fs'%(i+1, len(file_list), t2-t1))
            #t1 = time.time()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script with various functions for data processing')
    parser.add_argument('--generate_views', '--v', action = 'store_true', help = 'evaluates view measures on unifrom grids (req: --f --o --dimension --resolution --recalc --reverse)')
    parser.add_argument('--generate_labels', '--l', action = 'store_true', help = 'generates label files (req: --f --threshold)')
    parser.add_argument('--generate_random_points', '--rp', action = 'store_true', help = 'generates csv from obj files (req: --f --o)')
    parser.add_argument('--generate_uniform_points', '--up', action = 'store_true', help = 'generates csv from obj files (req: --f --o)')
    parser.add_argument('--generate_ptH', '--ptH',action = 'store_true', help = 'generates point hierarchy files (req: --f --numPtH)')
    parser.add_argument('--generate_images', '--img', action = 'store_true', help = 'saves images from best viewpoints (req: --f --o --resolution)')
    parser.add_argument('--generate_contour_plots', '--c', action = 'store_true', help = 'generates contour plots in 2D (req: --f, --o)')
    parser.add_argument('--generate_tet_meshing', '--t', action = 'store_true', help = 'generates tetrahedral meshing (TetWild)')
    parser.add_argument('--folder', '--f', default = 'SHREC15', help = 'Name of the dataset folder')
    parser.add_argument('--objfolder', '--o', default = 'objfiles', help = 'folder of the obj files')
    parser.add_argument('--dimension', default = '3D', help = 'dimension in which the uniform grid is created')
    parser.add_argument('--resolution', default = 1024, type=int, help='resolution of the image')
    parser.add_argument('--numPtH', default = 10, type=int, help='resolution of the image')
    parser.add_argument('--numPoints', default = 20000, type=int, help='number of points for point generation')
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--skip', action='store_true')
    parser.add_argument('--recalc', action='store_true')
    parser.add_argument('--threshold', default = 0.01, type=float, help = 'threshold for label calculation')
    parser.add_argument('--subset', default = 'full')
    parser.add_argument('--newvq', action = 'store_true')
    args = parser.parse_args()
    
    DATA_DIR = args.folder + '/'
    
    print('');print('##### Data Set: ' + DATA_DIR);print('')
    
    if args.newvq:
        newviewquality(DATA_DIR)
    
    if args.generate_views:
        generate_views(DATA_DIR, args.objfolder, args.dimension, args.resolution, recalc = args.recalc, reverse = args.reverse, subset = args.subset, skip = args.skip)
        
    if args.generate_labels:
        print('---- Generating Labels ----')
        generate_labels(DATA_DIR, args.threshold, recalc = args.recalc)
        
    if args.generate_random_points:
        generate_random_points(DATA_DIR, args.objfolder, subset = args.subset, recalc = args.recalc)


    if args.generate_uniform_points:
        generate_uniform_points(DATA_DIR, args.objfolder, numPoints = args.numPoints, subset = args.subset)
        
    if args.generate_ptH:
        generate_pointHierarchies(DATA_DIR, numPtH = args.numPtH, subset = args.subset, recalc = args.recalc)
        
    if args.generate_images:
        generate_images(DATA_DIR, args.objfolder, args.resolution, subset = args.subset)
        
    if args.generate_contour_plots:
        generate_contour_plots(DATA_DIR, args.objfolder)
    if args.generate_tet_meshing:
        generate_tet_meshing(DATA_DIR, args.objfolder)
    
