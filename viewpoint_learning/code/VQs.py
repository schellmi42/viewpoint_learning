# Measure calculation for viewpoint quality
import numpy as np
import tensorflow as tf
import os
import sys
BASE_DIR = os.path.abspath('.')
sys.path.append(BASE_DIR +'/helpers')
from MeshHelpers2 import read_model2, generate_rendering_buffers
#from Application import GLScene

import time

current_milli_time = lambda: time.time() * 1000.0

### helper functions ##

#def getIds(model_params, viewDir=[1,1,1], MyGL=GLScene(1024, 1024)):
    #""" Calls functions from MeshHelpers2 to get pixelwise plygon identifiers
        #--- Input: ---
        #model_params: output of read_model2 from MeshHelpers2.py
        #viewDir: list - vector of viewpoint, independent of scaling
        #MyGL: GLScene - from Application.py
        #--- Output ---
        #texIds: rxr array - Pixelwise polygon identifiers, 0s where no polygons are visible (size: resolution x resolution)
        #texNormals: rxr array - Pixelwise normal vectors (not used for speed improvement -- has to be activated in Application.py)
        #numVertex: int - number of vertices in the model
        #numFaces: int - number of polygons in the model
    #"""
    
    #rendVert, rendVertTrianIds, rendFaces, coordMin, coordMax = model_params    
    #texIds,_,_ = MyGL.generate_images(coordMin, coordMax, rendVert, rendVertTrianIds, rendFaces, np.array(viewDir), only_texIds = True)

    #numVertex = len(rendVert)//7
    #numFaces= len(rendFaces)//3
    
    #return texIds, 0, numVertex, numFaces

def getFaceIds(texIds, numFaces, resolution=1024):
    """ extracts a list of the visible polygons from a pixelwise identifier
        --- Input ---
        texIds: rxr array - Pixelwise polygon identifiers (size: resolution x resolution) 
        numFaces: int - number of polygons in the model
        resolution: int - resolution of the rendered image
        --- Output ---
        faceIds: nx list - list of IDs of the visible poylgons
        vis_z: mx list - indicator list of 0s and 1s for every polygon      
    """
    # initialize variables
    vis_z = np.zeros(numFaces)
    faceIds = np.unique(texIds[texIds!=0])-1
    # create identifier vector
    vis_z[faceIds] = 1
    return faceIds, vis_z

def getFaceIds_fast(texIds, numFaces, resolution=1024):
    """ extracts a list of the visible polygons from a pixelwise identifier
        --- Input ---
        texIds: rxr array - Pixelwise polygon identifiers (size: resolution x resolution) 
        numFaces: int - number of polygons in the model
        resolution: int - resolution of the rendered image
        --- Output ---
        faceIds: nx list - list of IDs of the visible poylgons    
    """
    # initialize variables
    ind = np.zeros(numFaces)
    faceIds = np.unique(texIds[texIds!=0])-1
    return faceIds

def getProb(texIds, numFaces):
    a_z = np.zeros(numFaces+1)
    unique, counts = np.unique(texIds[texIds!=0] , return_counts=True)
    a_z[unique] = counts
    a_t = np.sum(counts)
    return a_t, a_z

def getProb_and_visz(texIds, numFaces):
    ind = np.zeros(numFaces)
    a_z = np.zeros(numFaces+1)
    faceIds, counts = np.unique(texIds[texIds!=0] , return_counts=True)
    a_z[faceIds] = counts
    a_t = np.sum(counts)
    ind[faceIds-1] = 1
    return a_t, a_z, ind

def getAngles(texIds):
    """ Filter the image to get the turning angles per pixels
    """
    conv_filter = np.ones([3,3])
    conv = convolve(1*(texIds!=0), conv_filter)
    unique, counts = np.unique(conv[texIds!=0] , return_counts=True)

    unique = 0.5*(unique-6)
    ind = np.absolute(unique)<=1
    return unique[ind], counts[ind]

def getAs(model, recalc = False):
    """ returns the polygon areas of a model (only triangles)
        --- Input ---
        model: string - direction to the .obj file
        recalc: bool - ignores precalculated values
        -- Ouput ---
        At: float - total surface area of the model
        areas: mx array - areas of the triangles in the model
    """    
    # set direction for parameter files
    param_dir = os.path.dirname(os.path.dirname(model))+'/param/area/' + os.path.basename(model)[:-4] + '.npz'
    # if already existent load from disk
    if os.path.exists(param_dir) and not recalc:
        area = np.load(param_dir)['arr_0']
    else:
        # create output direction
        print('--- calculating areas ---')
        if not os.path.exists(os.path.dirname(param_dir)):
            os.makedirs(os.path.dirname(param_dir))
        # load model form .obj file
        vertices,_,triangle_indices,coordMin, coordMax = read_model2(model)
        triangle_indices = np.array(triangle_indices)
        triangle_indices = triangle_indices[:,:,0]
        # initialize array for the area of the triangles
        #area = np.zeros(triangle_indices.shape[0])
        ## claculate the area of all triangles
        #for triangle in range(triangle_indices.shape[0]):
            ## get the corners of the triangle
            #A = vertices[triangle_indices[triangle,0]]
            #B = vertices[triangle_indices[triangle,1]]
            #C = vertices[triangle_indices[triangle,2]]
            ## calculate the area of the triangle using Heron's formula
            #a = np.linalg.norm(C-B)
            #b = np.linalg.norm(C-A)
            #c = np.linalg.norm(B-A)
            #s = (a + b + c) / 2.0
            #if s * (s - a) * (s - b) * (s - c) <=0:
                #area[triangle] = 0
            #else:
                #area[triangle] = np.sqrt(s * (s - a) * (s - b) * (s - c))
        ABC = vertices[triangle_indices]
        a = np.linalg.norm(ABC[:,2] - ABC[:,1], axis=-1)
        b = np.linalg.norm(ABC[:,2] - ABC[:,0], axis=-1)
        c = np.linalg.norm(ABC[:,1] - ABC[:,0], axis=-1)
        s = (a+b+c)/2.0
        area = s*(s-a)*(s-b)*(s-c)
        area[area<0] = 0
        area = np.sqrt(area)
        np.savez(param_dir, area)
    At = np.sum(area)
    return At, area

def getPz(model):
    return np.load(os.path.dirname(os.path.dirname(model))+'/param/pz/' + os.path.basename(model)[:-4] + '.npz')['arr_0']

def getScan(texIds,numFaces):
    a_t, a_z, vis_z = getProb_and_visz(texIds,numFaces)
    percentage = sum(vis_z)/float(len(vis_z))
    return a_t, a_z, vis_z, percentage
    

### View measures ##

### VQ_4 Visibility Ratio ###
def vq4(A_z, A_t, vis_z):
    """ A_z: nx1, is the area of polygon z
        A_t: float, is the total area of the model
        vis_z: nx1, indicator for visible areas
    """
    v = np.sum(np.multiply(vis_z,A_z))/(A_t) 
    
    # slower method: v= np.sum(A_z[faceIds])/A_t
    #v = tf.truediv(tf.multiply(vis_z,A_z), A_t) 
    return v


### VQ_5 Viewpoint Entropy ###
def vq5(a_z, a_t):
    """ a_z: nx1, visible area of polygon z
        a_t: float, projected area of the model 
    """

    prob = a_z[a_z!=0]/float(a_t)
    v = -np.sum(np.multiply(prob, np.log2(prob)))
    #prob = tf.truediv(a_z,a_t)
    #v = tf.sum(tf.multiply(prob, np.log2(prob)))
    return v


### VQ_7 Viewpoint Kullback-Leibler  distance ###
def vq7(a_z, a_t, A_z, A_t):
    """ p_z: nx1, average porojected area of polygon z
        p_zv: nx1, normalized projected area of polygon z, given through a_z/a_t
    """
    ind = (a_z!=0)*(A_z!=0)
    prob = a_z[ind]/float(a_t)
    div = np.divide(prob, A_z[ind]/A_t)
    v = np.sum(np.multiply(prob,np.log2(div)))
    return v

### VQ_8 Viewpoint Mutual Information ###
def vq8(a_z, a_t, p_z):
    """ p_z: nx1, average porojected area of polygon z
        p_zv: nx1, normalized projected area of polygon z, given through a_z/a_t
    """
    ind = (a_z!=0)
    prob = a_z[ind]/a_t
    p_z = p_z[ind]
    div = np.divide(prob, p_z, out = np.zeros_like(prob), where=(p_z!=0))
    v = np.sum(np.multiply(prob,np.log2(div), out=np.zeros_like(prob), where=(div!=0)))
    #div = tf.truediv(p_zv, p_z)
    #v = tf.sum(tf.multiply(p_zv,np.log2(div)))
    return v

### VQ_12 Silhoutte Curvature ###
def vq12(angles, counts):
    """ angles: nx1 turning angles between consecutive pixels (range from -1 to +1 in 0.5 steps)
        counts: nx1 number of times the angles occured
    """
    v = np.sum(angles*np.absolute(unique))
    #N_c = c.shape[0]
    #v = sum(np.abs(c))*np.pi*N_c/2    
    #N_c = tf.shape(c)[0]
    #v = tf.sum(tf.abs(c))*np.pi*N_c/2
    return v


### VQ_14 Stoev and Strassler ###
def vq14(p, d, weights):
    """ p: float, normalized projection area
        d: float, normalized maximum depth
        weights: nx3, weighting parameters [alpha, beta, gamma]
    """
    
    v = weights[0] * p + weights[1] * d + weights[2] * (abs(d - p))
    return v


