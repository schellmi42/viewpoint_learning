'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \brief Code with dataset for DLDL learning

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# import important functions
import sys, copy,os, math
import numpy as np

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



# import read_model from MeshHelpers.py


# directions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
sys.path.append(ROOT_DIR + '/MCCNN/utils')
sys.path.append(os.path.join(BASE_DIR, 'helpers'))
from DataSet import DataSet
from collections import deque
from DataProcessing import generate_labels, load_data_from_disk
from MeshHelpers2 import read_model2, read_and_generate_buffers, generate_rendering_buffers
from VQs import getAs

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

def filter_categories(fileList1, link_table):
        for f in fileList_1:
            curr_cat = f.split('/')[0]
            if curr_cat in link_table:
                fileList.append(f)
        
        return fileList

def load_table(direction):
    out = {}
    with open(direction + '/link2Pascal3D+.txt','r') as inFile:
        for line in inFile:
            lineElements = line[:-1].split('-')
            out[lineElements[0]] = lineElements[1]
    return out

class VQDataSet(DataSet):
    """SHREC15 dataset.

    Attributes:
        useNormalsAsFeatures_ (bool): Boolean that indicates if the normals will be used as the input features.
        maxStoredNumPoints_ (int): Maximum number of points stored per model.
        objfileList_ (list of strings): List of directions to the .obj files
        fileList_ (list fo strings): List of directions to the point cloud files
        numPts_ (list of ints): List of number of points per model
        modelParams_(list): List of the output of the read_model function from MeshHelpers
        referenceValues_(list): List that contains the best and worst view quality loss
                                [0]: lowest view quality loss (Best View)
                                [1]: highest view quality loss (Worst View)
                                [2:5] Best View Direction
    """
    def __init__(self, train, maxStoredNumPoints, batchSize,
        augment = False, noise = False, aug_sym=False, useNormalsAsFeatures=False, VQs = ['4','5','7','8'], folders=["ShapeNet"], label_threshold = 0, calc_ptH = False, smallrotations = False, seed=None, filter_categories = False, categories = None, pointFeatures = True, rotation_axis = '012', pts_source = 'pts_unif'):
        """Constructor.

        Args:
            train (string): String that indicates if this is a training, test or validation dataset.
            maxStoredNumPoints (int): Maximum number of points stored per model.
            batchSize (int): Size of the batch used.
            augment (bool): Boolean that indicates if data augmentation will be used in the models.
            useNormalsAsFeatures (bool): Boolean that indicates if the normals will be used as the input features.
            folder (list of str): Folders in which the data is stored.
            seed (int): Seed used to initialize the random number generator. If None is provided instead, the current
                time on the machine will be used to initialize the number generator.
        """
        if train == 'train':
            inset = lambda(index): index%10 in [0,1,2,3,4,5,6,7]
        elif train == 'val':
            inset = lambda(index): index%10 ==8
        elif train == 'test':
            inset = lambda(index): index%10 ==9
        
        self.augment_symmetric_ = aug_sym
        self.noise_level_ = 0.01     
        self.onlyUV_ = VQs == ['UV']
        self.numVQs_ = len(VQs)
        if self.onlyUV_:
            use_dataset = '/objfiles.txt'
        else:
            use_dataset = '/small.txt'
        self.rotation_axis = rotation_axis
        
        # Store the parameters of the class.
        self.useNormalsAsFeatures_ = useNormalsAsFeatures
        self.maxStoredNumPoints_ = maxStoredNumPoints
        self.modelParams_ = []
        self.objfileList_=[]
        self.fileList_ = []
        self.referenceValues_ = [[] for i in range(self.numVQs_)]
        self.labels_ = [[] for i in range(self.numVQs_)]
        self.vqs_ = [[] for i in range(self.numVQs_)]
        self.vqs2D_ = [[] for i in range(self.numVQs_)]
        self.areas_ = []
        # Create the list of features that need to be augmented.
        augmentedFeatures = []
        if useNormalsAsFeatures:
            augmentedFeatures = [0]
        self.augment_noise_ = noise
        if categories != None:
            if 'aeroplane' in categories:
                categories.append('airplane')
            if 'diningtable' in categories:
                categories.append('table')
        
        # Call the constructor of the parent class.
        super(VQDataSet,self).__init__(0, 0, pointFeatures, 
            False, True, False, False, batchSize, 
            [0], 1e8, 0, augment, 1, smallrotations, False, augmentedFeatures, 
            [], seed)
        
        file_ext = '.txt'
        if pts_source == 'MCCNN2_pts':
            file_ext = '.obj.hdf5'
            
        for folder in folders:
            if label_threshold != 0 and not self.onlyUV_:
                print('---- Generating Labels: ' + folder + ' ----')
                generate_labels(os.path.join(DATA_DIR, folder) + '/', label_threshold, VQs)
            # List for files
            obj_dir = os.path.join(DATA_DIR,folder) + '/objfiles'
            csv_dir = os.path.join(DATA_DIR,folder) + '/' + pts_source
            #obj_files,_ = list_files(obj_dir)
            #csv_files,_ = list_files(csv_dir)
            if filter_categories and categories == None:
                categories = load_table(os.path.join(DATA_DIR,folder))
            #csv_files = filter_categories(csv_files, link_table)
            #obj_files = filter_categories(obj_files, link_table)
            #if filter_categories:
                #cat_table = self._load_table_(os.path.join(DATA_DIR, folder))
            if os.path.exists(os.path.join(DATA_DIR, folder)+'/small.txt'):
                #csv_dir = os.path.join(DATA_DIR, folder) + '/csv_rand'
                index = 0
                with open (os.path.join(DATA_DIR, folder)+'/small.txt', 'r') as f:
                    for line in f:
                        if 'objfiles' in line:
                            line = line[line.find('objfiles/')+len('objfiles/'):]
                        if filter_categories:
                            curr_cat = line.split('/')[0]
                            if not curr_cat in categories:
                                continue
                        lineElements = line.split('/')
                        if 'ModelNet' in folder:
                            if 'PointNet' in pts_source:
                                line_csv = lineElements[0] + '/' + lineElements[2] 
                            else:
                                line_csv = line
                        elif 'TetWild' in folder:
                            line_csv = line
                        if inset(index) and os.path.exists(csv_dir + '/' + line_csv[:-5] + file_ext):
                            self.objfileList_.append(obj_dir + '/' + line[:-1] )
                            self.fileList_.append(csv_dir + '/' + line_csv[:-5] + file_ext)
                        index += 1
            else:
                for index, filedir in os.listdir(csv_dir):
                    self.objfileList_.append(obj_dir + '/' + filedir[:-4] + '.obj')
                    self.fileList_.append(csv_dir + '/' + filedir)                

        for csv_file in self.fileList_:
            with open(csv_file, 'r') as curr_file:
                self.numPts_.append(len(curr_file.readlines()))

        #print('--- Loading Model Parameters ---')
        for i, filedir in enumerate(self.objfileList_):
            #if (i+1)%100 == 0 and train !='train' and not self.onlyUV_:
                #print('%3d / %3d' %(i+1,len(self.objfileList_)))
            if train == 'train' or self.onlyUV_:
                model_params =None
                area = None
            else:
                model_params =None
                area = None
                #model_params = read_and_generate_buffers(filedir)
                #_, area = getAs(filedir)
            self.modelParams_.append(model_params)
            self.areas_.append(area)
        if 'UV' in VQs:
            self.uv_index_ = VQs.index('UV') 
            self.labels_[self.uv_index_] = load_data_from_disk(os.path.join(DATA_DIR, folder)+'/uv_label.txt', delimiter = ',').reshape(-1,3)
            self.referenceValues_[self.uv_index_] = np.concatenate(([0,1],self.labels_[self.uv_index_].reshape(-1)))
        else:
            self.uv_index_ = None
        if 'FV' in VQs:
            self.fv_index_ = VQs.index('FV') 
            self.labels_[self.fv_index_] = load_data_from_disk(os.path.join(DATA_DIR, folder)+'/fv_label.txt', delimiter = ',').reshape(-1,3)
            self.referenceValues_[self.fv_index_] = np.concatenate(([0,1],self.labels_[self.fv_index_].reshape(-1)))
        else:
            self.fv_index_ = None
        # List for reference Values of the View Point Quality
        for i,filedir in enumerate(self.objfileList_):
            #file = filedir[len(obj_dir):]
            
            #dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(filedir))))
            dir, file = filedir.split('/objfiles/')
            file = file[:-4] + '.txt'
            for i,VQ in enumerate(VQs):
                if not (VQ == 'UV' or VQ == 'FV'):
                    self.referenceValues_[i].append(load_data_from_disk(dir + '/best_views/vq' + VQ + '/best/' + file, delimiter = ','))
                    if label_threshold == 0:
                       self.labels_[i].append(self.referenceValues_[i][-1][2:].reshape(-1,3)) 
                    else:
                        self.labels_[i].append(load_data_from_disk(dir + '/best_views/vq' + VQ + '/labels/' + file, delimiter = ',').reshape(-1,3))
                    vqs = load_data_from_disk(dir + '/best_views/vq' + VQ + '/3D/' + file.replace('.txt','_vq_list.txt'), delimiter = ',')
                    if '7' in VQ or '8' in VQ:
                        vqs = np.max(vqs)-vqs
                    m = np.min(vqs); M = np.max(vqs)
                    vqs = (vqs-m)/(M-m)
                    self.vqs_[i].append(vqs)
                    if not self.augment_:
                        vqs2D = load_data_from_disk(dir + '/best_views/vq' + VQ + '/2D/' + file.replace('.txt','_Z.txt'), delimiter = ',')
                        vqs2D = (vqs2D - np.min(vqs2D)) / (np.max(vqs2D)-np.min(vqs2D))
                        self.vqs2D_[i].append(vqs2D)
        if self.augment_:
            self.sphere_coordinates = fibonacci_sphere(1000)

    def _load_table(direction):
        out = {}
        with open(direction + 'link2Pascal3D+.txt','r') as inFile:
            for line in inFile:
                lineElements = line[:-1].split('-')
                out[lineElements[0]] = lineElements[1]
        return out
                
    def _load_model_from_disk_(self, modelPath):
        """Abstract method that should be implemented by child class which loads a model
            from disk.

        Args:
            modelPath (string): Path to the model that needs to be loaded.

        Returns:
            pts (nx3 np.array): List of points.
            normals (nx3 np.array): List of normals. If the dataset does not contain 
                normals, None should be returned.
            features (nxm np.array): List of features. If the dataset does not contain
                features, None should be returned.
            labels (nxl np.array): List of labels. If the dataset does not contain
                labels, None should be returned.
        """
        
        fileDataArray = []
        if modelPath.endswith('.txt'):
            with open(modelPath, 'r') as modelFile:        
                it = 0
                for line in modelFile:
                    if it < self.maxStoredNumPoints_:
                        line = line.replace("\n", "")
                        currPoint = line.split(',')
                        fileDataArray.append(currPoint)
                        it+=1
                    else:
                        break
            fileData = np.array(fileDataArray, dtype = float)
            pts = fileData[:,0:3]
            normals = fileData[:,3:6]
        else:
            h5File = h5py.File(modelPath, "r")
            pts = h5File["points"][()]
            normals = h5File["normals"][()]
            h5File.close()
            
        if self.useNormalsAsFeatures_:
            features = normals
        else:
            features = np.ones([pts.shape[0],1])
        labels = None
        #if self.useNormalsAsLabels_:
        #    labels = normals
            
        return  pts, normals, features, labels
    
    def _augment_data_rot_(self, inData, mainRotAxis = 1, smallRotations = False, inRotationMatrix = None):
        """Method to augment a list of vectors by rotating alogn an axis, and perform
        small rotations alogn all the 3 axes.

        Args:
            inData (nx3 np.array): List of vectors to augment. 
            mainRotAxis (int): Rotation axis. Allowed values (0, 1, 2). (deprecated)
            smallRotations (bool): Boolean that indicates if small rotations along all
                3 axes will be also applied. (Used to add noise to features)
            inRotationMatrix (3x3 np.array): Transformation matrix. If provided, no matrix is computed 
                and this parameter is used instead for the transformations.

        Returns:
            augData (nx3 np.array): List of transformed vectors.
            rotation_matrix (3x3 np.array): Transformation matrix used to augment the data (without small rotation)
            inv_rotation_matrix (3x3 np.array): Inverse of the transformation matrix (without small rotation)
        """
        rotationMatrix = inRotationMatrix
        invRotationMatrix = None
        if inRotationMatrix is None:
            # Compute the main rotation
            rotationMatrix = np.array([[1,0,0],[0,1,0],[0,0,1]])
            invRotationMatrix = np.array([[1,0,0],[0,1,0],[0,0,1]])
            if '0' in self.rotation_axis:
                rotationAngle = self.randomState_.uniform() * 2.0 * np.pi
                cosval = np.cos(rotationAngle)
                sinval = np.sin(rotationAngle)
                invCosval= np.cos(-rotationAngle)
                invSinval = np.sin(-rotationAngle)
                rotationMatrix = np.array([[1.0, 0.0, 0.0], [0.0, cosval, -sinval], [0.0, sinval, cosval]])
                invRotationMatrix = np.array([[1.0, 0.0, 0.0], [0.0, invCosval, -invSinval], [0.0, invSinval, invCosval]])
            if '1' in self.rotation_axis:
                rotationAngle = self.randomState_.uniform() * 2.0 * np.pi
                cosval = np.cos(rotationAngle)
                sinval = np.sin(rotationAngle)
                invCosval= np.cos(-rotationAngle)
                invSinval = np.sin(-rotationAngle)
                rotationMatrix = np.matmul(np.array([[cosval, 0.0, sinval], [0.0, 1.0, 0.0], [-sinval, 0.0, cosval]]),rotationMatrix)
                invRotationMatrix = np.matmul(invRotationMatrix,np.array([[invCosval, 0.0, invSinval], [0.0, 1.0, 0.0], [-invSinval, 0.0, invCosval]]))
            if '2' in self.rotation_axis:
                rotationAngle = self.randomState_.uniform() * 2.0 * np.pi
                cosval = np.cos(rotationAngle)
                sinval = np.sin(rotationAngle)
                invCosval= np.cos(-rotationAngle)
                invSinval = np.sin(-rotationAngle)
                rotationMatrix = np.matmul(np.array([[cosval, -sinval, 0.0], [sinval, cosval, 0.0], [0.0, 0.0, 1.0]]),rotationMatrix)
                invRotationMatrix = np.matmul(invRotationMatrix,np.array([[invCosval, -invSinval, 0.0], [invSinval, invCosval, 0.0], [0.0, 0.0, 1.0]]))
            # Compute small rotations.
        if smallRotations:
            angles = np.clip(0.15*self.randomState_.randn(3), -0.45, 0.45)
            Rx = np.array([[1.0, 0.0, 0.0],
                        [0.0, np.cos(angles[0]), -np.sin(angles[0])],
                        [0.0, np.sin(angles[0]), np.cos(angles[0])]])
            Ry = np.array([[np.cos(angles[1]), 0.0, np.sin(angles[1])],
                        [0.0, 1.0, 0.0],
                        [-np.sin(angles[1]), 0.0, np.cos(angles[1])]])
            Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0.0],
                        [np.sin(angles[2]), np.cos(angles[2]), 0.0],
                        [0.0, 0.0, 1.0]])
            R = np.dot(Rz, np.dot(Ry,Rx))

        if smallRotations:
            return np.dot(inData.reshape((-1, 3)), np.dot(rotationMatrix,R)), np.dot(rotationMatrix,R), np.dot(rotationMatrix,R).T
        else:
            return np.dot(inData.reshape((-1, 3)), rotationMatrix), rotationMatrix, invRotationMatrix
    
    def _get_2D_grid_with_sphere_coords(self, size):
        longitude = np.linspace(0,2*np.pi,size)
        latitude = np.linspace(-np.pi/2,np.pi/2,size)
        X, Y = np.meshgrid(longitude, latitude)
        x = np.reshape(np.cos(Y) * np.cos(X), [1024])
        y = np.reshape(np.cos(Y) * np.sin(X), [1024])
        z = np.reshape(np.sin(Y), [1024])
        return X,Y, np.stack((x,y,z), axis=1)

    def _get_2D_VQ_image(self, vq_sphere, sphere_coordinates, inRotationMatrix):
        size=32
        rotated_coords = np.dot(sphere_coordinates, inRotationMatrix)
        X, Y, image_coords = self._get_2D_grid_with_sphere_coords(size)
        rotated_vq_image = np.zeros([size, size])
        for i in range(size):
            for j in range(size):
                dists = np.linalg.norm(rotated_coords - image_coords[size*i+j].reshape(1,3), axis=1)
                closest_pt_id = np.argmin(dists)
                rotated_vq_image[i][j] = vq_sphere[closest_pt_id]
        return rotated_vq_image

        
        
    def _augment_data_noise_(self, inData, noise_level = 0.01, scale = None):
        """ Method to augment points with AWG-noise.
        
        Args:
            inData (nxm np.array): List of vectors to augment.
            noise_level (float): standard deviation of the noise
            scale (mx np.array): bounding box side lengths of inData
            
        Returns:
            augData (nxm np.array): List of transformed vectors
            
        """
        if scale is None:
            scale = np.max(inData,axis=0)-np.min(inData,axis=0)
        noise = np.clip(self.randomState_.randn(inData.shape[0], inData.shape[1]), -2, 2)*scale*noise_level
        return inData + noise
    
    def _augment_data_symmetric_deformation(self, inData, numDivisions = 4, noise_level = 0.33):
        
        coordMin = np.amin(inData,axis=0)
        coordMax = np.amax(inData,axis=0)
        
        outData = np.empty(inData.shape)
        
        for j in range(3):
            b = np.linspace(0,1,numDivisions)*(coordMax[j]-coordMin[j])+coordMin[j]
            s = np.zeros(numDivisions)
            s[1:numDivisions/2] = (np.random.randn(numDivisions/2 -1)*noise_level)/numDivisions
            s[numDivisions/2:-1] = -s[1:numDivisions/2][::-1]
            np.clip(s,-0.5/numDivisions, 0.5/numDivisions, out=s)
            s = s*(coordMax[j]-coordMin[j])
            bn = b + s

            for i in range(1, numDivisions):
                    aff_v = b[i-1]<=inData[:,j]
                    aff_v *= inData[:,j] <= b[i]
                    m = (bn[i-1]-bn[i])/(b[i-1]-b[i])
                    outData[aff_v,j] = inData[aff_v,j]*m + bn[i-1] - b[i-1] * m
                
        return outData
        
    #def get_next_batch_ptH(self, repeatModelInBatch = False, test = False):
        #"""Method to get the next batch of models.
        
        #Args:
            #repeatModelInBatch (bool): Boolean that indicates if the batch will be filled with
                #the same model.            

        #Returns:
            #numModelInBatch (int): Number of models in the batch.
            #accumPts (nx3 np.array): List of points of the batch.
            #accumBatchIds (nx1 np.array): List of model indentifiers within the batch 
                #for each point.
            #accumFeatures (nxm np.array): List of point features.
            #accumLabels (nxl np.array): List of point labels. If the dataset does not contain
                #point labels, None is returned instead.
            #accumCat (numModelInBatchx1 or nx1 np.array): List of categories of each model 
                #in the batch. If the dataset was initialize with pointCategories equal to True,
                #the category of each model is provided for each point. If the dataset does not
                #contain category information, None is returned instead.
            #accumPaths (array of strings): List of paths to the models used in the batch.
        #"""
        
        #if repeatModelInBatch:
            #self.ptH_index = 0
        ##initialize variables
        #accumLabels = [[] for i in range(self.numVQs_)]
        #accumLabels_mult = [[] for i in range(self.numVQs_)]
        #accumSigns = [ [] for i in range(self.numVQs_)]
        #accumModels = []
        #accumModelParams = []
        #accumAreas = []
        #accumReferenceValues = [[[],[],[]] for i in range(self.numVQs_)]
        #accumRotationMatrix = []
        #accumInvRotationMatrix = []
        #accumIndices = []
        #PtHier = [[],[],[],[],[],[]]
        #accumPtHierPts = [np.array([]) for i in range(5)]
        #accumPtHierBatchIds = [np.array([]) for i in range(5)]
        #accumPtHierSampledInd = [np.array([]) for i in range(4)]
        #accumPtHierFeatures = [np.array([]) for i in range(5)]
        #accumPtHierAabbMin = np.array([])
        #accumPtHierAabbMax = np.array([])
        
        #numLabels = [0 for i in range(self.numVQs_)]
        #numLabels_mult = [0 for i in range(self.numVQs_)]
        #numModelInBatch = 0
        #numPtsInBatch = 0
        #temp_aug = False

        ## Iterate over the elements on the batch.
        #for i in range(self.batchSize_):
            ## Check if there are enough models left.
            #if self.iterator_ < len(self.randomSelection_):
                #""" if first model in batch should not be rotated
                #"""
                ## do not augment first model in batch
                ##if repeatModelInBatch and i == 0 and self.augment_:
                    ##self.augment_ = False
                    ##temp_aug = True
                    ##accumInvRotationMatrix.append(np.array([[1,0,0],[0,1,0],[0,0,1]]))
                ##elif repeatModelInBatch and i!=0 and temp_aug == True:
                    ##self.augment_ = True
                    ##temp_aug = False
            
                ## Check if the model fit in the batch.
                #currModelIndex = self.randomSelection_[self.iterator_]
                #currModel = self.fileList_[currModelIndex]
                #currModelNumPts = self.numPts_[currModelIndex]
                #accumIndices.append(currModelIndex)
                #accumModels.append(self.objfileList_[currModelIndex])
                #accumModelParams.append(self.modelParams_[currModelIndex])
                #accumAreas.append(self.areas_[currModelIndex])
                #currLabels = []
                #currLabels_mult = []
                #currSignLabels = []
                #currRefView = []
                #for vq_i in range(self.numVQs_):                    
                    #if vq_i == self.uv_index_:
                        #currLabels.append(self.labels_[vq_i])
                        #numLabels[vq_i] = 1
                        #accumReferenceValues[vq_i][0].append(self.referenceValues_[vq_i][0])
                        #accumReferenceValues[vq_i][1].append(self.referenceValues_[vq_i][1])
                        #currRefView.append(self.referenceValues_[vq_i][2:])
                    #else:
                        #bestView = self.referenceValues_[vq_i][currModelIndex][2:]
                        #sign = np.sum((bestView<0).astype(int) * 2**np.array([0,1,2]))
                        #currLabels.append(bestView.reshape(-1,3))
                        #currSignLabels.append(sign)
                        #numLabels[vq_i] = max(numLabels[vq_i], len(currLabels[vq_i]))
                        #currLabels_mult.append(self.labels_[vq_i][currModelIndex])
                        #numLabels_mult[vq_i] = max(numLabels_mult[vq_i], len(currLabels_mult[vq_i]))
                        #accumReferenceValues[vq_i][0].append(self.referenceValues_[vq_i][currModelIndex][0])
                        #accumReferenceValues[vq_i][1].append(self.referenceValues_[vq_i][currModelIndex][1])
                        #currRefView.append(self.referenceValues_[vq_i][currModelIndex][2:])
                    
                
                ## If the batch has a limited number of points, check if the model fits in the batch.
                #if (self.maxPtsxBatch_ == 0) or ((numPtsInBatch+currModelNumPts) <= self.maxPtsxBatch_):

                    ## Determine the category of the model if it is necesary.
                    #currModelCat = None
                    #if self.useCategories_:
                        #currModelCat = self.categories_[currModelIndex]
                    #currPtHierPts, currPtHierSampledInd, currPtHierFeatures = self._load_ptH_(currModel)
                    ## Augment data.
                    #if self.augment_:
                        #currPtHierFeatures, rotationMatrix, invRotationMatrix= self._augment_data_rot_(currPtHierFeatures, self.augmentMainAxis_, self.augment_noise_)
                        #for vq_i in range(self.numVQs_):
                            #currLabels[vq_i],_, _= self._augment_data_rot_(currLabels[vq_i], self.augmentMainAxis_, self.augmentSmallRotations_, rotationMatrix)
                            #currRefView[vq_i] = np.dot(currRefView[vq_i].reshape(1,3), rotationMatrix)[0]                        
                        #for level in range(1,5):
                            #currPtHierPts[level],_, _= self._augment_data_rot_(currPtHierPts[level], self.augmentMainAxis_, self.augmentSmallRotations_, rotationMatrix) 
                        ## Append Rotation Matrix
                        #accumInvRotationMatrix.append(invRotationMatrix)
                    #currPtHierAabbMin = np.amin(currPtHierPts[1], axis=(0)).reshape(1,3)
                    #currPtHierAabbMax = np.amax(currPtHierPts[1], axis=(0)).reshape(1,3)
                    #if self.augment_noise_:
                        #for level in range(1,5):
                            #currPtHierPts[level] = self._augment_data_noise_(currPtHierPts[level], self.noise_level_, currPtHierAabbMax-currPtHierAabbMin)
                    ## Append the current model to the batch.  
                    #if self.pointFeatures_:
                        #accumPtHierFeatures[1] = np.concatenate((accumPtHierFeatures[1], currPtHierFeatures), axis=0) if accumPtHierFeatures[1].size else currPtHierFeatures
                    #else:
                        #auxFeatures = np.full([len(currPtHierPts[1]),1],1.0)
                        #accumPtHierFeatures[1] = np.concatenate((accumPtHierFeatures[1], auxFeatures), axis=0) if accumPtHierFeatures[1].size else auxFeatures
                    #for level in range(1,5):
                        #accumPtHierPts[level] = np.concatenate((accumPtHierPts[level], currPtHierPts[level]), axis=0) if accumPtHierPts[level].size else currPtHierPts[level]
                        #auxBatchIds = np.full([len(currPtHierPts[level]),1],i,dtype=int)
                        #accumPtHierBatchIds[level] = np.concatenate((accumPtHierBatchIds[level], auxBatchIds), axis=0) if accumPtHierBatchIds[level].size else auxBatchIds
                        #accumPtHierSampledInd[level-1] = np.concatenate((accumPtHierSampledInd[level-1], currPtHierSampledInd[level-1]), axis=0) if accumPtHierSampledInd[level-1].size else currPtHierSampledInd[level-1]
                    #accumPtHierAabbMin = np.concatenate((accumPtHierAabbMin, currPtHierAabbMin), axis=0) if accumPtHierAabbMin.size else currPtHierAabbMin
                    #accumPtHierAabbMax = np.concatenate((accumPtHierAabbMax, currPtHierAabbMax), axis=0) if accumPtHierAabbMax.size else currPtHierAabbMax
                    #for vq_i in range(self.numVQs_):
                        #accumLabels[vq_i].append(currLabels[vq_i])
                        #accumLabels_mult[vq_i].append(currLabels_mult[vq_i])
                        #accumSigns[vq_i].append(currSignLabels[vq_i])
                        #accumReferenceValues[vq_i][2].append(currRefView[vq_i])
                    ## Update the counters and the iterator.
                    #numPtsInBatch  += currModelNumPts
                    #numModelInBatch += 1
                    #if not repeatModelInBatch:
                        #self.iterator_ += 1
                    #else:
                        #self.reset_ptHcaches()
            
        #if repeatModelInBatch:
            #self.iterator_ += 1
            #accumAreas = [self.areas_[currModelIndex]]
            #accumModels = [self.objfileList_[currModelIndex]]
            #for vq_i in range(self.numVQs_):
                #if vq_i == self.uv_index_:
                    #accumReferenceValues[vq_i][0]= [self.referenceValues_[vq_i][0]]
                    #accumReferenceValues[vq_i][1] = [self.referenceValues_[vq_i][1]]
                #else:
                    #accumReferenceValues[vq_i][0]= [self.referenceValues_[vq_i][currModelIndex][0]]
                    #accumReferenceValues[vq_i][1] = [self.referenceValues_[vq_i][currModelIndex][1]]
                #accumReferenceValues[vq_i][2] = [currRefView[vq_i]]
                #accumModelParams = [self.modelParams_[currModelIndex]]
            
        #accumPtHierPts[0] = None
        #accumPtHierBatchIds[0] = None
        #accumPtHierFeatures[0] = None
        #while len(accumPtHierAabbMax) < self.batchSize_:
            #accumPtHierAabbMin = np.concatenate((accumPtHierAabbMin, np.ones([1,3])), axis=0)
            #accumPtHierAabbMax = np.concatenate((accumPtHierAabbMax, -np.ones([1,3])), axis=0)
        #for vq_i in range(self.numVQs_):
            #if not vq_i == self.uv_index_:
                #for i in range(len(accumLabels[vq_i])):
                    #add_label = accumLabels[vq_i][i][0].reshape(1,3)
                    #while len(accumLabels[vq_i][i]) < numLabels[vq_i]:
                        #accumLabels[vq_i][i] = np.concatenate((accumLabels[vq_i][i], add_label), axis=0)
        #for vq_i in range(self.numVQs_):
            #if not vq_i == self.uv_index_:
                #for i in range(len(accumLabels_mult[vq_i])):
                    #add_label = accumLabels_mult[vq_i][i][0].reshape(1,3)
                    #while len(accumLabels_mult[vq_i][i]) < numLabels_mult[vq_i]:
                        #accumLabels_mult[vq_i][i] = np.concatenate((accumLabels_mult[vq_i][i], add_label), axis=0)
                        
        
        #PtHier[0] = accumPtHierPts
        #PtHier[1] = accumPtHierSampledInd
        #PtHier[2] = accumPtHierFeatures
        #PtHier[3] = accumPtHierBatchIds
        #PtHier[4] = accumPtHierAabbMin
        #PtHier[5] = accumPtHierAabbMax
        
        #return numModelInBatch, PtHier, accumLabels, accumLabels_mult, accumSigns, accumModels, accumModelParams, accumAreas, accumReferenceValues, accumInvRotationMatrix

    def get_next_batch(self, repeatModelInBatch = False):
        """Method to get the next batch of models.
        
        Args:
            repeatModelInBatch (bool): Boolean that indicates if the batch will be filled with
                the same model.            

        Returns:
            numModelInBatch (int): Number of models in the batch.
            accumPts (nx3 np.array): List of points of the batch.
            accumBatchIds (nx1 np.array): List of model indentifiers within the batch 
                for each point.
            accumFeatures (nxm np.array): List of point features.
            accumLabels (nxl np.array): List of point labels. If the dataset does not contain
                point labels, None is returned instead.
            accumCat (numModelInBatchx1 or nx1 np.array): List of categories of each model 
                in the batch. If the dataset was initialize with pointCategories equal to True,
                the category of each model is provided for each point. If the dataset does not
                contain category information, None is returned instead.
            accumPaths (array of strings): List of paths to the models used in the batch.
        """
        #initialize variables
        accumLabels = [[] for i in range(self.numVQs_)]
        accumLabels_mult = [[] for i in range(self.numVQs_)]
        accumSigns = [ [] for i in range(self.numVQs_)]
        accumModels = []
        accumModelParams = []
        accumAreas = []
        accumReferenceValues = [[[],[],[]] for i in range(self.numVQs_)]
        accumVQs = [ [] for i in range(self.numVQs_)]
        accumVQs2D = [ [] for i in range(self.numVQs_)]
        accumRotationMatrix = []
        accumInvRotationMatrix = []
        accumRotationMatrix = []
        accumIndices = []
        accumPts = np.array([])
        accumFeatures = np.array([])
        accumBatchIds = np.array([])
        
        numLabels = [0 for i in range(self.numVQs_)]
        numLabels_mult = [0 for i in range(self.numVQs_)]
        numModelInBatch = 0
        numPtsInBatch = 0

        # Iterate over the elements on the batch.
        for i in range(self.batchSize_):
            # Check if there are enough models left.
            if self.iterator_ < len(self.randomSelection_):
            
                # Check if the model fit in the batch.
                currModelIndex = self.randomSelection_[self.iterator_]
                currModel = self.fileList_[currModelIndex]
                currModelNumPts = self.numPts_[currModelIndex]
                accumIndices.append(currModelIndex)
                accumModels.append(self.objfileList_[currModelIndex])
                accumModelParams.append(self.modelParams_[currModelIndex])
                accumAreas.append(self.areas_[currModelIndex])
                currLabels = []
                currLabels_mult = []
                currSignLabels = []
                currRefView = []
                for vq_i in range(self.numVQs_):
                    if vq_i == self.uv_index_ or vq_i == self.fv_index_:
                        #### fix UV
                        currLabels.append(self.labels_[vq_i])
                        sign = np.sum((self.labels_[vq_i]<0).astype(int) * 2**np.array([0,1,2]))
                        currSignLabels.append(sign)
                        currLabels_mult.append(self.labels_[vq_i])
                        numLabels[vq_i] = 1
                        accumReferenceValues[vq_i][0].append(self.referenceValues_[vq_i][0])
                        accumReferenceValues[vq_i][1].append(self.referenceValues_[vq_i][1])
                        currRefView.append(self.referenceValues_[vq_i][2:])
                    else:
                        bestView = self.referenceValues_[vq_i][currModelIndex][2:]
                        sign = np.sum((bestView<0).astype(int) * 2**np.array([0,1,2]))
                        currLabels.append(bestView.reshape(-1,3))
                        currSignLabels.append(sign)
                        numLabels[vq_i] = max(numLabels[vq_i], len(currLabels[vq_i]))
                        currLabels_mult.append(self.labels_[vq_i][currModelIndex])
                        numLabels_mult[vq_i] = max(numLabels_mult[vq_i], len(currLabels_mult[vq_i]))
                        accumReferenceValues[vq_i][0].append(self.referenceValues_[vq_i][currModelIndex][0])
                        accumReferenceValues[vq_i][1].append(self.referenceValues_[vq_i][currModelIndex][1])
                        currRefView.append(self.referenceValues_[vq_i][currModelIndex][2:])
                        accumVQs[vq_i].append(self.vqs_[vq_i][currModelIndex])
                        if not self.augment_:
                            accumVQs2D[vq_i].append(self.vqs2D_[vq_i][currModelIndex])
                
                
                # If the batch has a limited number of points, check if the model fits in the batch.
                if (self.maxPtsxBatch_ == 0) or ((numPtsInBatch+currModelNumPts) <= self.maxPtsxBatch_):

                    # Determine the category of the model if it is necesary.
                    currModelCat = None
                    if self.useCategories_:
                        currModelCat = self.categories_[currModelIndex]
                    currPts, _, currFeatures, _ = self._load_model_(currModel)
                    
                    # Augment data.
                    
                    if self.augment_symmetric_:
                        currPts = self._augment_data_symmetric_deformation(currPts)
                        
                    if self.augment_:
                        currPts, rotationMatrix, invRotationMatrix= self._augment_data_rot_(currPts, self.augmentMainAxis_, self.augmentSmallRotations_)
                        for vq_i in range(self.numVQs_):
                            currLabels[vq_i],_, _= self._augment_data_rot_(currLabels[vq_i], self.augmentMainAxis_, self.augmentSmallRotations_, rotationMatrix)
                            currSignLabels[vq_i] = np.sum(( currLabels[vq_i]<0).astype(int) * 2**np.array([0,1,2]))
                            currLabels_mult[vq_i],_, _= self._augment_data_rot_(currLabels_mult[vq_i], self.augmentMainAxis_, self.augmentSmallRotations_, rotationMatrix)
                            currRefView[vq_i] = np.dot(currRefView[vq_i].reshape(1,3), rotationMatrix)[0]        
                        if self.useNormalsAsFeatures_:
                            currFeatures,_, _= self._augment_data_rot_(currFeatures, self.augmentMainAxis_, self.augment_noise_, rotationMatrix) 
                        for vq_i in range(self.numVQs_):
                            currVQ2QD = self._get_2D_VQ_image(accumVQs[vq_i][-1], self.sphere_coordinates, rotationMatrix) 
                            accumVQs2D[vq_i].append(currVQ2QD)
                        # Append Rotation Matrix
                        accumInvRotationMatrix.append(invRotationMatrix)
                        accumRotationMatrix.append(rotationMatrix)
                    if self.augment_noise_:
                        currPts = self._augment_data_noise_(currPts, self.noise_level_)
                    # Append the current model to the batch.
                    accumPts = np.concatenate((accumPts, currPts), axis=0) if accumPts.size else currPts
                    auxBatchIds = np.full([len(currPts),1],i,dtype = int)
                    accumBatchIds = np.concatenate((accumBatchIds, auxBatchIds), axis=0) if accumBatchIds.size else auxBatchIds
                    accumFeatures = np.concatenate((accumFeatures, currFeatures), axis=0) if accumFeatures.size else currFeatures
                    
                    for vq_i in range(self.numVQs_):
                        accumLabels[vq_i].append(currLabels[vq_i])
                        accumLabels_mult[vq_i].append(currLabels_mult[vq_i])
                        accumSigns[vq_i].append(currSignLabels[vq_i])
                        accumReferenceValues[vq_i][2].append(currRefView[vq_i])
                    # Update the counters and the iterator.
                    numPtsInBatch  += currModelNumPts
                    numModelInBatch += 1
                    if not repeatModelInBatch:
                        self.iterator_ += 1
            
        if repeatModelInBatch:
            self.iterator_ += 1
            accumAreas = [self.areas_[currModelIndex]]
            accumModels = [self.objfileList_[currModelIndex]]
            for vq_i in range(self.numVQs_):
                accumReferenceValues[vq_i][0]= [self.referenceValues_[vq_i][currModelIndex][0]]
                accumReferenceValues[vq_i][1] = [self.referenceValues_[vq_i][currModelIndex][1]]
                accumReferenceValues[vq_i][2] = [currRefView[vq_i]]
                accumModelParams = [self.modelParams_[currModelIndex]]
            
        for vq_i in range(self.numVQs_):
            if not vq_i == self.uv_index_ or vq_i == self.fv_index_:
                for i in range(len(accumLabels[vq_i])):
                    add_label = accumLabels[vq_i][i][0].reshape(1,3)
                    while len(accumLabels[vq_i][i]) < numLabels[vq_i]:
                        accumLabels[vq_i][i] = np.concatenate((accumLabels[vq_i][i], add_label), axis=0)
        for vq_i in range(self.numVQs_):
            if not vq_i == self.uv_index_ or vq_i == self.fv_index_:
                for i in range(len(accumLabels_mult[vq_i])):
                    add_label = accumLabels_mult[vq_i][i][0].reshape(1,3)
                    while len(accumLabels_mult[vq_i][i]) < numLabels_mult[vq_i]:
                        accumLabels_mult[vq_i][i] = np.concatenate((accumLabels_mult[vq_i][i], add_label), axis=0)
                        

        return numModelInBatch, accumPts, accumFeatures, accumBatchIds, accumModels, accumReferenceValues, accumVQs2D, accumVQs, accumInvRotationMatrix
