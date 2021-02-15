'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \brief Code with helper function for rendering .objs

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np

def read_model2(myFile):
    # loads a mesh from a .obj file, centers the mesh around [0,0,0]
    
    # Args:
    #   myFile (str): direction to the .obj file
    
    # Returns:
    #   vertices (numpy nx3): array of vertex coordinates
    #   normals (numpy nx3): array of normal vectors (per vertex)
    #   faces (numpy mx3x2): array of triangle ids, format: [triangle_id, vertex_id, [coordinate, normal]]
    #   coordMin and coordMax (numpy x3): Edges of the bounding box
    
    vertices = []
    normals = []
    faces = []
    # load file
    with open(myFile, 'r') as modelFile:        
        for line in modelFile:
            lineElements = line.split()
            if len(lineElements) > 0:
                if lineElements[0] == "v":
                    vertices.append([lineElements[1], lineElements[2],lineElements[3]])
                elif lineElements[0] == "vn":
                    normals.append([lineElements[1], lineElements[2], lineElements[3]])
                elif lineElements[0] == "f":
                    vertex1 = lineElements[1].split('/')
                    vertex2 = lineElements[2].split('/')
                    vertex3 = lineElements[3].split('/')
                    faces.append([[vertex1[0],vertex1[2]], [vertex2[0], vertex2[2]], [vertex3[0], vertex3[2]]])
    # convert to numpy arrays
    vertices = np.array(vertices, dtype = float)
    normals = np.array(normals, dtype = float)
    faces = np.array(faces, dtype = int)-1
    # center the vertex coordinates
    coordMax = np.amax(vertices, axis=0)
    coordMin = np.amin(vertices, axis=0)
    center = (coordMax + coordMin)*0.5
    vertices = vertices-center
    
    return vertices, normals, faces, coordMin-center, coordMax-center

def read_and_scale(myFile, scale = 1):
    vertices, normals, faces, coordMin, coordMax = read_model2(myFile)
    resize = (coordMax[0] -coordMin[0])/scale
    vertices /= resize
    coordMin /= resize
    coordMax /= resize
    return [vertices, normals, faces, coordMin, coordMax], resize


def read_and_generate_buffers(myFile):
    # execute read_file and generate_rendering_buffers together 
    [vertexs, normals, faces, coordMin, coordMax] = read_model2(myFile)
    rendVert, rendVertTrianIds, rendFaces = generate_rendering_buffers(vertexs, normals, faces)
    
    return rendVert, rendVertTrianIds, rendFaces, coordMin, coordMax


def generate_rendering_buffers(vertexs, normals, faces):
    # inputs must be np.arrays
    # faces must be of dtype int
    numFaces = len(faces)

    rendVerts = np.concatenate((vertexs[faces[:,:,0]],np.ones([numFaces,3,1]),normals[faces[:,:,0]]),axis=2).reshape(-1)

    renderIndexs = np.linspace(0,3*numFaces-1, 3*numFaces, dtype = int)
    temp = np.linspace(1,numFaces,numFaces).reshape(-1,1)
    rendVertsTrianIds = np.concatenate((temp,temp,temp),axis=1).reshape(-1)

    return rendVerts, rendVertsTrianIds, renderIndexs


def generate_rendering_buffers_old(vertexs, normals, faces):
    rendVerts = []
    rendVertsTrianIds = []
    renderIndexs = []
    for it,face in enumerate(faces):
        vert1 = face[0]
        indexVert1 = len(rendVerts)//7
        rendVerts.append(vertexs[vert1[0]][0])
        rendVerts.append(vertexs[vert1[0]][1])
        rendVerts.append(vertexs[vert1[0]][2])
        rendVerts.append(1.0)
        rendVerts.append(normals[vert1[1]][0])
        rendVerts.append(normals[vert1[1]][1])
        rendVerts.append(normals[vert1[1]][2])
        rendVertsTrianIds.append(it+1)

        vert2 = face[1]
        indexVert2 = len(rendVerts)//7
        rendVerts.append(vertexs[vert2[0]][0])
        rendVerts.append(vertexs[vert2[0]][1])
        rendVerts.append(vertexs[vert2[0]][2])
        rendVerts.append(1.0)
        rendVerts.append(normals[vert2[1]][0])
        rendVerts.append(normals[vert2[1]][1])
        rendVerts.append(normals[vert2[1]][2])
        rendVertsTrianIds.append(it+1)


        vert3 = face[2]

        indexVert3 = len(rendVerts)//7
        rendVerts.append(vertexs[vert3[0]][0])
        rendVerts.append(vertexs[vert3[0]][1])
        rendVerts.append(vertexs[vert3[0]][2])
        rendVerts.append(1.0)
        rendVerts.append(normals[vert3[1]][0])
        rendVerts.append(normals[vert3[1]][1])
        rendVerts.append(normals[vert3[1]][2])
        rendVertsTrianIds.append(it+1)

        renderIndexs.append(indexVert1)
        renderIndexs.append(indexVert2)
        renderIndexs.append(indexVert3)

    return np.array(rendVerts), np.array(rendVertsTrianIds), np.array(renderIndexs)



