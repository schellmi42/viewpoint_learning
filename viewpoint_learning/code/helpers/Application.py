'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \brief Code to render an image of an .obj

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import sys
import math
import argparse
import os
import numpy as np
from PIL import Image

import OpenGL
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
from OpenGL import GL

import pygame as pg

from OpenGLUtils import ShaderLoader, MeshRenderer, Camera, FrameBuffer
from MeshHelpers2 import read_model2, generate_rendering_buffers_old

FLOAT_SIZE = 4
INT_SIZE = 4

class GLScene:
    
    def __init__(self, width, height):

        pg.init()
        SCREEN = pg.display.set_mode((width, height),  pg.OPENGL | pg.DOUBLEBUF)#pg.FULLSCREEN
        pg.display.iconify()
        
        # Configure OpenGL state.
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        GL.glFrontFace(GL.GL_CCW)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LEQUAL)        

        # Load the shaders.
        self.shaderLoader_ = ShaderLoader()
        self.shaderMesh_ = self.shaderLoader_.load_shader(
            ["vertexShader.glsl", "pixelShader.glsl"],
            [GL.GL_VERTEX_SHADER, GL.GL_FRAGMENT_SHADER])
        self.worldViewProjMatrixUnif_ = GL.glGetUniformLocation(self.shaderMesh_, "worldViewProjMatrix")
        self.worldViewMatrixUnif_ = GL.glGetUniformLocation(self.shaderMesh_, "worldViewMatrix")
        self.posLoc_ = GL.glGetAttribLocation(self.shaderMesh_, "sPos")
        self.normalLoc_ = GL.glGetAttribLocation(self.shaderMesh_, "sNormal")
        self.trianIdLoc_ = GL.glGetAttribLocation(self.shaderMesh_, "sId")

        # Resize viewport.
        self.width_ = width
        self.height_ = height
        GL.glViewport(0, 0, width, height)

        # Initialize frame buffer.
        self.frameBuffer_ = FrameBuffer([GL.GL_R32I, GL.GL_RGBA32F, GL.GL_RGBA32F], width, height)


    def generate_images(self, ptMin, ptMax, vertexs, vertsTrianIds, faces, viewDir, only_texIds = False):
        
        # Create the camera.
        aabbSize = math.sqrt(np.sum((ptMax-ptMin) ** 2))
        auxViewLength = math.sqrt(np.sum((viewDir) ** 2))
        distance = aabbSize/math.sin((45.0*math.pi)/180.0)
        self.camera_ = Camera(
                [0.0, 0.0, 0.0], 
                [(viewDir[0]*distance)/auxViewLength, (viewDir[1]*distance)/auxViewLength, (viewDir[2]*distance)/auxViewLength], 
                [0.0, -1.0, 0.0],
                float(self.width_)/float(self.height_),
                45.0, 0.1, aabbSize*5.0)
        self.viewMat_ = self.camera_.get_view_natrix()
        self.projMat_ = self.camera_.get_projection_matrix()
        self.worldViewProjMat_ = self.projMat_ * self.viewMat_
        
        # Load the mesh.
        self.mesh_ = MeshRenderer(vertexs, vertsTrianIds, faces, [self.posLoc_, self.normalLoc_, self.trianIdLoc_])
        
        self.frameBuffer_.bind()
        GL.glClearColor(0,0,0,0)
         
        #Render Mesh
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)     
        GL.glUseProgram(self.shaderMesh_)
        GL.glBindFragDataLocation(self.shaderMesh_, 0, "outId")
        GL.glBindFragDataLocation(self.shaderMesh_, 1, "outNormal")
        GL.glBindFragDataLocation(self.shaderMesh_, 2, "outColor")
        GL.glUniformMatrix4fv(self.worldViewProjMatrixUnif_, 1, GL.GL_TRUE, np.ascontiguousarray(self.worldViewProjMat_, dtype=np.float32))
        GL.glUniformMatrix4fv(self.worldViewMatrixUnif_, 1, GL.GL_TRUE, np.ascontiguousarray(self.viewMat_, dtype=np.float32))
        self.mesh_.render_mesh()
        GL.glUseProgram(0)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.frameBuffer_.get_texture(0))
        texIds = GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_RED_INTEGER, GL.GL_INT)
        if not only_texIds:
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.frameBuffer_.get_texture(1))
            texNormals = GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, GL.GL_FLOAT)

            GL.glBindTexture(GL.GL_TEXTURE_2D, self.frameBuffer_.get_texture(2))
            texColors = GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, GL.GL_FLOAT)

            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        else:
            texNormals = None
            texColors = None
        self.mesh_.delete_buffers()

        return texIds, texNormals, texColors
        

##################################################################### MAIN


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to train GINN')
    parser.add_argument('--in3DModel', help='3D input model', required=True)
    parser.add_argument('--inViewDir', help='Input view direction', nargs=3, type=float, required=True)
    args = parser.parse_args()    

    # #Load the model
    vertexs, normals, faces, coordMin, coordMax = read_model2(args.in3DModel)
    rendVert, rendVertTrianIds, rendFaces = generate_rendering_buffers_old(vertexs, normals, faces)

    print("Loaded model "+args.in3DModel)
    print("Vertexs: "+str(len(rendVert)/7))
    print("faces: "+str(len(rendFaces)/3))
    
    #Render
    MyGL = GLScene(1024, 1024)


    texIds, texNormals, texColors = MyGL.generate_images(coordMin, coordMax, rendVert, rendVertTrianIds, rendFaces, np.array(args.inViewDir))

    texNormals = ((texNormals * 0.5 + 0.5)*255.0)[:,:,:3]
    img = Image.fromarray(texNormals.astype('uint8'), 'RGB')
    img.save('normals.png')

    texIds = np.reshape(texIds, (1024, 1024, 1))
    texIdsExp = np.concatenate((texIds, texIds), axis=2)
    texIdsExp = np.concatenate((texIdsExp, texIds), axis=2)
    texIdsExp[:,:,0] = texIdsExp[:,:,0]%255
    texIdsExp[:,:,1] = texIdsExp[:,:,1]/255
    texIdsExp[:,:,2] = texIdsExp[:,:,2]*0
    img = Image.fromarray(texIdsExp.astype('uint8'), 'RGB')
    img.save('ids.png')

    texColorsExp = (texColors*255.0)[:,:,:3]
    img = Image.fromarray(texColorsExp.astype('uint8'), 'RGB')
    img.save('colors.png')
