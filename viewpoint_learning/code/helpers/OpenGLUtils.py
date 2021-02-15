'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \brief Code with openGL utils.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import math
import numpy as np

import OpenGL
from OpenGL import GL

FLOAT_SIZE = 4
INT_SIZE = 4
SHORT_SIZE = 2

class ShaderLoader:
    
    def __init__(self):
        pass

    def _link_(self, program):
        GL.glLinkProgram(program)

        status = GL.glGetProgramiv(program, GL.GL_LINK_STATUS)
        if not status:
            log = GL.glGetProgramInfoLog(program)
            raise RuntimeError("Linking failue: "+str(log))

    def _compile_(self, shaderPath, shaderCode, shaderType):
        shader = GL.glCreateShader(shaderType)

        GL.glShaderSource(shader, shaderCode)

        GL.glCompileShader(shader)

        status = GL.glGetShaderiv(shader,GL.GL_COMPILE_STATUS)
        if not status:
            log = GL.glGetShaderInfoLog(shader)
            raise RuntimeError("Compile failure in shader: "+shaderPath+ "\n "+str(log))

        return shader

    def _load_shader_code_(self, shaderPath):
        shaderCode = ""
        with open(shaderPath, 'r') as modelFile:        
            for line in modelFile:
                shaderCode += line
        return shaderCode
    
    def load_shader(self, shaderPathList, shaderTypes):
        currProgram = GL.glCreateProgram()

        shadeList = []
        for shaderPath, shaderType in zip(shaderPathList, shaderTypes):
            shadeList.append(self._compile_(shaderPath, 
                self._load_shader_code_(shaderPath), shaderType))
        
        for shade in shadeList:
            GL.glAttachShader(currProgram,shade)
        
        self._link_(currProgram)

        for shade in shadeList:
            GL.glDetachShader(currProgram,shade)
            GL.glDeleteShader(shade)

        return currProgram

    
class MeshRenderer:
    
    def __init__(self, verts, trianIds, trians, attribLoc):
        self.verts_ = verts
        self.trianIds_ = trianIds
        self.trians_ = trians
        self.attribLoc_ = attribLoc
        self._create_buffers_()
        self._init_vao_()

    def _create_buffers_(self):
        flattenVerts = self.verts_.tolist()
        self.vbo_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_)
        ArrayType = (GL.GLfloat*len(flattenVerts))
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(flattenVerts)*FLOAT_SIZE,
                        ArrayType(*flattenVerts), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER,0)

        flattenVertsTrianIds = self.trianIds_.astype('int32').tolist()
        self.vbo2_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo2_)
        ArrayType = (GL.GLint*len(flattenVertsTrianIds))
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(flattenVertsTrianIds)*INT_SIZE,
                        ArrayType(*flattenVertsTrianIds), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER,0)

        flattenIndexs = self.trians_.astype('int32').tolist()
        self.ibo_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ibo_)
        ArrayType = (GL.GLint*len(flattenIndexs))
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, len(flattenIndexs)*INT_SIZE,
                        ArrayType(*flattenIndexs), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER,0)

    def _init_vao_(self):
        self.vao_ = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao_)
        

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_)
        GL.glEnableVertexAttribArray(self.attribLoc_[0])
        GL.glVertexAttribPointer(self.attribLoc_[0], 4, GL.GL_FLOAT, GL.GL_FALSE, 
            7*FLOAT_SIZE, GL.GLvoidp(0))
        GL.glEnableVertexAttribArray(self.attribLoc_[1])
        GL.glVertexAttribPointer(self.attribLoc_[1], 3, GL.GL_FLOAT, GL.GL_FALSE, 
            7*FLOAT_SIZE, GL.GLvoidp(4*FLOAT_SIZE))

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo2_)
        GL.glEnableVertexAttribArray(self.attribLoc_[2])
        GL.glVertexAttribPointer(self.attribLoc_[2], 1, GL.GL_INT, GL.GL_FALSE, 
            0, GL.GLvoidp(0))
            

        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ibo_)
        GL.glBindVertexArray(0)

    def render_mesh(self):
        GL.glBindVertexArray(self.vao_)
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.trians_), GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)

    def delete_buffers(self):
        GL.glDeleteBuffers(3, [self.vbo_, self.vbo2_, self.ibo_])
        GL.glDeleteVertexArrays(1, [self.vao_])


class Camera:

    def __init__(self, vrp, obs, upVec, ar, fov, zNear, zFar):
        self.vrp_ = np.array(vrp)
        self.obs_ = np.array(obs)
        self.upVec_ = np.array(upVec)
        self.ar_ = ar
        self.fov_ = fov
        self.zNear_ = zNear
        self.zFar_ = zFar


    def _normalize_(self, v):
        m = math.sqrt(np.sum(v ** 2))
        if m == 0:
            return v
        return v / m


    def rotate_y(self, angle):
        cosVal = np.cos(angle)
        sinVal = np.sin(angle)
        T = np.array([[cosVal, 0.0, -sinVal],
                       [0.0, 1.0, 0.0],
                       [sinVal, 0.0, cosVal]])
        auxPos = self.obs_ - self.vrp_
        auxPos = np.dot(T, auxPos)[:3]
        self.obs_ = auxPos + self.vrp_


    def rotate_x(self, angle):
        F = self.vrp_ - self.obs_
        f = self._normalize_(F)
        U = self._normalize_(self.upVec_)
        axis = np.cross(f, U)
        
        x, y, z = self._normalize_(axis)
        s = math.sin(-angle)
        c = math.cos(-angle)
        nc = 1 - c
        T = np.array([[x*x*nc +   c, x*y*nc - z*s, x*z*nc + y*s],
                        [y*x*nc + z*s, y*y*nc +   c, y*z*nc - x*s],
                        [x*z*nc - y*s, y*z*nc + x*s, z*z*nc +   c]])
                        

        auxPos = self.obs_ - self.vrp_
        auxPos = np.dot(T, auxPos)
        self.obs_ = auxPos + self.vrp_


    def get_view_natrix(self):
        F = self.vrp_ - self.obs_
        f = self._normalize_(F)
        U = self._normalize_(self.upVec_)
        s = self._normalize_(np.cross(f, U))
        u = self._normalize_(np.cross(s, f))
        M = np.matrix(np.identity(4))
        M[:3,:3] = np.vstack([s,u,-f])
        T = np.matrix([[1.0, 0.0, 0.0, -self.obs_[0]],
                       [0.0, 1.0, 0.0, -self.obs_[1]],
                       [0.0, 0.0, 1.0, -self.obs_[2]],
                       [0.0, 0.0, 0.0, 1.0]])
        return  M * T


    def get_projection_matrix(self):
        s = 1.0/math.tan(math.radians(self.fov_)/2.0)
        sx, sy = s / self.ar_, s
        zz = (self.zFar_+self.zNear_)/(self.zNear_-self.zFar_)
        zw = (2*self.zFar_*self.zNear_)/(self.zNear_-self.zFar_)
        return np.matrix([[sx,0,0,0],
                        [0,sy,0,0],
                        [0,0,zz,zw],
                        [0,0,-1,0]])


class FrameBuffer:

    def __init__(self, bufferFormats, width, height):
        if len(bufferFormats) > 6:
            raise RuntimeError("Number of attachement to buffer too high: "+str(len(bufferFormats)))

        # Create the frame buffer.
        self.textures_ = []
        self.fbo_ = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo_)
        
        self.colorAttList_ = [GL.GL_COLOR_ATTACHMENT0,
                              GL.GL_COLOR_ATTACHMENT1,
                              GL.GL_COLOR_ATTACHMENT2,
                              GL.GL_COLOR_ATTACHMENT3,
                              GL.GL_COLOR_ATTACHMENT4,
                              GL.GL_COLOR_ATTACHMENT5,
                              GL.GL_COLOR_ATTACHMENT6]
        self.formatDict_ = {
                GL.GL_RGBA32F: (GL.GL_RGBA, GL.GL_FLOAT),
                GL.GL_RGBA: (GL.GL_RGBA, GL.GL_UNSIGNED_BYTE),
                GL.GL_R32I: (GL.GL_RED_INTEGER, GL.GL_INT),
                GL.GL_RGBA32I: (GL.GL_RGBA_INTEGER, GL.GL_INT)}

         # Create the textures of the frame buffer.
        for it, currFormat in enumerate(bufferFormats):
            if not(currFormat in self.formatDict_):
                raise RuntimeError("The texture format is not in the dictionary: "+str(currFormat))
            myFormat = self.formatDict_[currFormat]
            texture = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, currFormat, width, height, 0, myFormat[0], myFormat[1], None)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, self.colorAttList_[it], GL.GL_TEXTURE_2D, texture, 0)
            self.textures_.append(texture)

        # Creaet the render buffer.
        self.rbo_ = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.rbo_)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH24_STENCIL8, width, height)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_STENCIL_ATTACHMENT, GL.GL_RENDERBUFFER, self.rbo_)

        # Check if the frame buffer was created properly.
        if not GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Bind frame buffer failed")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, 0)

    def bind(self):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo_)
        GL.glDrawBuffers(len(self.textures_),  self.colorAttList_)

    def get_texture(self, index):
        return self.textures_[index]
