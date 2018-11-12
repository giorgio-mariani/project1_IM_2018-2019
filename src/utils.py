# -*- coding: utf-8 -*-
import os

import cv2 
import numpy as np

import params


def load_image_depth(filename, width, height):
    import struct
    with open(filename, "rb") as f:
        def read_float():
            bs = f.read(4)
            if bs=="":
                return None 
            else:
                return struct.unpack('f', bs)[0]
        
        I = np.zeros([height, width], dtype=np.float32)
        h, w = 0, 0
        disp = read_float()
        while disp is not None:
            I[h,w] = disp
            
            w = (w + 1) % width
            h = h + 1 if w==0 else h 
            disp = read_float()            
    return I

#------------------------------------------------------------------------------
class Camera():
    def __init__(self, filename, height, width):
        self.h = height
        self.w = width
        Ks, Rs, Ts = self._get_camera_matrices(filename)
        
        self.K = Ks[0]
        self.Rs = Rs
        self.Ts = Ts
        
        o_w = (self.K[0,2]+0.5)*2
        o_h = (self.K[1,2]+0.5)*2
              
        self.K[0, 0] = self.K[0,0]*width/o_w
        self.K[1, 1] = self.K[1,1]*height/o_h
        
        self.K[0, 2] = width/2.0 + 0.5
        self.K[1, 2] = height/2.0 + 0.5
       
    def _get_camera_matrices(self, filename):
        with open(filename,'r') as f:
            frame_num = int(f.readline())
            f.readline()
            
            def read_vector():
                vec = f.readline().split()
                return np.array([float(x) for x in vec], dtype=np.float32)
                
            def read_matrix():
                r1 = read_vector()
                r2 = read_vector()
                r3 = read_vector()
                return np.matrix([r1,r2,r3], dtype=np.float32)
            
            K_sequence = [None]*frame_num
            R_sequence = [None]*frame_num
            T_sequence = [None]*frame_num
            
            for i in range(frame_num):
                K_sequence[i] = read_matrix()
                R_sequence[i] = read_matrix()
                T_sequence[i] = np.asmatrix(read_vector())
                f.readline()
                f.readline()
        return K_sequence, R_sequence, T_sequence

#------------------------------------------------------------------------------
class PictureSequence():
    def __init__(self, 
                 directory, 
                 file_extension, 
                 height=None, 
                 width=None, 
                 maxlength=None):
        
        if height is not None: assert type(height) == int
        if width is not None: assert type(width) == int
        if maxlength is not None: assert type(maxlength) == int
        assert type(directory) == str
        assert type(file_extension) == str

        self._imgfiles = self._get_sequence_files(directory, file_extension)
        length = min(len(self._imgfiles), maxlength)
        self._imgfiles = self._imgfiles[:length]
            
        h, w, _ = cv2.imread(self._imgfiles[0], cv2.IMREAD_COLOR).shape
        self.height = h if height is None else height
        self.width = w if width is None else width
        
    def _get_sequence_files(self, directory_name, target_extension):
        sequence_files = []
        for f in os.listdir(directory_name):
            full_name = os.path.join(directory_name, f)
            if os.path.isfile(full_name):
                _, extension = os.path.splitext(full_name)
                if target_extension == extension:
                    sequence_files.append(full_name)
                    
        if len(sequence_files) == 0:
            raise StandardError('No sequence file in the directory')
        return sequence_files
    
    def get_filename(self, index, extension=True):
        filename = os.path.basename(self._imgfiles[index])
        if not extension:
            filename, extension = os.path.splitext(filename)
        return filename
        
    def __getitem__(self, index):
        img = cv2.imread(self._imgfiles[index])
        img = cv2.resize(img, (self.width,self.height))
        return img
    
    def __len__(self):
        return len(self._imgfiles)
    
    
    
#------------------------------------------------------------------------------
class DepthSequence():
    def __init__(self, directory, pic_sequence):
        
        assert isinstance(pic_sequence, PictureSequence)
        assert type(directory) == str
        
        self._depthfiles = self._get_depth_files(directory, pic_sequence)
        self.height = pic_sequence.height
        self.width = pic_sequence.width
        
    def _get_depth_files(self, directory_name, pic_sequence):
        ext = '.npy'
        seq_files = []

        for i in range(len(pic_sequence)):
            filename = pic_sequence.get_filename(i, extension=False)
            seq_files.append(os.path.join(directory_name, filename+ext))
                    
        if len(seq_files) == 0:
            raise StandardError('No sequence file in the directory')
        return seq_files
    
    def __getitem__(self, index):
        depthmap = np.load(self._depthfiles[index])
        
        assert depthmap.shape == (self.height, self.width)
        assert depthmap.dtype == np.uint16
        return depthmap
    
    def __len__(self):
        return len(self._depthfiles)
    
#------------------------------------------------------------------------------
        
def show_image(img, secs=0, close_after=True):
    cv2.imshow('image', cv2.UMat(img))
    cv2.waitKey(secs)
    if close_after:
        cv2.destroyAllWindows()

def show_depthmap(depth_map, secs=0, close_after=True):
    depth_map = np.array(depth_map)
    m = depth_map.min()
    M = depth_map.max()
    img = np.uint8((depth_map-m)/(M-m)*255)
    show_image(img, secs, close_after)

###############################################################################

def test_projection():
    import compute_energy as ce
    
    sequence = Sequence('../video sequences/Lawn/src/', '.jpg')
    camera = Camera('../video sequences/Lawn/cameras.txt', 
                          height=sequence.height, 
                          width=sequence.width)
    
    h, w = sequence.height, sequence.width
    
    [X, Y] = np.meshgrid(
                np.arange(w, dtype=np.float32),
                np.arange(h, dtype=np.float32),
                indexing='xy')
    
    X, Y, Z = (np.expand_dims(X,axis=0), 
               np.expand_dims(Y,axis=0), 
               np.ones([1, h, w], np.float32))
    
    coorsxy = np.concatenate([X, Y, Z], axis=0)
    coorsxy = np.reshape(coorsxy, [3,-1])
    
    i=40
    filename = '../video sequences/Lawn/depth/_depth'+str(0)+'.raw'
    depth = np.reshape(load_image_depth(filename, w, h), [1,h*w])
    print depth.max(), depth.min()
    remap = ce.compute_candidates(
            camera,
            0,
            i,
            coorsxy, 
            depth)
    
    # compute norm of difference
    img_i_projected = cv2.remap(sequence[i], remap, None, cv2.INTER_NEAREST, borderValue=[128,128,128])
    img = cv2.resize(img_i_projected,(w*2,h*2))
    show_image(img)
    
    
    

