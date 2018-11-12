# -*- coding: utf-8 -*-
import cv2 
import numpy as np
from time import time
from tqdm import tqdm

import utils 
import params

def compute_energy_init(
        sequence, 
        camera, 
        curr_frame):
    
    # get paramaters
    h, w = camera.h, camera.w
    sigma_c = params.SIGMA_C
    levels = params.DEPTH_LEVELS
    depth_vector = params.DEPTH_VEC
    
    # get image at frame cf
    img_o = np.float32(sequence[curr_frame])
    
    # initialize indices coordinates
    [X, Y] = np.float32(np.meshgrid(
                np.arange(w, dtype=np.float32),
                np.arange(h, dtype=np.float32),
                indexing='xy'))
    
    X, Y, Z = (np.expand_dims(X,axis=0), 
               np.expand_dims(Y,axis=0), 
               np.ones([1, h, w], np.float32))
    coor_hw = np.concatenate([X, Y, Z], axis=0)
    coor_hw = np.reshape(coor_hw, [3,-1])
    
    # initialize Likelihood table L [num. levels, num. pixel]
    L = np.zeros([levels, h*w, 1], dtype=np.float32)
    
    for i in tqdm(range(len(sequence))):
        # skip if same image
        if i == curr_frame:
            continue
        
        # get image at frame i
        img_i = np.float32(sequence[i])
        
        # compute candidate indices [d, h, w, 2]
        depth_hw = np.zeros([1, h*w], dtype=np.float32) #[1, h*w]
        for depth_i in range(levels):
            depth_hw[0, :] = depth_vector[depth_i]
            remap = compute_candidates(
                    camera,
                    curr_frame,
                    i,
                    coor_hw, 
                    depth_hw)

            # compute norm of difference
            img_i_projected = cv2.remap(
                    img_i, 
                    remap, 
                    None, 
                    cv2.INTER_NEAREST, 
                    borderValue=[128,128,128])
            
            pc = np.reshape(compute_norm(img_o, img_i_projected), [-1, 1])
            L[depth_i, :] += sigma_c/(sigma_c + pc)
            
    u = L.max(axis=0, keepdims=True)
    D = 1 - np.divide(L, u)
    D = np.reshape(D, [levels, h, w])
    return D

def compute_energy_bundle(
        pic_sequence, 
        dep_sequence, 
        camera, 
        frame_index):
    
    # get paramaters
    h, w = camera.h, camera.w
    sigma_c = params.SIGMA_C
    levels = params.DEPTH_LEVELS
    depth_vector = params.DEPTH_VEC
    
    # get image at frame cf
    img_o = np.float32(pic_sequence[frame_index])
    
    # initialize indices coordinates
    X, Y = np.float32(np.meshgrid(
                np.arange(w, dtype=np.float32),
                np.arange(h, dtype=np.float32),
                indexing='xy'))
    
    X, Y, Z = (np.expand_dims(X,axis=0), 
               np.expand_dims(Y,axis=0), 
               np.ones([1, h, w], np.float32))
    
    coor_hw = np.concatenate([X, Y, Z], axis=0) # [3, h, w] 
    depth_hw = np.zeros([h, w], dtype=np.float32) #[h, w]
    
    # initialize Likelihood table L [num. levels, h, w]
    L = np.zeros([levels, h, w], dtype=np.float32)
    
    for i in tqdm(range(len(pic_sequence))):
        # skip if same image
        if i == frame_index: continue
        
        # get image at frame i
        img_i = np.float32(pic_sequence[i])
        
        
        for depth_i in range(levels):
            # photo-consistency constraint ------------------------------------
            depth_hw[:,:] = depth_vector[depth_i]
            candidates = compute_candidates_2(
                    camera,
                    frame_index,
                    i,
                    coor_hw, 
                    depth_hw)
            
            remap = np.transpose(candidates[:2,:,:], [1,2,0])
            img_i_projected = cv2.remap(
                    img_i, 
                    remap, 
                    None, 
                    cv2.INTER_NEAREST, 
                    borderValue=[128, 128, 128])
            
            x = compute_norm(img_o, img_i_projected, keepdims=False)
            pc = sigma_c/(sigma_c + x)
            
            # geometric consistency constraint --------------------------------
            depth_indices = dep_sequence[i]
            depth_indices_projected = cv2.remap(
                    depth_indices, 
                    remap, 
                    None, 
                    cv2.INTER_NEAREST, 
                    borderValue=int(levels/2.0))
                        
            depth = np.take(depth_vector, depth_indices_projected)
            candidates_geom = compute_candidates_2(
                    camera,
                    i,
                    frame_index,
                    candidates,
                    depth)

            x = np.sum(
                np.square(coor_hw - candidates_geom), 
                axis=0,
                keepdims=False)
            
            pv = np.exp(x/(-2*params.SIGMA_D_SQUARED))
            #utils.show_depthmap(1-pv,1,close_after=False)
            
            # update likelyhood
            L[depth_i,:,:] += pc*pv
        #cv2.destroyAllWindows()

    u = L.max(axis=0, keepdims=True)
    D = 1 - np.divide(L, u)
    return D

def compute_candidates(camera, pose1, pose2, coorsxy, d):
    """
    Return the image indices with respect of two different camera poses.

    Arguments:
     * camera - utils.Camera instance
     
     * pose1 - index of the first 
       camera pose
       
     * pose2 - index of the second
       camera pose
       
     * coorsxy - homogeneous camera 
       coordinates: 
        + numpy.ndarray 
        + numpy.float32
        + [3, pixel count]
        
     * d - pixel depth:
        + numpy.ndarray
        + numpy.float32
        + [1, pixel count]
    
    Return:
     numpy.array with [h,w,2] dimension 
     representing the new coordinates.
    """
    
    assert type(pose1) == int
    assert type(pose2) == int
    assert isinstance(camera, utils.Camera)
    
    K = camera.K
    R1, T1 = camera.Rs[pose1], camera.Ts[pose1]
    R2, T2 = camera.Rs[pose2], camera.Ts[pose2]
    h, w = camera.h, camera.w
    
    assert isinstance(coorsxy, np.ndarray)
    assert isinstance(d, np.ndarray)
    
    assert coorsxy.dtype == np.float32
    assert d.dtype == np.float32
    
    assert coorsxy.shape == (3, h*w)
    assert d.shape == (1, h*w)

    depth = (T1-T2).T * d 
    remap = (K*R2.T) * ((R1*K.I) * coorsxy + depth)
    remap = np.divide(remap, remap[2, :])
    remap = np.transpose(remap, [1,0])
    remap = np.reshape(np.asarray(remap), [h, w, 3])
    return remap[:,:,:2]


def compute_norm(img_a, img_b, keepdims=True):
    '''
    Compute the norm of the per-pixel difference between the two images.
    
    Arguments:
      * img_a - first image, numpy.array 
        of type np.float32 [h, w, <x>]
        
      * uImg_b - second image, numpy.array 
        of type np.float32 [h, w, <x>]
    
    Returns:
      numpy.array representing the norm of the difference
      between the two images.
      
      shape [h, w, 1] if keepdims == True
      shape [h, w] if keepdims == False
    '''
    assert isinstance(img_a, np.ndarray)
    assert isinstance(img_b, np.ndarray)
    
    assert img_a.dtype == np.float32
    assert img_b.dtype == np.float32

    assert len(img_a.shape) == 3
    assert len(img_b.shape) == 3
    assert img_b.shape == img_a.shape

    return np.sqrt(np.sum(np.square(img_a-img_b), axis=-1, keepdims=keepdims))

#------------------------------------------------------------------------------
def compute_smoothness_weight(image):
    '''
    Compute the smoothness weight matrix for the LBP algorithm.
    
    Arguments:
     * image - numpy.array of type np.npfloat32 [h,w,3]
     
    Return:
      list of numpy.array containing the smoothness weights
      [h,w], one matrix per neighbor direction.
    '''
    assert isinstance(image, np.ndarray)
    assert image.dtype == np.float32
    assert len(image.shape) == 3
    
    h, w, _ = image.shape
    directions = params.DIRECTIONS
    img = [None]*len(directions)
    lamb = [None]*len(directions)
    
    up, do, le, ri = params.UP,params.DOWN, params.LEFT, params.RIGHT
    img[up] = cv2.warpAffine(image, params.AFFINE_DIR[do], dsize=(w, h))
    img[do] = cv2.warpAffine(image, params.AFFINE_DIR[up], dsize=(w, h))
    img[le] = cv2.warpAffine(image, params.AFFINE_DIR[ri], dsize=(w, h))
    img[ri] = cv2.warpAffine(image, params.AFFINE_DIR[le], dsize=(w, h))
    
    u = np.zeros([h,w], dtype=np.float32)
    ws = params.W_S
    
    for j in directions:
        tmp = np.float32(1)/(compute_norm(image, img[j], False) + params.EPSILON)
        u += tmp
        lamb[j] = ws*tmp
    
    u = np.float32(4)/u
    for j in directions:
        lamb[j] *= u
    return lamb




###############################################################################
    
def compute_candidates_2(camera, pose1, pose2, coorsxy, d):
    """
    Return the image indices with respect of two different camera poses.

    Arguments:
     * camera - utils.Camera instance
     
     * pose1 - index of the first 
       camera pose
       
     * pose2 - index of the second
       camera pose
       
     * coorsxy - homogeneous camera 
       coordinates: 
        + numpy.ndarray 
        + numpy.float32
        + [3, h, w]
        
     * d - pixel depth:
        + numpy.ndarray
        + numpy.float32
        + [h, w]
    
    Return:
     numpy.array with shape [3, h, w],
     representing the new coordinates.
    """
    
    assert type(pose1) == int
    assert type(pose2) == int
    assert isinstance(camera, utils.Camera)
    
    K = camera.K
    R1, T1 = camera.Rs[pose1], camera.Ts[pose1]
    R2, T2 = camera.Rs[pose2], camera.Ts[pose2]
    h, w = camera.h, camera.w
    
    assert isinstance(coorsxy, np.ndarray)
    assert isinstance(d, np.ndarray)
    
    assert coorsxy.dtype == np.float32
    assert d.dtype == np.float32
    
    assert coorsxy.shape == (3, h, w)
    assert d.shape == (h, w)
    
    coorsxy = np.reshape(coorsxy, [3,-1])
    d = np.reshape(d, [1, -1])

    depth = (T1-T2).T * d 
    remap = (K*R2.T) * ((R1*K.I) * coorsxy + depth)
    remap = np.divide(remap, remap[2, :])
    remap = np.reshape(np.asarray(remap), [3, h, w])
    return remap


def get_homogeneous_coord(h,w):
    X, Y = np.float32(np.meshgrid(
                np.arange(w, dtype=np.float32),
                np.arange(h, dtype=np.float32),
                indexing='xy'))
    
    X, Y, Z = (np.expand_dims(X,axis=0), 
               np.expand_dims(Y,axis=0), 
               np.ones([1, h, w], np.float32))
    
    coor = np.concatenate([X, Y, Z], axis=0)
    return coor