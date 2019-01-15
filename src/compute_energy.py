# -*- coding: utf-8 -*-
from time import time

import cv2 
import numpy as np
from tqdm import tqdm

import utils 
import params


###############################################################################
def compute_energy_data( 
        camera, 
        frame_index,
        pic_sequence, 
        dep_sequence=None):
    
    # TODO check input validity
    
    # check if it is possible to use bundle optimization
    usebundle = dep_sequence != None
    
    # set the range of possible frames 
    t = frame_index
    window_side = params.WINDOW_SIDE
    window_start = max(0,t - window_side)
    window_end = min(len(pic_sequence), window_start + window_side*2 + 1)
    
    # get paramaters
    h, w = camera.h, camera.w
    sigma_c = params.SIGMA_C
    levels = params.DEPTH_LEVELS
    depth_values = np.linspace(
        params.DEPTH_MIN,
        params.DEPTH_MAX,
        params.DEPTH_LEVELS,
        dtype=np.float32)
    
    # get image at frame cf
    I_t = np.float32(pic_sequence[t])
    
    # initialize indices coordinates
    x_h = homogeneous_coord_grid(h,w) #[3, h, w]
    d = np.zeros([h, w], dtype=np.float32) #[h, w]
    
    # initialize Likelihood table L 
    L = np.zeros([levels, h, w], dtype=np.float32) #[levels, h, w]

    
    for t_prime in tqdm(range(window_start, window_end)):
        # skip if same image
        if t_prime == t: continue  
    
        # get image at frame i
        I_t_prime = np.float32(pic_sequence[t_prime])  
        
        for level in range(levels):
            # photo-consistency constraint ------------------------------------
            
            # fill each position with the current level depht value
            d[:,:] = depth_values[level] 

            # compute conjugate pixel position using the depth level at time t
            x_prime_h = conujugate_coordinates(
                    camera=camera,
                    pose1=t,
                    pose2=t_prime,
                    coorsxy=x_h, 
                    d=d)
            
            # remap the image w.r.t. the projected pixels
            x_prime = np.transpose(x_prime_h[:2,:,:], [1,2,0])
            I_t_prime_projected = cv2.remap(
                    src=I_t_prime, 
                    map1=x_prime, 
                    map2=None, 
                    interpolation=cv2.INTER_NEAREST, 
                    borderValue=[128, 128, 128])
            
            # compute the photo constraint term (pc)
            color_difference = L2_norm(I_t, I_t_prime_projected, keepdims=False)
            pc = sigma_c/(sigma_c + color_difference)
            
            # check if bundle optimization is required
            if not usebundle:
                # update likelyhood using photo-consistency contraint
                L[level,:,:] += pc

            else:
                # geometric-consistency constraint ----------------------------
                
                # compute candidate pixels using previously estimated depth values
                depth_indices = dep_sequence[t_prime] # get prev. estimated depth 
                depth_indices_projected = cv2.remap(
                        src=depth_indices, 
                        map1=x_prime, 
                        map2=None, 
                        interpolation=cv2.INTER_NEAREST, 
                        borderValue=int(levels/2.0))
                
                # fill the matrix d with the conjugate pixels' depth value indices
                np.take(depth_values, depth_indices_projected, out=d)
                
                # project back from t_prime to t using prev. estimated depth values
                projected_x_prime_h = conujugate_coordinates(
                        camera=camera,
                        pose1=t_prime,
                        pose2=t,
                        coorsxy=x_prime_h,
                        d=d)
    
                # compute norm of diff. between original and projected coord 
                color_difference_norm = np.sum(
                    np.square(x_h - projected_x_prime_h), 
                    axis=0,
                    keepdims=False)
                
                # compute the bundle optimization term (pv)
                pv = np.exp(color_difference_norm/(-2*params.SIGMA_D_SQUARED))
                
                # update likelyhood using photo and geometric constraints
                L[level,:,:] += pc*pv
                
    # compute and return the computed energy values
    u = np.reciprocal(L.max(axis=0, keepdims=True))
    return 1 - u*L

#------------------------------------------------------------------------------
def conujugate_coordinates(camera, pose1, pose2, coorsxy, d):
    """
    Return the image pixel coordinates with respect of two different camera poses.

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
    
    # check input validity ----------------------------------------------------
    assert type(pose1) == int and 0 <= pose1 < len(camera)
    assert type(pose2) == int and 0 <= pose2 < len(camera)
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
    
    # compute pixel candidates w.r.t. the given camera parameters -------------
    coorsxy = np.reshape(coorsxy, [3,-1])
    d = np.reshape(d, [1, -1])

    depth = (T1-T2).T * d 
    remap = (K*R2.T) * ((R1*K.I) * coorsxy + depth)
    remap = np.divide(remap, remap[2, :])
    remap = np.reshape(np.asarray(remap), [3, h, w])
    return remap

#------------------------------------------------------------------------------
def L2_norm(img_a, img_b, keepdims=True):
    """
    Compute the norm of the per-pixel difference between the two images.
    
    Arguments:
      * img_a - first image, numpy.array 
        of type np.float32 [h, w, 3]
        
      * uImg_b - second image, numpy.array 
        of type np.float32 [h, w, 3]
    
    Returns:
      numpy.array representing the norm of the difference
      between the two images.
      
      shape [h, w, 1] if keepdims == True
      shape [h, w] if keepdims == False
    """

    assert isinstance(img_a, np.ndarray)
    assert isinstance(img_b, np.ndarray)
    
    assert img_a.dtype == np.float32
    assert img_b.dtype == np.float32

    assert len(img_a.shape) == 3
    assert len(img_b.shape) == 3
    assert img_b.shape == img_a.shape

    return np.sqrt(np.sum(np.square(img_a-img_b), axis=-1, keepdims=keepdims))

#------------------------------------------------------------------------------
def lambda_factor(image):
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
    ws = params.W_S
    directions = params.DIRECTIONS
    img = [None]*len(directions)
    lambda_factor = [None]*len(directions)
    
    up, do, le, ri = params.UP,params.DOWN, params.LEFT, params.RIGHT
    img[up] = cv2.warpAffine(image, params.AFFINE_DIR[do], dsize=(w, h))
    img[do] = cv2.warpAffine(image, params.AFFINE_DIR[up], dsize=(w, h))
    img[le] = cv2.warpAffine(image, params.AFFINE_DIR[ri], dsize=(w, h))
    img[ri] = cv2.warpAffine(image, params.AFFINE_DIR[le], dsize=(w, h))
    u_denominator = np.zeros([h,w], dtype=np.float32)
    
    # compute partial lambda factor and the denominator for the u_lambda term
    for j in directions:
        inverse_color_diff = np.float32(1)/(L2_norm(image, img[j], False) + params.EPSILON)
        u_denominator += inverse_color_diff # compute denominator for u_lambda term
        lambda_factor[j] = inverse_color_diff # compute partial lambda factor
    
    # compute u_lambda term (approximation of the term, incorrect on image borders)
    u_lambda = np.float32(4)/u_denominator
    
    # compute final lambda smoothness weight for each edge
    for j in directions:
        lambda_factor[j] = ws*u_lambda*lambda_factor[j]
    
    return lambda_factor

#------------------------------------------------------------------------------
def homogeneous_coord_grid(h,w):
    '''
    Compute grid of homogeneous coordinates having three dimensions.
    
    Arguments:
     * h - integer value
     * w - integer value
     
    Return:
        out, grid of homogeneous coordinates 
        with shape [3,h,w] and such that
         out[0,y,:] = y
         out[1,:,x] = x
         out[2,:,:] = 1
        '''
    assert type(h) == int
    assert type(w) == int
    
    X, Y = np.float32(np.meshgrid(
                np.arange(w, dtype=np.float32),
                np.arange(h, dtype=np.float32),
                indexing='xy'))
    
    X, Y, Z = (np.expand_dims(X,axis=0), 
               np.expand_dims(Y,axis=0), 
               np.ones([1, h, w], np.float32))
    
    return np.concatenate([X, Y, Z], axis=0)