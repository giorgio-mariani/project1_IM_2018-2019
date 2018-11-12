# -*- coding: utf-8 -*-
'''
Loopy Belief Propagation
''' 
import numpy as np
import cv2
import params
import compute_energy as ce
import utils



def LBP(energy_data, edge_weights):
    '''
    Loopy Belief Propagation algorithm.
    
    This is an implementation of the LBP optimization 
    algorithm, using as edge cost between two pixels p 
    and q the function:   
        V(d_p,d_q) = min(|d_p - d_q|, threshold)
    
    The function to optimize is the following:
    E(d) = sum_{p,q} V(d_p,d_q) + sum_{p} D(d_p)
    
    in this case d represent an assignment of the 
    parameters d_p, for all pixel p.
    There are m possible assignment for each par.
    d_p
    
    Arguments:
      * energy_data - data term in the optimizated
        energy funtion:
         + numpy.ndarray
         + numpy.float32
         + [m, height, width]
         
      * edge_weights - scalar weigth per image 
        edge:
         + list of numpy.ndarray
         + numpy.float32
         + [height, width] * 4 
    
    Return:
        numpy.ndarray containing the initialization
        depth index per pixel:
         + nuumpy.ndarray
         + numpy.uint16
         + [height,width] 
    '''
    assert isinstance(energy_data, np.ndarray)
    assert energy_data.dtype == np.float32
    assert len(energy_data.shape) == 3
    m, h, w = energy_data.shape

    
    assert isinstance(edge_weights, list)
    assert len(edge_weights) == 4
    for matrix in edge_weights:
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (h,w)
        assert matrix.dtype == np.float32

    M = []
    for _ in params.DIRECTIONS:
        M.append(np.zeros([h, w, m], dtype=np.float32))
        
    #compute per-pixel energy
    D = np.transpose(energy_data, [1,2,0])
    for i in range(params.ITERATION_NUMBER):
        print 'iteration number ', i
        M = _LBP_iteration(D, M,
                           edge_weights,
                           height=h, 
                           width=w, 
                           label_count=m)
    B = np.array(D)
    for i in params.DIRECTIONS:
        B += M[i]
    return np.uint16(B.argmin(axis=-1))



def _LBP_iteration(D, M, lambda_weights, height, width, label_count):
    h, w, k = height, width, label_count
    
    lambda_weights = np.expand_dims(lambda_weights, axis=-1)
    
    H = np.array(D) # [h,w,m]
    m, hf = [None]*4, [None]*4
    directions = params.DIRECTIONS
    up, do, le, ri = params.UP,params.DOWN, params.LEFT, params.RIGHT

    for j in directions:
        H += M[j] #[h,w,m] + [h,w,m]
        
    hf[up] = cv2.warpAffine(H-M[do], params.AFFINE_DIR[do], dsize=(w, h))
    hf[do] = cv2.warpAffine(H-M[up], params.AFFINE_DIR[up], dsize=(w, h))
    hf[le] = cv2.warpAffine(H-M[ri], params.AFFINE_DIR[ri], dsize=(w, h))
    hf[ri] = cv2.warpAffine(H-M[le], params.AFFINE_DIR[le], dsize=(w, h))

    for j in directions:
        m[j] = np.array(hf[j])
        
    for j in directions:    
        for i in range(1, k):
            m[j][:,:,i] = np.minimum(m[j][:,:,i], m[j][:,:,i-1] + lambda_weights[j,:,:,0]*params.STEP)
        for i in reversed(range(0, k-1)):
            m[j][:,:,i] = np.minimum(m[j][:,:,i], m[j][:,:,i+1] + lambda_weights[j,:,:,0]*params.STEP)
        
    for j in directions:
        tmp = hf[j].min(axis=-1, keepdims=True) + lambda_weights[j]*params.ETA
        m[j] = np.minimum(m[j], tmp)
    return m




###############################################################################
'''
def _LBP_iteration(D, M, h, w, m, step, threshold):
        
    up, down, left, right = 0,1,2,3
    dirs = [up, down, left, right]

    
    uH = cv2.UMat(D)
    uM = [None]*4
    um = [None]*4
    uh = [None]*4

    for i in dirs:
        uM[i] = cv2.UMat(M[i]) #[h,w,m] 
        cv2.add(uH, uM[i], uH) #[h,w,m] + [h,w,m]
    
    offsets = [np.array([[1, 0, 0], [0, 1,-1]], dtype=np.float32),
               np.array([[1, 0, 0], [0, 1, 1]], dtype=np.float32),
               np.array([[1, 0,-1], [0, 1, 0]], dtype=np.float32),
               np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)]
        
    for i in dirs:
        uh[i] = cv2.subtract(uH, uM[i])
        uh[i] = cv2.warpAffine(uh[i], offsets[i], dsize=(w, h)) 
        um[i] = uh[i]
        
    uTmp = cv2.UMat(np.zeros([h, w], dtype=np.float32))
    for j in dirs:
        um_sliced = []
        for x in cv2.split(um[j]):
            um_sliced.append(cv2.UMat(x))
    
        for i in range(1, m):
            uTmp = cv2.add(um_sliced[i-1], step)
            cv2.min(um_sliced[i], uTmp, um_sliced[i])
            
        for i in reversed(range(0, m-1)):
            uTmp = cv2.add(um_sliced[i+1], step)
            um_sliced[i] = cv2.min(um_sliced[i], uTmp, um_sliced[i])
        
        for i in range(m):
            um_sliced[i] = um_sliced[i].get()
        um[j] = cv2.UMat(cv2.merge(um_sliced))
        
    for i in dirs:
        uh[i] = cv2.UMat(np.reshape(uh[i].get(), [h*w,m]))
        uTmp = cv2.reduce(uh[i], 1, cv2.REDUCE_MIN)
        uTmp = cv2.repeat(uTmp, 1, m)
        cv2.add(uTmp, threshold, uTmp)
        uTmp = cv2.UMat(np.reshape(uTmp.get(), (h, w, m)))
        cv2.min(um[i], uTmp, um[i])    
    return um
'''