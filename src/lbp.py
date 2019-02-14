# -*- coding: utf-8 -*-
'''
Loopy Belief Propagation
''' 
import numpy as np

import cv2
import utils


"""
def lbp(data_cost, edge_weights, step=1.0, eta=1.0, iterations=4):
    ''' *Loopy Belief Propagation* (LBP) algorithm.
    
    LBP is a dynamic programming algorithm that can be used to find
    approximate solutions for energy minimization problems over labeling 
    of graphs. In particular, LBP works only with grid-graphs, and 
    this specific implementation works only with graphs representing
    images; each pixel is a vertex, adjacents pixel are assumed connected
    by an edge (only in the four cardinal directions, no oblique adjacents).

    Notes
    -----
    Given a graph with vertices (pixels) `P` and edges `N`, and a set of
    labels `L` (with cardinality `m`), the goal of LBP is to find a 
    labeling of the vertices :math:`\\{f_p\\}_{p \\in V}` such that 
    the energy function

    .. math::

        \\sum_{(p,q)\\in N} V(f_p,f_q) + \\sum_{p \in P}  D(p, f_p)

    is minimized. The terms :math:`V(\\cdot,\\cdot)` and :math:`D(\\cdot,\\cdot)`
    are rispecively names **discontinuity cost** and **data cost**.
    The **data cost** can be any arbitrary mapping between pixel-label pairs
    over real values (in this case it is passed as input through *data_cost*).
    On the other hand, the **discontinuity cost** between two pixels `p` 
    and `q` is defined as

    .. math::

        w_{p,q}\\cdot\\min(s||d_p - d_q||, \\eta) ,

    with :math:`\\eta` and :math:`s` positive constants, while :math:`w_{p,q}` is 
    an edge dependent scalar value (stored in *edge_weights*).

    Parameters
    ----------
    data_cost: numpy array, type float32
        Array with shape `[labels, height, width]` representing the **data cost** 
        of the energy funtion.
         
    edge_weights: list of numpy arrays, type float32
        List of four arrays with shape `[height, width]` representing the 
        weights used by the **discontinuity cost**.
        
        * The first array contains weights for edges of type :math:`(p, p_{up})`,
          with :math:`p=(y,x)` and :math:`p_{up}=(y+1, x)`
        * The second array contains weights for edges of type :math:`(p, p_{down})`,
          with :math:`p=(y,x)` and :math:`p_{down}=(y-1, x)`
        * The third array contains weights of type :math:`(p, p_{left})`,
          with :math:`p=(y,x)` and :math:`p_{left}=(y, x+1)`
        * The fourth array contains weights of type :math:`(p, p_{right})`,
          with :math:`p=(y,x)` and :math:`p_{right}=(y, x-1)`

    Returns
    -------
    numpy array, type uint16
        Array with shape `[height, width]` containing the depth-values labels 
        (that is, an integer that can be used to obtain the disparity value)
        per pixel.
    '''

    assert isinstance(data_cost, np.ndarray)
    assert data_cost.dtype == np.float32
    assert len(data_cost.shape) == 3
    k, h, w = data_cost.shape

    
    assert isinstance(edge_weights, list)
    assert len(edge_weights) == 4
    for matrix in edge_weights:
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (h,w)
        assert matrix.dtype == np.float32

    M = []
    for _ in utils.DIRECTIONS:
        M.append(np.zeros([h, w, k], dtype=np.float32))
    
    #compute per-pixel energy
    D = np.transpose(data_cost, [1,2,0])
    edge_weights = np.expand_dims(edge_weights, axis=-1)

    for i in range(iterations):
        print 'iteration number ', i

        H = np.array(D) # [h,w,m]
        m, hf = [None]*4, [None]*4
        directions = utils.DIRECTIONS
        up, do, le, ri = utils.UP,utils.DOWN, utils.LEFT, utils.RIGHT

        for j in directions:
            H += M[j] #[h,w,m] + [h,w,m]
            
        hf[up] = cv2.warpAffine(H-M[do], utils.AFFINE_DIR[do], dsize=(w, h))
        hf[do] = cv2.warpAffine(H-M[up], utils.AFFINE_DIR[up], dsize=(w, h))
        hf[le] = cv2.warpAffine(H-M[ri], utils.AFFINE_DIR[ri], dsize=(w, h))
        hf[ri] = cv2.warpAffine(H-M[le], utils.AFFINE_DIR[le], dsize=(w, h))
        
        for j in directions:
            m[j] = np.array(hf[j])

        for j in directions:
            for i in range(1, k):
                m[j][:,:,i] = np.minimum(m[j][:,:,i], m[j][:,:,i-1] + edge_weights[j,:,:,0]*step)
            for i in reversed(range(0, k-1)):
                m[j][:,:,i] = np.minimum(m[j][:,:,i], m[j][:,:,i+1] + edge_weights[j,:,:,0]*step)
            
        for j in directions:
            tmp = hf[j].min(axis=-1, keepdims=True) + edge_weights[j]*eta
            m[j] = np.minimum(m[j], tmp)
        M = m

    B = np.array(D)
    for i in utils.DIRECTIONS:
        B += M[i]
    return np.uint16(B.argmin(axis=-1))
"""

def lbp(data_cost, edge_weights, step=1.0, eta=1.0, inter_it=2, intra_it=2, scale=2):
    ''' *Loopy Belief Propagation* (LBP) algorithm.
    
    LBP is a dynamic programming algorithm that can be used to find
    approximate solutions for energy minimization problems over labeling 
    of graphs. In particular, LBP works only with grid-graphs, and 
    this specific implementation works only with graphs representing
    images; each pixel is a vertex, adjacents pixel are assumed connected
    by an edge (only in the four cardinal directions, no oblique adjacents).

    Notes
    -----
    Given a graph with vertices (pixels) `P` and edges `N`, and a set of
    labels `L` (with cardinality `m`), the goal of LBP is to find a 
    labeling of the vertices :math:`\\{f_p\\}_{p \\in V}` such that 
    the energy function

    .. math::

        \\sum_{(p,q)\\in N} V(f_p,f_q) + \\sum_{p \in P}  D(p, f_p)

    is minimized. The terms :math:`V(\\cdot,\\cdot)` and :math:`D(\\cdot,\\cdot)`
    are rispecively names **discontinuity cost** and **data cost**.
    The **data cost** can be any arbitrary mapping between pixel-label pairs
    over real values (in this case it is passed as input through *data_cost*).
    On the other hand, the **discontinuity cost** between two pixels `p` 
    and `q` is defined as

    .. math::

        w_{p,q}\\cdot\\min(s||d_p - d_q||, \\eta) ,

    with :math:`\\eta` and :math:`s` positive constants, while :math:`w_{p,q}` is 
    an edge dependent scalar value (stored in *edge_weights*).

    Parameters
    ----------
    data_cost: numpy array, type float32
        Array with shape `[labels, height, width]` representing the **data cost** 
        of the energy funtion.
         
    edge_weights: list of numpy arrays, type float32
        List of four arrays with shape `[height, width]` representing the 
        weights used by the **discontinuity cost**.
        
        * The first array contains weights for edges of type :math:`(p, p_{up})`,
          with :math:`p=(y,x)` and :math:`p_{up}=(y+1, x)`
        * The second array contains weights for edges of type :math:`(p, p_{down})`,
          with :math:`p=(y,x)` and :math:`p_{down}=(y-1, x)`
        * The third array contains weights of type :math:`(p, p_{left})`,
          with :math:`p=(y,x)` and :math:`p_{left}=(y, x+1)`
        * The fourth array contains weights of type :math:`(p, p_{right})`,
          with :math:`p=(y,x)` and :math:`p_{right}=(y, x-1)`

    Returns
    -------
    numpy array, type uint16
        Array with shape `[height, width]` containing the depth-values labels 
        (that is, an integer that can be used to obtain the disparity value)
        per pixel.
    '''
    D = np.transpose(data_cost, [1,2,0])
    D = _create_tables(D, iterations=inter_it, scale=scale)
    edge_weight = edge_weights[0].mean()
    labels, height, width = data_cost.shape

    M = []
    h,w,k = D[-1].shape
    for _ in utils.DIRECTIONS:
        M.append(np.zeros([h, w, k], dtype=np.float32))

    for i in reversed(range(1, inter_it)):
        M = _lbp(D[i], edge_weight, M, step, eta, intra_it)
        M = _increase_scale(M, scale=scale)

    for i in range(len(M)): M[i] = M[i][0:height,0:width]
    M = _lbp2(D[0][0:height,0:width], edge_weights, M, step, eta, intra_it)

    B = D[0][0:height,0:width]
    for i in utils.DIRECTIONS:
        B += M[i]
    return np.uint32(B.argmin(axis=-1))

def _increase_scale(M, scale=4):
    for i in range(len(M)):
        M[i] = M[i].repeat(repeats=scale, axis=0)
        M[i] = M[i].repeat(repeats=scale, axis=1)
    return M

def _create_tables(data_cost, iterations=3, scale=4):
    h, w, levels = data_cost.shape
    mul = scale*iterations 
    pad_h = 0 if h%mul == 0 else mul - h%mul
    pad_w = 0 if w%mul == 0 else mul - w%mul

    data_cost = np.pad(
        array=data_cost,
        pad_width=((0,pad_h), (0,pad_w), (0,0)),
        mode='constant')

    D = [data_cost]
    for _ in range(1, iterations):
        h, w, levels = data_cost.shape
        data_cost = np.reshape(data_cost, (h/scale, scale, w/scale, scale, levels))
        data_cost = data_cost.sum((1, 3))
        D.append(data_cost)
    return D

def _lbp(data_cost, edge_weight, messages, step=1.0, eta=1.0, iterations=4):
    h, w, k = data_cost.shape

    M = messages
    
    for i in range(iterations):
        print 'iteration number ', i

        H = np.array(data_cost) # [h,w,m]
        m, hf = [None]*4, [None]*4
        directions = utils.DIRECTIONS
        up, do, le, ri = utils.UP,utils.DOWN, utils.LEFT, utils.RIGHT

        for j in directions:
            H += M[j] #[h,w,m] + [h,w,m]
            
        hf[up] = cv2.warpAffine(H-M[do], utils.AFFINE_DIR[do], dsize=(w, h))
        hf[do] = cv2.warpAffine(H-M[up], utils.AFFINE_DIR[up], dsize=(w, h))
        hf[le] = cv2.warpAffine(H-M[ri], utils.AFFINE_DIR[ri], dsize=(w, h))
        hf[ri] = cv2.warpAffine(H-M[le], utils.AFFINE_DIR[le], dsize=(w, h))
        
        for j in directions:
            m[j] = np.array(hf[j])

        for j in directions:
            for i in range(1, k):
                m[j][:,:,i] = np.minimum(m[j][:,:,i], m[j][:,:,i-1] + edge_weight*step)
            for i in reversed(range(0, k-1)):
                m[j][:,:,i] = np.minimum(m[j][:,:,i], m[j][:,:,i+1] + edge_weight*step)
            
        for j in directions:
            tmp = hf[j].min(axis=-1, keepdims=True) + edge_weight*eta
            m[j] = np.minimum(m[j], tmp)
        M = m
    return M

def _lbp2(data_cost, edge_weights, messages, step=1.0, eta=1.0, iterations=4):
    h, w, k = data_cost.shape

    M = messages
    edge_weights = np.expand_dims(edge_weights, axis=-1)

    for i in range(iterations):
        print 'iteration number ', i

        H = np.array(data_cost) # [h,w,m]
        m, hf = [None]*4, [None]*4
        directions = utils.DIRECTIONS
        up, do, le, ri = utils.UP,utils.DOWN, utils.LEFT, utils.RIGHT

        for j in directions:
            H += M[j] #[h,w,m] + [h,w,m]
            
        hf[up] = cv2.warpAffine(H-M[do], utils.AFFINE_DIR[do], dsize=(w, h))
        hf[do] = cv2.warpAffine(H-M[up], utils.AFFINE_DIR[up], dsize=(w, h))
        hf[le] = cv2.warpAffine(H-M[ri], utils.AFFINE_DIR[ri], dsize=(w, h))
        hf[ri] = cv2.warpAffine(H-M[le], utils.AFFINE_DIR[le], dsize=(w, h))
        
        for j in directions:
            m[j] = np.array(hf[j])

        for j in directions:
            for i in range(1, k):
                m[j][:,:,i] = np.minimum(m[j][:,:,i], m[j][:,:,i-1] + edge_weights[j,:,:,0]*step)
            for i in reversed(range(0, k-1)):
                m[j][:,:,i] = np.minimum(m[j][:,:,i], m[j][:,:,i+1] + edge_weights[j,:,:,0]*step)
            
        for j in directions:
            tmp = hf[j].min(axis=-1, keepdims=True) + edge_weights[j]*eta
            m[j] = np.minimum(m[j], tmp)
        M = m
    return M
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