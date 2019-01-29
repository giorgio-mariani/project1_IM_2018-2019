# -*- coding: utf-8 -*-
from time import time

import cv2 
import numpy as np
from tqdm import tqdm

import utils 


###############################################################################
def compute_energy_data(
    frame_index, 
    sequence, 
    window_side=10, 
    sigma_c=1.0, 
    sigma_d=2.5):
    """Compute the data cost term, for frame *frame_index*, to be used 
    for the LBP algorithm.

    This function computes an array with shape `[m, h, w]`, where `m` is 
    the number of depth labels used, `h` and `w` are respectively the 
    pictures' height and width.
    The value in this array at position `(d, y, x)`, is inversely 
    proportional to the likelihood that pixel `(y,x)` has disparity `d`.
    this value is computed using multi-stereo photo consistency constraints,
    and, if ``sequence.use_bundle()==True``, geometric consistency with previously
    estimated depth-maps (*bundle optimization*).
    The result of this function should be used as **data cost** for the LBP
    algorithm (see :func:`lbp.lbp`).

    See Also
    --------
    lbp.lbp: *Loopy Belief Propagation* implementation.

    Parameters
    ----------
    frame_index : int
        Frame whose depth-map is estimated.

    sequence : utils.Sequence
        Object containing parameters necessary to the depth-maps estimation.
        It contains the camera matrices, picture arrays, length of the
        sequence, etc. If *sequence* contains also previously estimated 
        depth-maps, then the *bundle optimization* phase is also executed.

    Returns
    -------
    numpy array, type float32
        Array with shape `[m, h, w]`, with `m` the number of possible depth 
        labels, and `h` and `w` height and width of frame *frame_index*.
    """
    
    # TODO check input validity
    
    # set the range of possible frames 
    t, start, end = frame_index, 0, sequence.end - sequence.start
    if t - window_side < start :
        window_start = start
        window_end = min(end, window_start + window_side*2 + 1)
    elif t + window_side + 1> end:
        window_end = end
        window_start = max(start, window_end - window_side*2 - 1)
    else:
        window_start = t - window_side
        window_end = t + window_side + 1 
    
    # get paramaters
    h, w = sequence.height, sequence.width
    levels = sequence.depth_levels
    depth_values = np.linspace(
        sequence.depth_min,
        sequence.depth_max,
        sequence.depth_levels,
        dtype=np.float32)
    
    # get image at frame cf
    I_t = sequence.I[t]
    
    # initialize indices coordinates
    x_h = homogeneous_coord_grid(h,w) #[3, h, w]
    d = np.zeros([h, w], dtype=np.float32) #[h, w]
    
    # initialize Likelihood table L 
    L = np.zeros([levels, h, w], dtype=np.float32) #[levels, h, w]

    
    for t_prime in tqdm(range(window_start, window_end)):
        # skip if same image
        if t_prime == t: continue  
    
        # get image at frame i
        I_t_prime = sequence.I[t_prime]
        
        for level in range(levels):
            # photo-consistency constraint ------------------------------------
            
            # fill each position with the current level depht value
            d[:,:] = depth_values[level] 

            # compute conjugate pixel position using the depth level at time t
            x_prime_h = conujugate_coordinates(
                    sequence=sequence,
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
            if not sequence.use_bundle():
                # update likelyhood using photo-consistency contraint
                L[level,:,:] += pc

            else:
                # geometric-consistency constraint ----------------------------
                
                # compute candidate pixels using previously estimated depth values
                depth_indices = sequence.D[t_prime] # get prev. estimated depth 
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
                        sequence=sequence,
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
                pv = np.exp(color_difference_norm/(-2*sigma_d*sigma_d))
                
                # update likelyhood using photo and geometric constraints
                L[level,:,:] += pc*pv
                
    # compute and return the computed energy values
    u = np.reciprocal(L.max(axis=0, keepdims=True))
    return 1 - u*L

#------------------------------------------------------------------------------
def conujugate_coordinates(sequence, pose1, pose2, coorsxy, d):
    """Return the image pixel coordinates with respect of two different camera poses.

    Parameters
    ----------
    sequence : utils.Sequence
        Sequence object containing the camera parameters.
     
    pose1 : int
        Index of the first pose in *camera*
       
    pose2 : int
        Index of the second pose in *camera*
       
    coorsxy : numpy array, type float32
        Homogeneous camera coordinates of shape [3, h, w], the first axis
        represents the coordinate itself.
        
    d : numpy array, type float32
        Array of shape [h,w] that indicates the disparity to use for a 
        certain pixel while computing the conjugate point.
    
    Returns
    -------
    numpy array, type float32
        Array  with shape [3, h, w] representing the conjugate coordinates.
    """
    
    # check input validity ----------------------------------------------------
    #assert type(pose1) == int and 0 <= pose1 < len(camera)
    #assert type(pose2) == int and 0 <= pose2 < len(camera)
    #assert isinstance(camera, utils.Camera)
    
    K = sequence.K
    R1, T1 = sequence.Rs[pose1], sequence.Ts[pose1]
    R2, T2 = sequence.Rs[pose2], sequence.Ts[pose2]
    h, w = sequence.height, sequence.width
    
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
    """Compute the norm of the per-pixel difference between the two images.
    
    Parameters
    ----------
    img_a :  numpy array, type float32
        First image, shape [h, w, 3]
        
    img_b : numpy array, type float32
        Second image, shape [h, w, 3]
    
    Returns
    -------
    numpy array 
        Array representing the norm of the difference between the two images.
        The array has shape [h, w, 1] if ``keepdims==True``, [h, w] otherwise.
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
def homogeneous_coord_grid(h,w):
    '''Compute grid of homogeneous coordinates having three dimensions.
    
    Parameters
    ----------
    h : int
        height
    w : int
        width
     
    Returns
    -------
    out : numpy array, type float32
        Grid of homogeneous coordinates with shape [3, h, w], whose first
        axis indicates the coordinate itself, that is,
        
        | ``out[0, y, :] = y``
        | ``out[1, :, x] = x``
        | ``out[2, :, :] = 1``

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

#------------------------------------------------------------------------------
def lambda_factor(image, ws=1.0, epsilon=1.0):
    '''Compute the smoothness weights for the LBP algorithm.

    This functions computes the edge weights that will be used during the
    executiong of the *Loopy Belief Propagation* algorithm.
    The values are inversely proportional to the difference between 
    adjacent pixels' colors.

    Notes
    -----
    the math formula used to compute the lambda factor for an edge `(x,y)`, 
    is given by

    .. math::

        \\lambda(x,y) = w_s\\cdot \\frac{u_\\lambda(x)}{||I(x) - I(y)|| + \\epsilon},

    with :math:`w_s` and :math:`\\epsilon` positive constant values, :math:`I(x)`
    the image's color at pixel `x`, and :math:`u_\\lambda(x)` is

    .. math::

        u_\\lambda(x) = {|N(x)|}\\big/{\\sum_{y'\\in N(x)} \\frac{1}{||I(x) - I(y')||+\\epsilon}}.

    See Also
    --------
    lbp.lbp : *Loopy Belief Propagation* implementation.

    Parameters
    ----------
    image : numpy array, type float32
        Array representing an image (so with shape [`h`, `w`, 3]).
     
    Returns 
    -------
    list of numpy arrays, type float32
      Said list contains four smoothness weight arrays with shape 
      [`h`, `w`]; one matrix per neighbor direction. More precisely, 
      if we denote the four matrices as :math:`M_{up}, M_{down}, M_{left}, M_{right}`,
      we have that

      * :math:`M_{up}[y,x] = \\lambda((y,x), (y+1, x)),`
      * :math:`M_{down}[y,x] = \\lambda((y,x), (y-1, x)),`
      * :math:`M_{left}[y,x] = \\lambda((y,x), (y, x+1)),`
      * :math:`M_{right}[y,x] = \\lambda((y,x), (y, x-1)).`

    '''
    assert isinstance(image, np.ndarray)
    assert image.dtype == np.float32
    assert len(image.shape) == 3
    
    h, w, _ = image.shape
    directions = utils.DIRECTIONS
    img = [None]*len(directions)
    lambda_factor = [None]*len(directions)
    
    up, do, le, ri = utils.UP,utils.DOWN, utils.LEFT, utils.RIGHT
    img[up] = cv2.warpAffine(image, utils.AFFINE_DIR[do], dsize=(w, h))
    img[do] = cv2.warpAffine(image, utils.AFFINE_DIR[up], dsize=(w, h))
    img[le] = cv2.warpAffine(image, utils.AFFINE_DIR[ri], dsize=(w, h))
    img[ri] = cv2.warpAffine(image, utils.AFFINE_DIR[le], dsize=(w, h))
    u_denominator = np.zeros([h,w], dtype=np.float32)
    
    # compute partial lambda factor and the denominator for the u_lambda term
    for j in directions:
        inverse_color_diff = np.float32(1)/(L2_norm(image, img[j], False) + epsilon)
        u_denominator += inverse_color_diff # compute denominator for u_lambda term
        lambda_factor[j] = inverse_color_diff # compute partial lambda factor
    
    # compute u_lambda term (approximation of the term, incorrect on image borders)
    u_lambda = np.float32(4)/u_denominator
    
    # compute final lambda smoothness weight for each edge
    for j in directions:
        lambda_factor[j] = ws*u_lambda*lambda_factor[j]
    
    return lambda_factor
