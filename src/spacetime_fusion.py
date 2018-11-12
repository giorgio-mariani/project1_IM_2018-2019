# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse
import scipy.sparse.linalg



def compute_spatial_time_fusion(depths):
    '''
    compute the spatial-time fusion between depth values.
    
    Arguments:
     * depths - matri containing the depth values
       for a certain time window.
        + numpy.ndarray
        + numpy.float32
        + [h,w,t]
    '''
    
    h, w, t = depths.shape
    depths_flat = np.reshape(depths, [-1])
    depth_num = len(depths_flat)

    
    X, Y = np.int32(np.meshgrid(
        np.arange(w, dtype=np.int32),
        np.arange(h, dtype=np.int32),
        indexing='xy'))
    
    X = np.reshape(X, [-1])
    Y = np.reshape(Y, [-1])
    
    Xr = X+1
    Yu = Y+1
    
    mask_r = np.float32(Xr != w)
    mask_u = np.float32(Yu != h)
    
    Xr = np.minimum(Xr, w-1)
    Yu = np.minimum(Yu, h-1)
    
    
    coor = Y*w + X    
    coor_r = Y*w + Xr
    coor_u = Yu*w + X

    x_sparse_r = np.concatenate([coor, coor_r], axis=0)
    y_sparse_r = np.concatenate([coor, coor], axis=0)
    
    x_sparse_u = np.concatenate([coor, coor_u], axis=0)
    y_sparse_u = np.concatenate([coor, coor], axis=0)

    d1 = depths_flat
    d2r = np.take(depths_flat, coor_r)
    d2u = np.take(depths_flat, coor_u)
    
    dr = np.ones([depth_num])*mask_r
    dr = np.concatenate([-dr, dr],axis=0)
    
    du = np.ones([depth_num])*mask_u
    du = np.concatenate([-du, du],axis=0)
    
    data_r = (dr, (y_sparse_r, x_sparse_r))
    data_u = (du, (y_sparse_u, x_sparse_u))

    # define space matrix up
    A_r = scipy.sparse.csr_matrix(
        data_r, 
        shape=(depth_num, depth_num))

    A_u = scipy.sparse.csr_matrix(
        data_u, 
        shape=(depth_num, depth_num))
    
    x0 = np.arange(0,1.2, 0.1)
    x, info = scipy.sparse.linalg.cg(A_r, d2r-d1, x0=x0, maxiter=10)
   
    print A_r.toarray()
    print d2r-d1
    print
    print A_u.toarray()
    print d2u - d1
    print
    print x
    
    # define space matrix right
    
    
    # find reliable points

    # define temporal matrix
    
    # compute CG solver until convergence or the max num. of iteration is reached
        #update temporal matrix with new estimations
    
    # return the resulting depth values
    
###############################################################################
    
a_flat = np.arange(1,13,dtype=np.float32)
a = np.reshape(a_flat,[4,3,1])
compute_spatial_time_fusion(a)


def compute_timeconstraint_weights(
        camera, 
        start_i, 
        end_i, 
        coordinates,
        depths):
    
    K = camera.K
    
