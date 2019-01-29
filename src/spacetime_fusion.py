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
    x0 = np.arange(0.1, 1.0, 0.1)
    print x0.shape
    
    A = np.asarray(
        [[-1., 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
          [ 0.,-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
          [ 0.,-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
          [ 0., 0.,  0., -1.,  1.,  0.,  0.,  0.,  0.],
          [ 0., 0.,  0.,  0., -1.,  1.,  0.,  0.,  0.],
          [ 0., 0.,  0.,  0., -1.,  1.,  0.,  0.,  0.],
          [ 0., 0.,  0.,  0.,  0.,  0., -1.,  1.,  0.],
          [ 0., 0.,  0.,  0.,  0.,  0.,  0., -1.,  1.],
          [ 0., 0.,  0.,  0.,  0.,  0.,  0., -1.,  1.]],
         dtype=np.float32)
    
    print A.shape
    b = np.ones(9)
    x, info = scipy.sparse.linalg.bicgstab(A, b, x0=x0, maxiter=1)
   
    print x, info
    print b
    print np.around(x, 2)
    # define space matrix right
    
    
    
    # find reliable points

    # define temporal matrix
    
    # compute CG solver until convergence or the max num. of iteration is reached
        #update temporal matrix with new estimations
    
    # return the resulting depth values
    
###############################################################################
    
a_flat = np.arange(1, 10, dtype=np.float32)
a = np.reshape(a_flat, [3, 3, 1])

compute_spatial_time_fusion(a)