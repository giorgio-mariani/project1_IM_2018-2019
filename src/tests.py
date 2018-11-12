# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 13:24:43 2018

@author: giorg
"""
import cv2
import numpy as np

def test_warpaffine_is_generalized():
    '''
    This test checks whether or not the operation warpaffine can be generalized
    to 3 dim. matrices
    '''
    A = np.array([[[1,2,3,4],
                  [1,2,3,4]],
                  [[5,6,7,8],
                  [5,6,7,8]],
                  [[9,9,9,9],
                   [9,9,9,9]]], dtype=np.float32)
    
    warp = np.array([[1,0,-1],[0,1,0]], np.float32) #affine matrix of the index transformation
    A_warp = cv2.warpAffine(A, warp, dsize=(2,3))
