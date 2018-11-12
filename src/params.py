# -*- coding: utf-8 -*-
import numpy as np

# general parameters 1
SIGMA_C = 10.0
SIGMA_D_SQUARED = 2.5*2.5
EPSILON = 50
ITERATION_NUMBER = 5

# depth parameters
DEPTH_LEVELS = 40
DEPTH_MIN = np.float32(0.00)
DEPTH_MAX = np.float32(0.008)
DEPTH_VEC = np.linspace(
        DEPTH_MIN,
        DEPTH_MAX,
        DEPTH_LEVELS,
        dtype=np.float32)

# general parameters 2
ETA = 0.05*(DEPTH_MAX - DEPTH_MIN) 
STEP = (DEPTH_MAX - DEPTH_MIN)/(DEPTH_LEVELS)
W_S = 5/(DEPTH_MAX-DEPTH_MIN)


# neighbours parameters
UP, DOWN, LEFT, RIGHT = 0,1,2,3
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]
AFFINE_DIR = {UP:   np.array([[1, 0, 0], [0, 1,-1]], dtype=np.float32),
              DOWN: np.array([[1, 0, 0], [0, 1, 1]], dtype=np.float32),
              LEFT: np.array([[1, 0,-1], [0, 1, 0]], dtype=np.float32),
              RIGHT:np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)}