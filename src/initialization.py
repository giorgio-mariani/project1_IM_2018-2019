# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2

import compute_energy as ce
import lbp
import utils
import params

def compute_frame(frame_index,
                  pic_sequence, 
                  camera, 
                  depthmap_directory,
                  save=False):
    
    D = ce.compute_energy_init(picture_sequence, camera, frame_index)
    lambda_weights = ce.compute_smoothness_weight(
            np.float32(picture_sequence[frame_index]))
    
    depth_index_map = lbp.LBP(D, lambda_weights)
    #depthmap = np.take(params.DEPTH_VEC, B)
    
    if save:
        filename = os.path.join(
                depthmap_directory, 
                pic_sequence.get_filename(extension=False))
        
        np.save(filename, depth_index_map)
        
        M = np.float32(depth_index_map.max())
        img = np.uint8(depth_index_map/M*255)
        cv2.imwrite(filename+'.png', img)
    return M
