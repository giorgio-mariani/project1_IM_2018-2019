# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2

import compute_energy as ce
import lbp
import utils
import params

def compute_frame_init(frame_index,
                  pic_sequence, 
                  camera, 
                  directory,
                  save=False):
    
    assert isinstance(pic_sequence, utils.PictureSequence)
    assert isinstance(camera, utils.Camera)
    
    assert type(frame_index) == int
    assert type(directory) == str
    assert type(save) == bool
    
    #TODO check that os.path.exists(directory)
 
    initdir_name = (str(pic_sequence.height)+'h_'+
                    str(pic_sequence.width)+'w_'+
                    str(params.DEPTH_LEVELS)+'d_init')
    init_directory = os.path.join(directory, initdir_name)
    
    D = ce.compute_energy(camera, frame_index,pic_sequence)
    lambda_weights = ce.compute_smoothness_weight(
            np.float32(pic_sequence[frame_index]))
    
    depth_index_map = lbp.LBP(D, lambda_weights)
    
    if save:
        # create init directory
        if not os.path.exists(init_directory):
            os.makedirs(init_directory)
                                    
        # save depth info            
        filename = os.path.join(
                init_directory, 
                pic_sequence.get_filename(frame_index, extension=False))
        np.save(filename, depth_index_map)
        
        # save picture of image (for debug purpose)
        M = np.float32(depth_index_map.max())
        img = np.uint8(depth_index_map/M*255)
        cv2.imwrite(filename+'.png', img)
    return M

def compute_frame_bundle(frame_index,
                  pic_sequence,
                  depth_sequence,
                  camera, 
                  directory,
                  save=False):
    
    assert isinstance(pic_sequence, utils.PictureSequence)
    assert isinstance(depth_sequence, utils.DepthSequence)
    assert isinstance(camera, utils.Camera)
    
    assert type(frame_index) == int
    assert type(directory) == str
    assert type(save) == bool
    
    #TODO check that os.path.exists(directory)

    bundledir_name = (str(pic_sequence.height)+'h_'+
                      str(pic_sequence.width)+'w_'
                      +str(params.DEPTH_LEVELS)+'d_bundle')
    bundle_directory = os.path.join(directory, bundledir_name)
    
    # compute frame info
    D = ce.compute_energy_bundle(
            camera, 
            frame_index,
            pic_sequence,
            depth_sequence)
    
    lambda_weights = ce.compute_smoothness_weight(
            np.float32(pic_sequence[frame_index]))
    
    depth_index_map = lbp.LBP(D, lambda_weights)
    
    if save:
        # create init directory
        if not os.path.exists(bundle_directory):
            os.makedirs(bundle_directory)
                             
        # save depth info            
        filename = os.path.join(
                bundle_directory, 
                pic_sequence.get_filename(frame_index, extension=False))
        np.save(filename, depth_index_map)
        
        # save picture of image (for debug purpose)
        M = np.float32(depth_index_map.max())
        img = np.uint8(depth_index_map/M*255)
        cv2.imwrite(filename+'.png', img)
    return M
