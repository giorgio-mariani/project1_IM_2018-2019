# -*- coding: utf-8 -*-
import os

import numpy as np
import cv2

import lbp
import utils
import compute_energy as ce
import params


SCALE = 0.15
WIDTH = int(960*SCALE)
HEIGHT = int(540*SCALE)

PIC_DIRNAME = '../video sequences/lawn/'
DEP_DIRNAME =  'depths/405h_720w_40d_init/'

'''
utils.debug_depth_estimation(
        camera_file=PIC_DIRNAME+'cameras.txt',
        source_folder=PIC_DIRNAME+'src',
        #depths_folder=DEP_DIRNAME,
        height=HEIGHT,
        width=WIDTH,
        show_lambda=True,
        show_depth_levels=True,
        show_init=True,
        show_lbp=True,
        frame=10)'''



###############################################################################

def estimate(
        out_dir,
        camera,
        pic_sequence,
        dep_sequence=None):
    
    assert isinstance(pic_sequence, utils.PictureSequence)
    assert isinstance(camera, utils.Camera)
    assert type(out_dir) == str
    
    # check output directory constraints:
    if os.path.isdir(out_dir):
        raise "The output directory already exists, remove it in order to proceed!"
    if os.path.exists(out_dir):
        raise "The output directory name is already used by another file/resource, remove it in order to proceed!"
    
    # create output directory (TODO remove if an exception occur and the folder is empty)
    os.mkdir(out_dir)
    
    # create directory stat file
    '''stats = '\n'.join(
        ["height="+str(pic_sequence.height),
         "width="+str(pic_sequence.width),
         "levels="+str(params.DEPTH_LEVELS)])'''

    for i in range(len(pic_sequence)):
        print "Estimating depth-map for frame ", str(i)
        depthmap = compute_frame(i, camera, pic_sequence, dep_sequence)
    
        # save depth info            
        filename = os.path.join(out_dir, pic_sequence.get_filename(i, extension=False))
        np.save(filename, depthmap)
        
        # save picture of image (useful for debug purposes)
        M = np.float32(depthmap.max())
        cv2.imwrite(filename+'.png', np.uint8(depthmap/M*255))

def compute_frame(frame, camera, pic_sequence, dep_sequence=None):
    """
    Compute the (per-pixel) depth labels for a single frame in the sequence.
    
    Arguments:
     :frame: frame whose depth-map is computed
     :camera: utils.Camera containing the camera parameters for the sequence
     :pic_sequence: utils.PictureSequence representing the input sequence
     :depth_sequence: utils.DepthSequence representing the depth-maps used during bundle phase
     
    Return:
     a numpy.ndarray with type np.uint16 representing the per-pixel depth labels.
     The sahpe of such array is [height, width] (with h and w heigth and width of the input frame).
    """
    assert type(frame) == int

    # compute the per-pixel weight to be used during LBP
    pixels_weights = ce.compute_energy_data(camera, frame, pic_sequence, dep_sequence)
    
    # compute edges' weights for LBP
    edges_weights = ce.lambda_factor(np.float32(pic_sequence[frame]))
    
    # execute LBP algorithm
    depthmap_indices = lbp.LBP(pixels_weights, edges_weights)
    return depthmap_indices

#------------------------------------------------------------------------------

sequence = utils.PictureSequence(
        PIC_DIRNAME+'src', '.jpg',
        height=HEIGHT, 
        width=WIDTH,
        maxlength=50)

camera = utils.Camera(
        PIC_DIRNAME+'cameras.txt', 
        height=sequence.height, 
        width=sequence.width)

dep_sequence = utils.DepthSequence('out_bundle', sequence)

estimate('out_bundle2', camera, sequence, dep_sequence)
    