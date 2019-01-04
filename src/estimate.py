# -*- coding: utf-8 -*-
import numpy as np

import utils
import params

SCALE = 0.75
WIDTH = int(960*SCALE)
HEIGHT = int(540*SCALE)

PIC_DIRNAME = '../video sequences/lawn/'
DEP_DIRNAME =  'tmp/405h_720w_40d_bundle'#'405h_720w_40d_init/'


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
        frame=10)

###############################################################################
'''
dep_sequence = utils.DepthSequence(
        DEP_DIRNAME,
        sequence,
        maxlength=10)

for i in range(len(sequence)):
    #init.compute_frame_init(i, sequence, camera, './tmp/', save=True)
    init.compute_frame_bundle(i, sequence, dep_sequence, camera, './tmp/', save=True)
    print 'frame number', i
    
'''

'''

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
'''