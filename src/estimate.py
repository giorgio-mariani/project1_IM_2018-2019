# -*- coding: utf-8 -*-
import os

import numpy as np
import cv2

import lbp
import utils
import compute_energy as ce

###############################################################################

def estimate(configfile):
    """Estimate and store depthmaps from a picture sequence.

    The picture sequence directory, and a configuration parameters is 
    obtained by reading the configuration file *configfile*. The 
    estimated depth-maps are subsequently stored in an output directory 
    specified in the aforementioned file. To read a more in-depth explanation
    of the parameters in *configfile* see :ref:`config-file`.

    Notes
    -----
    If the configuration parameter ``depthmaps_directory`` is not **null** 
    then *bundle optimization* is performed during estimation.

    See Also
    --------
    compute_frame

    Parameters
    ----------
    configfile : str or unicode
        Filename of the configuration file.
    """

    configparams = utils.parse_configfile(configfile)
    
    # create necessary objects
    out_dir = configparams["output_directory"]
    sequence = utils.Sequence(configparams)
    
    # check output directory constraints:
    if os.path.isdir(out_dir):
        raise StandardError("The output directory already exists, remove it in order to proceed!")
    if os.path.exists(out_dir):
        raise StandardError("The output directory name is already used by another file/resource, remove it in order to proceed!")
    
    # create output directory (TODO remove if an exception occur and the folder is empty)
    os.mkdir(out_dir)
    
    for i in range(sequence.start, sequence.end):
        print "Estimating depth-map for frame ", str(i)
        depthmap = compute_frame(i-sequence.start, sequence, configparams)
        depthmap_filename = os.path.join(out_dir, "depth_"+str(i).zfill(4))

        # save depth info            
        np.save(depthmap_filename, depthmap)
        
        # save picture of image (useful for debug purposes)
        Max = np.float32(depthmap.max())
        cv2.imwrite(depthmap_filename+'.png', np.uint8(depthmap/Max*255))

def compute_frame(frame, sequence, configparams):
    """Estimate the (per-pixel) depth labels for a single frame in the sequence.

    This function executes the depth-map estimation for the frame *frame*, 
    given the input sequence *sequence*. If ``sequence.use_bundle()==True``
    then *bundle optimization* is performed using the depth-maps in
    *sequence*.

    Parameters
    ----------
    frame : int
        Frame whose depth-map is estimated.

    sequence : utils.Sequence
        Object containing parameters necessary to the depth-maps estimation.
        It contains the camera matrices, picture arrays, length of the
        sequence, etc. If the sequence instance contains also
        previously estimated depth-maps, then the *bundle optimization*
        phase is also executed.

    configparams : dict 
        Configuration parameters. It contains information about 
        necessary files directories, as well as the values for various
        parameters used by the system.

    Returns
    -------
    numpy array, type uint16
        An array representing the per-pixel depth labels. The shape of 
        such array is `[h, w]` (with `h` and `w` heigth and width of 
        the input frame).
    """
    assert type(frame) == int

    # compute parameters
    depth_range = sequence.depth_max - sequence.depth_min
    step = depth_range/sequence.depth_levels
    eta_default = 0.05*depth_range
    ws_default = 5.0/depth_range

    # read parameters from configuration file
    sigma_c = configparams["sigma_c"] or 10
    sigma_d = configparams["sigma_d"] or 2.5
    eta = configparams["eta"] or eta_default
    ws = configparams["w_s"] or ws_default
    epsilon = configparams["epsilon"] or 1.0
    window_side = configparams["window_side"] or 10

    # compute the per-pixel weight to be used during LBP
    pixels_weights = ce.compute_energy_data(
        frame_index=frame,
        sequence=sequence,
        window_side=window_side,
        sigma_c=sigma_c,
        sigma_d=sigma_d)
    
    # compute edges' weights for LBP
    edges_weights = ce.lambda_factor(
        image=sequence.I[frame], 
        ws=ws, 
        epsilon=epsilon)
    
    # execute LBP algorithm
    depthmap_indices = lbp.lbp(pixels_weights, edges_weights, eta=eta, step=step)
    return depthmap_indices

#------------------------------------------------------------------------------

#estimate('../configfile_example.txt')
