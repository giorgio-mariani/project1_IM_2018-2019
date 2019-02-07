# -*- coding: utf-8 -*-
import os

import cv2 
import numpy as np

import json


# neighbourhood parameters
UP, DOWN, LEFT, RIGHT = 0,1,2,3
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]
AFFINE_DIR = {UP:   np.array([[1, 0, 0], [0, 1,-1]], dtype=np.float32),
              DOWN: np.array([[1, 0, 0], [0, 1, 1]], dtype=np.float32),
              LEFT: np.array([[1, 0,-1], [0, 1, 0]], dtype=np.float32),
              RIGHT:np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)}

#------------------------------------------------------------------------
class Sequence():
    """

    Attributes
    ----------
    height : int
        height of the camera (important for intrinsic matrix).

    width : int
        width of the camera (important for intrinsic matrix).

    K : numpy matrix, type float32
        Matrix with shape `[3, 3]` representing the *intrinsic matrix* of
        the camera.

    Rs : list of numpy matrix, type float32
        List of numpy matrices with shape `[3, 3]` representing the 
        *rotations* of the camera during the sequence.

    Ts : list of numpy matrix, type float32
        List of numpy matrices with shape `[1, 3]` representing the 
        *translations* of the camera during the sequence.

    I : list of numpy arrays, type float32
        List of numpy arrays with shape `[h,w,3]`, representing the 
        images in the sequence. It must be true that `end - start = len(I)`

    D : list of numpy arrays, optional, type uint16
        List of numpy arrays with shape `[h,w]`, containing previously 
        estimated depth-maps of the sequence. If this parameter is passed
        as input then the sequence will use *bundle optimization*.
        It must be true that `end - start = len(I)`.

    Parameters
    ----------
    configparams : dict
        Configuration parameters. This dictionary contains information about 
        necessary files directories, as well as the values for various 
        parameters used by the system.

    """

    def __init__(self, configparams):

        camera_file = configparams["camera_file"]
        pictures_dir = configparams["pictures_directory"]
        pictures_ext = configparams["pictures_file_extension"]
        depthmap_dir = configparams["depthmaps_directory"]

        height = configparams["height"]
        width = configparams["width"]
        start = configparams["start_frame"]
        end = configparams["end_frame"]

        self.depth_levels = configparams["depth_levels"]
        self.depth_min = np.float32(configparams["depth_min"])
        self.depth_max = np.float32(configparams["depth_max"])

        self.height = height
        self.width = width
        self.start = start
        self.end = end

        # setup camera parameters-----------------------------------------
        Ks, Rs, Ts = self._load_camera_params(camera_file)
        
        if len(Rs) < end or len(Ts) < end:
            raise StandardError()

        self.K = Ks[0]
        self.Rs = Rs[start:end]
        self.Ts = Ts[start:end]

        orig_width = (self.K[0,2]+0.5)*2
        orig_height = (self.K[1,2]+0.5)*2
        
        # necessary if target resultion is different from the one used 
        # to estimate the camera parameters.
        self.K[0, 0] = self.K[0,0]*width/orig_width
        self.K[1, 1] = self.K[1,1]*height/orig_height
        
        self.K[0, 2] = width/2.0 + 0.5
        self.K[1, 2] = height/2.0 + 0.5
        #-----------------------------------------------------------------
        self.I = self._load_pictures(pictures_dir, pictures_ext, start, end)
        self.D = None if depthmap_dir is None else self._load_depthmaps(
            depthmap_dir, start, end)

    def _load_camera_params(self, filename):
        with open(filename,'r') as f:
            frame_num = int(f.readline())
            f.readline()
            
            def read_vector():
                vec = f.readline().split()
                return np.array([float(x) for x in vec], dtype=np.float32)
                
            def read_matrix():
                r1 = read_vector()
                r2 = read_vector()
                r3 = read_vector()
                return np.matrix([r1,r2,r3], dtype=np.float32)
            
            K_sequence = [None]*frame_num
            R_sequence = [None]*frame_num
            T_sequence = [None]*frame_num
            
            for i in range(frame_num):
                K_sequence[i] = read_matrix()
                R_sequence[i] = read_matrix()
                T_sequence[i] = np.asmatrix(read_vector())
                f.readline()
                f.readline()
        return K_sequence, R_sequence, T_sequence

    def _load_pictures(self, directory, file_ext, start, end):
        I = []
        for i in range(start, end):
            img_name = os.path.join(directory,"img_"+str(i).zfill(4)+file_ext)
            img = cv2.imread(img_name)
            if img is None:
                raise StandardError("Image "+img_name+" not found in picture sequence directory ("+directory+")!")
            I.append(np.float32(cv2.resize(img, (self.width,self.height))))
        return I

    def _load_depthmaps(self, directory, start, end):
        D = []
        for i in range(start, end):
            depthmap_name = os.path.join(directory, "depth_"+str(i).zfill(4)+".npy")
            try:
                depthmap = np.load(depthmap_name)
                D.append(depthmap)
            except IOError as e:
                raise StandardError("Depth-map file "+depthmap_name+" not found in depth sequence directory ("+directory+")!")

        # TODO check correct format shape height, width, labels
        return D
    #--------------------------------------------------------------------------
    
    def use_bundle(self):
        """Return if this sequence can be used for *bundle optimization*,
        that is, if it contains previously estimated depth-maps.

        Returns
        -------
        bool
            Whether the sequence can be used with *bundle optimization*.
        """
        return self.D != None


#------------------------------------------------------------------------------

def parse_configfile(filename):
    """Parse a configuration file and return a dictionary containing the
    various parameters used by the system.

    For a more in detail explanation of the parameters contained in the 
    configuration file, and its format see :ref:`config-file`.
    
    Parameters
    ----------
    filename : str or unicode
        Relative or absolute path to the configuration file.

    Returns
    -------
    dict
        Dictionary containing the system parameters. If a parameter-value
        pair was not in the configuration file then its value is ``None``.

    """

    assert isinstance(filename, basestring) 
    
    required_keys = {
        "camera_file":basestring,
        "pictures_directory":basestring,
        "output_directory":basestring,
        "pictures_file_extension":basestring,
        "height":int,
        "width":int,
        "start_frame":int,
        "end_frame":int,
        "depth_levels":int,
        "depth_min": float,
        "depth_max":float
    }

    optional_keys = {
        "depthmaps_directory":basestring,
        "sigma_c": float,
        "sigma_d":float,
        "eta": float,
        "w_s": float,
        "epsilon": float,
        "window_side": int,
    }
    
    # try to parse the file as a json object
    try:
        with open(filename,'r') as fp:
            jobj = json.load(fp=fp)
    except FileNotFoundError:
        raise StandardError("File "+str(filename)+"not found!")
    except IOError as e:
        raise StandardError("Unable to load file "+str(filename)+"!")
        
    # check if jason file contains a json object
    if type(jobj) != dict :
        raise StandardError("Configuration file should contain a single JSON object!")
    
    # check for correct type
    for key in required_keys:
        if key not in jobj:
            raise StandardError("parameter "+key+" is not in the configuration file JSON object!")
        elif not isinstance(jobj[key], required_keys[key]):
            raise StandardError("parameter "+key+" has the wrong type: expected "+str(required_keys[key])+", received "+str(type(jobj[key]))+"!")

    for key in optional_keys:
        if key not in jobj:
            jobj[key] = None
        elif not isinstance(jobj[key], optional_keys[key]):
            raise StandardError("parameter "+key+" has the wrong value: expected "+str(optional_keys[key])+", received "+str(type(jobj[key]))+"!")

    # check if there are unknown parameters in the config. file
    for key in jobj.keys():
        if key not in required_keys and key not in optional_keys:
            raise StandardError("Unknown parameter "+str(key)+" in the configuration file!")
    
    # convert relative paths into absolute paths
    absdirname = os.path.dirname(os.path.abspath(filename))
    for key in ["camera_file","depthmaps_directory","pictures_directory", "output_directory"]:
        if jobj[key] is not None:
            jobj[key] = os.path.join(absdirname,jobj[key])
    return jobj

###############################################################################

'''
def show_image(img, secs=0, close_after=True):
    cv2.imshow('image', cv2.UMat(img))
    cv2.waitKey(secs)
    if close_after:
        cv2.destroyAllWindows()

def show_depthmap(depth_map, secs=0, close_after=True):
    depth_map = np.array(depth_map)
    m = depth_map.min()
    M = depth_map.max()
    img = np.uint8((depth_map-m)/(M-m)*255)
    show_image(img, secs, close_after)

def load_image_depth(filename, width, height):
    import struct
    with open(filename, "rb") as f:
        def read_float():
            bs = f.read(4)
            if bs=="":
                return None 
            else:
                return struct.unpack('f', bs)[0]
        
        I = np.zeros([height, width], dtype=np.float32)
        h, w = 0, 0
        disp = read_float()
        while disp is not None:
            I[h,w] = disp
            
            w = (w + 1) % width
            h = h + 1 if w==0 else h 
            disp = read_float()
    return I
'''
###############################################################################