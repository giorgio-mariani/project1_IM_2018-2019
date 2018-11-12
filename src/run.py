# -*- coding: utf-8 -*-
import os
import pickle

import numpy as np
import cv2

import compute_energy as ce
import lbp
import utils
import params
import initialization as init

SCALE = 0.75
WIDTH = int(960*SCALE)
HEIGHT = int(540*SCALE)
MAXLENGTH = 30
SCALE_2 = 3
PIC_DIRNAME = '../video sequences/lawn/'
DEP_DIRNAME = 'depths/'

sequence = utils.PictureSequence(
        PIC_DIRNAME+'src', '.jpg',
        height=HEIGHT, 
        width=WIDTH, 
        maxlength=MAXLENGTH)

camera = utils.Camera(
        PIC_DIRNAME+'cameras.txt', 
        height=sequence.height, 
        width=sequence.width)

###############################################################################

USE_BUNDLE = True
SHOW_LAMBDA_WEIGTHS = False
SHOW_DEPTH_ESTIMATION_PROCESS = True
SHOW_DEPTH_INIT = True
SHOW_DEPTH_LBP = True


i = 3
if USE_BUNDLE:
    dep_sequence = utils.DepthSequence(DEP_DIRNAME, sequence)
    D = ce.compute_energy_bundle(sequence, dep_sequence, camera, i)
else:
    D = ce.compute_energy_init(sequence, camera, i)

lambda_weights = ce.compute_smoothness_weight(np.float32(sequence[i]))

if SHOW_LAMBDA_WEIGTHS:
    for i in range(4):
        img = cv2.resize(lambda_weights[i], (WIDTH*SCALE_2, HEIGHT*SCALE_2))
        utils.show_depthmap(img) 

if SHOW_DEPTH_ESTIMATION_PROCESS:
    for i in range(params.DEPTH_LEVELS):
        img = cv2.resize(D[i], (WIDTH*SCALE_2, HEIGHT*SCALE_2))
        utils.show_depthmap(img, 100, False)
    cv2.destroyAllWindows()

if SHOW_DEPTH_INIT:
    B = np.float32(D.argmin(axis=0))
    img = cv2.resize(B,(WIDTH*SCALE_2, HEIGHT*SCALE_2))
    utils.show_depthmap(img)

if SHOW_DEPTH_LBP:
    B = np.float32(lbp.LBP(D, lambda_weights))  
    img = cv2.resize(B,(WIDTH*SCALE_2, HEIGHT*SCALE_2))
    utils.show_depthmap(img)
    
filename = '../video sequences/Lawn/depth/_depth'+str(0)+'.raw'
depth = utils.load_image_depth(filename, 960, 540)
utils.show_depthmap(depth)

###############################################################################
    
'''
filename = '../video sequences/Lawn/depth/_depth'+str(0)+'.raw'
depth = utils.load_image_depth(filename, 960, 540)
utils.show_depthmap(depth)
'''

'''for i in range(len(sequence)):
    init.compute_frame(i, sequence, camera, 'depths', save=True)
    print 'frame number', i'''
    
#import sys
#sys.exit()