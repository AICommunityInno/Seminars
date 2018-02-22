from augmentation import SegmentationGenerator
# from model_checkpoint_parallel import ModelCheckpoint
import unet
import segnet

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras.utils.training_utils import multi_gpu_model
import keras.backend as K

import cv2
from skimage.io import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from os.path import join, basename, exists
from os import listdir, makedirs

SIZE = (256, 256)
N_GPUS = 1
BATCH_SIZE = 16*N_GPUS

SMOOTH = 1.

def dice(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)


def dice_coef_loss(y_true, y_pred):
    return -dice(y_true, y_pred)

DATA_PATHS = {
    'TRAIN': {
        'IMAGES': 'data/train/images',
        'MASKS': 'data/train/masks',
    },
    'TEST': {
        'IMAGES': 'data/test/images',
        'MASKS': 'data/test/masks',
    }
}

train_datagen = SegmentationGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=90.,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.2,
    shear_range=0.3,
    fill_mode='reflect'
)

train_iter = train_datagen.flow_from_directory(
    DATA_PATHS['TRAIN']['IMAGES'],
    DATA_PATHS['TRAIN']['MASKS'],
    target_size=SIZE, color_mode='grayscale',
    batch_size=BATCH_SIZE, shuffle=True
)

valid_datagen = SegmentationGenerator()

valid_iter = valid_datagen.flow_from_directory(
    DATA_PATHS['TEST']['IMAGES'],
    DATA_PATHS['TEST']['MASKS'],
    target_size=SIZE, color_mode='grayscale',
    batch_size=BATCH_SIZE, shuffle=False
)

model = segnet.build_model(*SIZE, n_channels=1)
if N_GPUS >= 2:
    model = multi_gpu_model(model, N_GPUS)
    
optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
model.compile(loss=dice_coef_loss, optimizer=optimizer, metrics=[dice])

model_idx = 0

experiment_dir = 'segnet_{}x{}'.format(*SIZE)
models_dir = join(experiment_dir, 'models')
log_dir = join(experiment_dir, 'log')
if not exists(models_dir):
    makedirs(models_dir)
if not exists(log_dir):
    makedirs(log_dir)
modelname = 'segnet' + str(model_idx)
modelpath = join(models_dir, modelname)
tb_log_dir = join(log_dir, modelname)

model.fit_generator(
    train_iter,
    epochs=200,
    callbacks=[
        ModelCheckpoint(filepath=modelpath, monitor='val_dice', period=1),
        TensorBoard(log_dir=tb_log_dir, batch_size=BATCH_SIZE)
    ],
    validation_data=valid_iter,
    use_multiprocessing=False
)
