#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 11:04:24 2018

@author: huzmorgoth
"""

#preparing the hdf5 datasets of the DRIVE database
#-------------------------------------------------

import os
import h5py
import numpy as np
from PIL import Image



def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


"""--------------PATH TO THE IMAGES-----------------------------------------"""

"""--------------TRAINING--------------"""

original_imgs_train = "./NEWDRIVE/training/images/"
groundTruth_imgs_train = "./NEWDRIVE/training/1st_manual/"
borderMasks_imgs_train = "./NEWDRIVE/training/mask/"

"""--------------TEST--------------"""

original_imgs_test = "./NEWDRIVE/test/images/"
groundTruth_imgs_test = "./NEWDRIVE/test/1st_manual/"
borderMasks_imgs_test = "./NEWDRIVE/test/mask/"

"""-------------------------------------------------------------------------"""

Nimgs = 20
channels = 3
height = 584
width = 565
dataset_path = "./NEWDRIVE_datasets_training_testing/"

def get_datasets(imgs_dir,groundTruth_dir,borderMasks_dir,train_test="null"):
    imgs = np.empty((Nimgs,height,width,channels))
    groundTruth = np.empty((Nimgs,height,width))
    border_masks = np.empty((Nimgs,height,width))
    for path, subdirs, files in os.walk(imgs_dir):#list path's directories and files
        for i in range(len(files)):
            
            #Actual Data
            print("Actual image: ",files[i])
            img = Image.open(imgs_dir+files[i])
            imgs[i] = np.asarray(img)
            
            #corresponding ground truth
            groundTruth_name = files[i][0:2] + "_manual1.gif"
            print("(NAME) Ground Truth: ",groundTruth_name)
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)
            
            #corresponding border masks
            border_masks_name = ""
            if train_test=="train":
                border_masks_name = files[i][0:2] + "_training_mask.gif"
            elif train_test=="test":
                border_masks_name = files[i][0:2] + "_test_mask.gif"
            else:
                print("Train or Test..")
                exit()
            print("(NAME) Border Masks: " + border_masks_name)
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            border_masks[i] = np.asarray(b_mask)

    print("Images maximum: ",str(np.max(imgs)))
    print("Images minimum: ",str(np.min(imgs)))
    assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
    assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
    print("Ground Truth and Border Masks are within defined pixels, 0 to 255 - BW")
    
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,height,width))
    groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
    border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
    assert(groundTruth.shape == (Nimgs,1,height,width))
    assert(border_masks.shape == (Nimgs,1,height,width))
    return imgs, groundTruth, border_masks

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

#getting the training datasets
imgs_train, groundTruth_train, border_masks_train = get_datasets(original_imgs_train,groundTruth_imgs_train,borderMasks_imgs_train,"train")
print("train datasets (Saving...)")
write_hdf5(imgs_train, dataset_path + "NEWDRIVE_dataset_imgs_train.hdf5")
write_hdf5(groundTruth_train, dataset_path + "NEWDRIVE_dataset_groundTruth_train.hdf5")
write_hdf5(border_masks_train,dataset_path + "NEWDRIVE_dataset_borderMasks_train.hdf5")

#getting the testing datasets
imgs_test, groundTruth_test, border_masks_test = get_datasets(original_imgs_test,groundTruth_imgs_test,borderMasks_imgs_test,"test")
print("test datasets (Saving...)")
write_hdf5(imgs_test,dataset_path + "NEWDRIVE_dataset_imgs_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + "NEWDRIVE_dataset_groundTruth_test.hdf5")
write_hdf5(border_masks_test,dataset_path + "NEWDRIVE_dataset_borderMasks_test.hdf5")
