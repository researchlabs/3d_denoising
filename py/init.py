#!/usr/bin/python

dicom1 = "../../dataset/d1_SE000002"

import dicom
import dicom_numpy

#1. load 3D imgage from dicom1
# size = X*Y*Z



#2. Add Gasian noise (PryNoise = 5-40%)
#random change pixel in 3D object = PryNoise*X*Y*Z/100

#Clear noise
#if 2D - http://docs.opencv.org/trunk/d5/d69/tutorial_py_non_local_means.html
#if 3D - byself


#Assesment of noise reduction (PSNR, SSIM)
#MSE=Count different pizels
#PSNR=10*lg<10>(X*Y*Z/MSE)

#Same noise reducted object to dicom 2D slices of 3d Voxel
#
