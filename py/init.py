#!/usr/bin/python

dicom1 = "../../dataset/d1_SE000002"
#dicom1 = "../../dataset/d0_test"

import Scipy_Median_Filter_Test
#import dicom_numpy

import fuzzy_convolution_3D

def main1():
    if len(sys.argv) < 2:
        print("Необходимо указать путь к папке с кт в параметрах.")
        sys.exit(1)
    try:
        sys.argv[2]
    except:
        window_size = 3
    else:
        window_size = int(sys.argv[2])

    print("Windows size ", window_size)

    image = read_dicom_files(sys.argv[1])
    
    npimage = sitk.GetArrayFromImage(image)
    print("Shape: %d, %d, %d" % npimage.shape)

    _3d_conv_results = fuzzy_convolution3D(npimage)
    pixels = pixels_changed(npimage, _3d_conv_results)
    mse_result = mse(npimage, filtered_image)

##    _3d_results = check_median_filter_on_3d(npimage, window_size)
#    layers_results = check_median_filter_on_layers(npimage, window_size)
    
    
####    print("\nМедианный фильтр всего изображения:")
####    print("Измененных пикселей: %d\nMean Squared Error: %f\n" % _3d_results)
    
    print("3D convolution фильтр:")
    print("Измененных пикселей: %d\nMean Squared Error: %f\n" % mse_result)
    print("pixels %d" % pixels)



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

if __name__ == '__main__':
    main()

