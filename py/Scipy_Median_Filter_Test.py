# -*- coding: utf-8 -*-

import sys
from operator import mul
from functools import reduce

import SimpleITK as sitk
from scipy import ndimage
import numpy as np
import math


def read_dicom_files(path):
    reader = sitk.ImageSeriesReader()
    filenames = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(filenames)
    image = reader.Execute()
    return image


def mse(image1, image2):
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    return err / reduce(mul, image1.shape)


def pixels_changed(image1, image2):
    pixels_total = reduce(mul, image1.shape)
    return pixels_total - (image1 == image2).sum()


def check_median_filter_on_3d(image, size):
    filtered_image = ndimage.filters.median_filter(image, size)
    pixels = pixels_changed(image, filtered_image)
    mse_result = mse(image, filtered_image)
    return pixels, mse_result


def check_median_filter_on_layers(image, size):
    filtered_image = np.ndarray(shape=image.shape, dtype=image.dtype)
    for i in range(image.shape[0]):
        filtered_image[i] = ndimage.filters.median_filter(image[i], size)
    pixels = pixels_changed(image, filtered_image)
    mse_result = mse(image, filtered_image)
    return pixels, mse_result


def check_F_transform_filter_on_3d(image, size):
    filtered_image = ndimage.filters.median_filter(image, size)
    pixels = pixels_changed(image, filtered_image)
    mse_result = mse(image, filtered_image)
    return pixels, mse_result








def main1():
##    window_size= 3
    
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
    
    _3d_results = check_median_filter_on_3d(npimage, window_size)
#    layers_results = check_median_filter_on_layers(npimage, window_size)
    
    if _3d_results == layers_results:
        print("Медианный фильтр работает одинаково для всего изображения целиком и послойно")
    else:
        print("Медианный фильтр работает по разному для всего изображения целиком и послойно")
    
    print("\nМедианный фильтр всего изображения:")
    print("Измененных пикселей: %d\nMean Squared Error: %f\n" % _3d_results)
    
#    print("Медианный фильтр по слоям:")
#    print("Измененных пикселей: %d\nMean Squared Error: %f\n" % layers_results)


#if __name__ == '__main__':
#    main()

