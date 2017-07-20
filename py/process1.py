#!/usr/bin/python
# %matplotlib inline

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

##from mayavi import mlab
from matplotlib import pyplot, cm

#from skimage import measure, morphology
import skimage
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import sys
print(sys.version)
bashCommand = "conda info -e"
os.system(bashCommand)


# Some constants
#INPUT_FOLDER = '/home/oleksii/kaggle/Bowl/input/sample_images/'
INPUT_FOLDER = '../../dataset/d0_test'
OUTPUT_FOLDER = '../../dataset/d0_del'
#INPUT_FOLDER = '../../dataset/d1_SE000002'
#INPUT_FOLDER = '../../dataset/d2_SE000001'
# INPUT_FOLDER = '/home/oleksii/kaggle/Bowl/trio/'
# INPUT_FOLDER = '/home/oleksii/kaggle/Bowl/input3/'


# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s, force=True) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

def save_scan(path, dataset):
	for i in range(0, len(dataset)-1):	
		dicom.write_file(path+"/slice"+i, dataset[i])
	return 0

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)

first_patient = load_scan(INPUT_FOLDER)
##save_scan(OUTPUT_FOLDER, first_patient)
exit();

first_patient_pixels = get_pixels_hu(first_patient)


plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()


# Show some slice in the middle
##plt.imshow(first_patient_pixels[5], cmap=plt.cm.gray)
plt.figure(1)
plt.subplot(321)
plt.imshow(first_patient_pixels[5], cmap=plt.cm.gray)
plt.subplot(322)
plt.imshow(first_patient_pixels[10], cmap=plt.cm.gray)

plt.subplot(323)
plt.imshow(first_patient_pixels[15], cmap=plt.cm.gray)
plt.subplot(324)
plt.imshow(first_patient_pixels[20], cmap=plt.cm.gray)

plt.subplot(325)
plt.imshow(first_patient_pixels[25], cmap=plt.cm.gray)
plt.subplot(326)
plt.imshow(first_patient_pixels[30], cmap=plt.cm.gray)

##plt.show()
##print("SliceThickness", first_patient[0].SliceThickness)
##print("PixelSpacing  ", first_patient[0].PixelSpacing)


def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing

    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))

    spacing = np.array(list(spacing))

    resize_factor 		= spacing / new_spacing
    new_real_shape 		= image.shape * resize_factor
    new_shape 			= np.round(new_real_shape)
    real_resize_factor 	= new_shape / image.shape
    new_spacing 		= spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing

def noise_adding(image, psy):
	img = np.array(image, copy=True)
	noise = np.random.random(image.shape)
	noise[noise >= 1-psy] = 1
	noise[noise < 1-psy] = 0
	std_let = image.std()
	img = image + np.ndarray.round(noise*std_let, decimals=0)

	return img

def noise_reduction(image):
	img = np.array(image, copy=True)  	
#http://lagrange.univ-lyon1.fr/docs/scipy/0.17.1/ndimage.html
	img = scipy.ndimage.filters.gaussian_filter(image, sigma=0.2, truncate=3.0)
#	img = scipy.ndimage.fourier_gaussian(image, sigma=0.2)

#	img = scipy.ndimage.filters.gaussian_gradient_magnitude(image, sigma=0.2, truncate=3.0)
#	img = scipy.ndimage.filters.median_filter(image, 3)
##	img = scipy.ndimage.filters.uniform_filter(image, size=3)

	#size of image
	#fix
	return img

def distance_calculation(image1, image2):
#	scipy.ndimage.distance_transform_bf()
	result = abs(image1 - image2)
	unique, counts = np.unique(result, return_counts=True)
	return counts[1:5].sum()

####pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1, 1, 1])
#print("Shape before resampling\t", first_patient_pixels.shape)
#print("Shape after resampling\t", pix_resampled.shape)
#print("spacing", spacing)


image_with_noise = noise_adding(first_patient_pixels, 0.02)
image_without_noise = noise_reduction(image_with_noise)

how_much1 = distance_calculation(image_with_noise, image_without_noise)

image_without_noise1 = noise_reduction(first_patient_pixels)
how_much2 = distance_calculation(image_without_noise1, first_patient_pixels)

how_much3 = distance_calculation(image_without_noise1, image_without_noise)

image_without_noise2 = noise_reduction(image_without_noise1)
how_much4 = distance_calculation(image_without_noise2, first_patient_pixels)
how_much5 = distance_calculation(image_without_noise2, image_without_noise1)

print ("how_much")
print ("how_much with noise && reduction nose        :", how_much1)
print ("how_much with reduction origin and origin    :", how_much2)
print ("how_much reduction noise and reduction origin:", how_much3)

print ("how_much 2 - origin                          :", how_much4)
print ("how_much 2 - 1                               :", how_much5)


exit();



def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)
    p = p[:, :, ::-1]

#    verts, faces = measure.marching_cubes(p, threshold)
    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


plot_3d(pix_resampled, 400)
print("exit")



def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image


##segmented_lungs = segment_lung_mask(pix_resampled, False)
##segmented_lungs_fill = segment_lung_mask(pix_resampled, True)

##plot_3d(segmented_lungs, 0)

##plot_3d(segmented_lungs_fill, 0)

##plot_3d(segmented_lungs_fill - segmented_lungs, 0)
