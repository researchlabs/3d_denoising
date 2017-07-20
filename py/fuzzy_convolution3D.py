#import cv2
import numpy as np
#from scipy import ndimage
#from matplotlib import pyplot as plt
import math

def calculate_3d_transform(image, p, kernel, x_dim,y_dim, z_dim,k_x, k_y, k_z, x_step, y_step, z_step):
    padded_image = np.zeros((x_dim + 2*k_x, y_dim + 2*k_y, z_dim +2*k_z))
    padded_image[k_x: x_dim + k_x, k_y: y_dim + k_y, k_z: z_dim+k_z] = image
#    print (p, x_step, y_step, z_step, k_x, k_y, k_z, padded_image.shape)
    patch = padded_image[(p[0] + k_x - x_step): p[0] + k_x + x_step+1,
                         (p[1] + k_y - y_step): p[1] + k_y + y_step+1,
                         (p[2] + k_z - z_step): p[2] + k_z + z_step+1]    
    
#    print (patch.shape, kernel.shape)

    s1 = patch * kernel
    
    s2 = 0
    #sum for only elements of memebership function inside the 3D image for this patch
    for i in range(0,k_x):
        if ((p[0] + k_x - x_step + i) >= k_x) and ((p[0] + k_x - x_step + i) < x_dim+k_x):
            for j in range(0, k_y):
                if ((p[1] + k_y - y_step + j) >= k_y) and ((p[1] + k_y - y_step + j) < y_dim+k_y):
                    for k in range(0, k_z):
                        if ((p[2] + k_z - z_step + k) >= k_z) and ((p[2] + k_z - z_step + k) < z_dim+k_z): 
                            s2 = s2 + kernel[i][j][k]

    Fk = s1.sum()/s2        
    if (math.isnan(Fk)):
        Fk = 0
        print ("Fk is Nan debug it")
    return Fk

def inverse_transform(image, F, i, j, k, kernel, x_dim,y_dim, z_dim,k_x, k_y, k_z, x_step, y_step, z_step):    
    Inv = 0
    for l in range(0, len(F)):
        if ((F[l][0] <= (i + x_step)) and (F[l][0] > (i - x_step))
            and (F[l][1] <= (j + y_step)) and (F[l][1] > (j - y_step))
            and (F[l][2] <= (k + z_step)) and (F[l][2] > (k - z_step))):

            Fk = F[l][3]
            i0 = F[l][0]
            j0 = F[l][1]
            k0 = F[l][2]
            Inv = Fk*kernel[i - (i0-x_step)][j - (j0 - y_step)][k - (k0 - z_step)] + Inv   
    return Inv

#remove /25
kernel = np.ones((5,5,5),np.float32)
membership = [0, 0.5, 1, 0.5, 0]
for i in  range(0,5):
    for j in range(0,5):
        for k in range(0,5):
            kernel[i][j][k] = membership[i]*membership[j]*membership[k]

source_array = np.ones((7,7,7),np.float32)
dest_array = np.zeros((7,7,7),np.float32)
source_array[4,4,4] = 2
#print(kernel)

#Array_for_F_components
F = list()

#Kernel shape variables
k_x, k_y, k_z = kernel.shape
print (k_x, k_y, k_z)
x_dim, y_dim, z_dim = source_array.shape 

#Step between centers of Ruspini partirions 
#It defines also count of the Ruspini partitions and F-components.
#For example - step equals to 1/2 of kernel size
x_step = int(k_x/2)
y_step = int(k_y/2)
z_step = int(k_z/2)

#Calculate direct F-ransform for each patch equals to kernel size

for i in range(0,x_dim, x_step):
    for j in range(0, y_dim, y_step):
        for k in range(0, z_dim, z_step):
            F.append([i,j,k,calculate_3d_transform(
                source_array, (i, j, k), kernel, x_dim, y_dim, z_dim, k_x, k_y, k_z, x_step, y_step, z_step)])            

print (F)

for i in range(0,x_dim):
    for j in range(0, y_dim):
        for k in range(0, z_dim):
            R = inverse_transform(source_array, F, i, j, k, kernel,x_dim, y_dim, z_dim, k_x, k_y, k_z, x_step, y_step, z_step)            
            dest_array[i, j, k] = R

print (dest_array)
