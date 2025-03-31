import os
import cv2
import numpy as np
import glob

# small pipeline to show image
def imshow(img):
    cv2.namedWindow("out", cv2.WINDOW_NORMAL)
    cv2.imshow("out",(img).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# empty array for appending small images
small_imgs = []
size = 300 # size we want to resize to

# where's folder
path = "/home/maiki/Documents/TUW/imgs_to_reduce"

# IMPORTANT: COLUMN THAT WE WANT TO SHOW
col = "11_p"

# Read and sort the list of names: a1, a2, a3...
f_list = glob.glob(f"{path}/2reduce/*{col}.jpg")
f_list.sort()

# how many images?
print(len(f_list))

# list to an array of images
imgs = [cv2.imread(file) for file in f_list]

# resize it all, to a new array
for i in imgs:
    small_imgs.append(cv2.resize(i, (size, size)))


# size of the row
width_img_matrix = 4

# frame to put the images into (just a row with three channels)
aux_vertical = np.empty((1, size*width_img_matrix, 3))

# white image to fill blank spaces 
white_img = 255*np.ones((size, size, 3))

for i in range(int(len(small_imgs)/width_img_matrix)+1): # divide: 7 images/4 images per row +1 = 2 rows
    try:
        aux_horizontal = small_imgs[i*width_img_matrix] # first image to append to it
        for j in range(width_img_matrix-1): # now append images 
            try:# append next image in line
                aux_horizontal = np.concatenate((aux_horizontal, small_imgs[i*width_img_matrix+j+1]), axis = 1)
            except: # append a white image
                aux_horizontal = np.concatenate((aux_horizontal, white_img), axis = 1)
        # now append the whole row
        aux_vertical = np.concatenate((aux_vertical, aux_horizontal))
        
    except: pass # try except for exact division (8/4+1 = 3)


imshow(aux_vertical)

# save
cv2.imwrite(f"{path}/reduced/{col}.jpg", aux_vertical[1:,:,:])
# cv2.imwrite("./reduced/3.jpg", aux_horizontal)