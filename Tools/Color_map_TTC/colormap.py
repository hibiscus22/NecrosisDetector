import cv2
import glob
import numpy as np

def applyCustomColorMap(im_gray):

    lut = np.zeros((256, 1, 3), dtype=np.uint8)

    #Red
    lut[:, 0, 2] = np.full(shape=256, fill_value=153)

    #Green
    lut[:, 0, 1] = np.arange(256)

    #Blue
    lut[:, 0, 0] = np.arange(256)

    #Apply custom colormap through LUT
    im_color = cv2.LUT(im_gray, lut)
    
    return im_color;




# array with all images
img = [cv2.imread(file) for file in glob.glob("./*.JPG")]

c = 'e'

for i in img:
    cm = applyCustomColorMap(i)
    
    #cv2.namedWindow("Colormap", cv2.WINDOW_NORMAL)
    #cv2.imshow('Colormap',cm)
    cv2.imwrite(c+'5_colormap.jpg', cm)
    c = chr(ord(c)+1)
    #cv2.waitKey(0)


