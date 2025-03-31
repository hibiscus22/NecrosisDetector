import cv2
import numpy as np


def imshow(img):
    cv2.namedWindow("out", cv2.WINDOW_NORMAL)
    cv2.imshow("out",(img).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():

    opt = cv2.IMREAD_GRAYSCALE

    path_tst = "/home/maiki/Documents/TUW/ht29/pi/h3.jpg"
    path_pred = "/home/maiki/Documents/TUW/ht29/results_dt_regressor/h3.jpg"
    
    
    tst = cv2.resize(cv2.imread(path_tst, opt), (256,256))
    
    # READ FLUOR IMAGE
    pred = cv2.resize(cv2.imread(path_pred, opt), (256,256))
        
    print(cv2.matchTemplate(pred.astype(np.uint8), tst.astype(np.uint8), method = cv2.TM_CCOEFF_NORMED)[0][0])

        
if __name__ == '__main__':
    main()