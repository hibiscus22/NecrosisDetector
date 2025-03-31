import cv2
import numpy as np

def imshow(img):
    cv2.namedWindow("out", cv2.WINDOW_NORMAL)
    cv2.imshow("out",(img).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def jaccard(img0, img1): 
    intersection = np.logical_and(img0, img1).sum()
    union = np.logical_or(img0, img1).sum()
    return intersection/union

def main():

    opt = cv2.IMREAD_GRAYSCALE

    path_tst = "F:/TUW/images/ht29/pi_mask/f4.jpg"
    path_pred = "F:/TUW/images/ht29/results_dt_classifier/f4.jpg"

    tst = cv2.resize(cv2.imread(path_tst, opt), (256,256))

    # READ FLUOR IMAGE
    pred = cv2.resize(cv2.imread(path_pred, opt), (256,256))
      
    imshow(np.concatenate((tst,pred), axis=1))

    print(jaccard(tst, pred))

        
if __name__ == '__main__':
    main()