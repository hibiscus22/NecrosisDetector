import cv2
import numpy as np


def imshow(img):
    cv2.namedWindow("out", cv2.WINDOW_NORMAL)
    cv2.imshow("out",(img).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extrapolate_kmeans (img):
    # For some reason (methinks jpg compression) the values are not just 3 but an interpolation. Back to three values:
    # hi_th = np.unique(img)[1:][np.diff(np.unique(img)) != 1]
    unique, counts = np.unique(img, return_counts = True)
    hi_th = unique[np.argpartition(counts, -3)[-3:]]
    hi_th.sort()
    img[img <= hi_th[0]] = 0
    #img[np.logical_and(img > hi_th[0], img < hi_th[1])] = hi_th[0] 
    img[np.logical_and(img > hi_th[0], img <= hi_th[1])] = 1
    #img[img >= hi_th[1]] = hi_th[1]
    img[img > hi_th[1]] = 2
    return img

def jaccard(img0, img1): 
    intersection = np.logical_and(img0, img1).sum()
    union = np.logical_or(img0, img1).sum()
    return intersection/union

def main():

    opt = cv2.IMREAD_GRAYSCALE

    path_tst = "/home/maiki/Documents/TUW/ht29/pi_mask_3/g5.jpg"
    path_pred = "/home/maiki/Documents/TUW/ht29/results_lr_classifier_3/g5.jpg"
    
    
    tst = cv2.resize(cv2.imread(path_tst, opt), (256,256))
    tst = extrapolate_kmeans(tst)
    tst1 = (tst == 1)
    tst2 = (tst == 2)


    # READ FLUOR IMAGE
    pred = cv2.resize(cv2.imread(path_pred, opt), (256,256))
    pred = extrapolate_kmeans(pred)
    pred1 = (pred == 1)
    pred2 = (pred == 2)
        
    imshow(np.concatenate((tst,pred), axis=1)*127)

    print(f"{jaccard(tst1, pred1)}\n{jaccard(tst2, pred2)}")

        
if __name__ == '__main__':
    main()