import cv2
import numpy as np


def otsu_equalize(bf):
    # take the negative of the image
    bf_neg = 255 - bf

    bf_neg = cv2.GaussianBlur(bf_neg, (5, 5), 0)

    (T, threshInv) = cv2.threshold(bf_neg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    threshInv = cv2.bitwise_and(bf_neg, bf_neg, mask=threshInv)

    # increase the contrast
    alpha = 0.7
    threshInv = np.clip(alpha * threshInv, 0, 255).astype(np.uint8)
    threshInv = cv2.equalizeHist(
        threshInv
    )  # now equalize the histogram that has got narrowed by a factor alpha

    (T, bf_final) = cv2.threshold(threshInv, 100, 255, cv2.THRESH_BINARY)
    # (T, bf_final) = cv2.threshold(bf_neg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return bf_final


def watershed(bf):

    bf_neg = 255 - bf
    (T, threshInv) = cv2.threshold(bf_neg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # threshInv = cv2.bitwise_and(img[i], img[i], mask=threshInv)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(threshInv, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = cv2.dilate(
        opening, kernel, iterations=2
    )  # probably results improve when we play with this

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(
        opening, cv2.DIST_L2, 3
    )  # probably results improve when we play with this
    ret, sure_fg = cv2.threshold(
        dist_transform, 0.5 * dist_transform.max(), 255, 0
    )  # probably results improve when we play with this
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(opening, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    bf_neg = cv2.cvtColor(bf_neg, cv2.COLOR_GRAY2RGB)
    markers = cv2.watershed(bf_neg, markers)

    res = 255 * (markers == 2)

    # res = np.logical_and(bf_neg, res)

    return res


def kmeans(bf, K=2):
    # take the negative of the image
    # bf_neg = 255 - bf
    bf_neg = bf

    z = np.float32(bf_neg.reshape((-1, 1)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
    compactness, labels, center = cv2.kmeans(
        z, K, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
    )
    center = np.uint8(center)
    res = center[labels.flatten()]
    bf_final = res.reshape((bf_neg.shape))

    # bf_final = cv2.threshold(bf_final, 150, 255, cv2.THRESH_BINARY)[1]

    return bf_final


def water_means(img):
    w = watershed(img).astype(np.uint8)
    wi = cv2.bitwise_and(255 - img, 255 - img, mask=w)
    ki = kmeans(wi, K=3)
    return 255 * (ki == np.max(ki))


dict_methods = {
    "Otsu": otsu_equalize,
    "Watershed": watershed,
    "Watershed + KMeans": water_means,
    "Logistic Regression": None,
    "Decision Tree": None,
    "UNet": None,
}