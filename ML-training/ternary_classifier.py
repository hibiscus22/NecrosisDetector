import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import sys

sys.path.append("/home/maiki/Documents/TUW/Tools/")
from tools_ML import *


def imshow(img):
    cv2.namedWindow("out", cv2.WINDOW_NORMAL)
    cv2.imshow("out", (img).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extrapolate_kmeans(img):
    # For some reason (methinks jpg compression) the values are not just 3 but an interpolation. Back to three values:
    # hi_th = np.unique(img)[1:][np.diff(np.unique(img)) != 1]
    unique, counts = np.unique(img, return_counts=True)
    hi_th = unique[np.argpartition(counts, -3)[-3:]]
    hi_th.sort()
    img[img <= hi_th[0]] = 0
    # img[np.logical_and(img > hi_th[0], img < hi_th[1])] = hi_th[0]
    img[np.logical_and(img > hi_th[0], img <= hi_th[1])] = 1
    # img[img >= hi_th[1]] = hi_th[1]
    img[img > hi_th[1]] = 2
    return img


def quality_metrics(y_true, y_pred, THRESHOLD=0.5):
    ##### METRICS #####
    # Area under ROC curve
    try:
        AUC = metrics.roc_auc_score(y_true, y_pred)
    except Exception as e:
        print("All values predicited one class!")
        AUC = 0

    # Discrete metrics
    y_pred = y_pred >= THRESHOLD

    TP = np.logical_and(y_pred, y_true).sum()
    TN = np.logical_and(np.logical_not(y_pred), np.logical_not(y_true)).sum()
    FP = np.logical_and(y_pred, np.logical_not(y_true)).sum()
    FN = np.logical_and(np.logical_not(y_pred), y_true).sum()

    # According to definition:
    accuracy = (TP + TN) / len(y_pred)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    # accuracy = (sensitivity+specificity)/2

    return AUC, accuracy, sensitivity, specificity


def jaccard(img0, img1):
    return np.logical_and(img0, img1).sum() / np.logical_or(img0, img1).sum()


def main():
    bf_arr = []
    pi_m1_arr = []
    pi_m2_arr = []

    opt = cv2.IMREAD_GRAYSCALE

    path = "/home/maiki/Documents/TUW/Images/"

    print("Select the group of cells to be used: 1. ht29 2. pancreas 3. both")
    group = input()
    if group == "1":
        group = "ht29"
    elif group == "2":
        group = "pancreas"
    elif group == "3":
        group = "both"

    print("Select the dye to be used: 1. pi 2. dapi")
    dye = input()
    if dye == "1":
        dye = "pi"
    elif dye == "2":
        dye = "dapi"

    path = path + group

    for img in os.listdir(f"{path}/imgs"):
        print(img + "...")
        bf = cv2.resize(cv2.imread(f"{path}/imgs/{img}", opt), (256, 256))
        bf = np.float32((255 - bf.reshape((-1, 1))))
        bf = bf / max(bf)
        bf_arr.append(bf)

    for img in os.listdir(f"{path}/pi_mask_3"):
        print(img + "...")
        pi_m = cv2.resize(cv2.imread(f"{path}/{dye}_mask_3/{img}", opt), (256, 256))
        pi_m = extrapolate_kmeans(pi_m)
        pi_m1 = (pi_m == 1).reshape((-1, 1))
        pi_m2 = (pi_m == 2).reshape((-1, 1))

        pi_m1_arr.append(pi_m1)
        pi_m2_arr.append(pi_m2)

    for i in range(len(pi_m1_arr)):
        pi_m1_arr[i] = pi_m1_arr[i][:, 0]
        pi_m2_arr[i] = pi_m2_arr[i][:, 0]

    x1_tr, x1_tst, y1_tr, y1_tst = train_test_split(
        bf_arr, pi_m1_arr, test_size=0.3, random_state=23
    )
    x2_tr, x2_tst, y2_tr, y2_tst = train_test_split(
        bf_arr, pi_m2_arr, test_size=0.3, random_state=23
    )

    x1_tr = np.concatenate((x1_tr[:]))
    y1_tr = np.concatenate((y1_tr[:]))
    x1_tst = np.concatenate((x1_tst[:]))
    y1_tst = np.concatenate((y1_tst[:]))
    x2_tr = np.concatenate((x2_tr[:]))
    y2_tr = np.concatenate((y2_tr[:]))
    x2_tst = np.concatenate((x2_tst[:]))
    y2_tst = np.concatenate((y2_tst[:]))

    for n in [x1_tr, x1_tst, y1_tr, y1_tst, x2_tr, x2_tst, y2_tr, y2_tst]:
        print(n.shape)

    print("Select the method to be used: 1. Logistic Regression 2. Decision Tree")
    method = input()
    if method == "1":
        model_background = LogisticRegression(max_iter=1000)
        model_foreground = LogisticRegression(max_iter=1000)
        model_name = "lr"
    elif method == "2":
        model_background = DecisionTreeClassifier()
        model_foreground = DecisionTreeClassifier()
        model_name = "dt"

    model_background.fit(x1_tr, y1_tr)
    y1_pred = model_background.predict_proba(x1_tst)[::, 1] > 0.5

    qm = quality_metrics(y1_tst, y1_pred)
    print(f"Background: AUC:{qm[0]} ACC:{qm[1]} SEN:{qm[2]} SEL:{qm[3]}")

    model_foreground.fit(x2_tr, y2_tr)
    y2_pred = model_foreground.predict_proba(x2_tst)[::, 1] > 0.3

    qm = quality_metrics(y2_tst, y2_pred)
    print(f"Foreground: AUC:{qm[0]} ACC:{qm[1]} SEN:{qm[2]} SEL:{qm[3]}")

    sim = []
    for i in range(int(len(y1_tst) / (256 * 256))):
        lo = i * 256 * 256
        hi = (i + 1) * 256 * 256
        pred = (y1_pred[lo:hi]).reshape(256, 256) * 255
        tst = (y1_tst[lo:hi]).reshape(256, 256) * 255
        sim.append(jaccard(tst, pred))
        imshow(np.concatenate((tst, pred), axis=1))
        print(sim[len(sim) - 1])

    print("average1: " + str(np.average(sim)))

    sim = []
    for i in range(int(len(y2_tst) / (256 * 256))):
        lo = i * 256 * 256
        hi = (i + 1) * 256 * 256
        pred = (y2_pred[lo:hi]).reshape(256, 256) * 255
        tst = (y2_tst[lo:hi]).reshape(256, 256) * 255
        sim.append(jaccard(tst, pred))
        imshow(np.concatenate((tst, pred), axis=1))
        print(sim[len(sim) - 1])

    print("average2: " + str(np.average(sim)))

    path_save = "/home/maiki/Documents/TUW/NecrosisDetector/ModelsML"
    save_model(
        model_background,
        f"{path_save}/ternary_{model_name}_classifier_{group}_{dye}_background.pkl",
    )
    save_model(
        model_foreground,
        f"{path_save}/ternary_{model_name}_classifier_{group}_{dye}_foreground.pkl",
    )

    save_images_ternary(
        f"{path}/results_{model_name}_classifier_3/",
        dye,
        model_background,
        model_foreground,
        bf_arr,
    )


if __name__ == "__main__":
    main()
