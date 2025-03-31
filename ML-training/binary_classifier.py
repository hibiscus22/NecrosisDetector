import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import sys

#sys.path.append("/home/maiki/Documents/TUW/Tools/")
sys.path.append("D:/TUW/Tools")
from tools_ML import *


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
    bf_array = []
    dye_array = []

    opt = cv2.IMREAD_GRAYSCALE

    #path = "/home/maiki/Documents/TUW/Images/"
    path = "D:/TUW/Images/"

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

    for img in os.listdir(f"{path}/brightfield"):
        print(img + "...")
        # APPEND MASK
        bf = cv2.resize(cv2.imread(f"{path}/brightfield/{img}", opt), (256, 256))
        bf = np.float32((255 - bf.reshape((-1, 1))))
        bf = bf / max(bf)
        bf_array.append(bf)
        # APPEND MASK
        dye_image = cv2.resize(cv2.imread(f"{path}/{dye}_mask/{img}", opt), (256, 256))
        dye_image = np.float32((dye_image.reshape((-1, 1))))
        dye_array.append(np.round(dye_image / 255))

    for i in range(len(dye_array)):
        dye_array[i] = dye_array[i][:, 0]

    x_tr, x_tst, y_tr, y_tst = train_test_split(
        bf_array, dye_array, test_size=0.3, random_state=23
    )

    x_tr = np.concatenate((x_tr[:]))
    y_tr = np.concatenate((y_tr[:]))
    x_tst = np.concatenate((x_tst[:]))
    y_tst = np.concatenate((y_tst[:]))

    print(x_tr.shape)
    print(y_tr.shape)
    print(x_tst.shape)
    print(y_tst.shape)

    print("Select the method to be used: 1. Logistic Regression 2. Decision Tree")
    method = input()
    if method == "1":
        model = LogisticRegression(max_iter=1000)
        model_name = "lr"
    elif method == "2":
        model = DecisionTreeClassifier()
        model_name = "dt"

    model.fit(x_tr, y_tr)
    y_pred = model.predict_proba(x_tst)[::, 1]

    qm = quality_metrics(y_tst, y_pred)
    print(f"AUC:{qm[0]}ACC:{qm[1]}SEN:{qm[2]}SEL:{qm[3]}")

    sim = []
    for i in range(int(len(y_tst) / (256 * 256))):
        lo = i * 256 * 256
        hi = (i + 1) * 256 * 256
        pred = (y_pred[lo:hi] > 0.5).reshape(256, 256) * 255
        tst = (y_tst[lo:hi]).reshape(256, 256) * 255
        sim.append(jaccard(tst, pred))
        imshow(np.concatenate((tst, pred), axis=1))
        print(sim[len(sim) - 1])

    print(f"average:{np.average(sim)}")

    # path_save = "/home/maiki/Documents/TUW/NecrosisDetector/ModelsML"

    # save_model(model, f"{path_save}/binary_{model_name}_classifier_{group}_{dye}.pkl")

    # save_images(f"{path}/results_{model_name}_classifier/", dye, model, bf_array)


if __name__ == "__main__":
    main()
