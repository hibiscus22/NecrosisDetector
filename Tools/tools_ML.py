import cv2
import numpy as np
import os
import pickle
from sklearn import metrics

def save_model(model: object, name: str) -> None:
    with open(name, "wb") as f:
        pickle.dump(model, f)


def load_model(name: str) -> object:
    with open(name, "rb") as f:
        model = pickle.load(f)
    return model


def imshow(img: np.ndarray) -> None:
    cv2.namedWindow("out", cv2.WINDOW_NORMAL)
    cv2.imshow("out", (img).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_images(folder: str, dye: str, model: object, bf_arr: np.ndarray) -> None:
    if not (os.path.exists(folder)):
        os.mkdir(folder)

    i = 0
    for img in os.listdir(f"{folder}/../{dye}"):
        img = img[:-4] + ".png"
        #print(img + "...")
        y_pred = model.predict_proba(bf_arr[i])[::, 1].reshape(256, 256)
        # cv2.imwrite(path+'/results_lr_classifier/'+img, y_pred)
        cv2.imwrite(folder + img, 255 * (y_pred > 0.5))
        i += 1


def save_images_ternary(
    folder: str,
    dye: str,
    model_background: object,
    model_foreground: object,
    bf_arr: np.ndarray,
) -> None:
    if not (os.path.exists(folder)):
        os.mkdir(folder)

    i = 0
    for img in os.listdir(f"{folder}/../{dye}"):
        print(img + "...")
        y1_pred = model_background.predict_proba(bf_arr[i])[::, 1].reshape(256, 256)
        y2_pred = model_foreground.predict_proba(bf_arr[i])[::, 1].reshape(256, 256)
        y_pred = np.zeros((256, 256))
        y_pred[y1_pred > 0.5] = 1
        y_pred[y2_pred > 0.3] = 2

        y_pred = y_pred * 127

        cv2.imwrite(folder + img, y_pred)
        i += 1


def save_images_continuous(
    folder: str, dye: str, model: object, bf_arr: np.ndarray
) -> None:
    if not (os.path.exists(folder)):
        os.mkdir(folder)

    i = 0
    for img in os.listdir(f"{folder}/../{dye}"):
        img = img[:-4] + ".png"

        y_pred = model.predict(bf_arr[i]).reshape(256, 256)
        cv2.imwrite(folder + img, y_pred * 255)
        i += 1
        printProgressBar(i, len(os.listdir(f"{folder}/../{dye}")), prefix = f"Saving {dye}:", suffix = '', length = 50)

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()




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