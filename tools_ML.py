import cv2
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pickle


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
        print(img + "...")
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
