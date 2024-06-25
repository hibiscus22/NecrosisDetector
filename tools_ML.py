import cv2
import numpy as np
import os
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


def save_images(path: str, dye: str, model: object, bf_arr: np.ndarray) -> None:
    folder = path + "/results_dt_classifier/"
    # folder = path + "/results_lr_classifier/"
    if not (os.path.exists(folder)):
        os.mkdir(folder)

    i = 0
    for img in os.listdir(f"{path}/{dye}"):
        print(img + "...")
        y_pred = model.predict_proba(bf_arr[i])[::, 1].reshape(256, 256)
        # cv2.imwrite(path+'/results_lr_classifier/'+img, y_pred)
        cv2.imwrite(folder + img, 255 * (y_pred > 0.5))
        i += 1
