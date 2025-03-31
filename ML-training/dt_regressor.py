import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import os
from sklearn.metrics import r2_score
import sys

sys.path.append("/home/maiki/Documents/TUW/Tools/")
from tools_ML import *


def imshow(img):
    cv2.namedWindow("out", cv2.WINDOW_NORMAL)
    cv2.imshow("out", (img).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    bf_arr = []
    dye_array = []

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
        # READ BF IMAGE
        bf = cv2.resize(cv2.imread(f"{path}/imgs/{img}", opt), (256, 256))
        bf = np.float32((255 - bf.reshape((-1, 1))))
        bf = bf / max(bf)
        bf_arr.append(bf)

        # READ FLUOR IMAGE
        dye_image = cv2.resize(cv2.imread(f"{path}/{dye}/{img}", opt), (256, 256))
        dye_image = np.float32((dye_image.reshape((-1, 1))))
        dye_image = dye_image / max(dye_image)
        dye_array.append(dye_image)

    for i in range(len(dye_array)):
        dye_array[i] = dye_array[i][:, 0]

    x_tr, x_tst, y_tr, y_tst = train_test_split(
        bf_arr, dye_array, test_size=0.3, random_state=23
    )

    x_tr = np.concatenate((x_tr[:]))
    y_tr = np.concatenate((y_tr[:]))
    x_tst = np.concatenate((x_tst[:]))
    y_tst = np.concatenate((y_tst[:]))

    print(x_tr.shape)
    print(y_tr.shape)
    print(x_tst.shape)
    print(y_tst.shape)

    model = DecisionTreeRegressor()

    model.fit(x_tr, y_tr)

    y_pred = model.predict(x_tst)

    sim = []
    for i in range(int(len(y_tst) / (256 * 256))):
        lo = i * 256 * 256
        hi = (i + 1) * 256 * 256
        pred = (y_pred[lo:hi]).reshape(256, 256) * 255
        tst = (y_tst[lo:hi]).reshape(256, 256) * 255
        try:
            sim.append(
                cv2.matchTemplate(
                    pred.astype(np.uint8),
                    tst.astype(np.uint8),
                    method=cv2.TM_CCOEFF_NORMED,
                )[0][0]
            )
        except:
            sim.append(0)
        # imshow(np.concatenate((tst, pred),axis = 1))
        # print(sim[len(sim)-1])

    print(f"average:{np.average(sim)}")
    print(f"r2: {r2_score(y_tst, y_pred)}")

    print(f"average:{np.average(sim)}")

    path_save = "/home/maiki/Documents/TUW/NecrosisDetector/ModelsML"

    save_model(model, f"{path_save}/dt_regressor_{group}_{dye}.pkl")

    save_images_continuous(f"{path}/results_dt_regressor/", dye, model, bf_arr)


if __name__ == "__main__":
    main()
