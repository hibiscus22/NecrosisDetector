import os
import cv2


height = 256  # pixels

opt = cv2.IMREAD_GRAYSCALE

path = "/home/maiki/Documents/TUW/imgs_to_reduce"

if not (os.path.exists(f"{path}/reduced/")):
    os.mkdir(f"{path}/reduced/")

for image_name in os.listdir(f"{path}/2reduce/"):
    print(image_name + "...")
    image = cv2.imread(f"{path}/2reduce/{image_name}")
    width = int(image.shape[1] / image.shape[0] * height)
    # print(width)
    image = cv2.resize(image, (width, height))
    cv2.imwrite(f"{path}/reduced/{image_name}", image)
