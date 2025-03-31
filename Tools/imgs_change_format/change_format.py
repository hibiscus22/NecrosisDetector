from PIL import Image
import os

########### IMPORTANT! IMAGES HAVE TO BE JPG, OTHERWISE SCALEBAR RENDERS WEIRD ###############
#path = "/home/maiki/Documents/TUW/imgs_to_scalebar/"
path = "D:/TUW/Tools/imgs_change_format/" 

for img in os.listdir(path+"2format"):    
    
    #if img.endswith(".jpg"):
    # Import image
    img_base = Image.open(path+"2format/"+img)

    img = img.strip(".jpg")

    # Let's save
    img_base.save(f"{path}formatted/{img}.tiff")

