from PIL import Image
import os

########### IMPORTANT! IMAGES HAVE TO BE JPG, OTHERWISE SCALEBAR RENDERS WEIRD ###############
#path = "/home/maiki/Documents/TUW/imgs_to_scalebar/"
path = "D:/TUW/Tools/imgs_to_scalebar/" 

for img in os.listdir(path+"2scalebar"):    
    
    # Import image
    img_base = Image.open(path+"2scalebar/"+img)
    #img_base = img_base.resize((762, 256))

    # Import Scalebar
    scalebar = Image.open(f"{path}/scalebar_white_x10.png")
    #scalebar = Image.open(f"{path}/scalebar_white_256_x10.png")
    #scalebar = Image.open(r"/home/maiki/Documents/TUW/scalebar_white_x40.png")

    # Put scalebar on top: MASK is important because scalebar has transparent background

    img_base.paste(scalebar, (1600, 1900), mask=scalebar ) # 2048
    #img_base.paste(scalebar, (200, 230), mask=scalebar ) # 256
    #img_base.paste(scalebar, (450, 230), mask=scalebar ) # 256
    #img_base.paste(scalebar, (700, 230), mask=scalebar ) # 256


    # Let's see
    img_base.show()

    # Let's save
    img_base.save(path+"scalebard/"+img)
