import os
import shutil

path = "C:/Users/Miguel/Downloads/ex1"

dye = "Calcein"

for cell_number in ["2000", "5000", "10000"]:
    for day_number in os.listdir(f"{path}/{cell_number}"):
        for picture in os.listdir(f"{path}/{cell_number}/{day_number}/{dye}"):
            if "jpg" in picture:
                print(f"Copying {picture}...")
                shutil.copy(f"{path}/{cell_number}/{day_number}/{dye}/{picture}",
                            f"{path}/Calcein/{cell_number}_{day_number}_{picture}")