import PySimpleGUI as sg
import cv2
from functions_masks import *
from gui_setup import window_setup


def main() -> None:
    names_combo = ["Select an option..."]

    layout = window_setup(names_combo)
    window = sg.Window("Necrosis Detector", layout)

    while True:
        event, values = window.read(timeout=20)

        if event == sg.WIN_CLOSED:
            break

        if values is not None:
            if values["-PI-"]:
                dye = "pi"
            elif values["-DAPI-"]:
                dye = "dapi"

            if values["-CRC-"]:
                group = "ht29"
            elif values["-PANCREAS-"]:
                group = "pancreas"
            elif values["-BOTH-"]:
                group = "both"

            if values["-MASK-"]:
                names_combo = [
                    "Otsu",
                    "Watershed",
                    "Watershed + KMeans",
                    "Logistic Regression",
                    "Decision Tree",
                    "UNet",
                ]

            if values["-MASK3-"]:
                names_combo = ["Logistic Regression", "Decision Tree", "UNet"]

            if values["-CONTINUOUS-"]:
                names_combo = ["Decision Tree", "UNet"]

            window["-METHOD-"].update(values=names_combo, value=values["-METHOD-"])

            if event == "-FILE_BF-":  # Open
                file_path = values["-FILE_BF-"]
                # Read with OpenCV
                brightfield = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                brightfield = cv2.resize(brightfield, (256, 256))
                # Display
                brightfield_bytes = cv2.imencode(".png", brightfield)[1].tobytes()
                window["-IMAGE_BF-"].update(data=brightfield_bytes)

            if event == "-FILE_DYE-":  # Save
                file_path = values["-FILE_DYE-"]
                cv2.imwrite(file_path, dye_image)

            if event == "Convert":
                if values["-MASK-"]:
                    # Get and apply the function
                    method = dict_methods[values["-METHOD-"]]

                elif values["-MASK3-"]:
                    method = dict_methods_3[values["-METHOD-"]]

                elif values["-CONTINUOUS-"]:
                    method = dict_continuous[values["-METHOD-"]]

                else:
                    break

                dye_image = method(brightfield, group, dye)
                dye_btyes = cv2.imencode(".png", dye_image)[1].tobytes()  # png format
                window["-IMAGE_DYE-"].update(data=dye_btyes)


if __name__ == "__main__":
    main()
