import PySimpleGUI as sg
import cv2
import numpy as np
from functions_masks import *


def window_setup(names) -> list:
    sg.theme("TanBlue")

    image_viewer_brightfield = [
        [sg.Text("Brightfield")],
        [sg.Image(key="-IMAGE_BF-")],
        # Dummy Input field to create an event when browsing
        [sg.Input(key="-FILE_BF-", enable_events=True, visible=False)],
        [sg.FileBrowse(target="-FILE_BF-")],
    ]

    image_viewer_stain = [
        [sg.Text("Stain")],
        [sg.Image(key="-IMAGE_DYE-")],
        [
            sg.Radio("PI", "Dye", size=(10, 2), key="-PI-", default=True),
            sg.Radio("DAPI", "Dye", size=(10, 2), key="-DAPI-"),
        ],
        # Another dummy but for the save
        [sg.Input(key="-FILE_DYE-", enable_events=True, visible=False)],
        [sg.FileSaveAs(target="-FILE_DYE-", file_types=(("PNG", "*.png"),))],
    ]

    mode_selector = [
        [
            sg.Radio("CRC", "Cell line", size=(10, 2), key="-CRC-"),
            sg.Radio("Pancreas", "Cell line", size=(10, 2), key="-PANCREAS-"),
            sg.Radio("Generic", "Cell line", size=(10, 2), key="-BOTH-", default=True),
        ],
        [
            sg.Radio("Binary Masks", "Solution", size=(10, 2), key="-MASK-"),
            sg.Radio("Ternary Masks", "Solution", size=(10, 2), key="-MASK3-"),
            sg.Radio(
                "Continuous", "Solution", size=(10, 2), key="-CONTINUOUS-", default=True
            ),
        ],
        [sg.Combo(names, enable_events=True, key="-METHOD-")],
    ]

    layout = [
        [
            sg.Column(image_viewer_brightfield),
            # sg.VSeparator(),
            sg.Button("Convert"),
            sg.Column(image_viewer_stain),
        ],
        mode_selector,
    ]

    return layout


def main() -> None:
    names_combo = ["Select an option..."]

    layout = window_setup(names_combo)
    window = sg.Window("Necrosis Detector", layout)

    while True:

        event, values = window.read(timeout=20)

        if values["-PI-"]:
            dye = "pi"
        else:
            dye = "dapi"

        if event == sg.WIN_CLOSED:
            break

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
                dye_image = method(brightfield, "ht29", dye)

                dye_btyes = cv2.imencode(".png", dye_image)[1].tobytes()  # png format
                window["-IMAGE_DYE-"].update(data=dye_btyes)

            elif values["-MASK3-"]:
                pass

            elif values["-CONTINUOUS-"]:
                pass

            else:
                pass


if __name__ == "__main__":
    main()
