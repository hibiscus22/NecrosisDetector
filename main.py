import PySimpleGUI as sg
import cv2
import numpy as np


def main() -> None:

    image_viewer_brightfield = [
        [sg.Text("Brightfield")],
        [sg.Image(key="-IMAGE_BF-")],
        # Dummy Input field to create an event when browsing
        [sg.Input(key="-FILE_BF-", enable_events=True, visible=False)],
        [sg.FileBrowse(target="-FILE_BF-")],
    ]

    image_viewer_stain = [[sg.Text("Stain")], [sg.Image(key="-IMAGE_DYE-")]]

    layout = [
        [
            sg.Column(image_viewer_brightfield),
            sg.VSeparator(),
            sg.Column(image_viewer_stain),
        ],
        [sg.Radio("Binary Masks", "Radio", size=(10, 2), key="-MASK-")],
        [sg.Radio("Ternary Masks", "Radio", size=(10, 2), key="-MASK3-")],
        [sg.Radio("Continuous", "Radio", size=(10, 2), key="-CONTINUOUS-")],
    ]

    window = sg.Window("Necrosis Detector", layout)

    while True:
        event, values = window.read(timeout=20)
        if event == sg.WIN_CLOSED:
            break

        if event == "-FILE_BF-":
            file_path = values["-FILE_BF-"]
            # Read with OpenCV
            brightfield = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            brightfield = cv2.resize(brightfield, (256, 256))
            # Display
            brightfield_bytes = cv2.imencode(".png", brightfield)[1].tobytes()
            window["-IMAGE_BF-"].update(data=brightfield_bytes)

        if values["-MASK-"]:
            # Apply method to estimate the dye image
            _, dye = cv2.threshold(
                brightfield, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )

            dye_btyes = cv2.imencode(".png", dye)[1].tobytes()
            window["-IMAGE_DYE-"].update(data=dye_btyes)


if __name__ == "__main__":
    main()
