import PySimpleGUI as sg
import cv2
import numpy as np


def main() -> None:

    image_viewer_brightfield = [
        [sg.Text("Brightfield")],
        [sg.Image(key="-IMAGE_BF-")],
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
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break

        if event == "-FILE_BF-":
            file_path = values["-FILE_BF-"]
            print(file_path)
            image = cv2.imread(file_path)
            image = cv2.resize(image, (256, 256))

            img_bytes = cv2.imencode(".png", image)[1].tobytes()
            window["-IMAGE_BF-"].update(data=img_bytes)


if __name__ == "__main__":
    main()
