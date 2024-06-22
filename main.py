import PySimpleGUI as sg
import cv2
import numpy as np


def main() -> None:

    image_viewer_brightfield = [
        [sg.Text("Brightfield")],
        [sg.Image(key="-IMAGE_BF-")],
        [sg.FileBrowse()],
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
        event, _ = window.read()
        if event == sg.WIN_CLOSED:
            break


if __name__ == "__main__":
    main()
