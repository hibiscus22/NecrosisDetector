import PySimpleGUI as sg


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
