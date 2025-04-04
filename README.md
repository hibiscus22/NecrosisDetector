# Description
Hi! This repository contains a simple Python Application to Segment Necrosis in Cancer Spheroids, as part of the ELEVATE Project of the TU Wien (Austria).
The code is validated in Windows and Linux (Ubuntu).


# Installation: Clone this repository 
```PowerShell
git clone https://github.com/hibiscus22/NecrosisDetector.git
```

# Download Python 3.10.10

### Option 1 (recommended): Conda
Either [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-installation) or [anaconda](https://www.anaconda.com/download?utm_source=anacondadocs&utm_medium=documentation&utm_campaign=download&utm_content=installwindows)

Create an environment and activate it (`nec` is the name of the environment, it can be changed):

```PowerShell
conda create -n nec python=3.10.10
```

### Option 2: Python Package
Install [Python 3.10.10](https://www.python.org/downloads/) for your distribution.

# Installing Dependencies

PySimpleGUI 5.0.9 (you'll get a free, non-commercial License).
```PowerShell
python.exe -m pip install --upgrade --extra-index-url https://PySimpleGUI.net/install PySimpleGUI
```

Install the rest of requirements
```PowerShell
python.exe -m pip install -r requirements.txt
```

# Start the program
```PowerShell
python.exe main.py
```

# Additional Features

Validated with `pyinstaller`, you can also find folders with algorithms to Train ML and DL models.

# Additional Information
Contact: miguelmlozano@gmail.com

Project Responsible: sonia.lopez@tuwien.ac.at