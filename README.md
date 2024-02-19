# Video frame color palette generator

A Python based program, which consists of a VLC player instance, that can generate color palettes for a specified number of video frames and ultimately combine them into a final color palette group image. (See the visual example below)

![Screenshot](https://andris.gauracs.com/images/3f5a7922-675e-4156-9c6b-7dc129af684e.jpg)

##### Example of the result image
![Screenshot](https://andris.gauracs.com/images/88bd79a3-0b4d-428d-9c0f-114d84b37f76.jpg)

### Prerequisites

In order to run this program, several Python packages must be installed first using pip and Homebrew:

```
pip install python-vlc
brew install pyqt@4
pip install matplotlib
pip install numpy
pip install scikit-learn
pip install opencv-python
```

**Note:** This project is using the 4th version of PyQt - PyQt4, because this project is built on top of the existing [PyQt VLC player instance example](https://github.com/oaubert/python-vlc/blob/master/examples/qtvlc.py), which uses PyQt4 specifically.

### Run the program

```
python main.py
```

## License

MIT
