# Video frame color palette generator

A Python based program, which consists of a VLC player instance, that can generate color palettes for a specified number of video frames and ultimately combine them into a final color palette group image. (See the visual example below)

![Screenshot](https://andris.gauracs.com/api/files/projects/color_palette_generator/color_palette_generator_frame.jpg)

##### Example of the result image
![Screenshot](https://andris.gauracs.com/api/files/projects/color_palette_generator/color_palette_generator.jpg)

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
