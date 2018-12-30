import cv2
import numpy as np
from sklearn.cluster import KMeans
from vlc_player import Player
import sys
import os
from PyQt4 import QtGui, QtCore

# Global variables
framesTaken = 0
framesTakenList = []
framesSpecified = 0
clusters = 5
margin = 5
borderSize = 60
offset = 2

# We define our custom VLC Player class


class Custom_VLC_Player(Player):
    def __init__(self):
        # We inherit all the attributes of the original class
        super(Custom_VLC_Player, self).__init__()

        # We want the width and height to be fixed, so the layout dimensions
        # don't change in a weird way
        self.videoframe.setFixedWidth(640)
        self.videoframe.setFixedHeight(360)

        # We create a new layout, where we will place our custom buttons
        self.snapbox = QtGui.QHBoxLayout()

        # We create a button, that will execute our snapshot taking function
        self.snapbutton = QtGui.QPushButton("Take Snapshot")

        # We will set it disabled for the start of the program
        self.snapbutton.setEnabled(0)

        # Let's add the button to our layout
        self.snapbox.addWidget(self.snapbutton)

        # We will connect a snapshot taking function to the button later.
        # Let's leave this command commented out for now
        self.connect(self.snapbutton, QtCore.SIGNAL("clicked()"),
                     self.take_snapshot)

        # We place a label for specifing the frame count to use
        self.l1 = QtGui.QLabel("Number of frames:")

        # We add it to the layout
        self.snapbox.addWidget(self.l1)

        # We create a spin box, where we can choose
        # how many frames we want to take
        self.sp = QtGui.QSpinBox()

        # We will limit the number to 10 frames in this program
        self.sp.setMaximum(10)
        self.sp.setMinimum(0)

        # We add it to the layout
        self.snapbox.addWidget(self.sp)
        # Connect a value change function to the spinbox.
        self.sp.valueChanged.connect(self.valuechange)

        # We add an empty space that streches to the right side
        # that way the next added element will be aligned to the right
        self.snapbox.addStretch(1)

        # This label will hold information,
        # how many frames have been taken so far
        self.l2 = QtGui.QLabel("Frames taken: "+str(framesTaken))

        # This is needed for the layout not to break
        self.l2.setFixedHeight(24)
        self.snapbox.addWidget(self.l2)

        # While the process hasn't started yet, we can hide the label
        self.l2.setVisible(0)

        # We add it to the layout
        self.vboxlayout.addLayout(self.snapbox)

        # This is a layout which will consist of 10 labels, each label
        # will hold a thumbnail of the frame taken
        self.imageareaWidget = QtGui.QWidget(self)
        self.imageareaWidget.setFixedHeight(80)

        # This is a wrapper around our image label widget
        self.imagearea = QtGui.QHBoxLayout(self.imageareaWidget)

        # Let's create an array of 10 label objects and add them
        # to the widget we just created
        self.imageBoxes = []
        for i in range(0, 10):
            self.imageBoxes.append(QtGui.QLabel(str(i)))
            self.imageBoxes[len(self.imageBoxes)-1]
            self.imagearea.addWidget(self.imageBoxes[len(self.imageBoxes)-1])

        # We will add this area to our layout, but initially we will set it
        # invisible, while the snapshot capture process hasn't started yet
        self.vboxlayout.addWidget(self.imageareaWidget)
        self.imageareaWidget.setVisible(0)

    def valuechange(self):
        # We access our global variables within the function
        global framesTaken
        global framesSpecified

        # We set the framesSpecified value to
        # whatever is specified on the spinbox
        framesSpecified = self.sp.value()

        # We modify our label to give us info, how many
        # frames have been captured so far
        self.l2.setText(
            "Frames taken: "+str(framesTaken)+" from "+str(framesSpecified))

        # Once the snapshot taking process has begun,
        # we need to disable the spinbox
        self.sp.setEnabled(0) if (framesTaken > 0) else self.sp.setEnabled(1)

        # We enable our snapshot trigger button, if frames have been specified
        # and the process is ongoing.
        # If the process ends, we disable the button again
        if (self.sp.value() > 0
            and framesTaken < 10
                and framesTaken < framesSpecified):
            self.snapbutton.setEnabled(1)
        else:
            self.snapbutton.setEnabled(0)

        if (self.sp.value() > 0):
            self.l2.setVisible(1)
        else:
            self.l2.setVisible(0)

    def take_snapshot(self):
        # Import the global variables we'll be using
        global framesTaken, clusters, borderSize, offset
        # This will be needed to check if the player was playing at the
        # time of the button press
        wasPlaying = None
        # We need to get the width and height of the video file
        videoSize = self.mediaplayer.video_get_size()
        # This is the VLC function, that let's us
        # take a snap shot of the video frame and save it in the directory
        self.mediaplayer.video_take_snapshot(
            0, "./img_"+str(framesTaken)+".png",
            videoSize[0],
            videoSize[1])

        # While we do the image processing, let's pause the video
        if self.mediaplayer.is_playing():
            self.PlayPause()
            wasPlaying = True

        # Let's fetch the video frame we just captured
        imagePath = os.getcwd() + "/img_"+str(framesTaken)+".png"
        # Transform the image to an OpenCV readable image
        image = cv2.imread(imagePath)

        # Let's make a copy of this image
        # to use for the color palette generation
        image_copy = image_resize(cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB), width=100)

        # Since the K-means algorithm we're about to do,
        # is very labour intensive, we will do it on a smaller image copy
        # This will not affect the quality of the algorithm
        pixelImage = image_copy.reshape(
            (image_copy.shape[0] * image_copy.shape[1], 3))

        # We use the sklearn K-Means algorithm to find the color histogram
        # from our small size image copy
        clt = KMeans(n_clusters=clusters+offset)
        clt.fit(pixelImage)

        # build a histogram of clusters and then create a figure
        # representing the number of pixels labeled to each color
        hist = centroid_histogram(clt)

        # Let's plot the retrieved colors. See the plot_colors function
        # for more details
        bar = plot_colors(hist, clt.cluster_centers_)

        # Resize the color bar to be even width with the video frame
        barImage = image_resize(
            cv2.cvtColor(bar, cv2.COLOR_RGB2BGR),
            width=int(videoSize[0]))

        # This is just a whitespace to put between the image and the color bar
        im = np.zeros((borderSize/2, int(videoSize[0]), 3), np.uint8)
        cv2.rectangle(im, (0, 0), (int(videoSize[0]), borderSize/2),
                      (255, 255, 255), -1)

        # Now we combine the video frame and the color bar into one image
        newImg = np.concatenate([image, im, barImage], axis=0)
        cv2.imwrite(imagePath, newImg)

        # To show a thumbnail version of this image, we need to store it
        # in a pixmap and then add it to a label widget
        pixmap = QtGui.QPixmap()
        pixmap.load(imagePath)
        pixmap = pixmap.scaledToWidth(50)
        self.imageBoxes[framesTaken].setPixmap(pixmap)
        framesTaken = framesTaken + 1

        # Now that we have at least one image captured and processed,
        # we can go ahead and show the bottom layout with our thumbnail labels
        self.imageareaWidget.setVisible(1)
        self.valuechange()

        # Let's now add the full size image to a list of all captured images
        framesTakenList.append(imagePath)

        # Once the list is full with all the images we need, we can start to
        # combine it into the final output image
        if (framesTaken == framesSpecified):
            resultImgs = []
            for i in framesTakenList:
                # here we just add a whitespace rectangle as a top margin
                resultImgs.append(cv2.imread(i))
                im = np.zeros((
                    borderSize*2,
                    cv2.imread(i).shape[1],
                    3), np.uint8)
                cv2.rectangle(
                    im,
                    (0, 0),
                    (cv2.imread(i).shape[1], borderSize * 2),
                    (255, 255, 255), -1)
                # here we just add a whitespace rectangle as a bottom margin
                resultImgs.append(im)

            # Now that we have the list of all the needed images,
            # we concatinate them vertically and form the final image
            final = np.concatenate(resultImgs, axis=0)

            # The final image still needs an outside border, so the last task
            # is to create this border line
            finalShape = final.shape
            w = finalShape[1]
            h = finalShape[0]
            base_size = h+borderSize, w+borderSize, 3
            base = np.zeros(base_size, dtype=np.uint8)
            # We combine our main image with the border line
            cv2.rectangle(
                base,
                (0, 0),
                (w + borderSize, h + borderSize),
                (255, 255, 255), borderSize)
            base[
                (borderSize/2):h+(borderSize/2),
                (borderSize/2):w+(borderSize/2)
                ] = final

            # The final output image is ready.
            # We now export it to the root directory
            cv2.imwrite("result_image.png", base)
            sys.exit(app.exec_())

        # If the image capture process continues, we resume playing the video
        self.PlayPause() if wasPlaying else None


# Courtesy of https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist


# Courtesy of https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # Sort the centroids to form a gradient color look
    centroids = sorted(centroids, key=lambda x: sum(x))

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids[offset:]):
        # plot the relative percentage of each cluster
        # endX = startX + (percent * 300)

        # Instead of plotting the relative percentage,
        # we will make a n=clusters number of color rectangles
        # we will also seperate them by a margin
        new_length = 300 - margin * (clusters - 1)
        endX = startX + new_length/clusters
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        cv2.rectangle(bar, (int(endX), 0), (int(endX + margin), 50),
                      (255, 255, 255), -1)
        startX = endX + margin

    # return the bar chart
    return bar


# A helper function to resize images
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


# Initialize the QtGUI Application instance
app = QtGui.QApplication(sys.argv)

# Initialize our custom VLC Player instance
vlc = Custom_VLC_Player()
vlc.show()
# Let's change the size of the window. We will make it 660px by 530px
vlc.resize(660, 530)

if sys.argv[1:]:
    vlc.OpenFile(sys.argv[1])
sys.exit(app.exec_())
