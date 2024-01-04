import os, sys
import numpy as np

def imadjust(img, lb=0, ub=.999):
    p,q = np.histogram(img.flatten(), bins=np.linspace(np.quantile(img, lb), np.quantile(img, ub), 256))
    adj_img = np.empty(img.shape)
    
    for cnt, (i,j) in enumerate(zip(q[:-1], q[1:])):
        if cnt == 0:
            adj_img[img <= i] = 0
        adj_img[(img > i) & (img <= j)] = cnt+1
    
    adj_img[img > q[-1]] = 255
    return adj_img.astype(int)

img_f = imadjust(np.load('/Users/hanmanhyuk/Desktop/testimg_f.npy'))
img_i1 = imadjust(np.load('/Users/hanmanhyuk/Desktop/testimg_i1.npy'))
img_i2 = imadjust(np.load('/Users/hanmanhyuk/Desktop/testimg_i2.npy'))
img_x = imadjust(np.load('/Users/hanmanhyuk/Desktop/testimg_x.npy'))

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')

app = pg.mkQApp()

## Create window with ImageView widget
win = QtWidgets.QMainWindow()
win.resize(800,800)
imv = pg.ImageView()
win.setCentralWidget(imv)
win.show()
#win.setWindowTitle('pyqtgraph example: ImageView')
#imv.setHistogramLabel("Histogram label goes here")

## Create random 3D data set with time varying signals
dataRed = np.concatenate([img_f[:,:,None], np.zeros(img_f.shape)[:,:,None], np.zeros(img_f.shape)[:,:,None]], axis=2)
dataCyan = np.concatenate([np.zeros(img_i1.shape)[:,:,None], img_i1[:,:,None], img_i1[:,:,None]], axis=2)

#data = dataRed + dataCyan
data = img_f
#data = np.concatenate(
#    [dataRed,dataCyan], axis=2#    [img_f[:,:,None],img_f[:,:,None],img_f[:,:,None]], axis=2
#)

# Display the data and assign each frame a time value from 1.0 to 3.0
imv.setImage(data, xvals=np.linspace(1., 3., data.shape[0]))
#imv.play(10)

## Set a custom color map
colors = [
    (0, 0, 0),
    (45, 5, 61),
    (84, 42, 55),
    (150, 87, 60),
    (208, 171, 141),
    (255, 255, 255)
]
#cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
#imv.setColorMap(cmap)

# Start up with an ROI
#imv.ui.roiBtn.setChecked(True)
#imv.roiClicked()

if __name__ == '__main__':
    pg.exec()