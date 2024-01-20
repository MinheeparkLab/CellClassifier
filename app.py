## UNIVERSAL (version-independent)
import os, sys
import xml.etree.ElementTree as eT

import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy.ndimage as scind
import scipy.spatial.distance as ssd
from skimage.feature import peak_local_max
import skimage.measure
import skimage.morphology
import skimage.segmentation
import skimage.filters

from shapely.geometry import Polygon, Point

import warnings
warnings.filterwarnings('ignore')

from copy import deepcopy
from itertools import product

from pickle import load, dump

### IMPORTANT ###
import pyqtgraph as pg ### VERSION 0.13.3
from PyQt5 import QtWidgets, QtCore, QtGui ### VERSION 5.15.2

__version__ = '2024.01.17'

path = '/Users/hanmanhyuk/Documents/GitHub/MinHeeLab/CellClassifier/'    
config_name = 'default_config.xml'

translate = QtCore.QCoreApplication.translate
colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
          '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
          '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
          '#000075', '#808080', '#000000']

hex2rgb = lambda hex: (int(hex[1:3],16),int(hex[3:5],16),int(hex[5:7],16))

def construct_cmap_fromblack(color, gamma = 1.5):
    c = np.array(color) / 255
    cdict = {'red': [(0.0,  0.0, 0.0),
                     (0.5, c[0]/2, c[0]/2),
                     (1.0, c[0],   c[0])],

            'green': [(0.0,  0.0, 0.0),
                     (0.5, c[1]/2, c[1]/2),
                     (1.0, c[1],   c[1])],

            'blue':  [(0.0,  0.0, 0.0),
                     (0.5, c[2]/2, c[2]/2),
                     (1.0, c[2],   c[2])]}
    return mcolors.LinearSegmentedColormap('',cdict,gamma=gamma)

def imadjust(img, lb=0, ub=1):
    #p,q = np.histogram(img.flatten(), bins=np.linspace(np.quantile(img, lb), np.quantile(img, ub), 256))
    p,q = np.histogram(img.flatten(), bins=np.linspace(img.max()*lb, img.max()*ub, 256))
    adj_img = np.empty(img.shape)
    
    for cnt, (i,j) in enumerate(zip(q[:-1], q[1:])):
        if cnt == 0:
            adj_img[img <= i] = 0
        adj_img[(img > i) & (img <= j)] = cnt+1
    
    adj_img[img > q[-1]] = 255
    return adj_img.astype(int)

def bootstrap(arr):
    while True:
        yield np.random.choice(arr, arr.size, replace = True)
        
def mv_avg_center(x, w, err=False):
    if w ==1: return x
    x_pad = np.full((x.size+w,1), np.nan)
    lw = int(w/2)
    rw = w - lw
    x_pad[lw:-rw,0] = x
    
    if err:
        return np.nanmean(np.concatenate([x_pad[i:(i-w)] for i in range(w-1)], axis=1),1), np.nanstd(np.concatenate([x_pad[i:(i-w)] for i in range(w-1)], axis=1),1) / np.sqrt(w)
    else:
        return np.nanmean(np.concatenate([x_pad[i:(i-w)] for i in range(w-1)], axis=1),1)

def _listWidgetValueToLabel(listWidget, Label):
    def __changeLabel(listWidget, Label):
        selectedItems = [item.text().lower() for item in listWidget.selectedItems()]
        if 'all' in selectedItems:
            text = 'all'
        else:
            text = ','.join(selectedItems)
        Label.setText(text)    
    listWidget.itemClicked.connect(lambda:__changeLabel(listWidget,Label))
    
class CellSegmentPanelUI(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(677, 717)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(Form)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setGeometry(QtCore.QRect(10, 10, 311, 231))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.label = QtWidgets.QLabel(self.frame_2)
        self.label.setGeometry(QtCore.QRect(0, 0, 311, 31))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.line = QtWidgets.QFrame(self.frame_2)
        self.line.setGeometry(QtCore.QRect(0, 15, 311, 31))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.ActivateCellBoundaries_cellBoundariesLineEdit = QtWidgets.QLineEdit(self.frame_2)
        self.ActivateCellBoundaries_cellBoundariesLineEdit.setGeometry(QtCore.QRect(10, 40, 291, 21))
        self.ActivateCellBoundaries_cellBoundariesLineEdit.setObjectName("ActivateCellBoundaries_cellBoundariesLineEdit")
        self.ActivateCellBoundaries_cellBoundariesListWidget = QtWidgets.QListWidget(self.frame_2)
        self.ActivateCellBoundaries_cellBoundariesListWidget.setGeometry(QtCore.QRect(10, 70, 211, 151))
        self.ActivateCellBoundaries_cellBoundariesListWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.ActivateCellBoundaries_cellBoundariesListWidget.setObjectName("ActivateCellBoundaries_cellBoundariesListWidget")
        item = QtWidgets.QListWidgetItem()
        self.ActivateCellBoundaries_cellBoundariesListWidget.addItem(item)
        self.ActivateCellBoundaries_cellBoundariesShowPushButton = QtWidgets.QPushButton(self.frame_2)
        self.ActivateCellBoundaries_cellBoundariesShowPushButton.setGeometry(QtCore.QRect(220, 70, 91, 32))
        self.ActivateCellBoundaries_cellBoundariesShowPushButton.setObjectName("ActivateCellBoundaries_cellBoundariesShowPushButton")
        self.ActivateCellBoundaries_cellBoundariesHidePushButton = QtWidgets.QPushButton(self.frame_2)
        self.ActivateCellBoundaries_cellBoundariesHidePushButton.setGeometry(QtCore.QRect(220, 110, 91, 32))
        self.ActivateCellBoundaries_cellBoundariesHidePushButton.setObjectName("ActivateCellBoundaries_cellBoundariesHidePushButton")
        self.line.raise_()
        self.label.raise_()
        self.ActivateCellBoundaries_cellBoundariesLineEdit.raise_()
        self.ActivateCellBoundaries_cellBoundariesListWidget.raise_()
        self.ActivateCellBoundaries_cellBoundariesShowPushButton.raise_()
        self.ActivateCellBoundaries_cellBoundariesHidePushButton.raise_()
        self.frame_3 = QtWidgets.QFrame(self.frame)
        self.frame_3.setGeometry(QtCore.QRect(10, 250, 631, 431))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.label_3 = QtWidgets.QLabel(self.frame_3)
        self.label_3.setGeometry(QtCore.QRect(0, 0, 311, 31))
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.line_3 = QtWidgets.QFrame(self.frame_3)
        self.line_3.setGeometry(QtCore.QRect(10, 15, 301, 31))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.frame_8 = QtWidgets.QFrame(self.frame_3)
        self.frame_8.setGeometry(QtCore.QRect(10, 40, 141, 80))
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.label_11 = QtWidgets.QLabel(self.frame_8)
        self.label_11.setGeometry(QtCore.QRect(0, 0, 141, 51))
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setWordWrap(True)
        self.label_11.setObjectName("label_11")
        self.ClassifyCells_methodComboBox = QtWidgets.QComboBox(self.frame_8)
        self.ClassifyCells_methodComboBox.setGeometry(QtCore.QRect(10, 50, 121, 26))
        self.ClassifyCells_methodComboBox.setObjectName("ClassifyCells_methodComboBox")
        self.ClassifyCells_methodComboBox.addItem("")
        self.ClassifyCells_methodComboBox.addItem("")
        self.ClassifyCells_methodComboBox.addItem("")
        self.ClassifyCells_methodComboBox.addItem("")
        self.ClassifyCells_methodComboBox.addItem("")
        self.frame_5 = QtWidgets.QFrame(self.frame_3)
        self.frame_5.setGeometry(QtCore.QRect(160, 40, 141, 80))
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.label_6 = QtWidgets.QLabel(self.frame_5)
        self.label_6.setGeometry(QtCore.QRect(0, 0, 141, 51))
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setWordWrap(True)
        self.label_6.setObjectName("label_6")
        self.ClassifyCells_minDistanceBetweenCellsSpinBox = QtWidgets.QSpinBox(self.frame_5)
        self.ClassifyCells_minDistanceBetweenCellsSpinBox.setGeometry(QtCore.QRect(10, 50, 121, 24))
        self.ClassifyCells_minDistanceBetweenCellsSpinBox.setMaximum(1000000000)
        self.ClassifyCells_minDistanceBetweenCellsSpinBox.setProperty("value", 15)
        self.ClassifyCells_minDistanceBetweenCellsSpinBox.setObjectName("ClassifyCells_minDistanceBetweenCellsSpinBox")
        self.frame_6 = QtWidgets.QFrame(self.frame_3)
        self.frame_6.setGeometry(QtCore.QRect(10, 130, 141, 80))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.label_8 = QtWidgets.QLabel(self.frame_6)
        self.label_8.setGeometry(QtCore.QRect(0, 0, 141, 51))
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setWordWrap(True)
        self.label_8.setObjectName("label_8")
        self.ClassifyCells_minSizeOfCellsSpinBox = QtWidgets.QSpinBox(self.frame_6)
        self.ClassifyCells_minSizeOfCellsSpinBox.setGeometry(QtCore.QRect(10, 50, 121, 24))
        self.ClassifyCells_minSizeOfCellsSpinBox.setMaximum(1000000000)
        self.ClassifyCells_minSizeOfCellsSpinBox.setProperty("value", 100)
        self.ClassifyCells_minSizeOfCellsSpinBox.setObjectName("ClassifyCells_minSizeOfCellsSpinBox")
        self.frame_7 = QtWidgets.QFrame(self.frame_3)
        self.frame_7.setGeometry(QtCore.QRect(160, 130, 141, 80))
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.label_9 = QtWidgets.QLabel(self.frame_7)
        self.label_9.setGeometry(QtCore.QRect(0, 0, 141, 51))
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setWordWrap(True)
        self.label_9.setObjectName("label_9")
        self.ClassifyCells_maxSizeOfCellsSpinBox = QtWidgets.QSpinBox(self.frame_7)
        self.ClassifyCells_maxSizeOfCellsSpinBox.setGeometry(QtCore.QRect(10, 50, 121, 24))
        self.ClassifyCells_maxSizeOfCellsSpinBox.setMaximum(1000000000)
        self.ClassifyCells_maxSizeOfCellsSpinBox.setProperty("value", 30000)
        self.ClassifyCells_maxSizeOfCellsSpinBox.setObjectName("ClassifyCells_maxSizeOfCellsSpinBox")
        self.frame_4 = QtWidgets.QFrame(self.frame_3)
        self.frame_4.setGeometry(QtCore.QRect(10, 220, 291, 131))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.label_4 = QtWidgets.QLabel(self.frame_4)
        self.label_4.setGeometry(QtCore.QRect(0, 0, 291, 31))
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.classifyCells_binarizationFracHorizontalSlider = QtWidgets.QSlider(self.frame_4)
        self.classifyCells_binarizationFracHorizontalSlider.setGeometry(QtCore.QRect(70, 40, 151, 31))
        self.classifyCells_binarizationFracHorizontalSlider.setMaximum(100)
        self.classifyCells_binarizationFracHorizontalSlider.setProperty("value", 60)
        self.classifyCells_binarizationFracHorizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.classifyCells_binarizationFracHorizontalSlider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.classifyCells_binarizationFracHorizontalSlider.setObjectName("classifyCells_binarizationFracHorizontalSlider")
        self.classifyCells_binarizationFracLineEdit = QtWidgets.QLineEdit(self.frame_4)
        self.classifyCells_binarizationFracLineEdit.setGeometry(QtCore.QRect(230, 40, 51, 31))
        self.classifyCells_binarizationFracLineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.classifyCells_binarizationFracLineEdit.setObjectName("classifyCells_binarizationFracLineEdit")
        self.label_5 = QtWidgets.QLabel(self.frame_4)
        self.label_5.setGeometry(QtCore.QRect(10, 40, 51, 31))
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.classifyCells_binarizationAbsLineEdit = QtWidgets.QLineEdit(self.frame_4)
        self.classifyCells_binarizationAbsLineEdit.setGeometry(QtCore.QRect(70, 70, 131, 31))
        self.classifyCells_binarizationAbsLineEdit.setObjectName("classifyCells_binarizationAbsLineEdit")
        self.label_12 = QtWidgets.QLabel(self.frame_4)
        self.label_12.setGeometry(QtCore.QRect(10, 70, 51, 31))
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.line_4 = QtWidgets.QFrame(self.frame_4)
        self.line_4.setGeometry(QtCore.QRect(0, 13, 291, 41))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.line_5 = QtWidgets.QFrame(self.frame_4)
        self.line_5.setGeometry(QtCore.QRect(50, 40, 20, 61))
        self.line_5.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.classifyCells_binarizationShowSketchPushButton = QtWidgets.QPushButton(self.frame_4)
        self.classifyCells_binarizationShowSketchPushButton.setGeometry(QtCore.QRect(70, 100, 111, 32))
        self.classifyCells_binarizationShowSketchPushButton.setObjectName("classifyCells_binarizationShowSketchPushButton")
        self.classifyCells_binarizationShowInitializePushButton = QtWidgets.QPushButton(self.frame_4)
        self.classifyCells_binarizationShowInitializePushButton.setGeometry(QtCore.QRect(180, 100, 111, 32))
        self.classifyCells_binarizationShowInitializePushButton.setObjectName("classifyCells_binarizationShowInitializePushButton")
        self.line_5.raise_()
        self.line_4.raise_()
        self.label_4.raise_()
        self.classifyCells_binarizationFracHorizontalSlider.raise_()
        self.classifyCells_binarizationFracLineEdit.raise_()
        self.label_5.raise_()
        self.classifyCells_binarizationAbsLineEdit.raise_()
        self.label_12.raise_()
        self.classifyCells_binarizationShowSketchPushButton.raise_()
        self.classifyCells_binarizationShowInitializePushButton.raise_()
        self.line_7 = QtWidgets.QFrame(self.frame_3)
        self.line_7.setGeometry(QtCore.QRect(306, 0, 20, 431))
        self.line_7.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.frame_12 = QtWidgets.QFrame(self.frame_3)
        self.frame_12.setGeometry(QtCore.QRect(10, 360, 291, 61))
        self.frame_12.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_12.setObjectName("frame_12")
        self.classifyCells_autoRunPushButton = QtWidgets.QPushButton(self.frame_12)
        self.classifyCells_autoRunPushButton.setGeometry(QtCore.QRect(0, 0, 141, 32))
        self.classifyCells_autoRunPushButton.setObjectName("classifyCells_autoRunPushButton")
        self.classifyCells_manualModeONPushButton = QtWidgets.QPushButton(self.frame_12)
        self.classifyCells_manualModeONPushButton.setGeometry(QtCore.QRect(0, 30, 141, 32))
        self.classifyCells_manualModeONPushButton.setObjectName("classifyCells_manualModeONPushButton")
        self.classifyCells_manualModeOFFPushButton = QtWidgets.QPushButton(self.frame_12)
        self.classifyCells_manualModeOFFPushButton.setEnabled(False)
        self.classifyCells_manualModeOFFPushButton.setGeometry(QtCore.QRect(150, 30, 141, 32))
        self.classifyCells_manualModeOFFPushButton.setObjectName("classifyCells_manualModeOFFPushButton")
        self.classifyCells_addNewCellPushButton = QtWidgets.QPushButton(self.frame_12)
        self.classifyCells_addNewCellPushButton.setGeometry(QtCore.QRect(150, 0, 141, 32))
        self.classifyCells_addNewCellPushButton.setObjectName("classifyCells_addNewCellPushButton")
        self.frame_10 = QtWidgets.QFrame(self.frame_3)
        self.frame_10.setGeometry(QtCore.QRect(330, 10, 291, 201))
        self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.label_13 = QtWidgets.QLabel(self.frame_10)
        self.label_13.setGeometry(QtCore.QRect(0, 0, 291, 31))
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.transientCellContainer_textBrowser = QtWidgets.QTextBrowser(self.frame_10)
        self.transientCellContainer_textBrowser.setGeometry(QtCore.QRect(10, 40, 271, 131))
        self.transientCellContainer_textBrowser.setObjectName("transientCellContainer_textBrowser")
        self.transientCellContainer_saveCurrentCells = QtWidgets.QPushButton(self.frame_10)
        self.transientCellContainer_saveCurrentCells.setGeometry(QtCore.QRect(10, 170, 113, 32))
        self.transientCellContainer_saveCurrentCells.setObjectName("transientCellContainer_saveCurrentCells")
        self.transientCellContainer_ClearCurrentCells = QtWidgets.QPushButton(self.frame_10)
        self.transientCellContainer_ClearCurrentCells.setGeometry(QtCore.QRect(120, 170, 113, 32))
        self.transientCellContainer_ClearCurrentCells.setObjectName("transientCellContainer_ClearCurrentCells")
        self.frame_11 = QtWidgets.QFrame(self.frame_3)
        self.frame_11.setGeometry(QtCore.QRect(330, 220, 291, 201))
        self.frame_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_11.setObjectName("frame_11")
        self.label_14 = QtWidgets.QLabel(self.frame_11)
        self.label_14.setGeometry(QtCore.QRect(0, 0, 291, 31))
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.permanentCellContainer_textBrowser = QtWidgets.QTextBrowser(self.frame_11)
        self.permanentCellContainer_textBrowser.setGeometry(QtCore.QRect(10, 40, 271, 131))
        self.permanentCellContainer_textBrowser.setObjectName("permanentCellContainer_textBrowser")
        self.permanentCellContainer_SendCurrentCells = QtWidgets.QPushButton(self.frame_11)
        self.permanentCellContainer_SendCurrentCells.setGeometry(QtCore.QRect(10, 170, 113, 32))
        self.permanentCellContainer_SendCurrentCells.setObjectName("permanentCellContainer_SendCurrentCells")
        self.frame_9 = QtWidgets.QFrame(self.frame)
        self.frame_9.setGeometry(QtCore.QRect(330, 10, 311, 231))
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.label_15 = QtWidgets.QLabel(self.frame_9)
        self.label_15.setGeometry(QtCore.QRect(0, 0, 311, 31))
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.line_6 = QtWidgets.QFrame(self.frame_9)
        self.line_6.setGeometry(QtCore.QRect(0, 15, 311, 31))
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.ModulateCell_modulateCellLineEdit = QtWidgets.QLineEdit(self.frame_9)
        self.ModulateCell_modulateCellLineEdit.setGeometry(QtCore.QRect(10, 40, 291, 21))
        self.ModulateCell_modulateCellLineEdit.setObjectName("ModulateCell_modulateCellLineEdit")
        self.ModulateCell_modulateCellListWidget = QtWidgets.QListWidget(self.frame_9)
        self.ModulateCell_modulateCellListWidget.setGeometry(QtCore.QRect(10, 70, 211, 151))
        self.ModulateCell_modulateCellListWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.ModulateCell_modulateCellListWidget.setObjectName("ModulateCell_modulateCellListWidget")
        item = QtWidgets.QListWidgetItem()
        self.ModulateCell_modulateCellListWidget.addItem(item)
        self.ModulateCell_modulateCellExpandPushButton = QtWidgets.QPushButton(self.frame_9)
        self.ModulateCell_modulateCellExpandPushButton.setGeometry(QtCore.QRect(220, 70, 91, 32))
        self.ModulateCell_modulateCellExpandPushButton.setObjectName("ModulateCell_modulateCellExpandPushButton")
        self.ModulateCell_modulateCellShrinkPushButton = QtWidgets.QPushButton(self.frame_9)
        self.ModulateCell_modulateCellShrinkPushButton.setGeometry(QtCore.QRect(220, 110, 91, 32))
        self.ModulateCell_modulateCellShrinkPushButton.setObjectName("ModulateCell_modulateCellShrinkPushButton")
        self.ModulateCell_modulateCellCombinePushButton = QtWidgets.QPushButton(self.frame_9)
        self.ModulateCell_modulateCellCombinePushButton.setGeometry(QtCore.QRect(220, 150, 91, 32))
        self.ModulateCell_modulateCellCombinePushButton.setObjectName("ModulateCell_modulateCellCombinePushButton")
        self.ModulateCell_modulateCellDeletePushButton = QtWidgets.QPushButton(self.frame_9)
        self.ModulateCell_modulateCellDeletePushButton.setGeometry(QtCore.QRect(220, 190, 91, 32))
        self.ModulateCell_modulateCellDeletePushButton.setObjectName("ModulateCell_modulateCellDeletePushButton")
        self.verticalLayout.addWidget(self.frame)
        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Activate Cell Boundaries"))
        __sortingEnabled = self.ActivateCellBoundaries_cellBoundariesListWidget.isSortingEnabled()
        self.ActivateCellBoundaries_cellBoundariesListWidget.setSortingEnabled(False)
        item = self.ActivateCellBoundaries_cellBoundariesListWidget.item(0)
        item.setText(_translate("Form", "All"))
        self.ActivateCellBoundaries_cellBoundariesListWidget.setSortingEnabled(__sortingEnabled)
        self.ActivateCellBoundaries_cellBoundariesShowPushButton.setText(_translate("Form", "Show"))
        self.ActivateCellBoundaries_cellBoundariesHidePushButton.setText(_translate("Form", "Hide"))
        self.label_3.setText(_translate("Form", "Classify cells"))
        self.label_11.setText(_translate("Form", "Method"))
        self.ClassifyCells_methodComboBox.setItemText(0, _translate("Form", "Otsu"))
        self.ClassifyCells_methodComboBox.setItemText(1, _translate("Form", "Yen"))
        self.ClassifyCells_methodComboBox.setItemText(2, _translate("Form", "Li"))
        self.ClassifyCells_methodComboBox.setItemText(3, _translate("Form", "Triangle"))
        self.ClassifyCells_methodComboBox.setItemText(4, _translate("Form", "Manual"))
        self.label_6.setText(_translate("Form", "min distance b/w cells (default: 15)"))
        self.label_8.setText(_translate("Form", "min size of cells (default: 100)"))
        self.label_9.setText(_translate("Form", "max size of cells (default: 30000)"))
        self.label_4.setText(_translate("Form", "Binarization"))
        self.classifyCells_binarizationFracLineEdit.setText(_translate("Form", "0.6"))
        self.label_5.setText(_translate("Form", "frac"))
        self.label_12.setText(_translate("Form", "abs"))
        self.classifyCells_binarizationShowSketchPushButton.setText(_translate("Form", "Show sketch"))
        self.classifyCells_binarizationShowInitializePushButton.setText(_translate("Form", "Initialize"))
        self.classifyCells_autoRunPushButton.setText(_translate("Form", "Auto Run"))
        self.classifyCells_manualModeONPushButton.setText(_translate("Form", "manual mode ON"))
        self.classifyCells_manualModeOFFPushButton.setText(_translate("Form", "manual mode OFF"))
        self.classifyCells_addNewCellPushButton.setText(_translate("Form", "Add new cell"))
        self.label_13.setText(_translate("Form", "Transient Cell Container"))
        self.transientCellContainer_saveCurrentCells.setText(_translate("Form", "Save"))
        self.transientCellContainer_ClearCurrentCells.setText(_translate("Form", "Clear"))
        self.label_14.setText(_translate("Form", "Permanent Cell Container"))
        self.permanentCellContainer_SendCurrentCells.setText(_translate("Form", "Send"))
        self.label_15.setText(_translate("Form", "Modulate Cell"))
        __sortingEnabled = self.ModulateCell_modulateCellListWidget.isSortingEnabled()
        self.ModulateCell_modulateCellListWidget.setSortingEnabled(False)
        item = self.ModulateCell_modulateCellListWidget.item(0)
        item.setText(_translate("Form", "All"))
        self.ModulateCell_modulateCellListWidget.setSortingEnabled(__sortingEnabled)
        self.ModulateCell_modulateCellExpandPushButton.setText(_translate("Form", "Expand"))
        self.ModulateCell_modulateCellShrinkPushButton.setText(_translate("Form", "Shrink"))
        self.ModulateCell_modulateCellCombinePushButton.setText(_translate("Form", "Combine"))
        self.ModulateCell_modulateCellDeletePushButton.setText(_translate("Form", "Delete"))

class SpotAnalysisPanelUI(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(718, 569)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(Form)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.line = QtWidgets.QFrame(self.frame)
        self.line.setGeometry(QtCore.QRect(250, 0, 20, 261))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(0, 0, 261, 41))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.line_2 = QtWidgets.QFrame(self.frame)
        self.line_2.setGeometry(QtCore.QRect(-3, 30, 811, 20))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(260, 0, 211, 41))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setGeometry(QtCore.QRect(470, 0, 221, 41))
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.line_3 = QtWidgets.QFrame(self.frame)
        self.line_3.setGeometry(QtCore.QRect(460, 0, 20, 261))
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setGeometry(QtCore.QRect(10, 50, 241, 91))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.label_4 = QtWidgets.QLabel(self.frame_2)
        self.label_4.setGeometry(QtCore.QRect(0, 0, 241, 31))
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.frame_2)
        self.label_5.setGeometry(QtCore.QRect(0, 30, 41, 31))
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.line_4 = QtWidgets.QFrame(self.frame_2)
        self.line_4.setGeometry(QtCore.QRect(0, 15, 241, 31))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.spotLocalizationParameters_intensityThresholdSlider = QtWidgets.QSlider(self.frame_2)
        self.spotLocalizationParameters_intensityThresholdSlider.setGeometry(QtCore.QRect(49, 30, 121, 31))
        self.spotLocalizationParameters_intensityThresholdSlider.setMaximum(100)
        self.spotLocalizationParameters_intensityThresholdSlider.setProperty("value", 60)
        self.spotLocalizationParameters_intensityThresholdSlider.setOrientation(QtCore.Qt.Horizontal)
        self.spotLocalizationParameters_intensityThresholdSlider.setObjectName("spotLocalizationParameters_intensityThresholdSlider")
        self.spotLocalizationParameters_intensityThreshold_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.frame_2)
        self.spotLocalizationParameters_intensityThreshold_doubleSpinBox.setGeometry(QtCore.QRect(177, 35, 61, 21))
        self.spotLocalizationParameters_intensityThreshold_doubleSpinBox.setMaximum(100.0)
        self.spotLocalizationParameters_intensityThreshold_doubleSpinBox.setProperty("value", 60.0)
        self.spotLocalizationParameters_intensityThreshold_doubleSpinBox.setObjectName("spotLocalizationParameters_intensityThreshold_doubleSpinBox")
        self.label_6 = QtWidgets.QLabel(self.frame_2)
        self.label_6.setGeometry(QtCore.QRect(0, 60, 41, 31))
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.line_5 = QtWidgets.QFrame(self.frame_2)
        self.line_5.setGeometry(QtCore.QRect(30, 30, 31, 111))
        self.line_5.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.line_6 = QtWidgets.QFrame(self.frame_2)
        self.line_6.setGeometry(QtCore.QRect(0, 40, 241, 41))
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.spotLocalizationParameters_intensityThresholdAbsoluteLineEdit = QtWidgets.QLineEdit(self.frame_2)
        self.spotLocalizationParameters_intensityThresholdAbsoluteLineEdit.setGeometry(QtCore.QRect(50, 65, 181, 21))
        self.spotLocalizationParameters_intensityThresholdAbsoluteLineEdit.setObjectName("spotLocalizationParameters_intensityThresholdAbsoluteLineEdit")
        self.line_6.raise_()
        self.line_4.raise_()
        self.label_4.raise_()
        self.label_5.raise_()
        self.spotLocalizationParameters_intensityThresholdSlider.raise_()
        self.spotLocalizationParameters_intensityThreshold_doubleSpinBox.raise_()
        self.label_6.raise_()
        self.line_5.raise_()
        self.spotLocalizationParameters_intensityThresholdAbsoluteLineEdit.raise_()
        self.frame_3 = QtWidgets.QFrame(self.frame)
        self.frame_3.setGeometry(QtCore.QRect(10, 150, 241, 71))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.label_7 = QtWidgets.QLabel(self.frame_3)
        self.label_7.setGeometry(QtCore.QRect(0, 0, 241, 31))
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.spotLocalizationMinimumDistance_spinBox = QtWidgets.QSpinBox(self.frame_3)
        self.spotLocalizationMinimumDistance_spinBox.setGeometry(QtCore.QRect(20, 40, 211, 21))
        self.spotLocalizationMinimumDistance_spinBox.setMinimum(1)
        self.spotLocalizationMinimumDistance_spinBox.setMaximum(100000)
        self.spotLocalizationMinimumDistance_spinBox.setObjectName("spotLocalizationMinimumDistance_spinBox")
        self.line_7 = QtWidgets.QFrame(self.frame_3)
        self.line_7.setGeometry(QtCore.QRect(0, 10, 241, 41))
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.line_7.raise_()
        self.label_7.raise_()
        self.spotLocalizationMinimumDistance_spinBox.raise_()
        self.runSpotLocalization_pushButton = QtWidgets.QPushButton(self.frame)
        self.runSpotLocalization_pushButton.setGeometry(QtCore.QRect(10, 230, 241, 32))
        self.runSpotLocalization_pushButton.setObjectName("runSpotLocalization_pushButton")
        self.SpotAnalysisPanel_TransientSpotContainerClearChannelPushButton = QtWidgets.QPushButton(self.frame)
        self.SpotAnalysisPanel_TransientSpotContainerClearChannelPushButton.setGeometry(QtCore.QRect(120, 510, 111, 32))
        self.SpotAnalysisPanel_TransientSpotContainerClearChannelPushButton.setObjectName("SpotAnalysisPanel_TransientSpotContainerClearChannelPushButton")
        self.frame_4 = QtWidgets.QFrame(self.frame)
        self.frame_4.setGeometry(QtCore.QRect(270, 50, 191, 91))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.label_11 = QtWidgets.QLabel(self.frame_4)
        self.label_11.setGeometry(QtCore.QRect(0, 0, 191, 31))
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.frame_4)
        self.label_12.setGeometry(QtCore.QRect(0, 30, 41, 31))
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.line_11 = QtWidgets.QFrame(self.frame_4)
        self.line_11.setGeometry(QtCore.QRect(0, 15, 241, 31))
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")
        self.spotSizeMeasurementParameters_intensityThresholdSlider = QtWidgets.QSlider(self.frame_4)
        self.spotSizeMeasurementParameters_intensityThresholdSlider.setGeometry(QtCore.QRect(49, 30, 71, 31))
        self.spotSizeMeasurementParameters_intensityThresholdSlider.setMaximum(100)
        self.spotSizeMeasurementParameters_intensityThresholdSlider.setProperty("value", 70)
        self.spotSizeMeasurementParameters_intensityThresholdSlider.setOrientation(QtCore.Qt.Horizontal)
        self.spotSizeMeasurementParameters_intensityThresholdSlider.setObjectName("spotSizeMeasurementParameters_intensityThresholdSlider")
        self.spotSizeMeasurementParameters_intensityThreshold_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.frame_4)
        self.spotSizeMeasurementParameters_intensityThreshold_doubleSpinBox.setGeometry(QtCore.QRect(120, 35, 61, 21))
        self.spotSizeMeasurementParameters_intensityThreshold_doubleSpinBox.setMaximum(100.0)
        self.spotSizeMeasurementParameters_intensityThreshold_doubleSpinBox.setProperty("value", 70.0)
        self.spotSizeMeasurementParameters_intensityThreshold_doubleSpinBox.setObjectName("spotSizeMeasurementParameters_intensityThreshold_doubleSpinBox")
        self.label_13 = QtWidgets.QLabel(self.frame_4)
        self.label_13.setGeometry(QtCore.QRect(0, 60, 41, 31))
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.line_12 = QtWidgets.QFrame(self.frame_4)
        self.line_12.setGeometry(QtCore.QRect(30, 30, 31, 111))
        self.line_12.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_12.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_12.setObjectName("line_12")
        self.line_13 = QtWidgets.QFrame(self.frame_4)
        self.line_13.setGeometry(QtCore.QRect(0, 40, 241, 41))
        self.line_13.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_13.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_13.setObjectName("line_13")
        self.spotSizeMeasurementParameters_intensityThresholdAbsoluteLineEdit = QtWidgets.QLineEdit(self.frame_4)
        self.spotSizeMeasurementParameters_intensityThresholdAbsoluteLineEdit.setGeometry(QtCore.QRect(50, 65, 131, 21))
        self.spotSizeMeasurementParameters_intensityThresholdAbsoluteLineEdit.setObjectName("spotSizeMeasurementParameters_intensityThresholdAbsoluteLineEdit")
        self.line_13.raise_()
        self.label_11.raise_()
        self.label_12.raise_()
        self.line_11.raise_()
        self.spotSizeMeasurementParameters_intensityThresholdSlider.raise_()
        self.spotSizeMeasurementParameters_intensityThreshold_doubleSpinBox.raise_()
        self.label_13.raise_()
        self.line_12.raise_()
        self.spotSizeMeasurementParameters_intensityThresholdAbsoluteLineEdit.raise_()
        self.frame_5 = QtWidgets.QFrame(self.frame)
        self.frame_5.setGeometry(QtCore.QRect(270, 150, 91, 71))
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.label_14 = QtWidgets.QLabel(self.frame_5)
        self.label_14.setGeometry(QtCore.QRect(0, 0, 91, 41))
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setWordWrap(True)
        self.label_14.setObjectName("label_14")
        self.spotQualityControl_minimumSpotSizeSpinBox = QtWidgets.QSpinBox(self.frame_5)
        self.spotQualityControl_minimumSpotSizeSpinBox.setGeometry(QtCore.QRect(10, 44, 71, 21))
        self.spotQualityControl_minimumSpotSizeSpinBox.setMaximum(100000)
        self.spotQualityControl_minimumSpotSizeSpinBox.setProperty("value", 1)
        self.spotQualityControl_minimumSpotSizeSpinBox.setObjectName("spotQualityControl_minimumSpotSizeSpinBox")
        self.line_14 = QtWidgets.QFrame(self.frame_5)
        self.line_14.setGeometry(QtCore.QRect(0, 20, 91, 41))
        self.line_14.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_14.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_14.setObjectName("line_14")
        self.line_14.raise_()
        self.label_14.raise_()
        self.spotQualityControl_minimumSpotSizeSpinBox.raise_()
        self.frame_7 = QtWidgets.QFrame(self.frame)
        self.frame_7.setGeometry(QtCore.QRect(370, 150, 91, 71))
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.label_17 = QtWidgets.QLabel(self.frame_7)
        self.label_17.setGeometry(QtCore.QRect(0, 0, 91, 41))
        self.label_17.setAlignment(QtCore.Qt.AlignCenter)
        self.label_17.setWordWrap(True)
        self.label_17.setObjectName("label_17")
        self.spotQualityControl_maximumSpotSizeSpinBox = QtWidgets.QSpinBox(self.frame_7)
        self.spotQualityControl_maximumSpotSizeSpinBox.setGeometry(QtCore.QRect(10, 44, 71, 21))
        self.spotQualityControl_maximumSpotSizeSpinBox.setMaximum(100000)
        self.spotQualityControl_maximumSpotSizeSpinBox.setProperty("value", 100)
        self.spotQualityControl_maximumSpotSizeSpinBox.setObjectName("spotQualityControl_maximumSpotSizeSpinBox")
        self.line_17 = QtWidgets.QFrame(self.frame_7)
        self.line_17.setGeometry(QtCore.QRect(0, 20, 91, 41))
        self.line_17.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_17.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_17.setObjectName("line_17")
        self.line_17.raise_()
        self.label_17.raise_()
        self.spotQualityControl_maximumSpotSizeSpinBox.raise_()
        self.line_15 = QtWidgets.QFrame(self.frame)
        self.line_15.setGeometry(QtCore.QRect(0, 220, 811, 20))
        self.line_15.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_15.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_15.setObjectName("line_15")
        self.runSpotQualityControl_pushButton = QtWidgets.QPushButton(self.frame)
        self.runSpotQualityControl_pushButton.setGeometry(QtCore.QRect(270, 230, 191, 32))
        self.runSpotQualityControl_pushButton.setObjectName("runSpotQualityControl_pushButton")
        self.label_15 = QtWidgets.QLabel(self.frame)
        self.label_15.setGeometry(QtCore.QRect(480, 40, 111, 31))
        self.label_15.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_15.setObjectName("label_15")
        self.spotLocalization_channelListWidget = QtWidgets.QListWidget(self.frame)
        self.spotLocalization_channelListWidget.setGeometry(QtCore.QRect(480, 71, 201, 131))
        self.spotLocalization_channelListWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.spotLocalization_channelListWidget.setObjectName("spotLocalization_channelListWidget")
        self.spotVisualization_showSpotsPushButton = QtWidgets.QPushButton(self.frame)
        self.spotVisualization_showSpotsPushButton.setGeometry(QtCore.QRect(480, 200, 101, 32))
        self.spotVisualization_showSpotsPushButton.setObjectName("spotVisualization_showSpotsPushButton")
        self.spotVisualization_hideSpotsPushButton = QtWidgets.QPushButton(self.frame)
        self.spotVisualization_hideSpotsPushButton.setGeometry(QtCore.QRect(580, 200, 101, 32))
        self.spotVisualization_hideSpotsPushButton.setObjectName("spotVisualization_hideSpotsPushButton")
        self.line_18 = QtWidgets.QFrame(self.frame)
        self.line_18.setGeometry(QtCore.QRect(482, 200, 201, 61))
        self.line_18.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_18.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_18.setObjectName("line_18")
        self.line_19 = QtWidgets.QFrame(self.frame)
        self.line_19.setGeometry(QtCore.QRect(0, 250, 811, 20))
        self.line_19.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_19.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_19.setObjectName("line_19")
        self.label_18 = QtWidgets.QLabel(self.frame)
        self.label_18.setGeometry(QtCore.QRect(0, 260, 351, 41))
        self.label_18.setAlignment(QtCore.Qt.AlignCenter)
        self.label_18.setObjectName("label_18")
        self.label_19 = QtWidgets.QLabel(self.frame)
        self.label_19.setGeometry(QtCore.QRect(350, 260, 341, 41))
        self.label_19.setAlignment(QtCore.Qt.AlignCenter)
        self.label_19.setObjectName("label_19")
        self.line_20 = QtWidgets.QFrame(self.frame)
        self.line_20.setGeometry(QtCore.QRect(-1, 260, 701, 281))
        self.line_20.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_20.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_20.setObjectName("line_20")
        self.line_21 = QtWidgets.QFrame(self.frame)
        self.line_21.setGeometry(QtCore.QRect(0, 290, 701, 20))
        self.line_21.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_21.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_21.setObjectName("line_21")
        self.SpotAnalysisPanel_TransientSpotContainerTestBrowser = QtWidgets.QTextBrowser(self.frame)
        self.SpotAnalysisPanel_TransientSpotContainerTestBrowser.setGeometry(QtCore.QRect(15, 310, 321, 192))
        self.SpotAnalysisPanel_TransientSpotContainerTestBrowser.setObjectName("SpotAnalysisPanel_TransientSpotContainerTestBrowser")
        self.SpotAnalysisPanel_PermanentSpotContainerTestBrowser = QtWidgets.QTextBrowser(self.frame)
        self.SpotAnalysisPanel_PermanentSpotContainerTestBrowser.setGeometry(QtCore.QRect(360, 310, 321, 192))
        self.SpotAnalysisPanel_PermanentSpotContainerTestBrowser.setObjectName("SpotAnalysisPanel_PermanentSpotContainerTestBrowser")
        self.line_22 = QtWidgets.QFrame(self.frame)
        self.line_22.setGeometry(QtCore.QRect(0, 500, 701, 20))
        self.line_22.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_22.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_22.setObjectName("line_22")
        self.SpotAnalysisPanel_PermanentSpotContainerSendPushButton = QtWidgets.QPushButton(self.frame)
        self.SpotAnalysisPanel_PermanentSpotContainerSendPushButton.setGeometry(QtCore.QRect(360, 510, 111, 32))
        self.SpotAnalysisPanel_PermanentSpotContainerSendPushButton.setObjectName("SpotAnalysisPanel_PermanentSpotContainerSendPushButton")
        self.SpotAnalysisPanel_TransientSpotContainerSavePushButton = QtWidgets.QPushButton(self.frame)
        self.SpotAnalysisPanel_TransientSpotContainerSavePushButton.setGeometry(QtCore.QRect(10, 510, 111, 32))
        self.SpotAnalysisPanel_TransientSpotContainerSavePushButton.setObjectName("SpotAnalysisPanel_TransientSpotContainerSavePushButton")
        self.spotVisualization_TurnONManualModePushButton = QtWidgets.QPushButton(self.frame)
        self.spotVisualization_TurnONManualModePushButton.setGeometry(QtCore.QRect(480, 230, 101, 32))
        self.spotVisualization_TurnONManualModePushButton.setObjectName("spotVisualization_TurnONManualModePushButton")
        self.spotVisualization_TurnOFFManualModePushButton = QtWidgets.QPushButton(self.frame)
        self.spotVisualization_TurnOFFManualModePushButton.setEnabled(False)
        self.spotVisualization_TurnOFFManualModePushButton.setGeometry(QtCore.QRect(580, 230, 101, 32))
        self.spotVisualization_TurnOFFManualModePushButton.setObjectName("spotVisualization_TurnOFFManualModePushButton")
        self.SpotAnalysisPanel_TransientSpotContainerClearAllPushButton = QtWidgets.QPushButton(self.frame)
        self.SpotAnalysisPanel_TransientSpotContainerClearAllPushButton.setGeometry(QtCore.QRect(230, 510, 111, 32))
        self.SpotAnalysisPanel_TransientSpotContainerClearAllPushButton.setObjectName("SpotAnalysisPanel_TransientSpotContainerClearAllPushButton")
        self.line_20.raise_()
        self.SpotAnalysisPanel_TransientSpotContainerTestBrowser.raise_()
        self.line_18.raise_()
        self.line.raise_()
        self.label.raise_()
        self.line_2.raise_()
        self.label_2.raise_()
        self.label_3.raise_()
        self.line_3.raise_()
        self.frame_2.raise_()
        self.frame_3.raise_()
        self.runSpotLocalization_pushButton.raise_()
        self.SpotAnalysisPanel_TransientSpotContainerClearChannelPushButton.raise_()
        self.frame_4.raise_()
        self.frame_5.raise_()
        self.frame_7.raise_()
        self.line_15.raise_()
        self.runSpotQualityControl_pushButton.raise_()
        self.label_15.raise_()
        self.spotLocalization_channelListWidget.raise_()
        self.spotVisualization_showSpotsPushButton.raise_()
        self.spotVisualization_hideSpotsPushButton.raise_()
        self.line_19.raise_()
        self.label_18.raise_()
        self.label_19.raise_()
        self.line_21.raise_()
        self.SpotAnalysisPanel_PermanentSpotContainerTestBrowser.raise_()
        self.line_22.raise_()
        self.SpotAnalysisPanel_PermanentSpotContainerSendPushButton.raise_()
        self.SpotAnalysisPanel_TransientSpotContainerSavePushButton.raise_()
        self.spotVisualization_TurnONManualModePushButton.raise_()
        self.spotVisualization_TurnOFFManualModePushButton.raise_()
        self.SpotAnalysisPanel_TransientSpotContainerClearAllPushButton.raise_()
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Spot Localization Parameters"))
        self.label_2.setText(_translate("Form", "Spot Quality Control Panel"))
        self.label_3.setText(_translate("Form", "Spot Visualization Panel"))
        self.label_4.setText(_translate("Form", "Minimum intensity threshold"))
        self.label_5.setText(_translate("Form", "Ratio"))
        self.label_6.setText(_translate("Form", "Abs"))
        self.label_7.setText(_translate("Form", "Minimum distance (pixel)"))
        self.runSpotLocalization_pushButton.setText(_translate("Form", "Run Spot Localization"))
        self.SpotAnalysisPanel_TransientSpotContainerClearChannelPushButton.setText(_translate("Form", "Clear channel"))
        self.label_11.setText(_translate("Form", "Minimum intensity threshold"))
        self.label_12.setText(_translate("Form", "Ratio"))
        self.label_13.setText(_translate("Form", "Abs"))
        self.label_14.setText(_translate("Form", "min spot size (default: 1)"))
        self.label_17.setText(_translate("Form", "max spot size (default: 100)"))
        self.runSpotQualityControl_pushButton.setText(_translate("Form", "Run Spot Q.C."))
        self.label_15.setText(_translate("Form", "Channel Selector"))
        self.spotVisualization_showSpotsPushButton.setText(_translate("Form", "Show"))
        self.spotVisualization_hideSpotsPushButton.setText(_translate("Form", "Hide"))
        self.label_18.setText(_translate("Form", "Transient Spot Container"))
        self.label_19.setText(_translate("Form", "Permanent Spot Container"))
        self.SpotAnalysisPanel_PermanentSpotContainerSendPushButton.setText(_translate("Form", "Send"))
        self.SpotAnalysisPanel_TransientSpotContainerSavePushButton.setText(_translate("Form", "Save"))
        self.spotVisualization_TurnONManualModePushButton.setText(_translate("Form", "Manual ON"))
        self.spotVisualization_TurnOFFManualModePushButton.setText(_translate("Form", "Manual OFF"))
        self.SpotAnalysisPanel_TransientSpotContainerClearAllPushButton.setText(_translate("Form", "Clear all"))

class AnalysisPanelUI(object):
    def setupUi(self, Analysis):
        Analysis.setObjectName("Analysis")
        Analysis.resize(1118, 888)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(Analysis)
        self.horizontalLayout_3.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout_3.setSpacing(5)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(5)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(5)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtWidgets.QGroupBox(Analysis)
        self.groupBox.setObjectName("groupBox")
        self.AnalysisPanelWorkBox_SpotDistanceAddConditionPushButton = QtWidgets.QPushButton(self.groupBox)
        self.AnalysisPanelWorkBox_SpotDistanceAddConditionPushButton.setGeometry(QtCore.QRect(380, 820, 141, 32))
        self.AnalysisPanelWorkBox_SpotDistanceAddConditionPushButton.setObjectName("AnalysisPanelWorkBox_SpotDistanceAddConditionPushButton")
        self.gridLayoutWidget_4 = QtWidgets.QWidget(self.groupBox)
        self.gridLayoutWidget_4.setGeometry(QtCore.QRect(10, 20, 511, 351))
        self.gridLayoutWidget_4.setObjectName("gridLayoutWidget_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.gridLayoutWidget_4)
        self.gridLayout_4.setContentsMargins(5, 5, 5, 5)
        self.gridLayout_4.setSpacing(5)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout_4.addWidget(self.label_2, 1, 0, 1, 1)
        self.line_3 = QtWidgets.QFrame(self.gridLayoutWidget_4)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.gridLayout_4.addWidget(self.line_3, 2, 0, 1, 2)
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout_4.addWidget(self.label_4, 1, 1, 1, 1)
        self.line_20 = QtWidgets.QFrame(self.gridLayoutWidget_4)
        self.line_20.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_20.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_20.setObjectName("line_20")
        self.gridLayout_4.addWidget(self.line_20, 0, 0, 1, 2)
        self.AnalysisPanelWorkBox_SpotContainerStateTextBrowser = QtWidgets.QTextBrowser(self.gridLayoutWidget_4)
        self.AnalysisPanelWorkBox_SpotContainerStateTextBrowser.setObjectName("AnalysisPanelWorkBox_SpotContainerStateTextBrowser")
        self.gridLayout_4.addWidget(self.AnalysisPanelWorkBox_SpotContainerStateTextBrowser, 3, 1, 1, 1)
        self.AnalysisPanelWorkBox_LinkSpotsAndCellsPushButton = QtWidgets.QPushButton(self.gridLayoutWidget_4)
        self.AnalysisPanelWorkBox_LinkSpotsAndCellsPushButton.setObjectName("AnalysisPanelWorkBox_LinkSpotsAndCellsPushButton")
        self.gridLayout_4.addWidget(self.AnalysisPanelWorkBox_LinkSpotsAndCellsPushButton, 5, 0, 1, 2)
        self.line_30 = QtWidgets.QFrame(self.gridLayoutWidget_4)
        self.line_30.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_30.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_30.setObjectName("line_30")
        self.gridLayout_4.addWidget(self.line_30, 4, 0, 1, 2)
        self.AnalysisPanelWorkBox_CellContainerStateTextBrowser = QtWidgets.QTextBrowser(self.gridLayoutWidget_4)
        self.AnalysisPanelWorkBox_CellContainerStateTextBrowser.setObjectName("AnalysisPanelWorkBox_CellContainerStateTextBrowser")
        self.gridLayout_4.addWidget(self.AnalysisPanelWorkBox_CellContainerStateTextBrowser, 3, 0, 1, 1)
        self.frame_6 = QtWidgets.QFrame(self.groupBox)
        self.frame_6.setGeometry(QtCore.QRect(10, 530, 311, 291))
        self.frame_6.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.label_13 = QtWidgets.QLabel(self.frame_6)
        self.label_13.setGeometry(QtCore.QRect(0, 0, 311, 31))
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.line_23 = QtWidgets.QFrame(self.frame_6)
        self.line_23.setGeometry(QtCore.QRect(0, 30, 466, 3))
        self.line_23.setLineWidth(2)
        self.line_23.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_23.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_23.setObjectName("line_23")
        self.label_14 = QtWidgets.QLabel(self.frame_6)
        self.label_14.setGeometry(QtCore.QRect(10, 40, 131, 16))
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.frame_6)
        self.label_15.setGeometry(QtCore.QRect(170, 40, 131, 16))
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.AnalysisPanelWorkBox_SpotDistanceChannel1ListWidget = QtWidgets.QListWidget(self.frame_6)
        self.AnalysisPanelWorkBox_SpotDistanceChannel1ListWidget.setGeometry(QtCore.QRect(10, 60, 131, 131))
        self.AnalysisPanelWorkBox_SpotDistanceChannel1ListWidget.setObjectName("AnalysisPanelWorkBox_SpotDistanceChannel1ListWidget")
        self.AnalysisPanelWorkBox_SpotDistanceChannel2ListWidget = QtWidgets.QListWidget(self.frame_6)
        self.AnalysisPanelWorkBox_SpotDistanceChannel2ListWidget.setGeometry(QtCore.QRect(170, 60, 131, 131))
        self.AnalysisPanelWorkBox_SpotDistanceChannel2ListWidget.setObjectName("AnalysisPanelWorkBox_SpotDistanceChannel2ListWidget")
        self.line_15 = QtWidgets.QFrame(self.frame_6)
        self.line_15.setGeometry(QtCore.QRect(140, 30, 31, 221))
        self.line_15.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_15.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_15.setObjectName("line_15")
        self.label_16 = QtWidgets.QLabel(self.frame_6)
        self.label_16.setGeometry(QtCore.QRect(10, 200, 131, 21))
        self.label_16.setAlignment(QtCore.Qt.AlignCenter)
        self.label_16.setObjectName("label_16")
        self.AnalysisPanelWorkBox_SpotDistanceMindistanceDoubleSpinBox = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.AnalysisPanelWorkBox_SpotDistanceMindistanceDoubleSpinBox.setGeometry(QtCore.QRect(30, 220, 101, 31))
        self.AnalysisPanelWorkBox_SpotDistanceMindistanceDoubleSpinBox.setMinimum(-1000000.0)
        self.AnalysisPanelWorkBox_SpotDistanceMindistanceDoubleSpinBox.setMaximum(1000000.0)
        self.AnalysisPanelWorkBox_SpotDistanceMindistanceDoubleSpinBox.setObjectName("AnalysisPanelWorkBox_SpotDistanceMindistanceDoubleSpinBox")
        self.label_17 = QtWidgets.QLabel(self.frame_6)
        self.label_17.setGeometry(QtCore.QRect(170, 200, 131, 20))
        self.label_17.setAlignment(QtCore.Qt.AlignCenter)
        self.label_17.setObjectName("label_17")
        self.AnalysisPanelWorkBox_SpotDistanceMaxdistanceDoubleSpinBox = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.AnalysisPanelWorkBox_SpotDistanceMaxdistanceDoubleSpinBox.setGeometry(QtCore.QRect(190, 220, 101, 31))
        self.AnalysisPanelWorkBox_SpotDistanceMaxdistanceDoubleSpinBox.setMinimum(-1000000.0)
        self.AnalysisPanelWorkBox_SpotDistanceMaxdistanceDoubleSpinBox.setMaximum(10000000.0)
        self.AnalysisPanelWorkBox_SpotDistanceMaxdistanceDoubleSpinBox.setProperty("value", 100000.0)
        self.AnalysisPanelWorkBox_SpotDistanceMaxdistanceDoubleSpinBox.setObjectName("AnalysisPanelWorkBox_SpotDistanceMaxdistanceDoubleSpinBox")
        self.AnalysisPanelWorkBox_SpotDistanceCenterToCenterRadioButton = QtWidgets.QRadioButton(self.frame_6)
        self.AnalysisPanelWorkBox_SpotDistanceCenterToCenterRadioButton.setGeometry(QtCore.QRect(90, 260, 131, 31))
        self.AnalysisPanelWorkBox_SpotDistanceCenterToCenterRadioButton.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.AnalysisPanelWorkBox_SpotDistanceCenterToCenterRadioButton.setChecked(True)
        self.AnalysisPanelWorkBox_SpotDistanceCenterToCenterRadioButton.setObjectName("AnalysisPanelWorkBox_SpotDistanceCenterToCenterRadioButton")
        self.line_2 = QtWidgets.QFrame(self.frame_6)
        self.line_2.setGeometry(QtCore.QRect(0, 240, 311, 31))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.line_4 = QtWidgets.QFrame(self.frame_6)
        self.line_4.setGeometry(QtCore.QRect(0, 180, 311, 31))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.frame_4 = QtWidgets.QFrame(self.groupBox)
        self.frame_4.setGeometry(QtCore.QRect(330, 680, 189, 139))
        self.frame_4.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.label_6 = QtWidgets.QLabel(self.frame_4)
        self.label_6.setGeometry(QtCore.QRect(0, 0, 191, 31))
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.line_5 = QtWidgets.QFrame(self.frame_4)
        self.line_5.setGeometry(QtCore.QRect(0, 30, 229, 3))
        self.line_5.setLineWidth(3)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.label_18 = QtWidgets.QLabel(self.frame_4)
        self.label_18.setGeometry(QtCore.QRect(10, 40, 61, 21))
        self.label_18.setObjectName("label_18")
        self.AnalysisPanelWorkBox_SpotSizesListWidget = QtWidgets.QListWidget(self.frame_4)
        self.AnalysisPanelWorkBox_SpotSizesListWidget.setGeometry(QtCore.QRect(10, 60, 81, 71))
        self.AnalysisPanelWorkBox_SpotSizesListWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.AnalysisPanelWorkBox_SpotSizesListWidget.setObjectName("AnalysisPanelWorkBox_SpotSizesListWidget")
        item = QtWidgets.QListWidgetItem()
        self.AnalysisPanelWorkBox_SpotSizesListWidget.addItem(item)
        self.label_7 = QtWidgets.QLabel(self.frame_4)
        self.label_7.setGeometry(QtCore.QRect(110, 40, 61, 21))
        self.label_7.setWordWrap(False)
        self.label_7.setObjectName("label_7")
        self.AnalysisPanelWorkBox_SpotSizesMinimumSpinBox = QtWidgets.QSpinBox(self.frame_4)
        self.AnalysisPanelWorkBox_SpotSizesMinimumSpinBox.setGeometry(QtCore.QRect(110, 60, 71, 24))
        self.AnalysisPanelWorkBox_SpotSizesMinimumSpinBox.setMaximum(1000000)
        self.AnalysisPanelWorkBox_SpotSizesMinimumSpinBox.setObjectName("AnalysisPanelWorkBox_SpotSizesMinimumSpinBox")
        self.AnalysisPanelWorkBox_SpotSizesMaximumSpinBox = QtWidgets.QSpinBox(self.frame_4)
        self.AnalysisPanelWorkBox_SpotSizesMaximumSpinBox.setGeometry(QtCore.QRect(110, 110, 71, 24))
        self.AnalysisPanelWorkBox_SpotSizesMaximumSpinBox.setMaximum(1000000)
        self.AnalysisPanelWorkBox_SpotSizesMaximumSpinBox.setProperty("value", 100000)
        self.AnalysisPanelWorkBox_SpotSizesMaximumSpinBox.setObjectName("AnalysisPanelWorkBox_SpotSizesMaximumSpinBox")
        self.label_9 = QtWidgets.QLabel(self.frame_4)
        self.label_9.setGeometry(QtCore.QRect(110, 90, 71, 21))
        self.label_9.setWordWrap(False)
        self.label_9.setObjectName("label_9")
        self.line_6 = QtWidgets.QFrame(self.frame_4)
        self.line_6.setGeometry(QtCore.QRect(90, 30, 16, 111))
        self.line_6.setLineWidth(2)
        self.line_6.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.frame_7 = QtWidgets.QFrame(self.groupBox)
        self.frame_7.setGeometry(QtCore.QRect(330, 530, 189, 139))
        self.frame_7.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.label_42 = QtWidgets.QLabel(self.frame_7)
        self.label_42.setGeometry(QtCore.QRect(0, 0, 191, 31))
        self.label_42.setAlignment(QtCore.Qt.AlignCenter)
        self.label_42.setObjectName("label_42")
        self.line_13 = QtWidgets.QFrame(self.frame_7)
        self.line_13.setGeometry(QtCore.QRect(0, 30, 229, 3))
        self.line_13.setLineWidth(3)
        self.line_13.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_13.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_13.setObjectName("line_13")
        self.label_43 = QtWidgets.QLabel(self.frame_7)
        self.label_43.setGeometry(QtCore.QRect(10, 40, 61, 21))
        self.label_43.setObjectName("label_43")
        self.label_44 = QtWidgets.QLabel(self.frame_7)
        self.label_44.setGeometry(QtCore.QRect(110, 40, 61, 21))
        self.label_44.setWordWrap(False)
        self.label_44.setObjectName("label_44")
        self.label_45 = QtWidgets.QLabel(self.frame_7)
        self.label_45.setGeometry(QtCore.QRect(110, 90, 71, 21))
        self.label_45.setWordWrap(False)
        self.label_45.setObjectName("label_45")
        self.line_14 = QtWidgets.QFrame(self.frame_7)
        self.line_14.setGeometry(QtCore.QRect(90, 30, 16, 111))
        self.line_14.setLineWidth(2)
        self.line_14.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_14.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_14.setObjectName("line_14")
        self.AnalysisPanelWorkBox_BrightnessListWidget = QtWidgets.QListWidget(self.frame_7)
        self.AnalysisPanelWorkBox_BrightnessListWidget.setGeometry(QtCore.QRect(10, 60, 81, 71))
        self.AnalysisPanelWorkBox_BrightnessListWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.AnalysisPanelWorkBox_BrightnessListWidget.setObjectName("AnalysisPanelWorkBox_BrightnessListWidget")
        item = QtWidgets.QListWidgetItem()
        self.AnalysisPanelWorkBox_BrightnessListWidget.addItem(item)
        self.AnalysisPanelWorkBox_BrightnessMinimumSpinBox = QtWidgets.QSpinBox(self.frame_7)
        self.AnalysisPanelWorkBox_BrightnessMinimumSpinBox.setGeometry(QtCore.QRect(110, 60, 71, 24))
        self.AnalysisPanelWorkBox_BrightnessMinimumSpinBox.setMaximum(1000000)
        self.AnalysisPanelWorkBox_BrightnessMinimumSpinBox.setObjectName("AnalysisPanelWorkBox_BrightnessMinimumSpinBox")
        self.AnalysisPanelWorkBox_BrightnessMaximumSpinBox = QtWidgets.QSpinBox(self.frame_7)
        self.AnalysisPanelWorkBox_BrightnessMaximumSpinBox.setGeometry(QtCore.QRect(110, 110, 71, 24))
        self.AnalysisPanelWorkBox_BrightnessMaximumSpinBox.setMaximum(1000000)
        self.AnalysisPanelWorkBox_BrightnessMaximumSpinBox.setProperty("value", 100000)
        self.AnalysisPanelWorkBox_BrightnessMaximumSpinBox.setObjectName("AnalysisPanelWorkBox_BrightnessMaximumSpinBox")
        self.frame_5 = QtWidgets.QFrame(self.groupBox)
        self.frame_5.setGeometry(QtCore.QRect(330, 380, 189, 139))
        self.frame_5.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.label_23 = QtWidgets.QLabel(self.frame_5)
        self.label_23.setGeometry(QtCore.QRect(0, 0, 191, 31))
        self.label_23.setAlignment(QtCore.Qt.AlignCenter)
        self.label_23.setObjectName("label_23")
        self.line_9 = QtWidgets.QFrame(self.frame_5)
        self.line_9.setGeometry(QtCore.QRect(0, 30, 229, 3))
        self.line_9.setLineWidth(3)
        self.line_9.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_9.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_9.setObjectName("line_9")
        self.label_35 = QtWidgets.QLabel(self.frame_5)
        self.label_35.setGeometry(QtCore.QRect(10, 40, 61, 21))
        self.label_35.setObjectName("label_35")
        self.label_36 = QtWidgets.QLabel(self.frame_5)
        self.label_36.setGeometry(QtCore.QRect(110, 40, 61, 21))
        self.label_36.setWordWrap(False)
        self.label_36.setObjectName("label_36")
        self.label_37 = QtWidgets.QLabel(self.frame_5)
        self.label_37.setGeometry(QtCore.QRect(110, 90, 71, 21))
        self.label_37.setWordWrap(False)
        self.label_37.setObjectName("label_37")
        self.line_10 = QtWidgets.QFrame(self.frame_5)
        self.line_10.setGeometry(QtCore.QRect(90, 30, 16, 111))
        self.line_10.setLineWidth(2)
        self.line_10.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_10.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_10.setObjectName("line_10")
        self.AnalysisPanelWorkBox_NumberOfSpotsListWidget = QtWidgets.QListWidget(self.frame_5)
        self.AnalysisPanelWorkBox_NumberOfSpotsListWidget.setGeometry(QtCore.QRect(10, 60, 81, 71))
        self.AnalysisPanelWorkBox_NumberOfSpotsListWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.AnalysisPanelWorkBox_NumberOfSpotsListWidget.setObjectName("AnalysisPanelWorkBox_NumberOfSpotsListWidget")
        item = QtWidgets.QListWidgetItem()
        self.AnalysisPanelWorkBox_NumberOfSpotsListWidget.addItem(item)
        self.AnalysisPanelWorkBox_NumberOfSpotsMinimumSpinBox = QtWidgets.QSpinBox(self.frame_5)
        self.AnalysisPanelWorkBox_NumberOfSpotsMinimumSpinBox.setGeometry(QtCore.QRect(110, 60, 71, 24))
        self.AnalysisPanelWorkBox_NumberOfSpotsMinimumSpinBox.setMaximum(1000000)
        self.AnalysisPanelWorkBox_NumberOfSpotsMinimumSpinBox.setObjectName("AnalysisPanelWorkBox_NumberOfSpotsMinimumSpinBox")
        self.AnalysisPanelWorkBox_NumberOfSpotsMaximumSpinBox = QtWidgets.QSpinBox(self.frame_5)
        self.AnalysisPanelWorkBox_NumberOfSpotsMaximumSpinBox.setGeometry(QtCore.QRect(110, 110, 71, 24))
        self.AnalysisPanelWorkBox_NumberOfSpotsMaximumSpinBox.setMaximum(1000000)
        self.AnalysisPanelWorkBox_NumberOfSpotsMaximumSpinBox.setProperty("value", 100000)
        self.AnalysisPanelWorkBox_NumberOfSpotsMaximumSpinBox.setObjectName("AnalysisPanelWorkBox_NumberOfSpotsMaximumSpinBox")
        self.frame_12 = QtWidgets.QFrame(self.groupBox)
        self.frame_12.setGeometry(QtCore.QRect(170, 380, 151, 141))
        self.frame_12.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_12.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_12.setObjectName("frame_12")
        self.label_5 = QtWidgets.QLabel(self.frame_12)
        self.label_5.setGeometry(QtCore.QRect(0, 0, 151, 31))
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.AnalysisPanelWorkBox_FieldOfViewsLabel = QtWidgets.QLabel(self.frame_12)
        self.AnalysisPanelWorkBox_FieldOfViewsLabel.setGeometry(QtCore.QRect(10, 30, 131, 20))
        self.AnalysisPanelWorkBox_FieldOfViewsLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.AnalysisPanelWorkBox_FieldOfViewsLabel.setFrameShadow(QtWidgets.QFrame.Plain)
        self.AnalysisPanelWorkBox_FieldOfViewsLabel.setLineWidth(1)
        self.AnalysisPanelWorkBox_FieldOfViewsLabel.setObjectName("AnalysisPanelWorkBox_FieldOfViewsLabel")
        self.AnalysisPanelWorkBox_FieldOfViewsListWidget = QtWidgets.QListWidget(self.frame_12)
        self.AnalysisPanelWorkBox_FieldOfViewsListWidget.setGeometry(QtCore.QRect(10, 60, 131, 71))
        self.AnalysisPanelWorkBox_FieldOfViewsListWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.AnalysisPanelWorkBox_FieldOfViewsListWidget.setObjectName("AnalysisPanelWorkBox_FieldOfViewsListWidget")
        item = QtWidgets.QListWidgetItem()
        self.AnalysisPanelWorkBox_FieldOfViewsListWidget.addItem(item)
        self.frame_13 = QtWidgets.QFrame(self.groupBox)
        self.frame_13.setGeometry(QtCore.QRect(10, 380, 151, 141))
        self.frame_13.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_13.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_13.setObjectName("frame_13")
        self.label_3 = QtWidgets.QLabel(self.frame_13)
        self.label_3.setGeometry(QtCore.QRect(0, 0, 141, 31))
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_8 = QtWidgets.QLabel(self.frame_13)
        self.label_8.setGeometry(QtCore.QRect(10, 30, 131, 20))
        self.label_8.setFrameShape(QtWidgets.QFrame.Box)
        self.label_8.setObjectName("label_8")
        self.AnalysisPanelWorkBox_CellTypesListWidget = QtWidgets.QListWidget(self.frame_13)
        self.AnalysisPanelWorkBox_CellTypesListWidget.setGeometry(QtCore.QRect(10, 60, 131, 71))
        self.AnalysisPanelWorkBox_CellTypesListWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.AnalysisPanelWorkBox_CellTypesListWidget.setObjectName("AnalysisPanelWorkBox_CellTypesListWidget")
        item = QtWidgets.QListWidgetItem()
        self.AnalysisPanelWorkBox_CellTypesListWidget.addItem(item)
        self.verticalLayout.addWidget(self.groupBox)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.line = QtWidgets.QFrame(Analysis)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout.addWidget(self.line)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSpacing(5)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.frame = QtWidgets.QFrame(Analysis)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(20, 0, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label.setFont(font)
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setObjectName("label")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setGeometry(QtCore.QRect(10, 20, 511, 351))
        self.frame_2.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.AnalysisPanelFrame_ConditionLineEdit = QtWidgets.QLineEdit(self.frame_2)
        self.AnalysisPanelFrame_ConditionLineEdit.setGeometry(QtCore.QRect(10, 310, 331, 31))
        self.AnalysisPanelFrame_ConditionLineEdit.setObjectName("AnalysisPanelFrame_ConditionLineEdit")
        self.AnalysisPanelFrame_ConditionRemoveConditionPushButton = QtWidgets.QPushButton(self.frame_2)
        self.AnalysisPanelFrame_ConditionRemoveConditionPushButton.setGeometry(QtCore.QRect(340, 310, 81, 32))
        self.AnalysisPanelFrame_ConditionRemoveConditionPushButton.setObjectName("AnalysisPanelFrame_ConditionRemoveConditionPushButton")
        self.AnalysisPanelFrame_ExportCellsAndSpotsPushButton = QtWidgets.QPushButton(self.frame_2)
        self.AnalysisPanelFrame_ExportCellsAndSpotsPushButton.setGeometry(QtCore.QRect(420, 310, 81, 32))
        self.AnalysisPanelFrame_ExportCellsAndSpotsPushButton.setObjectName("AnalysisPanelFrame_ExportCellsAndSpotsPushButton")
        self.AnalysisPanelFrame_ConditionTextBrowser = QtWidgets.QTextBrowser(self.frame_2)
        self.AnalysisPanelFrame_ConditionTextBrowser.setGeometry(QtCore.QRect(10, 10, 491, 291))
        self.AnalysisPanelFrame_ConditionTextBrowser.setFrameShape(QtWidgets.QFrame.Box)
        self.AnalysisPanelFrame_ConditionTextBrowser.setFrameShadow(QtWidgets.QFrame.Plain)
        self.AnalysisPanelFrame_ConditionTextBrowser.setObjectName("AnalysisPanelFrame_ConditionTextBrowser")
        self.frame_3 = QtWidgets.QFrame(self.frame)
        self.frame_3.setGeometry(QtCore.QRect(10, 380, 511, 91))
        self.frame_3.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.label_22 = QtWidgets.QLabel(self.frame_3)
        self.label_22.setGeometry(QtCore.QRect(10, 0, 131, 21))
        self.label_22.setObjectName("label_22")
        self.AnalysisPanelFrame_targetCellsListWidget = QtWidgets.QListWidget(self.frame_3)
        self.AnalysisPanelFrame_targetCellsListWidget.setGeometry(QtCore.QRect(10, 20, 421, 61))
        self.AnalysisPanelFrame_targetCellsListWidget.setObjectName("AnalysisPanelFrame_targetCellsListWidget")
        self.AnalysisPanelFrame_ShowSelectedCellPushButton = QtWidgets.QPushButton(self.frame_3)
        self.AnalysisPanelFrame_ShowSelectedCellPushButton.setGeometry(QtCore.QRect(430, 20, 81, 32))
        self.AnalysisPanelFrame_ShowSelectedCellPushButton.setObjectName("AnalysisPanelFrame_ShowSelectedCellPushButton")
        self.AnalysisPanelFrame_ExportSelectedCellPushButton = QtWidgets.QPushButton(self.frame_3)
        self.AnalysisPanelFrame_ExportSelectedCellPushButton.setGeometry(QtCore.QRect(430, 50, 81, 32))
        self.AnalysisPanelFrame_ExportSelectedCellPushButton.setObjectName("AnalysisPanelFrame_ExportSelectedCellPushButton")
        self.frame_8 = QtWidgets.QFrame(self.frame)
        self.frame_8.setGeometry(QtCore.QRect(10, 480, 151, 161))
        self.frame_8.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.label_26 = QtWidgets.QLabel(self.frame_8)
        self.label_26.setGeometry(QtCore.QRect(0, 4, 151, 21))
        self.label_26.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_26.setAlignment(QtCore.Qt.AlignCenter)
        self.label_26.setObjectName("label_26")
        self.label_27 = QtWidgets.QLabel(self.frame_8)
        self.label_27.setGeometry(QtCore.QRect(0, 30, 151, 16))
        self.label_27.setAlignment(QtCore.Qt.AlignCenter)
        self.label_27.setObjectName("label_27")
        self.AnalysisPanelFrame_CountSpotsPerCellChannelListWidget = QtWidgets.QListWidget(self.frame_8)
        self.AnalysisPanelFrame_CountSpotsPerCellChannelListWidget.setGeometry(QtCore.QRect(10, 50, 131, 81))
        self.AnalysisPanelFrame_CountSpotsPerCellChannelListWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.AnalysisPanelFrame_CountSpotsPerCellChannelListWidget.setObjectName("AnalysisPanelFrame_CountSpotsPerCellChannelListWidget")
        item = QtWidgets.QListWidgetItem()
        self.AnalysisPanelFrame_CountSpotsPerCellChannelListWidget.addItem(item)
        self.AnalysisPanelFrame_CountSpotsPerCellPushButton = QtWidgets.QPushButton(self.frame_8)
        self.AnalysisPanelFrame_CountSpotsPerCellPushButton.setGeometry(QtCore.QRect(10, 130, 131, 32))
        self.AnalysisPanelFrame_CountSpotsPerCellPushButton.setObjectName("AnalysisPanelFrame_CountSpotsPerCellPushButton")
        self.line_8 = QtWidgets.QFrame(self.frame_8)
        self.line_8.setGeometry(QtCore.QRect(0, 20, 151, 16))
        self.line_8.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.frame_9 = QtWidgets.QFrame(self.frame)
        self.frame_9.setGeometry(QtCore.QRect(170, 480, 191, 161))
        self.frame_9.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.label_24 = QtWidgets.QLabel(self.frame_9)
        self.label_24.setGeometry(QtCore.QRect(0, 5, 191, 16))
        self.label_24.setAlignment(QtCore.Qt.AlignCenter)
        self.label_24.setObjectName("label_24")
        self.label_25 = QtWidgets.QLabel(self.frame_9)
        self.label_25.setGeometry(QtCore.QRect(10, 30, 87, 16))
        self.label_25.setAlignment(QtCore.Qt.AlignCenter)
        self.label_25.setObjectName("label_25")
        self.AnalysisPanelFrame_CalculateSpotSizesChannelListWidget = QtWidgets.QListWidget(self.frame_9)
        self.AnalysisPanelFrame_CalculateSpotSizesChannelListWidget.setGeometry(QtCore.QRect(10, 50, 85, 101))
        self.AnalysisPanelFrame_CalculateSpotSizesChannelListWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.AnalysisPanelFrame_CalculateSpotSizesChannelListWidget.setObjectName("AnalysisPanelFrame_CalculateSpotSizesChannelListWidget")
        item = QtWidgets.QListWidgetItem()
        self.AnalysisPanelFrame_CalculateSpotSizesChannelListWidget.addItem(item)
        self.line_7 = QtWidgets.QFrame(self.frame_9)
        self.line_7.setGeometry(QtCore.QRect(100, 30, 16, 141))
        self.line_7.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.label_20 = QtWidgets.QLabel(self.frame_9)
        self.label_20.setGeometry(QtCore.QRect(110, 30, 80, 16))
        self.label_20.setAlignment(QtCore.Qt.AlignCenter)
        self.label_20.setObjectName("label_20")
        self.AnalysisPanelFrame_CalculateSpotSizesThresholdDoubleSpinBox = QtWidgets.QDoubleSpinBox(self.frame_9)
        self.AnalysisPanelFrame_CalculateSpotSizesThresholdDoubleSpinBox.setGeometry(QtCore.QRect(120, 50, 61, 24))
        self.AnalysisPanelFrame_CalculateSpotSizesThresholdDoubleSpinBox.setMaximum(100.0)
        self.AnalysisPanelFrame_CalculateSpotSizesThresholdDoubleSpinBox.setProperty("value", 80.0)
        self.AnalysisPanelFrame_CalculateSpotSizesThresholdDoubleSpinBox.setObjectName("AnalysisPanelFrame_CalculateSpotSizesThresholdDoubleSpinBox")
        self.AnalysisPanelFrame_CalculateSpotSizesPushButton = QtWidgets.QPushButton(self.frame_9)
        self.AnalysisPanelFrame_CalculateSpotSizesPushButton.setGeometry(QtCore.QRect(110, 90, 81, 32))
        self.AnalysisPanelFrame_CalculateSpotSizesPushButton.setObjectName("AnalysisPanelFrame_CalculateSpotSizesPushButton")
        self.AnalysisPanelFrame_InitializePushButton = QtWidgets.QPushButton(self.frame_9)
        self.AnalysisPanelFrame_InitializePushButton.setGeometry(QtCore.QRect(110, 120, 81, 32))
        self.AnalysisPanelFrame_InitializePushButton.setObjectName("AnalysisPanelFrame_InitializePushButton")
        self.line_11 = QtWidgets.QFrame(self.frame_9)
        self.line_11.setGeometry(QtCore.QRect(0, 20, 191, 16))
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")
        self.line_12 = QtWidgets.QFrame(self.frame_9)
        self.line_12.setGeometry(QtCore.QRect(110, 80, 118, 3))
        self.line_12.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_12.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_12.setObjectName("line_12")
        self.frame_10 = QtWidgets.QFrame(self.frame)
        self.frame_10.setGeometry(QtCore.QRect(370, 480, 151, 161))
        self.frame_10.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.label_33 = QtWidgets.QLabel(self.frame_10)
        self.label_33.setGeometry(QtCore.QRect(0, 0, 151, 31))
        self.label_33.setAlignment(QtCore.Qt.AlignCenter)
        self.label_33.setObjectName("label_33")
        self.line_16 = QtWidgets.QFrame(self.frame_10)
        self.line_16.setGeometry(QtCore.QRect(-10, 20, 181, 16))
        self.line_16.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_16.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_16.setObjectName("line_16")
        self.label_34 = QtWidgets.QLabel(self.frame_10)
        self.label_34.setGeometry(QtCore.QRect(10, 30, 131, 16))
        self.label_34.setAlignment(QtCore.Qt.AlignCenter)
        self.label_34.setObjectName("label_34")
        self.AnalysisPanelFrame_ShowSpotBrightnessListWidget = QtWidgets.QListWidget(self.frame_10)
        self.AnalysisPanelFrame_ShowSpotBrightnessListWidget.setGeometry(QtCore.QRect(10, 50, 131, 81))
        self.AnalysisPanelFrame_ShowSpotBrightnessListWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.AnalysisPanelFrame_ShowSpotBrightnessListWidget.setObjectName("AnalysisPanelFrame_ShowSpotBrightnessListWidget")
        item = QtWidgets.QListWidgetItem()
        self.AnalysisPanelFrame_ShowSpotBrightnessListWidget.addItem(item)
        self.AnalysisPanelFrame_showSpotBrightnessShowPushButton = QtWidgets.QPushButton(self.frame_10)
        self.AnalysisPanelFrame_showSpotBrightnessShowPushButton.setGeometry(QtCore.QRect(10, 130, 131, 32))
        self.AnalysisPanelFrame_showSpotBrightnessShowPushButton.setObjectName("AnalysisPanelFrame_showSpotBrightnessShowPushButton")
        self.frame_11 = QtWidgets.QFrame(self.frame)
        self.frame_11.setGeometry(QtCore.QRect(110, 650, 301, 191))
        self.frame_11.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_11.setObjectName("frame_11")
        self.label_32 = QtWidgets.QLabel(self.frame_11)
        self.label_32.setGeometry(QtCore.QRect(0, 5, 301, 21))
        self.label_32.setAlignment(QtCore.Qt.AlignCenter)
        self.label_32.setObjectName("label_32")
        self.line_17 = QtWidgets.QFrame(self.frame_11)
        self.line_17.setGeometry(QtCore.QRect(0, 20, 511, 16))
        self.line_17.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_17.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_17.setObjectName("line_17")
        self.label_29 = QtWidgets.QLabel(self.frame_11)
        self.label_29.setGeometry(QtCore.QRect(10, 30, 131, 16))
        self.label_29.setAlignment(QtCore.Qt.AlignCenter)
        self.label_29.setObjectName("label_29")
        self.label_28 = QtWidgets.QLabel(self.frame_11)
        self.label_28.setGeometry(QtCore.QRect(160, 30, 124, 16))
        self.label_28.setAlignment(QtCore.Qt.AlignCenter)
        self.label_28.setObjectName("label_28")
        self.AnalysisPanelFrame_MeasureSpotDistanceChannel1ListWidget = QtWidgets.QListWidget(self.frame_11)
        self.AnalysisPanelFrame_MeasureSpotDistanceChannel1ListWidget.setGeometry(QtCore.QRect(10, 50, 131, 101))
        self.AnalysisPanelFrame_MeasureSpotDistanceChannel1ListWidget.setObjectName("AnalysisPanelFrame_MeasureSpotDistanceChannel1ListWidget")
        self.AnalysisPanelFrame_MeasureSpotDistanceChannel2ListWidget = QtWidgets.QListWidget(self.frame_11)
        self.AnalysisPanelFrame_MeasureSpotDistanceChannel2ListWidget.setGeometry(QtCore.QRect(160, 50, 131, 101))
        self.AnalysisPanelFrame_MeasureSpotDistanceChannel2ListWidget.setObjectName("AnalysisPanelFrame_MeasureSpotDistanceChannel2ListWidget")
        self.line_19 = QtWidgets.QFrame(self.frame_11)
        self.line_19.setGeometry(QtCore.QRect(140, 30, 21, 201))
        self.line_19.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_19.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_19.setObjectName("line_19")
        self.AnalysisPanelFrame_MeasureSpotDistanceCenterToCenterRadioButton = QtWidgets.QRadioButton(self.frame_11)
        self.AnalysisPanelFrame_MeasureSpotDistanceCenterToCenterRadioButton.setGeometry(QtCore.QRect(10, 160, 131, 31))
        self.AnalysisPanelFrame_MeasureSpotDistanceCenterToCenterRadioButton.setChecked(True)
        self.AnalysisPanelFrame_MeasureSpotDistanceCenterToCenterRadioButton.setObjectName("AnalysisPanelFrame_MeasureSpotDistanceCenterToCenterRadioButton")
        self.AnalysisPanelFrame_MeasureSpotDistancePushButton = QtWidgets.QPushButton(self.frame_11)
        self.AnalysisPanelFrame_MeasureSpotDistancePushButton.setGeometry(QtCore.QRect(160, 160, 131, 32))
        self.AnalysisPanelFrame_MeasureSpotDistancePushButton.setObjectName("AnalysisPanelFrame_MeasureSpotDistancePushButton")
        self.line_21 = QtWidgets.QFrame(self.frame_11)
        self.line_21.setGeometry(QtCore.QRect(0, 150, 301, 21))
        self.line_21.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_21.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_21.setObjectName("line_21")
        self.verticalLayout_2.addWidget(self.frame)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.horizontalLayout_3.addLayout(self.horizontalLayout)

        self.retranslateUi(Analysis)
        self.AnalysisPanelWorkBox_SpotSizesListWidget.setCurrentRow(0)
        self.AnalysisPanelWorkBox_BrightnessListWidget.setCurrentRow(0)
        self.AnalysisPanelWorkBox_NumberOfSpotsListWidget.setCurrentRow(0)
        self.AnalysisPanelWorkBox_FieldOfViewsListWidget.setCurrentRow(0)
        self.AnalysisPanelWorkBox_CellTypesListWidget.setCurrentRow(0)
        QtCore.QMetaObject.connectSlotsByName(Analysis)

    def retranslateUi(self, Analysis):
        _translate = QtCore.QCoreApplication.translate
        Analysis.setWindowTitle(_translate("Analysis", "AnalysisPanel"))
        self.groupBox.setTitle(_translate("Analysis", " WorkBox"))
        self.AnalysisPanelWorkBox_SpotDistanceAddConditionPushButton.setText(_translate("Analysis", "Add Condition"))
        self.label_2.setText(_translate("Analysis", "CellContainer"))
        self.label_4.setText(_translate("Analysis", "SpotContainer"))
        self.AnalysisPanelWorkBox_LinkSpotsAndCellsPushButton.setText(_translate("Analysis", "Link spots and cells"))
        self.label_13.setText(_translate("Analysis", "Spot Distance "))
        self.label_14.setText(_translate("Analysis", "Channel 1"))
        self.label_15.setText(_translate("Analysis", "Channel 2"))
        self.label_16.setText(_translate("Analysis", "Minimum (px)"))
        self.label_17.setText(_translate("Analysis", "Maximum (px)"))
        self.AnalysisPanelWorkBox_SpotDistanceCenterToCenterRadioButton.setText(_translate("Analysis", "CenterToCenter"))
        self.label_6.setText(_translate("Analysis", "Spot Sizes"))
        self.label_18.setText(_translate("Analysis", "Channel"))
        __sortingEnabled = self.AnalysisPanelWorkBox_SpotSizesListWidget.isSortingEnabled()
        self.AnalysisPanelWorkBox_SpotSizesListWidget.setSortingEnabled(False)
        item = self.AnalysisPanelWorkBox_SpotSizesListWidget.item(0)
        item.setText(_translate("Analysis", "All"))
        self.AnalysisPanelWorkBox_SpotSizesListWidget.setSortingEnabled(__sortingEnabled)
        self.label_7.setText(_translate("Analysis", "Minimum"))
        self.label_9.setText(_translate("Analysis", "Maximum"))
        self.label_42.setText(_translate("Analysis", "Brightness"))
        self.label_43.setText(_translate("Analysis", "Channel"))
        self.label_44.setText(_translate("Analysis", "Minimum"))
        self.label_45.setText(_translate("Analysis", "Maximum"))
        __sortingEnabled = self.AnalysisPanelWorkBox_BrightnessListWidget.isSortingEnabled()
        self.AnalysisPanelWorkBox_BrightnessListWidget.setSortingEnabled(False)
        item = self.AnalysisPanelWorkBox_BrightnessListWidget.item(0)
        item.setText(_translate("Analysis", "All"))
        self.AnalysisPanelWorkBox_BrightnessListWidget.setSortingEnabled(__sortingEnabled)
        self.label_23.setText(_translate("Analysis", "Number of Spots"))
        self.label_35.setText(_translate("Analysis", "Channel"))
        self.label_36.setText(_translate("Analysis", "Minimum"))
        self.label_37.setText(_translate("Analysis", "Maximum"))
        __sortingEnabled = self.AnalysisPanelWorkBox_NumberOfSpotsListWidget.isSortingEnabled()
        self.AnalysisPanelWorkBox_NumberOfSpotsListWidget.setSortingEnabled(False)
        item = self.AnalysisPanelWorkBox_NumberOfSpotsListWidget.item(0)
        item.setText(_translate("Analysis", "All"))
        self.AnalysisPanelWorkBox_NumberOfSpotsListWidget.setSortingEnabled(__sortingEnabled)
        self.label_5.setText(_translate("Analysis", "Field Of Views"))
        self.AnalysisPanelWorkBox_FieldOfViewsLabel.setText(_translate("Analysis", "All"))
        __sortingEnabled = self.AnalysisPanelWorkBox_FieldOfViewsListWidget.isSortingEnabled()
        self.AnalysisPanelWorkBox_FieldOfViewsListWidget.setSortingEnabled(False)
        item = self.AnalysisPanelWorkBox_FieldOfViewsListWidget.item(0)
        item.setText(_translate("Analysis", "All"))
        self.AnalysisPanelWorkBox_FieldOfViewsListWidget.setSortingEnabled(__sortingEnabled)
        self.label_3.setText(_translate("Analysis", "Cell types"))
        self.label_8.setText(_translate("Analysis", "All"))
        __sortingEnabled = self.AnalysisPanelWorkBox_CellTypesListWidget.isSortingEnabled()
        self.AnalysisPanelWorkBox_CellTypesListWidget.setSortingEnabled(False)
        item = self.AnalysisPanelWorkBox_CellTypesListWidget.item(0)
        item.setText(_translate("Analysis", "All"))
        self.AnalysisPanelWorkBox_CellTypesListWidget.setSortingEnabled(__sortingEnabled)
        self.label.setText(_translate("Analysis", "Conditions"))
        self.AnalysisPanelFrame_ConditionRemoveConditionPushButton.setText(_translate("Analysis", "Remove"))
        self.AnalysisPanelFrame_ExportCellsAndSpotsPushButton.setText(_translate("Analysis", "Export"))
        self.AnalysisPanelFrame_ConditionTextBrowser.setHtml(_translate("Analysis", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'.AppleSystemUIFont\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.label_22.setText(_translate("Analysis", "Show selected cells"))
        self.AnalysisPanelFrame_ShowSelectedCellPushButton.setText(_translate("Analysis", "Show"))
        self.AnalysisPanelFrame_ExportSelectedCellPushButton.setText(_translate("Analysis", "Export"))
        self.label_26.setText(_translate("Analysis", "Count Spots Per Cell"))
        self.label_27.setText(_translate("Analysis", "Channel"))
        __sortingEnabled = self.AnalysisPanelFrame_CountSpotsPerCellChannelListWidget.isSortingEnabled()
        self.AnalysisPanelFrame_CountSpotsPerCellChannelListWidget.setSortingEnabled(False)
        item = self.AnalysisPanelFrame_CountSpotsPerCellChannelListWidget.item(0)
        item.setText(_translate("Analysis", "All"))
        self.AnalysisPanelFrame_CountSpotsPerCellChannelListWidget.setSortingEnabled(__sortingEnabled)
        self.AnalysisPanelFrame_CountSpotsPerCellPushButton.setText(_translate("Analysis", "Count"))
        self.label_24.setText(_translate("Analysis", "Calculate Spot Sizes"))
        self.label_25.setText(_translate("Analysis", "Channel"))
        __sortingEnabled = self.AnalysisPanelFrame_CalculateSpotSizesChannelListWidget.isSortingEnabled()
        self.AnalysisPanelFrame_CalculateSpotSizesChannelListWidget.setSortingEnabled(False)
        item = self.AnalysisPanelFrame_CalculateSpotSizesChannelListWidget.item(0)
        item.setText(_translate("Analysis", "All"))
        self.AnalysisPanelFrame_CalculateSpotSizesChannelListWidget.setSortingEnabled(__sortingEnabled)
        self.label_20.setText(_translate("Analysis", "Threshold"))
        self.AnalysisPanelFrame_CalculateSpotSizesPushButton.setText(_translate("Analysis", "Calculate"))
        self.AnalysisPanelFrame_InitializePushButton.setText(_translate("Analysis", "Initialize"))
        self.label_33.setText(_translate("Analysis", "Show Spot Brightness"))
        self.label_34.setText(_translate("Analysis", "Channel"))
        __sortingEnabled = self.AnalysisPanelFrame_ShowSpotBrightnessListWidget.isSortingEnabled()
        self.AnalysisPanelFrame_ShowSpotBrightnessListWidget.setSortingEnabled(False)
        item = self.AnalysisPanelFrame_ShowSpotBrightnessListWidget.item(0)
        item.setText(_translate("Analysis", "All"))
        self.AnalysisPanelFrame_ShowSpotBrightnessListWidget.setSortingEnabled(__sortingEnabled)
        self.AnalysisPanelFrame_showSpotBrightnessShowPushButton.setText(_translate("Analysis", "Show"))
        self.label_32.setText(_translate("Analysis", "Measure Spot Distance "))
        self.label_29.setText(_translate("Analysis", "Channel 1"))
        self.label_28.setText(_translate("Analysis", "Channel 2"))
        self.AnalysisPanelFrame_MeasureSpotDistanceCenterToCenterRadioButton.setText(_translate("Analysis", "CenterToCenter"))
        self.AnalysisPanelFrame_MeasureSpotDistancePushButton.setText(_translate("Analysis", "Measure"))

class CelltypeDeterminationUI(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(526, 674)
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(Form)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setGeometry(QtCore.QRect(10, 10, 181, 251))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.label = QtWidgets.QLabel(self.frame_2)
        self.label.setGeometry(QtCore.QRect(0, 0, 181, 31))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.CellTypeDeterminePanel_CelltypeRemoveSelectedPushButton = QtWidgets.QPushButton(self.frame_2)
        self.CellTypeDeterminePanel_CelltypeRemoveSelectedPushButton.setGeometry(QtCore.QRect(10, 130, 161, 31))
        self.CellTypeDeterminePanel_CelltypeRemoveSelectedPushButton.setObjectName("CellTypeDeterminePanel_CelltypeRemoveSelectedPushButton")
        self.CellTypeDeterminePanel_CelltypeAddTextEdit = QtWidgets.QTextEdit(self.frame_2)
        self.CellTypeDeterminePanel_CelltypeAddTextEdit.setGeometry(QtCore.QRect(10, 170, 161, 20))
        self.CellTypeDeterminePanel_CelltypeAddTextEdit.setObjectName("CellTypeDeterminePanel_CelltypeAddTextEdit")
        self.CellTypeDeterminePanel_CelltypeAddnewcelltypePushButton = QtWidgets.QPushButton(self.frame_2)
        self.CellTypeDeterminePanel_CelltypeAddnewcelltypePushButton.setGeometry(QtCore.QRect(10, 190, 161, 31))
        self.CellTypeDeterminePanel_CelltypeAddnewcelltypePushButton.setObjectName("CellTypeDeterminePanel_CelltypeAddnewcelltypePushButton")
        self.line = QtWidgets.QFrame(self.frame_2)
        self.line.setGeometry(QtCore.QRect(0, 150, 181, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.frame_2)
        self.line_2.setGeometry(QtCore.QRect(0, 210, 181, 21))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.CellTypeDeterminePanel_FOVPushButton = QtWidgets.QPushButton(self.frame_2)
        self.CellTypeDeterminePanel_FOVPushButton.setEnabled(False)
        self.CellTypeDeterminePanel_FOVPushButton.setGeometry(QtCore.QRect(0, 220, 71, 31))
        self.CellTypeDeterminePanel_FOVPushButton.setObjectName("CellTypeDeterminePanel_FOVPushButton")
        self.CellTypeDeterminePanel_CellBarcodePushButton = QtWidgets.QPushButton(self.frame_2)
        self.CellTypeDeterminePanel_CellBarcodePushButton.setEnabled(True)
        self.CellTypeDeterminePanel_CellBarcodePushButton.setGeometry(QtCore.QRect(70, 220, 111, 31))
        self.CellTypeDeterminePanel_CellBarcodePushButton.setObjectName("CellTypeDeterminePanel_CellBarcodePushButton")
        self.line_5 = QtWidgets.QFrame(self.frame_2)
        self.line_5.setGeometry(QtCore.QRect(0, 20, 181, 16))
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.CellTypeDeterminePanel_CelltypeListWidget = QtWidgets.QListWidget(self.frame_2)
        self.CellTypeDeterminePanel_CelltypeListWidget.setGeometry(QtCore.QRect(10, 30, 161, 101))
        self.CellTypeDeterminePanel_CelltypeListWidget.setObjectName("CellTypeDeterminePanel_CelltypeListWidget")
        self.frame_3 = QtWidgets.QFrame(self.frame)
        self.frame_3.setGeometry(QtCore.QRect(200, 10, 291, 251))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.label_2 = QtWidgets.QLabel(self.frame_3)
        self.label_2.setGeometry(QtCore.QRect(0, 0, 61, 31))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.line_3 = QtWidgets.QFrame(self.frame_3)
        self.line_3.setGeometry(QtCore.QRect(0, 210, 291, 21))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.line_4 = QtWidgets.QFrame(self.frame_3)
        self.line_4.setGeometry(QtCore.QRect(0, 20, 291, 16))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.label_3 = QtWidgets.QLabel(self.frame_3)
        self.label_3.setGeometry(QtCore.QRect(10, 30, 131, 31))
        self.label_3.setObjectName("label_3")
        self.CellTypeDeterminePanel_FOVModulateSelectedCellTypePushButton = QtWidgets.QPushButton(self.frame_3)
        self.CellTypeDeterminePanel_FOVModulateSelectedCellTypePushButton.setEnabled(True)
        self.CellTypeDeterminePanel_FOVModulateSelectedCellTypePushButton.setGeometry(QtCore.QRect(10, 220, 271, 31))
        self.CellTypeDeterminePanel_FOVModulateSelectedCellTypePushButton.setObjectName("CellTypeDeterminePanel_FOVModulateSelectedCellTypePushButton")
        self.CellTypeDeterminePanel_FOVCelltypesListWidget = QtWidgets.QListWidget(self.frame_3)
        self.CellTypeDeterminePanel_FOVCelltypesListWidget.setGeometry(QtCore.QRect(10, 60, 131, 151))
        self.CellTypeDeterminePanel_FOVCelltypesListWidget.setObjectName("CellTypeDeterminePanel_FOVCelltypesListWidget")
        self.CellTypeDeterminePanel_FOVListWidget = QtWidgets.QListWidget(self.frame_3)
        self.CellTypeDeterminePanel_FOVListWidget.setGeometry(QtCore.QRect(150, 60, 131, 151))
        self.CellTypeDeterminePanel_FOVListWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.CellTypeDeterminePanel_FOVListWidget.setObjectName("CellTypeDeterminePanel_FOVListWidget")
        self.CellTypeDeterminePanel_FOVLineEdit = QtWidgets.QLineEdit(self.frame_3)
        self.CellTypeDeterminePanel_FOVLineEdit.setGeometry(QtCore.QRect(150, 30, 131, 21))
        self.CellTypeDeterminePanel_FOVLineEdit.setObjectName("CellTypeDeterminePanel_FOVLineEdit")
        self.frame_4 = QtWidgets.QFrame(self.frame)
        self.frame_4.setGeometry(QtCore.QRect(10, 270, 481, 371))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.label_5 = QtWidgets.QLabel(self.frame_4)
        self.label_5.setGeometry(QtCore.QRect(0, 0, 101, 31))
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.line_6 = QtWidgets.QFrame(self.frame_4)
        self.line_6.setGeometry(QtCore.QRect(0, 20, 481, 16))
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.label_6 = QtWidgets.QLabel(self.frame_4)
        self.label_6.setGeometry(QtCore.QRect(10, 30, 131, 31))
        self.label_6.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.frame_4)
        self.label_7.setGeometry(QtCore.QRect(150, 30, 321, 31))
        self.label_7.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_7.setObjectName("label_7")
        self.CellTypeDeterminePanel_CellBarcodeUpdatePushButton = QtWidgets.QPushButton(self.frame_4)
        self.CellTypeDeterminePanel_CellBarcodeUpdatePushButton.setGeometry(QtCore.QRect(10, 310, 131, 31))
        self.CellTypeDeterminePanel_CellBarcodeUpdatePushButton.setObjectName("CellTypeDeterminePanel_CellBarcodeUpdatePushButton")
        self.CellTypeDeterminePanel_CellBarcodeSetupPushButton = QtWidgets.QPushButton(self.frame_4)
        self.CellTypeDeterminePanel_CellBarcodeSetupPushButton.setGeometry(QtCore.QRect(200, 310, 231, 31))
        self.CellTypeDeterminePanel_CellBarcodeSetupPushButton.setObjectName("CellTypeDeterminePanel_CellBarcodeSetupPushButton")
        self.CellTypeDeterminePanel_CellBarcodeChannelListWidget = QtWidgets.QListWidget(self.frame_4)
        self.CellTypeDeterminePanel_CellBarcodeChannelListWidget.setGeometry(QtCore.QRect(10, 60, 131, 71))
        self.CellTypeDeterminePanel_CellBarcodeChannelListWidget.setObjectName("CellTypeDeterminePanel_CellBarcodeChannelListWidget")
        self.CellTypeDeterminePanel_CellBarcodeTotalChannelListWidget = QtWidgets.QListWidget(self.frame_4)
        self.CellTypeDeterminePanel_CellBarcodeTotalChannelListWidget.setGeometry(QtCore.QRect(10, 160, 131, 141))
        self.CellTypeDeterminePanel_CellBarcodeTotalChannelListWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.CellTypeDeterminePanel_CellBarcodeTotalChannelListWidget.setObjectName("CellTypeDeterminePanel_CellBarcodeTotalChannelListWidget")
        self.label_9 = QtWidgets.QLabel(self.frame_4)
        self.label_9.setGeometry(QtCore.QRect(10, 130, 131, 31))
        self.label_9.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_9.setObjectName("label_9")
        self.line_7 = QtWidgets.QFrame(self.frame_4)
        self.line_7.setGeometry(QtCore.QRect(0, 300, 481, 20))
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracLineEdit = QtWidgets.QLineEdit(self.frame_4)
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracLineEdit.setGeometry(QtCore.QRect(260, 250, 51, 21))
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracLineEdit.setObjectName("CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracLineEdit")
        self.CellTypeDeterminePanel_CellBarcodeChannelDistributionTextBrowser = QtWidgets.QTextBrowser(self.frame_4)
        self.CellTypeDeterminePanel_CellBarcodeChannelDistributionTextBrowser.setGeometry(QtCore.QRect(150, 60, 221, 151))
        self.CellTypeDeterminePanel_CellBarcodeChannelDistributionTextBrowser.setObjectName("CellTypeDeterminePanel_CellBarcodeChannelDistributionTextBrowser")
        self.label_8 = QtWidgets.QLabel(self.frame_4)
        self.label_8.setGeometry(QtCore.QRect(210, 210, 101, 31))
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.line_9 = QtWidgets.QFrame(self.frame_4)
        self.line_9.setGeometry(QtCore.QRect(0, 330, 481, 20))
        self.line_9.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_9.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_9.setObjectName("line_9")
        self.CellTypeDeterminePanel_ShowCellBarcodePushButton = QtWidgets.QPushButton(self.frame_4)
        self.CellTypeDeterminePanel_ShowCellBarcodePushButton.setGeometry(QtCore.QRect(322, 340, 151, 32))
        self.CellTypeDeterminePanel_ShowCellBarcodePushButton.setObjectName("CellTypeDeterminePanel_ShowCellBarcodePushButton")
        self.label_11 = QtWidgets.QLabel(self.frame_4)
        self.label_11.setGeometry(QtCore.QRect(380, 30, 91, 31))
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.line_10 = QtWidgets.QFrame(self.frame_4)
        self.line_10.setGeometry(QtCore.QRect(300, 220, 41, 91))
        self.line_10.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_10.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_10.setObjectName("line_10")
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVListWidget = QtWidgets.QListWidget(self.frame_4)
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVListWidget.setGeometry(QtCore.QRect(380, 90, 91, 121))
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVListWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVListWidget.setObjectName("CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVListWidget")
        item = QtWidgets.QListWidgetItem()
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVListWidget.addItem(item)
        self.label_12 = QtWidgets.QLabel(self.frame_4)
        self.label_12.setGeometry(QtCore.QRect(10, 340, 41, 31))
        self.label_12.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_12.setObjectName("label_12")
        self.CellTypeDeterminePanel_AlphaHorizontalSlider = QtWidgets.QSlider(self.frame_4)
        self.CellTypeDeterminePanel_AlphaHorizontalSlider.setGeometry(QtCore.QRect(50, 340, 201, 31))
        self.CellTypeDeterminePanel_AlphaHorizontalSlider.setMaximum(100)
        self.CellTypeDeterminePanel_AlphaHorizontalSlider.setProperty("value", 30)
        self.CellTypeDeterminePanel_AlphaHorizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.CellTypeDeterminePanel_AlphaHorizontalSlider.setObjectName("CellTypeDeterminePanel_AlphaHorizontalSlider")
        self.CellTypeDeterminePanel_AlphaDoubleSpinBox = QtWidgets.QDoubleSpinBox(self.frame_4)
        self.CellTypeDeterminePanel_AlphaDoubleSpinBox.setGeometry(QtCore.QRect(260, 340, 61, 31))
        self.CellTypeDeterminePanel_AlphaDoubleSpinBox.setMaximum(1.0)
        self.CellTypeDeterminePanel_AlphaDoubleSpinBox.setSingleStep(0.01)
        self.CellTypeDeterminePanel_AlphaDoubleSpinBox.setProperty("value", 0.3)
        self.CellTypeDeterminePanel_AlphaDoubleSpinBox.setObjectName("CellTypeDeterminePanel_AlphaDoubleSpinBox")
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVLineEdit = QtWidgets.QLineEdit(self.frame_4)
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVLineEdit.setGeometry(QtCore.QRect(380, 60, 91, 21))
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVLineEdit.setText("")
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVLineEdit.setObjectName("CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVLineEdit")
        self.label_13 = QtWidgets.QLabel(self.frame_4)
        self.label_13.setGeometry(QtCore.QRect(330, 210, 101, 31))
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracLineEdit = QtWidgets.QLineEdit(self.frame_4)
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracLineEdit.setGeometry(QtCore.QRect(380, 250, 51, 21))
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracLineEdit.setObjectName("CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracLineEdit")
        self.line_11 = QtWidgets.QFrame(self.frame_4)
        self.line_11.setGeometry(QtCore.QRect(150, 230, 331, 20))
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundAbsLineEdit = QtWidgets.QLineEdit(self.frame_4)
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundAbsLineEdit.setGeometry(QtCore.QRect(260, 280, 51, 21))
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundAbsLineEdit.setObjectName("CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundAbsLineEdit")
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundAbsLineEdit = QtWidgets.QLineEdit(self.frame_4)
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundAbsLineEdit.setGeometry(QtCore.QRect(380, 280, 51, 21))
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundAbsLineEdit.setObjectName("CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundAbsLineEdit")
        self.label_18 = QtWidgets.QLabel(self.frame_4)
        self.label_18.setGeometry(QtCore.QRect(150, 250, 41, 21))
        self.label_18.setAlignment(QtCore.Qt.AlignCenter)
        self.label_18.setObjectName("label_18")
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationScaleLineEdit = QtWidgets.QLineEdit(self.frame_4)
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationScaleLineEdit.setGeometry(QtCore.QRect(150, 280, 41, 21))
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationScaleLineEdit.setObjectName("CellTypeDeterminePanel_CellBarcodeCelltypeClassificationScaleLineEdit")
        self.line_12 = QtWidgets.QFrame(self.frame_4)
        self.line_12.setGeometry(QtCore.QRect(180, 220, 41, 91))
        self.line_12.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_12.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_12.setObjectName("line_12")
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracRadioButton = QtWidgets.QRadioButton(self.frame_4)
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracRadioButton.setEnabled(True)
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracRadioButton.setGeometry(QtCore.QRect(210, 250, 51, 20))
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracRadioButton.setChecked(True)
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracRadioButton.setAutoExclusive(False)
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracRadioButton.setObjectName("CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracRadioButton")
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundAbsRadioButton = QtWidgets.QRadioButton(self.frame_4)
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundAbsRadioButton.setEnabled(True)
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundAbsRadioButton.setGeometry(QtCore.QRect(210, 280, 51, 20))
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundAbsRadioButton.setAutoExclusive(False)
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundAbsRadioButton.setObjectName("CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundAbsRadioButton")
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracRadioButton = QtWidgets.QRadioButton(self.frame_4)
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracRadioButton.setGeometry(QtCore.QRect(330, 250, 51, 20))
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracRadioButton.setChecked(True)
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracRadioButton.setAutoExclusive(False)
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracRadioButton.setObjectName("CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracRadioButton")
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundAbsRadioButton = QtWidgets.QRadioButton(self.frame_4)
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundAbsRadioButton.setGeometry(QtCore.QRect(330, 280, 51, 20))
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundAbsRadioButton.setAutoExclusive(False)
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundAbsRadioButton.setObjectName("CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundAbsRadioButton")
        self.line_13 = QtWidgets.QFrame(self.frame_4)
        self.line_13.setGeometry(QtCore.QRect(130, 310, 41, 31))
        self.line_13.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_13.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_13.setObjectName("line_13")
        self.line_13.raise_()
        self.line_11.raise_()
        self.line_10.raise_()
        self.label_5.raise_()
        self.line_6.raise_()
        self.label_6.raise_()
        self.label_7.raise_()
        self.CellTypeDeterminePanel_CellBarcodeUpdatePushButton.raise_()
        self.CellTypeDeterminePanel_CellBarcodeSetupPushButton.raise_()
        self.CellTypeDeterminePanel_CellBarcodeChannelListWidget.raise_()
        self.CellTypeDeterminePanel_CellBarcodeTotalChannelListWidget.raise_()
        self.label_9.raise_()
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracLineEdit.raise_()
        self.CellTypeDeterminePanel_CellBarcodeChannelDistributionTextBrowser.raise_()
        self.label_8.raise_()
        self.line_9.raise_()
        self.CellTypeDeterminePanel_ShowCellBarcodePushButton.raise_()
        self.label_11.raise_()
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVListWidget.raise_()
        self.label_12.raise_()
        self.CellTypeDeterminePanel_AlphaHorizontalSlider.raise_()
        self.CellTypeDeterminePanel_AlphaDoubleSpinBox.raise_()
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVLineEdit.raise_()
        self.label_13.raise_()
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracLineEdit.raise_()
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundAbsLineEdit.raise_()
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundAbsLineEdit.raise_()
        self.label_18.raise_()
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationScaleLineEdit.raise_()
        self.line_12.raise_()
        self.line_7.raise_()
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracRadioButton.raise_()
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundAbsRadioButton.raise_()
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracRadioButton.raise_()
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundAbsRadioButton.raise_()
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.retranslateUi(Form)
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVListWidget.setCurrentRow(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Cell types"))
        self.CellTypeDeterminePanel_CelltypeRemoveSelectedPushButton.setText(_translate("Form", "Remove selected"))
        self.CellTypeDeterminePanel_CelltypeAddnewcelltypePushButton.setText(_translate("Form", "Add new celltype"))
        self.CellTypeDeterminePanel_FOVPushButton.setText(_translate("Form", "FOV"))
        self.CellTypeDeterminePanel_CellBarcodePushButton.setText(_translate("Form", "Cell Barcode"))
        self.label_2.setText(_translate("Form", "FOV"))
        self.label_3.setText(_translate("Form", "Cell types"))
        self.CellTypeDeterminePanel_FOVModulateSelectedCellTypePushButton.setText(_translate("Form", "Modulate Selected Cell Type"))
        self.label_5.setText(_translate("Form", "Cell Barcode"))
        self.label_6.setText(_translate("Form", "Cell Barcode Channel"))
        self.label_7.setText(_translate("Form", "Cell Barcode Channel Distribution"))
        self.CellTypeDeterminePanel_CellBarcodeUpdatePushButton.setText(_translate("Form", "Update"))
        self.CellTypeDeterminePanel_CellBarcodeSetupPushButton.setText(_translate("Form", "Setup"))
        self.label_9.setText(_translate("Form", "Total Channel"))
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracLineEdit.setText(_translate("Form", "0.3"))
        self.label_8.setText(_translate("Form", "Lower bound"))
        self.CellTypeDeterminePanel_ShowCellBarcodePushButton.setText(_translate("Form", "Show Cell Barcode"))
        self.label_11.setText(_translate("Form", "FOV"))
        __sortingEnabled = self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVListWidget.isSortingEnabled()
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVListWidget.setSortingEnabled(False)
        item = self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVListWidget.item(0)
        item.setText(_translate("Form", "All"))
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVListWidget.setSortingEnabled(__sortingEnabled)
        self.label_12.setText(_translate("Form", "Alpha"))
        self.label_13.setText(_translate("Form", "Upper bound"))
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracLineEdit.setText(_translate("Form", "0.999"))
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundAbsLineEdit.setText(_translate("Form", "0"))
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundAbsLineEdit.setText(_translate("Form", "15000"))
        self.label_18.setText(_translate("Form", "scale"))
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationScaleLineEdit.setText(_translate("Form", "1"))
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracRadioButton.setText(_translate("Form", "frac"))
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundAbsRadioButton.setText(_translate("Form", "abs"))
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracRadioButton.setText(_translate("Form", "frac"))
        self.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundAbsRadioButton.setText(_translate("Form", "abs"))

class ChannelSelection(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(295, 373)
        Form.setMinimumSize(QtCore.QSize(1, 2))
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(Form)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(0, 0, 131, 31))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.ChannelSelection_CurrentChannelListWidget = QtWidgets.QListWidget(self.frame)
        self.ChannelSelection_CurrentChannelListWidget.setGeometry(QtCore.QRect(10, 50, 111, 21))
        self.ChannelSelection_CurrentChannelListWidget.setToolTipDuration(-1)
        self.ChannelSelection_CurrentChannelListWidget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.ChannelSelection_CurrentChannelListWidget.setObjectName("ChannelSelection_CurrentChannelListWidget")
        self.line = QtWidgets.QFrame(self.frame)
        self.line.setGeometry(QtCore.QRect(120, 0, 20, 341))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.ChannelSelection_MinimumSlider = QtWidgets.QSlider(self.frame)
        self.ChannelSelection_MinimumSlider.setGeometry(QtCore.QRect(140, 40, 51, 261))
        self.ChannelSelection_MinimumSlider.setMaximum(10000)
        self.ChannelSelection_MinimumSlider.setProperty("value", 3000)
        self.ChannelSelection_MinimumSlider.setOrientation(QtCore.Qt.Vertical)
        self.ChannelSelection_MinimumSlider.setObjectName("ChannelSelection_MinimumSlider")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(140, 0, 51, 31))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setGeometry(QtCore.QRect(210, 0, 51, 31))
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.ChannelSelection_MaximumSlider = QtWidgets.QSlider(self.frame)
        self.ChannelSelection_MaximumSlider.setGeometry(QtCore.QRect(211, 40, 51, 261))
        self.ChannelSelection_MaximumSlider.setMaximum(10000)
        self.ChannelSelection_MaximumSlider.setSingleStep(1)
        self.ChannelSelection_MaximumSlider.setProperty("value", 9999)
        self.ChannelSelection_MaximumSlider.setOrientation(QtCore.Qt.Vertical)
        self.ChannelSelection_MaximumSlider.setObjectName("ChannelSelection_MaximumSlider")
        self.line_2 = QtWidgets.QFrame(self.frame)
        self.line_2.setGeometry(QtCore.QRect(190, 0, 21, 251))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.ChannelSelection_ChannelListWidget = QtWidgets.QListWidget(self.frame)
        self.ChannelSelection_ChannelListWidget.setGeometry(QtCore.QRect(10, 90, 111, 131))
        self.ChannelSelection_ChannelListWidget.setToolTipDuration(-1)
        self.ChannelSelection_ChannelListWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.ChannelSelection_ChannelListWidget.setObjectName("ChannelSelection_ChannelListWidget")
        self.ChannelSelection_ChangeChannelPushButton = QtWidgets.QPushButton(self.frame)
        self.ChannelSelection_ChangeChannelPushButton.setGeometry(QtCore.QRect(10, 310, 111, 32))
        self.ChannelSelection_ChangeChannelPushButton.setObjectName("ChannelSelection_ChangeChannelPushButton")
        self.label_4 = QtWidgets.QLabel(self.frame)
        self.label_4.setGeometry(QtCore.QRect(9, 30, 111, 21))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.frame)
        self.label_5.setGeometry(QtCore.QRect(10, 70, 111, 21))
        self.label_5.setObjectName("label_5")
        self.ChannelSelection_MinimumLineEdit = QtWidgets.QLineEdit(self.frame)
        self.ChannelSelection_MinimumLineEdit.setGeometry(QtCore.QRect(140, 310, 51, 21))
        self.ChannelSelection_MinimumLineEdit.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.ChannelSelection_MinimumLineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.ChannelSelection_MinimumLineEdit.setObjectName("ChannelSelection_MinimumLineEdit")
        self.ChannelSelection_MaximumLineEdit = QtWidgets.QLineEdit(self.frame)
        self.ChannelSelection_MaximumLineEdit.setGeometry(QtCore.QRect(210, 310, 51, 21))
        self.ChannelSelection_MaximumLineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.ChannelSelection_MaximumLineEdit.setObjectName("ChannelSelection_MaximumLineEdit")
        self.ChannelSelection_ChannelListWidgetColor = QtWidgets.QListWidget(self.frame)
        self.ChannelSelection_ChannelListWidgetColor.setEnabled(True)
        self.ChannelSelection_ChannelListWidgetColor.setGeometry(QtCore.QRect(10, 230, 111, 71))
        self.ChannelSelection_ChannelListWidgetColor.setToolTipDuration(-1)
        self.ChannelSelection_ChannelListWidgetColor.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.ChannelSelection_ChannelListWidgetColor.setObjectName("ChannelSelection_ChannelListWidgetColor")
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Channel"))
        self.label_2.setText(_translate("Form", "Min"))
        self.label_3.setText(_translate("Form", "Max"))
        self.ChannelSelection_ChangeChannelPushButton.setText(_translate("Form", "Change"))
        self.label_4.setText(_translate("Form", "Working"))
        self.label_5.setText(_translate("Form", "Total"))
        self.ChannelSelection_MinimumLineEdit.setText(_translate("Form", "0.3"))
        self.ChannelSelection_MaximumLineEdit.setText(_translate("Form", "0.9999"))

class MainWindowUI(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1339, 919)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.GraphicWidget = QtWidgets.QWidget(self.centralwidget)
        self.GraphicWidget.setGeometry(QtCore.QRect(10, 10, 1041, 841))
        self.GraphicWidget.setObjectName("GraphicWidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(1060, 260, 271, 581))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(0, 0, 271, 31))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.ChannelSelection_CurrentChannelListWidget = QtWidgets.QListWidget(self.frame)
        self.ChannelSelection_CurrentChannelListWidget.setGeometry(QtCore.QRect(10, 50, 111, 101))
        self.ChannelSelection_CurrentChannelListWidget.setToolTipDuration(-1)
        self.ChannelSelection_CurrentChannelListWidget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.ChannelSelection_CurrentChannelListWidget.setObjectName("ChannelSelection_CurrentChannelListWidget")
        self.label_4 = QtWidgets.QLabel(self.frame)
        self.label_4.setGeometry(QtCore.QRect(9, 30, 111, 21))
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.frame)
        self.label_5.setGeometry(QtCore.QRect(10, 160, 251, 21))
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.ChannelSelection_ChannelListWidget = QtWidgets.QListWidget(self.frame)
        self.ChannelSelection_ChannelListWidget.setGeometry(QtCore.QRect(10, 180, 251, 151))
        self.ChannelSelection_ChannelListWidget.setToolTipDuration(-1)
        self.ChannelSelection_ChannelListWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.ChannelSelection_ChannelListWidget.setObjectName("ChannelSelection_ChannelListWidget")
        self.label_8 = QtWidgets.QLabel(self.frame)
        self.label_8.setGeometry(QtCore.QRect(150, 30, 111, 21))
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.line = QtWidgets.QFrame(self.frame)
        self.line.setGeometry(QtCore.QRect(0, 40, 271, 111))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.ChannelSelection_ChannelListWidgetColor = QtWidgets.QListWidget(self.frame)
        self.ChannelSelection_ChannelListWidgetColor.setEnabled(True)
        self.ChannelSelection_ChannelListWidgetColor.setGeometry(QtCore.QRect(150, 50, 111, 101))
        self.ChannelSelection_ChannelListWidgetColor.setToolTipDuration(-1)
        self.ChannelSelection_ChannelListWidgetColor.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.ChannelSelection_ChannelListWidgetColor.setObjectName("ChannelSelection_ChannelListWidgetColor")
        self.ChannelSelection_ChangeChannelPushButton = QtWidgets.QPushButton(self.frame)
        self.ChannelSelection_ChangeChannelPushButton.setGeometry(QtCore.QRect(20, 330, 231, 32))
        self.ChannelSelection_ChangeChannelPushButton.setObjectName("ChannelSelection_ChangeChannelPushButton")
        self.line_3 = QtWidgets.QFrame(self.frame)
        self.line_3.setGeometry(QtCore.QRect(10, 355, 251, 21))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.line_2 = QtWidgets.QFrame(self.frame)
        self.line_2.setGeometry(QtCore.QRect(0, 390, 271, 191))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.ChannelSelection_MinimumSlider = QtWidgets.QSlider(self.frame)
        self.ChannelSelection_MinimumSlider.setGeometry(QtCore.QRect(10, 400, 51, 171))
        self.ChannelSelection_MinimumSlider.setMaximum(10000)
        self.ChannelSelection_MinimumSlider.setProperty("value", 3000)
        self.ChannelSelection_MinimumSlider.setOrientation(QtCore.Qt.Vertical)
        self.ChannelSelection_MinimumSlider.setObjectName("ChannelSelection_MinimumSlider")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(10, 370, 121, 31))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.ChannelSelection_MinimumLineEdit = QtWidgets.QLineEdit(self.frame)
        self.ChannelSelection_MinimumLineEdit.setGeometry(QtCore.QRect(60, 420, 61, 21))
        self.ChannelSelection_MinimumLineEdit.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.ChannelSelection_MinimumLineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.ChannelSelection_MinimumLineEdit.setObjectName("ChannelSelection_MinimumLineEdit")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setGeometry(QtCore.QRect(140, 370, 121, 31))
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.ChannelSelection_MaximumSlider = QtWidgets.QSlider(self.frame)
        self.ChannelSelection_MaximumSlider.setGeometry(QtCore.QRect(150, 400, 51, 171))
        self.ChannelSelection_MaximumSlider.setMaximum(10000)
        self.ChannelSelection_MaximumSlider.setSingleStep(1)
        self.ChannelSelection_MaximumSlider.setProperty("value", 9999)
        self.ChannelSelection_MaximumSlider.setOrientation(QtCore.Qt.Vertical)
        self.ChannelSelection_MaximumSlider.setObjectName("ChannelSelection_MaximumSlider")
        self.ChannelSelection_MaximumLineEdit = QtWidgets.QLineEdit(self.frame)
        self.ChannelSelection_MaximumLineEdit.setGeometry(QtCore.QRect(200, 420, 61, 21))
        self.ChannelSelection_MaximumLineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.ChannelSelection_MaximumLineEdit.setObjectName("ChannelSelection_MaximumLineEdit")
        self.label_9 = QtWidgets.QLabel(self.frame)
        self.label_9.setGeometry(QtCore.QRect(60, 400, 61, 21))
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.label_11 = QtWidgets.QLabel(self.frame)
        self.label_11.setGeometry(QtCore.QRect(200, 400, 61, 21))
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.label_10 = QtWidgets.QLabel(self.frame)
        self.label_10.setGeometry(QtCore.QRect(60, 450, 61, 21))
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.label_12 = QtWidgets.QLabel(self.frame)
        self.label_12.setGeometry(QtCore.QRect(200, 450, 61, 21))
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.ChannelSelection_MinimumAbsLineEdit = QtWidgets.QLineEdit(self.frame)
        self.ChannelSelection_MinimumAbsLineEdit.setEnabled(False)
        self.ChannelSelection_MinimumAbsLineEdit.setGeometry(QtCore.QRect(60, 470, 61, 21))
        self.ChannelSelection_MinimumAbsLineEdit.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.ChannelSelection_MinimumAbsLineEdit.setText("")
        self.ChannelSelection_MinimumAbsLineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.ChannelSelection_MinimumAbsLineEdit.setObjectName("ChannelSelection_MinimumAbsLineEdit")
        self.ChannelSelection_MaximumAbsLineEdit = QtWidgets.QLineEdit(self.frame)
        self.ChannelSelection_MaximumAbsLineEdit.setEnabled(False)
        self.ChannelSelection_MaximumAbsLineEdit.setGeometry(QtCore.QRect(200, 470, 61, 21))
        self.ChannelSelection_MaximumAbsLineEdit.setText("")
        self.ChannelSelection_MaximumAbsLineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.ChannelSelection_MaximumAbsLineEdit.setObjectName("ChannelSelection_MaximumAbsLineEdit")
        self.line.raise_()
        self.label.raise_()
        self.ChannelSelection_CurrentChannelListWidget.raise_()
        self.label_4.raise_()
        self.label_5.raise_()
        self.ChannelSelection_ChannelListWidget.raise_()
        self.label_8.raise_()
        self.ChannelSelection_ChannelListWidgetColor.raise_()
        self.ChannelSelection_ChangeChannelPushButton.raise_()
        self.line_3.raise_()
        self.line_2.raise_()
        self.ChannelSelection_MinimumSlider.raise_()
        self.label_2.raise_()
        self.ChannelSelection_MinimumLineEdit.raise_()
        self.label_3.raise_()
        self.ChannelSelection_MaximumSlider.raise_()
        self.ChannelSelection_MaximumLineEdit.raise_()
        self.label_9.raise_()
        self.label_11.raise_()
        self.label_10.raise_()
        self.label_12.raise_()
        self.ChannelSelection_MinimumAbsLineEdit.raise_()
        self.ChannelSelection_MaximumAbsLineEdit.raise_()
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(1200, 10, 131, 241))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.label_7 = QtWidgets.QLabel(self.frame_2)
        self.label_7.setGeometry(QtCore.QRect(10, 0, 111, 31))
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.SelectCell_ListWidget = QtWidgets.QListWidget(self.frame_2)
        self.SelectCell_ListWidget.setGeometry(QtCore.QRect(10, 30, 111, 181))
        self.SelectCell_ListWidget.setObjectName("SelectCell_ListWidget")
        self.SelectCell_ChangeCellPushButton = QtWidgets.QPushButton(self.frame_2)
        self.SelectCell_ChangeCellPushButton.setGeometry(QtCore.QRect(10, 210, 111, 32))
        self.SelectCell_ChangeCellPushButton.setObjectName("SelectCell_ChangeCellPushButton")
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(1060, 10, 131, 241))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.label_6 = QtWidgets.QLabel(self.frame_3)
        self.label_6.setGeometry(QtCore.QRect(10, 0, 111, 31))
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.SelectFOV_ListWidget = QtWidgets.QListWidget(self.frame_3)
        self.SelectFOV_ListWidget.setGeometry(QtCore.QRect(10, 30, 111, 181))
        self.SelectFOV_ListWidget.setObjectName("SelectFOV_ListWidget")
        self.SelectFOV_ChangeFOVPushButton = QtWidgets.QPushButton(self.frame_3)
        self.SelectFOV_ChangeFOVPushButton.setGeometry(QtCore.QRect(10, 210, 111, 32))
        self.SelectFOV_ChangeFOVPushButton.setObjectName("SelectFOV_ChangeFOVPushButton")
        self.FixViewPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.FixViewPushButton.setGeometry(QtCore.QRect(1060, 840, 131, 32))
        self.FixViewPushButton.setObjectName("FixViewPushButton")
        self.InitializeViewPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.InitializeViewPushButton.setEnabled(False)
        self.InitializeViewPushButton.setGeometry(QtCore.QRect(1200, 840, 131, 32))
        self.InitializeViewPushButton.setObjectName("InitializeViewPushButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1339, 24))
        self.menubar.setNativeMenuBar(False)
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuSave = QtWidgets.QMenu(self.menubar)
        self.menuSave.setObjectName("menuSave")
        self.menuLoad = QtWidgets.QMenu(self.menubar)
        self.menuLoad.setObjectName("menuLoad")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoad_Mosaic_Tile = QtWidgets.QAction(MainWindow)
        self.actionLoad_Mosaic_Tile.setObjectName("actionLoad_Mosaic_Tile")
        self.actionLoad_Mosaic_File = QtWidgets.QAction(MainWindow)
        self.actionLoad_Mosaic_File.setObjectName("actionLoad_Mosaic_File")
        self.actionExit_Cntrl_Q = QtWidgets.QAction(MainWindow)
        self.actionExit_Cntrl_Q.setObjectName("actionExit_Cntrl_Q")
        self.actionSave_Cells = QtWidgets.QAction(MainWindow)
        self.actionSave_Cells.setObjectName("actionSave_Cells")
        self.actionSave_Spots = QtWidgets.QAction(MainWindow)
        self.actionSave_Spots.setObjectName("actionSave_Spots")
        self.actionSave_Metadata = QtWidgets.QAction(MainWindow)
        self.actionSave_Metadata.setObjectName("actionSave_Metadata")
        self.actionLoad_Cells = QtWidgets.QAction(MainWindow)
        self.actionLoad_Cells.setObjectName("actionLoad_Cells")
        self.actionLoad_Spots = QtWidgets.QAction(MainWindow)
        self.actionLoad_Spots.setObjectName("actionLoad_Spots")
        self.actionLoad_Spot_Metadata = QtWidgets.QAction(MainWindow)
        self.actionLoad_Spot_Metadata.setObjectName("actionLoad_Spot_Metadata")
        self.menuFile.addAction(self.actionLoad_Mosaic_Tile)
        self.menuFile.addAction(self.actionLoad_Mosaic_File)
        self.menuFile.addAction(self.actionExit_Cntrl_Q)
        self.menuSave.addAction(self.actionSave_Cells)
        self.menuSave.addAction(self.actionSave_Spots)
        self.menuSave.addAction(self.actionSave_Metadata)
        self.menuLoad.addAction(self.actionLoad_Cells)
        self.menuLoad.addAction(self.actionLoad_Spots)
        self.menuLoad.addAction(self.actionLoad_Spot_Metadata)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuSave.menuAction())
        self.menubar.addAction(self.menuLoad.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Channel"))
        self.label_4.setText(_translate("MainWindow", "Working"))
        self.label_5.setText(_translate("MainWindow", "Total"))
        self.label_8.setText(_translate("MainWindow", "Color"))
        self.ChannelSelection_ChangeChannelPushButton.setText(_translate("MainWindow", "Change Channel"))
        self.label_2.setText(_translate("MainWindow", "Min"))
        self.ChannelSelection_MinimumLineEdit.setText(_translate("MainWindow", "0.3"))
        self.label_3.setText(_translate("MainWindow", "Max"))
        self.ChannelSelection_MaximumLineEdit.setText(_translate("MainWindow", "0.9999"))
        self.label_9.setText(_translate("MainWindow", "frac"))
        self.label_11.setText(_translate("MainWindow", "frac"))
        self.label_10.setText(_translate("MainWindow", "abs"))
        self.label_12.setText(_translate("MainWindow", "abs"))
        self.label_7.setText(_translate("MainWindow", "Select Cell"))
        self.SelectCell_ChangeCellPushButton.setText(_translate("MainWindow", "Change Cell"))
        self.label_6.setText(_translate("MainWindow", "Select FOV"))
        self.SelectFOV_ChangeFOVPushButton.setText(_translate("MainWindow", "Change FOV"))
        self.FixViewPushButton.setText(_translate("MainWindow", "Fix View"))
        self.InitializeViewPushButton.setText(_translate("MainWindow", "Initialize View"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuSave.setTitle(_translate("MainWindow", "Save"))
        self.menuLoad.setTitle(_translate("MainWindow", "Load"))
        self.actionLoad_Mosaic_Tile.setText(_translate("MainWindow", "Load Mosaic Tile"))
        self.actionLoad_Mosaic_File.setText(_translate("MainWindow", "Load Mosaic File"))
        self.actionExit_Cntrl_Q.setText(_translate("MainWindow", "Exit (Cntrl+Q)"))
        self.actionSave_Cells.setText(_translate("MainWindow", "Save Cells"))
        self.actionSave_Spots.setText(_translate("MainWindow", "Save Spots"))
        self.actionSave_Metadata.setText(_translate("MainWindow", "Save Metadata"))
        self.actionLoad_Cells.setText(_translate("MainWindow", "Load Cells"))
        self.actionLoad_Spots.setText(_translate("MainWindow", "Load Spots"))
        self.actionLoad_Spot_Metadata.setText(_translate("MainWindow", "Load Spot Metadata"))

class removable_target_item(pg.TargetItem):
    _delete_spot_signal = QtCore.pyqtSignal()
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.movable = True
    def mouseClickEvent(self, ev):
        if (not self.moving) and (ev.button() == QtCore.Qt.MouseButton.RightButton):
            self._delete_spot_signal.emit()
            
        else: super().mouseClickEvent(ev)

class removable_spot_item(pg.ScatterPlotItem):
    
    _update_new_spots_signal = QtCore.pyqtSignal(bool)
    
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.target = removable_target_item()
        self.target.setParentItem(self)
        self.target.sigPositionChanged.connect(self.targetMoved)
        self.target._delete_spot_signal.connect(self._removeSpot)
        self.target.hide()
        self.selectedPoint = None
        self.coordinateLabel = pg.TextItem()
        self.coordinateLabel.setParentItem(self.target)
        self.coordinateLabel.setAnchor((0,1))
        self.manual_modification = False
        self.imagesize = (0,0)
        self.brush = kargs['Brush'] if 'Brush' in kargs else pg.mkBrush(255,255,0,255)
        self.pen = kargs['Pen'] if 'Pen' in kargs else pg.mkPen(255,255,0,255)

    def _setImageSize(self, imagesize):
        self.imagesize = imagesize

    def _turnONManualModification(self):
        self.manual_modification = True
        
    def _turnOFFManualModification(self):
        self.manual_modification = False

    def dataBounds(self, *args, **kargs):
        if self.manual_modification and 'ax' in kargs:
            if kargs['ax'] == 0:
                return np.array([0,self.imagesize[1]])
            elif kargs['ax'] == 1:
                return np.array([0,self.imagesize[0]])
        else: return super().dataBounds(*args, **kargs)
    
    def _removeSpot(self):
        if self.target.isVisible() and self.selectedPoint is not None:
            self.data = np.delete(self.data, self.current_id)
            self.updateSpots()
            self.invalidate()
            self.target.hide()
            self.selectedPoint = None
            self.current_id = None
            self._update_new_spots_signal.emit(False)
            self.target.setParentItem(self)
    
    def mouseClickEvent(self, ev):
        if not self.manual_modification: super().mouseClickEvent(ev)
        else:
            points = self.pointsAt(ev.pos())
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                if self.target.isVisible(): self.target.hide()
                else: 
                    if len(points) == 0:
                        x,y = np.atleast_2d(ev.pos()).astype(int).flatten()
                        spot = [{'pos': (x + 1/2, y + 1/2),
                                 'data': 255, 'size':10,
                                 'brush':self.brush,
                                 'pen':self.pen}]
                        self.addPoints(spots = spot)
                        self._update_new_spots_signal.emit(True)
                        print(self.data.shape)
                ev.accept()
            elif ev.button() == QtCore.Qt.MouseButton.RightButton:
                if len(points) > 0:
                    self.selectedPoint = points[-1]
                    self.target.setPos(self.selectedPoint.pos())
                    x,y = np.atleast_2d(self.selectedPoint.pos()).flatten()
                    self.current_id = np.arange(len(self.data))[(self.data['x'] == x) & (self.data['y'] == y)]
                    label = f'{tuple(map(lambda n:n-.5, (x,y)))}'
                    self.coordinateLabel.setHtml("<div style='color: red; background: black;'>%s</div>" % label)
                    self.target.show()
                    ev.accept()
    
    def targetMoved(self, target):
        if self.target.isVisible() and self.selectedPoint is not None:
            self.data[['x','y']][self.current_id] = tuple(map(lambda el: round(el,0)+1/2, target.pos()))
            self.updateSpots()
            self.invalidate()
            label = f'{tuple(map(lambda el: round(el,0), target.pos()))}'
            self.coordinateLabel.setHtml("<div style='color: red; background: black;'>%s</div>" % label)

class MainImageCanvas(pg.ImageView):
    
    save_new_classified_cell_signal = QtCore.pyqtSignal()
    request_stored_cell_signal = QtCore.pyqtSignal()
    save_new_localized_spot_signal = QtCore.pyqtSignal()
    request_stored_spots_signal = QtCore.pyqtSignal()
    link_spot_cell_signal = QtCore.pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.ui.roiBtn.setVisible(False)
        
        self.current_image = np.zeros((1024,1024), dtype=int)
        self.stored_images = {'1':{'All': {'channel 0': self.current_image}}}
        self.fov_list = ['1']
        self.channel_list = ['channel 0']
        self.cell_list = ['All']
        self.celltypes = []
        self.celltype_determination = {'fov':{},
                                       'barcode':{}}
        self.celltype_mode = 'fov'
        self._autoRange = True
        
        self.current_fov = self.fov_list[0]
        self.current_channel = [self.channel_list[-1]]
        self.working_channel = self.channel_list[-1]
        self.current_cell = self.cell_list[0]
        
        self.lb = {ch:.3 for ch in self.channel_list}
        self.hb = {ch:.9999 for ch in self.channel_list}
        self.w = 8
        self.ymin, self.ymax = 0,0
        self.xmin, self.xmax = 0,0

        # visualizer
        self.boundaries = pg.ScatterPlotItem()
        self.rough_boundaries = pg.ScatterPlotItem()
        self.manual_boundaries = removable_spot_item(Brush = pg.mkBrush(255,0,255,255), Pen = pg.mkPen(255,0,255,255))
        self.spots = {ch: removable_spot_item() for ch in self.channel_list}
        self.auxiliary = [self.boundaries, self.rough_boundaries, self.manual_boundaries]
        self.texts = []
        
        # transient memory for spots, cells
        self.SpotContainer = SpotContainer(self.fov_list, self.channel_list)
        self.CellContainer = CellContainer(self.fov_list)
        # permanent memory for spots, cells
        self.SpotContainer_permanent = SpotContainer(self.fov_list, self.channel_list)
        self.CellContainer_permanent = CellContainer(self.fov_list)
        self.SpotMetaDataAnalyzer = None
        self.conditions = []

        self.current_spot_indices = []
                
        self.view = self.getView()
        for aux in self.auxiliary:
            self.view.addItem(aux)
        for spots in self.spots.values():
            self.view.addItem(spots)
        if len(self.texts) > 0:
            for text in self.texts:
                self.view.addItem(text)
        self._createGUI()
    
    def _load_celltype_determination(self, celltypes, celltype_determination):
        celltypes_fov = celltype_determination['fov']['raw']
        celltypes_barcode = celltype_determination['barcode']['raw']
        self.celltypes = celltypes
        
        celltype_from_fov = {}
        fov_from_celltype = {}
        if celltypes_fov[0] != '':
            for i,ct in enumerate(self.celltypes):
                string = celltypes_fov[i].split()
                for j in range(len(string)):
                    line = string[j].split('-')
                    st,end = int(line[0]), int(line[1])
                    for k in range(st,end):
                        celltype_from_fov[k] = ct
            
            fov_from_celltype = self._fov_celltype_cross(celltype_from_fov = celltype_from_fov)
        
        self.celltype_determination = {'fov':{'raw': celltypes_fov,
                                              'fov_from_celltype': fov_from_celltype,
                                              'celltype_from_fov': celltype_from_fov},
                                       'barcode':{'raw': celltypes_barcode,
                                                  'barcode_channel': celltypes_barcode,
                                                  'barcode_celltype': celltypes,
                                                  'lower_bound':{ch:{int(fov):0 for fov in self.fov_list}
                                                                 for ch in self.channel_list},
                                                  'upper_bound':{ch:{int(fov):0 for fov in self.fov_list}
                                                                 for ch in self.channel_list},
                                                  'scale':{ch:{int(fov):1 for fov in self.fov_list}
                                                           for ch in self.channel_list},
                                                  'image':{int(fov):[] for fov in self.fov_list}}}
        
    @staticmethod
    def _fov_celltype_cross(**kwargs):
        if 'fov_from_celltype' in kwargs:
            fov_from_celltype = kwargs['fov_from_celltype']
            celltypes = np.unique(list(fov_from_celltype.keys()))
            celltype_from_fov = {}
            for k,v in fov_from_celltype:
                for f in v:
                    celltype_from_fov[int(f)] = str(k)
            return celltype_from_fov
            
        elif 'celltype_from_fov' in kwargs:
            celltype_from_fov = kwargs['celltype_from_fov']
            celltypes = np.unique([str(v) for v in celltype_from_fov.values()])
            fov_from_celltype = {ct:[] for ct in celltypes}
            for k,v in celltype_from_fov.items():
                fov_from_celltype[str(v)].append(int(k))
            return fov_from_celltype

    def buildMenu(self):
        self.menu = QtWidgets.QMenu()

        self.SegmentCells = QtWidgets.QAction(translate('ImageView', 'Segment Cells'), self.menu)
        self.SegmentCells.setCheckable(True)
        self.SegmentCells.toggled.connect(self._SegmentCellsClicked)
        self.menu.addAction(self.SegmentCells)
        
        self.SpotAnalysis = QtWidgets.QAction(translate('ImageView', 'Spot Localization'), self.menu)
        self.SpotAnalysis.setCheckable(True)
        self.SpotAnalysis.toggled.connect(self._SpotAnalysisClicked)
        self.menu.addAction(self.SpotAnalysis)
        
        self.AnalysisPanel = QtWidgets.QAction(translate('ImageView', 'Analysis Panel'), self.menu)
        self.AnalysisPanel.setCheckable(True)
        self.AnalysisPanel.toggled.connect(self._AnalysisPanelClicked)
        self.menu.addAction(self.AnalysisPanel)
        
        self.CelltypePanel = QtWidgets.QAction(translate('IamgeView', 'Celltype Panel'), self.menu)
        self.CelltypePanel.setCheckable(True)
        self.CelltypePanel.toggled.connect(self._CelltypePanelClicked)
        self.menu.addAction(self.CelltypePanel)

    def _createGUI(self):
        ## General ##
        self.font = QtGui.QFont()
        self.font.setWeight(75)
        self.font.setBold(True)
        self.SegmentCellsWidget = CellSegmentPanelUI()
        self.SpotAnalysisWidget = SpotAnalysisPanelUI()
        self.AnalysisPanelWidget = AnalysisPanelUI()
        self.CelltypePanelWidget = CelltypeDeterminationUI()
        
        self.SegmentCellsMainWindow = QtWidgets.QMainWindow()
        self.SegmentCellsCentralWidget = QtWidgets.QWidget()
        self.SegmentCellsWidget.setupUi(self.SegmentCellsCentralWidget)
        self.SegmentCellsMainWindow.setCentralWidget(self.SegmentCellsCentralWidget)
        self.SegmentCellsMainWindow.setWindowTitle('Segment Cell Panel')
        self.SegmentCellsMainWindow.setGeometry(self.SegmentCellsCentralWidget.geometry())
        
        self.SpotAnalysisMainWindow = QtWidgets.QMainWindow()
        self.SpotAnalysisCentralWidget = QtWidgets.QWidget()
        self.SpotAnalysisWidget.setupUi(self.SpotAnalysisCentralWidget)
        self.SpotAnalysisMainWindow.setCentralWidget(self.SpotAnalysisCentralWidget)
        self.SpotAnalysisMainWindow.setWindowTitle('Spot Analysis Panel')
        self.SpotAnalysisMainWindow.setGeometry(self.SpotAnalysisCentralWidget.geometry())
        
        self.AnalysisPanelMainWindow = QtWidgets.QMainWindow()
        self.AnalysisPanelCentralWidget = QtWidgets.QWidget()
        self.AnalysisPanelWidget.setupUi(self.AnalysisPanelCentralWidget)
        self.AnalysisPanelMainWindow.setCentralWidget(self.AnalysisPanelCentralWidget)
        self.AnalysisPanelMainWindow.setWindowTitle('Analysis Panel')
        re = QtCore.QRegExp('[0-9 ]+')
        self.AnalysisPanelWidget.AnalysisPanelFrame_ConditionLineEdit.setValidator(QtGui.QRegExpValidator(re))
        self.AnalysisPanelMainWindow.setGeometry(self.AnalysisPanelCentralWidget.geometry())
        
        self.CelltypePanelMainWindow = QtWidgets.QMainWindow()
        self.CelltypePanelCentralWidget = QtWidgets.QWidget()
        self.CelltypePanelWidget.setupUi(self.CelltypePanelCentralWidget)
        self.CelltypePanelMainWindow.setCentralWidget(self.CelltypePanelCentralWidget)
        self.CelltypePanelMainWindow.setWindowTitle('Celltype Determination Panel')
        self.CelltypePanelMainWindow.setGeometry(self.CelltypePanelCentralWidget.geometry())
        
        self._SegmentCellsConnection()
        self._SpotAnalysisConnection()
        self._AnalysisPanelConnection()
        self._CelltypePanelConnection()

    def _SegmentCellsClicked(self,b):
        if b == True:
            self.SegmentCellsMainWindow.show()
        else:
            self.SegmentCellsMainWindow.close()
            
    def _SpotAnalysisClicked(self, b):
        if b == True:
            self.SpotAnalysisMainWindow.show()
        else:
            self.SpotAnalysisMainWindow.close()
    
    def _AnalysisPanelClicked(self, b):
        if b == True:
            self.AnalysisPanelMainWindow.show()
        else:
            self.AnalysisPanelMainWindow.close()
    
    def _CelltypePanelClicked(self, b):
        if b == True:
            self.CelltypePanelMainWindow.show()
        else:
            self.CelltypePanelMainWindow.close()

    def _updateMinimumIntensityAbsoluteValue_FromSlider(self, slider, doubleSpinBox, lineEdit):
        slidervalue = float(slider.value())
        doubleSpinBox.setValue(float(slidervalue))
        absolute_value = int(self.current_image.max() * slidervalue * .01)
        lineEdit.setText(f'{int(absolute_value)}')
        
    def _updateMinimumIntensityAbsoluteValue_FromDoubleSpinBox(self, slider, doubleSpinBox, lineEdit):
        doubleSpinValue = float(doubleSpinBox.value())
        slider.setValue(int(doubleSpinValue))
        absolute_value = int(self.current_image.max() * doubleSpinValue *.01)
        lineEdit.setText(f'{int(absolute_value)}')
        
    def _updateBinarizationAbsoluteValue_FromSlider(self, slider, lineEdit, absLineEdit):
        slidervalue = float(slider.value())
        lineEdit.setText(f'{round(slidervalue * .01 ,2)}')
        absolute_value = int(self.current_image.max() * slidervalue * .01)
        absLineEdit.setText(f'{int(absolute_value)}')
        
    def _updateBinarizationAbsoluteValue_FromLineEdit(self, slider, lineEdit, absLineEdit):
        lineEditValue = min(float(lineEdit.text()), 1)
        slider.setValue(float(lineEditValue))
        absolute_value = int(self.current_image.max() * lineEditValue)
        absLineEdit.setText(f'{int(absolute_value)}')
        
    def _SegmentCellsConnection(self):
        """
        set-up connections in SegmentCellsWidget
        
        List:
         Classify cells method comboBox
         Run cell classification pushbutton
         
        """
        def __changeLineEdit(listWidget, lineedit):
            selectedItems = [item.text() for item in listWidget.selectedItems()]
            if 'All' in selectedItems: selectedItems = [str(e) for e in deepcopy(self.cell_list[1:])]
            text = ' '.join(selectedItems)
            lineedit.setText(text)
        re = QtCore.QRegExp('[0-9 ]+')
        self.SegmentCellsWidget.ActivateCellBoundaries_cellBoundariesLineEdit.setValidator(QtGui.QRegExpValidator(re))
        self.SegmentCellsWidget.ActivateCellBoundaries_cellBoundariesListWidget.itemClicked.connect(lambda:__changeLineEdit(self.SegmentCellsWidget.ActivateCellBoundaries_cellBoundariesListWidget,
                                                                                                                            self.SegmentCellsWidget.ActivateCellBoundaries_cellBoundariesLineEdit))
        self.SegmentCellsWidget.ActivateCellBoundaries_cellBoundariesShowPushButton.clicked.connect(self._showBoundaries)
        self.SegmentCellsWidget.ActivateCellBoundaries_cellBoundariesHidePushButton.clicked.connect(self._hideBoundaries)
        self.SegmentCellsWidget.ModulateCell_modulateCellLineEdit.setValidator(QtGui.QRegExpValidator(re))
        self.SegmentCellsWidget.ModulateCell_modulateCellListWidget.itemClicked.connect(lambda:__changeLineEdit(self.SegmentCellsWidget.ModulateCell_modulateCellListWidget,
                                                                                                                self.SegmentCellsWidget.ModulateCell_modulateCellLineEdit))
        self.SegmentCellsWidget.ModulateCell_modulateCellExpandPushButton.clicked.connect(self._runExpandCells)
        self.SegmentCellsWidget.ModulateCell_modulateCellShrinkPushButton.clicked.connect(self._runShrinkCells)
        self.SegmentCellsWidget.ModulateCell_modulateCellCombinePushButton.clicked.connect(self._runCombineCells)
        self.SegmentCellsWidget.ModulateCell_modulateCellDeletePushButton.clicked.connect(self._runRemoveCells)
        
        re = QtCore.QRegExp('\d+\.\d{4}')
        self.SegmentCellsWidget.classifyCells_binarizationFracLineEdit.setValidator(QtGui.QRegExpValidator(re))
        self.SegmentCellsWidget.classifyCells_binarizationFracHorizontalSlider.valueChanged.connect(lambda:self._updateBinarizationAbsoluteValue_FromSlider(self.SegmentCellsWidget.classifyCells_binarizationFracHorizontalSlider,
                                                                                                                                                            self.SegmentCellsWidget.classifyCells_binarizationFracLineEdit,
                                                                                                                                                            self.SegmentCellsWidget.classifyCells_binarizationAbsLineEdit))
        self.SegmentCellsWidget.classifyCells_binarizationShowSketchPushButton.clicked.connect(lambda: self._updateImage(boundaries = True))
        self.SegmentCellsWidget.classifyCells_binarizationShowInitializePushButton.clicked.connect(self._hideBoundaries)
        self.SegmentCellsWidget.classifyCells_autoRunPushButton.clicked.connect(self._runCellSegment)
        self.SegmentCellsWidget.classifyCells_addNewCellPushButton.clicked.connect(self._addNewCell)
        self.SegmentCellsWidget.classifyCells_manualModeONPushButton.clicked.connect(self._updateManualCellClassficationMode)
        self.SegmentCellsWidget.classifyCells_manualModeOFFPushButton.clicked.connect(self._updateManualCellClassficationMode)
        self.SegmentCellsWidget.transientCellContainer_saveCurrentCells.clicked.connect(self._saveCurrentCells)
        self.SegmentCellsWidget.transientCellContainer_ClearCurrentCells.clicked.connect(self._discardCurrentCells)
        self.SegmentCellsWidget.permanentCellContainer_SendCurrentCells.clicked.connect(self._sendPermanentCellContainerToTransient)
        
    def _sendPermanentCellContainerToTransient(self):
        self.request_stored_cell_signal.emit()
        self._reinitializeCellSegmentListWidget(self.cell_list) 
        self._showBoundaries()
    
    def _updateImage(self, boundaries = False):
        """
        update current image
        
        raised when:
         current fov, current cell, current channel changed
         binarization updated
        
        """
        # channel view
        total_image = self.stored_images[self.current_fov][self.working_channel]

        # cell view
        self._checkCellRegulation()
        print(f'ymin: {self.ymin}, ymax: {self.ymax}\nxmin: {self.xmin}, xmax: {self.xmax}')
        if self.current_cell.lower() == 'all':
            self._hideSpots() 
        else:
            self._hideAuxiliary()
        self.current_image = total_image[self.ymin:self.ymax,self.xmin:self.xmax]
        lb = self.current_image[np.isfinite(self.current_image)].max() * self.lb[self.working_channel]#np.quantile(self.current_image[np.isfinite(self.current_image)],self.lb[self.working_channel])
        hb = self.current_image[np.isfinite(self.current_image)].max() * self.hb[self.working_channel]#np.quantile(self.current_image[np.isfinite(self.current_image)],self.hb[self.working_channel])
         
        # putative cell boundaries
        if boundaries:
            self.view.removeItem(self.rough_boundaries)
            self.rough_boundaries = pg.ScatterPlotItem()
            self.binary = np.zeros_like(self.current_image)
            method = self.SegmentCellsWidget.ClassifyCells_methodComboBox.currentText().lower()
            
            if method == 'manual':
                cutoff = float(self.SegmentCellsWidget.classifyCells_binarizationAbsLineEdit.text())
            elif method == 'yen':
                cutoff = skimage.filters.threshold_yen(self.current_image)
            elif method == 'otsu':
                cutoff = skimage.filters.threshold_otsu(self.current_image)
            elif method == 'triangle':
                cutoff = skimage.filters.threshold_triangle(self.current_image)
            elif method == 'li':
                cutoff = skimage.filters.threshold_li(self.current_image)
                
            self.binary[self.current_image > cutoff] = 1
            y,x = np.where(self.binary > skimage.morphology.erosion(self.binary, scind.generate_binary_structure(2,1)).astype(int))
            spots = [{'pos':(y[i],x[i]),
                      'data': 255, 'size': 1,
                      'brush': 'b', 'pen': 'b'} for i in range(len(x))]
            self.rough_boundaries.addPoints(spots)
            self.view.addItem(self.rough_boundaries)

        self.clear()
        if len(self.current_channel) == 1:
            self.setImage(self.current_image,
                        levels=(lb,hb),
                        autoRange = self._autoRange
                        )
        else:
            super_image = np.empty((len(self.current_channel),self.ymax-self.ymin,self.xmax-self.xmin,4))
            for i,ch in enumerate(self.current_channel):
                single_image_original = self.stored_images[self.current_fov][ch][self.ymin:self.ymax,self.xmin:self.xmax]
                cmap = construct_cmap_fromblack(hex2rgb(colors[i]))
                single_image_adjust = imadjust(single_image_original,self.lb[ch],self.hb[ch])
                single_image = 255 * np.array(list(map(cmap, single_image_adjust)))
                super_image[i,:,:,:] = single_image
            self.setImage(super_image.max(0),autoRange = self._autoRange)
    
    def _reinitializeCellSegmentListWidget(self, cell_list):
        self.SegmentCellsWidget.ActivateCellBoundaries_cellBoundariesListWidget.clear()
        self.SegmentCellsWidget.ModulateCell_modulateCellListWidget.clear()
        self.SegmentCellsWidget.ActivateCellBoundaries_cellBoundariesLineEdit.clear()
        self.SegmentCellsWidget.ModulateCell_modulateCellLineEdit.clear()
        
        for i in cell_list:
            self.SegmentCellsWidget.ActivateCellBoundaries_cellBoundariesListWidget.addItem(str(i))
            self.SegmentCellsWidget.ModulateCell_modulateCellListWidget.addItem(str(i))
        
        self.SegmentCellsWidget.ActivateCellBoundaries_cellBoundariesLineEdit.setText(' '.join(self.cell_list[1:]))
    
    def _runCellSegment(self):
        """
        run cell classification
        Watershed method starting from binarization
        
        parameters
         Method | str | Yen, Otsu, Manual
         Minimum_distance_between_cells | int | default 7
         Minimum_size_of_cells | int | default 15
         Maximum_size_of_cells | int | default 10000
        
        """
        # Input parameters
        method = self.SegmentCellsWidget.ClassifyCells_methodComboBox.currentText().lower()
        Minimum_distance_between_cells = int(self.SegmentCellsWidget.ClassifyCells_minDistanceBetweenCellsSpinBox.value())
        Minimum_size_of_cells = int(self.SegmentCellsWidget.ClassifyCells_minSizeOfCellsSpinBox.value())
        Maximum_size_of_cells = int(self.SegmentCellsWidget.ClassifyCells_maxSizeOfCellsSpinBox.value())
        print(f'Classify cell with minimum distance {Minimum_distance_between_cells} and method {method}')
        
        self.binary = np.zeros_like(self.current_image)
        if method == 'manual':
            cutoff = float(self.SegmentCellsWidget.classifyCells_binarizationAbsLineEdit.text())
        elif method == 'yen':
            cutoff = skimage.filters.threshold_yen(self.current_image)
        elif method == 'otsu':
            cutoff = skimage.filters.threshold_otsu(self.current_image)
        elif method == 'triangle':
            cutoff = skimage.filters.threshold_triangle(self.current_image)
        elif method == 'li':
            cutoff = skimage.filters.threshold_li(self.current_image)
        self.binary[self.current_image > cutoff] = 1
        self.opened_binary = skimage.morphology.opening(self.binary)
        self.distance_transformed = scind.morphology.distance_transform_edt(self.opened_binary)
        self.localMax = peak_local_max(self.distance_transformed, indices=False,
                                       min_distance=Minimum_distance_between_cells)
        self.markers = scind.label(self.localMax, structure = np.ones((3,3)))[0]
        self.watershed = skimage.segmentation.watershed(-self.distance_transformed, self.markers, mask = self.opened_binary)
        
        sizes = []
        for i in np.unique(self.watershed)[1:]:
            size = (self.watershed == i).sum()
            if (size < Minimum_size_of_cells) or (size > Maximum_size_of_cells):
                self.watershed[self.watershed == i] = 0
            else:
                sizes.append(size)
        print(f'Mean size of the cell: {np.mean(sizes)}')
        
        self.CellContainer.load_new_cells(self.current_fov, self.watershed)
        self._reindexCells()
        self.SegmentCellsWidget.ActivateCellBoundaries_cellBoundariesLineEdit.setText(' '.join(self.cell_list[1:]))
        self._showBoundaries()
        self.SegmentCellsWidget.transientCellContainer_textBrowser.setText(f'Cell-containing FOVs: {[k for k,v in self.CellContainer.num_cells.items() if v > 0]}\n' +
                                                                           f'Total cells: {np.sum([v for v in self.CellContainer.num_cells.values()])}\n' +
                                                                           ''.join([f'FOV {k}: {v} cells\n' for k,v in self.CellContainer.num_cells.items()]))

    def _saveCurrentCells(self):
        reply = QtWidgets.QMessageBox.warning(self, 'Overwrite cell', 'This will overwrite current cell container.\n Are you sure to proceed?',
                                              QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.No: return
        
        self.save_new_classified_cell_signal.emit()
        self._reinitializeCellSegmentListWidget(self.cell_list)
        
    def _discardCurrentCells(self):
        self.CellContainer.initialize(self.current_fov)
        self.cell_list = ['All']
        self._reinitializeCellSegmentListWidget(self.cell_list)   
        self.SegmentCellsWidget.transientCellContainer_textBrowser.setText(f'Cell-containing FOVs: {[k for k,v in self.CellContainer.num_cells.items() if v > 0]}\n' +
                                                                           f'Total cells: {np.sum([v for v in self.CellContainer.num_cells.values()])}\n' +
                                                                           ''.join([f'FOV {k}: {v} cells\n' for k,v in self.CellContainer.num_cells.items()]))
        self._hideBoundaries()

    def _hideBoundaries(self):
        self.view.removeItem(self.boundaries)
        self.view.removeItem(self.rough_boundaries)
        self.view.removeItem(self.manual_boundaries)
        for text in self.texts: self.view.removeItem(text)
        self.boundaries = pg.ScatterPlotItem()
        self.rough_boundaries = pg.ScatterPlotItem()
        self.manual_boundaries = removable_spot_item(Brush = pg.mkBrush(255,0,255,255), Pen = pg.mkPen(255,0,255,255))
        self.auxiliary = [self.boundaries, self.rough_boundaries, self.manual_boundaries]
        self.texts = []
        self.view.addItem(self.manual_boundaries)
        
    def _showBoundaries(self):
        self._hideBoundaries()
        if self.current_cell.lower() == 'all':
            showCells = self.SegmentCellsWidget.ActivateCellBoundaries_cellBoundariesLineEdit.text().split()
            if len(showCells) == 0: return
            cellIDs = [int(c) - 1 for c in showCells]
            x,y = self.CellContainer.make_boundaries(self.current_fov,cellIDs)
            
            spots = [{'pos':(y[i]+1/2,x[i]+1/2),
                    'data': 255, 'size': 1,
                    'brush': 'r', 'pen': 'r'} for i in range(len(x))]
            self.boundaries.addPoints(spots)
        
            for cid in cellIDs:
                x,y = self.CellContainer.data[self.current_fov][cid]
                text = pg.TextItem('', color = (255,255,255))
                text.setText(str(cid + 1))
                text.setPos(np.mean(y), np.mean(x)) 
                self.texts.append(text)
                    
            self.view.addItem(self.boundaries)
            for text in self.texts: self.view.addItem(text)
        else:
            self._checkCellRegulation()
            cid = int(self.current_cell) - 1
            x,y = self.CellContainer.make_boundaries(self.current_fov,cid)
            spots = [{'pos':(y[i] - self.ymin, x[i] - self.xmin),
                      'data': 255, 'size': 10,
                      'brush':'r', 'pen': 'r'} for i in range(len(x))]
            self.boundaries.addPoints(spots)
            self.view.addItem(self.boundaries)      
        self._updateImage()
    
    def _reindexCells(self):
        print('Reindexing Cells...')
        current_cells = self.CellContainer.data[self.current_fov]
        print(f'Cell number before: {self.CellContainer.num_cells[self.current_fov]}')
        nice_cells = []
        for cells in current_cells:
            assert isinstance(cells, tuple)
            if len(cells) > 0:
                if len(cells[0]): nice_cells.append(cells)
        self.CellContainer.data[self.current_fov] = nice_cells
        self.CellContainer.num_cells[self.current_fov] = len(self.CellContainer.data[self.current_fov])
        print(f'Cell number after: {self.CellContainer.num_cells[self.current_fov]}')
        self.cell_list = ['All'] + [str(e) for e in list(range(1,self.CellContainer.num_cells[self.current_fov]+1))]
        self._reinitializeCellSegmentListWidget(self.cell_list)

    def _runExpandCells(self):
        expands = self.SegmentCellsWidget.ModulateCell_modulateCellLineEdit.text().split()
        if len(expands) == 0: return
        expand_cells = [int(e)-1 for e in expands]
        h,w = self.CellContainer.h[self.current_fov], self.CellContainer.w[self.current_fov]
        canvas = np.zeros((h,w))
        
        for c in expand_cells:
            x,y = self.CellContainer.data[self.current_fov][c]
            canvas[y,x] = 1
            dilated_cell = skimage.morphology.dilation(canvas, scind.generate_binary_structure(2,1))
            self.CellContainer.update_cell(self.current_fov, c, dilated_cell)
            canvas[y,x] = 0
            
        self._showBoundaries()

    def _runShrinkCells(self):
        shrinks = self.SegmentCellsWidget.ModulateCell_modulateCellLineEdit.text().split()
        if len(shrinks) == 0: return
        shrink_cells = [int(e)-1 for e in shrinks]

        h,w = self.CellContainer.h[self.current_fov], self.CellContainer.w[self.current_fov]
        canvas = np.zeros((h,w))
        
        for c in shrink_cells:
            x,y = self.CellContainer.data[self.current_fov][c]
            canvas[y,x] = 1
            erosed_cell = skimage.morphology.erosion(canvas, scind.generate_binary_structure(2,1))
            self.CellContainer.update_cell(self.current_fov, c, erosed_cell)
            canvas[y,x] = 0

        self._showBoundaries()

    def _runCombineCells(self):
        self._hideBoundaries()
        combines = self.SegmentCellsWidget.ModulateCell_modulateCellLineEdit.text().split()
        if len(combines) == 0: return
        combined_cells = [int(e)-1 for e in combines]
        min_combined_cell = np.min(combined_cells)
        x,y = self.CellContainer.data[self.current_fov][min_combined_cell]
        
        for c in combined_cells:
            if c != min_combined_cell:
                xx,yy = self.CellContainer.data[self.current_fov][c]
                y = np.concatenate((y.reshape(-1,1),yy.reshape(-1,1)),axis=0)
                x = np.concatenate((x.reshape(-1,1),xx.reshape(-1,1)),axis=0)
                self.CellContainer.data[self.current_fov][c] = ()
        self.CellContainer.data[self.current_fov][min_combined_cell] = (x,y)
        self.CellContainer.num_cells[self.current_fov] -= len(combined_cells) - 1
        
        self._reindexCells()
        self._showBoundaries()
        self.SegmentCellsWidget.transientCellContainer_textBrowser.setText(f'Cell-containing FOVs: {[k for k,v in self.CellContainer.num_cells.items() if v > 0]}\n' +
                                                                           f'Total cells: {np.sum([v for v in self.CellContainer.num_cells.values()])}\n' +
                                                                           ''.join([f'FOV {k}: {v} cells\n' for k,v in self.CellContainer.num_cells.items()]))
        
    def _runRemoveCells(self):
        self._hideBoundaries()
        removes = self.SegmentCellsWidget.ModulateCell_modulateCellLineEdit.text().split()
        if len(removes) == 0: return
        removed_cells = [int(e)-1 for e in removes]
        
        for c in removed_cells:
            self.CellContainer.data[self.current_fov][c] = ()
        self.CellContainer.num_cells[self.current_fov] -= len(removed_cells)
        
        self._reindexCells()
        self._showBoundaries()
        self.SegmentCellsWidget.transientCellContainer_textBrowser.setText(f'Cell-containing FOVs: {[k for k,v in self.CellContainer.num_cells.items() if v > 0]}\n' +
                                                                           f'Total cells: {np.sum([v for v in self.CellContainer.num_cells.values()])}\n' +
                                                                           ''.join([f'FOV {k}: {v} cells\n' for k,v in self.CellContainer.num_cells.items()]))

    def _updateManualCellClassficationMode(self):
        if self.SegmentCellsWidget.classifyCells_manualModeONPushButton.isEnabled():
            self.SegmentCellsWidget.classifyCells_manualModeOFFPushButton.setEnabled(True)
            self.SegmentCellsWidget.classifyCells_manualModeONPushButton.setEnabled(False)
            self.manual_boundaries._setImageSize(self.current_image.shape)
            self.manual_boundaries._turnONManualModification()
            print('Turning on manual cell classification mode...')
            
        elif self.SegmentCellsWidget.classifyCells_manualModeOFFPushButton.isEnabled():
            self.SegmentCellsWidget.classifyCells_manualModeOFFPushButton.setEnabled(False)
            self.SegmentCellsWidget.classifyCells_manualModeONPushButton.setEnabled(True)      
            self.manual_boundaries._turnOFFManualModification()  
            print('Turning off manual cell classification mode...')
        
    def _addNewCell(self):
        y,x = (np.array(self.manual_boundaries.data['x']) - .5).astype(int), (np.array(self.manual_boundaries.data['y']) - .5).astype(int)
        self._checkCellRegulation()
        polygon = Polygon([[y[i] + self.ymin, x[i] + self.xmin] for i in range(len(y))])
        for img in self.stored_images[self.current_fov].values():
            h,w = img.shape
            if self.CellContainer.h[self.current_fov] == 0:
                self.CellContainer.h[self.current_fov] = h
            if self.CellContainer.w[self.current_fov] == 0:
                self.CellContainer.w[self.current_fov] = w
        
        ymin,xmin,ymax,xmax = [int(b) for b in polygon.bounds]
        ymin = max(0,ymin)
        xmin = max(0,xmin)
        ymax = min(h,ymax)
        xmax = min(w,xmax)
        yy,xx = [],[]
        for y in range(ymin,ymax+1):
            for x in range(xmin,xmax+1):
                if Point(y,x).within(polygon):
                    yy.append(y)
                    xx.append(x)
        self.CellContainer.add_new_cell(self.current_fov, (np.array(xx),np.array(yy)))
        
        self.view.removeItem(self.manual_boundaries)            
        self.manual_boundaries = removable_spot_item(Brush = pg.mkBrush(255,0,255,255), Pen = pg.mkPen(255,0,255,255))
        self.auxiliary = [self.boundaries, self.rough_boundaries, self.manual_boundaries]
        self.view.addItem(self.manual_boundaries)
        self._reindexCells()
        self._showBoundaries()
        self._updateManualCellClassficationMode()
        self._updateManualCellClassficationMode()
        self.SegmentCellsWidget.transientCellContainer_textBrowser.setText(f'Cell-containing FOVs: {[k for k,v in self.CellContainer.num_cells.items() if v > 0]}\n' +
                                                                           f'Total cells: {np.sum([v for v in self.CellContainer.num_cells.values()])}\n' +
                                                                           ''.join([f'FOV {k}: {v} cells\n' for k,v in self.CellContainer.num_cells.items()]))
    
    def _SpotAnalysisConnection(self):
        """
        set-up connections in SpotAnalysisWidget
        
        List:
         Run spot location button
         Run spot size measurement button
         Save current spot locations
         Delete current spot locations
         Save current spot sizes
         Delete current spot locations
        """
        re = QtCore.QRegExp('^\d+$')
        self.SpotAnalysisWidget.spotLocalizationParameters_intensityThresholdSlider.valueChanged.connect(lambda:
            self._updateMinimumIntensityAbsoluteValue_FromSlider(self.SpotAnalysisWidget.spotLocalizationParameters_intensityThresholdSlider,
                                                                 self.SpotAnalysisWidget.spotLocalizationParameters_intensityThreshold_doubleSpinBox,
                                                                 self.SpotAnalysisWidget.spotLocalizationParameters_intensityThresholdAbsoluteLineEdit))
        self.SpotAnalysisWidget.spotLocalizationParameters_intensityThreshold_doubleSpinBox.valueChanged.connect(lambda:
            self._updateMinimumIntensityAbsoluteValue_FromDoubleSpinBox(self.SpotAnalysisWidget.spotLocalizationParameters_intensityThresholdSlider,
                                                                 self.SpotAnalysisWidget.spotLocalizationParameters_intensityThreshold_doubleSpinBox,
                                                                 self.SpotAnalysisWidget.spotLocalizationParameters_intensityThresholdAbsoluteLineEdit))
        self.SpotAnalysisWidget.spotLocalizationParameters_intensityThresholdAbsoluteLineEdit.setValidator(QtGui.QRegExpValidator(re))
        self.SpotAnalysisWidget.runSpotLocalization_pushButton.clicked.connect(self._runSpotLocalization)
        
        self.SpotAnalysisWidget.spotSizeMeasurementParameters_intensityThresholdSlider.valueChanged.connect(lambda:
            self._updateMinimumIntensityAbsoluteValue_FromSlider(self.SpotAnalysisWidget.spotSizeMeasurementParameters_intensityThresholdSlider,
                                                                 self.SpotAnalysisWidget.spotSizeMeasurementParameters_intensityThreshold_doubleSpinBox,
                                                                 self.SpotAnalysisWidget.spotSizeMeasurementParameters_intensityThresholdAbsoluteLineEdit))
        self.SpotAnalysisWidget.spotSizeMeasurementParameters_intensityThreshold_doubleSpinBox.valueChanged.connect(lambda:
            self._updateMinimumIntensityAbsoluteValue_FromDoubleSpinBox(self.SpotAnalysisWidget.spotSizeMeasurementParameters_intensityThresholdSlider,
                                                                 self.SpotAnalysisWidget.spotSizeMeasurementParameters_intensityThreshold_doubleSpinBox,
                                                                 self.SpotAnalysisWidget.spotSizeMeasurementParameters_intensityThresholdAbsoluteLineEdit))
        self.SpotAnalysisWidget.spotSizeMeasurementParameters_intensityThresholdAbsoluteLineEdit.setValidator(QtGui.QRegExpValidator(re))
        self.SpotAnalysisWidget.runSpotQualityControl_pushButton.clicked.connect(self._runSpotQualityControl)
        
        self.SpotAnalysisWidget.SpotAnalysisPanel_TransientSpotContainerSavePushButton.clicked.connect(self._runSaveSpots)
        self.SpotAnalysisWidget.SpotAnalysisPanel_TransientSpotContainerClearChannelPushButton.clicked.connect(self._runInitializeCurrentSpots)
        self.SpotAnalysisWidget.SpotAnalysisPanel_TransientSpotContainerClearAllPushButton.clicked.connect(self._runClearAllCurrentSpots)
        self.SpotAnalysisWidget.spotVisualization_showSpotsPushButton.clicked.connect(self._showSpots)
        self.SpotAnalysisWidget.spotVisualization_hideSpotsPushButton.clicked.connect(self._hideSpots)
        
        self.SpotAnalysisWidget.spotVisualization_TurnONManualModePushButton.clicked.connect(self._updateManualSpotModification)
        self.SpotAnalysisWidget.spotVisualization_TurnOFFManualModePushButton.clicked.connect(self._updateManualSpotModification)
        self.SpotAnalysisWidget.SpotAnalysisPanel_PermanentSpotContainerSendPushButton.clicked.connect(self._sendPermanentSpotContainerToTransient)

    def _checkCellRegulation(self):
        # regulation check    
        if self.current_cell.lower() == 'all':
            self.xmin, self.ymin = 0,0
            self.ymax, self.xmax = self.stored_images[self.current_fov][self.working_channel].shape
        else:
        #current_image = self.current_image
            id = int(self.current_cell) - 1
            x,y = self.CellContainer.data[self.current_fov][id]
            h,w = self.CellContainer.h[self.current_fov], self.CellContainer.w[self.current_fov]
            self.ymin, self.ymax = max(0,y.min()-self.w), min(h, y.max()+self.w)
            self.xmin, self.xmax = max(0,x.min()-self.w), min(w, x.max()+self.w)

    def _runSpotLocalization(self):
        """
        run Spot Localization
        
        parameters
         threshold | float (0-100) | minimum intensity threshold (% quantile) 
         minimum_distance | int | minimum distance between spots
        
        """
        print(f'analyze spot location...\nworking channel: {self.working_channel}')
        self._updateMinimumIntensityAbsoluteValue_FromDoubleSpinBox(self.SpotAnalysisWidget.spotLocalizationParameters_intensityThresholdSlider,
                                                                    self.SpotAnalysisWidget.spotLocalizationParameters_intensityThreshold_doubleSpinBox,
                                                                    self.SpotAnalysisWidget.spotLocalizationParameters_intensityThresholdAbsoluteLineEdit)
        if len(self.current_channel) > 1:
            self.current_channel = self.working_channel
            self._updateImage()
        self.view.removeItem(self.spots[self.working_channel])
        cutoff_value = int(self.SpotAnalysisWidget.spotLocalizationParameters_intensityThresholdAbsoluteLineEdit.text())
        minimum_distance = int(self.SpotAnalysisWidget.spotLocalizationMinimumDistance_spinBox.value())
        self._checkCellRegulation()

        coordinates = peak_local_max(self.current_image, min_distance = minimum_distance, threshold_abs = cutoff_value)
        x = coordinates[:,1]
        y = coordinates[:,0]
        
        if len(x) > 500:
            reply = QtWidgets.QMessageBox.warning(self, 'Spot localization', f'{len(x)} spots are detected. Are you sure to proceed?',
                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.No: return
        
        spots = [{'pos': (y[i] + 1/2,x[i] + 1/2),
                  'data': 255, 'size':10,
                  'brush':pg.mkBrush(255,255,0,0),
                  'pen':pg.mkPen(255,255,0,255)} for i in range(len(x))]
        self.spots[self.working_channel] = removable_spot_item()
        self.spots[self.working_channel].addPoints(spots)
        self.view.addItem(self.spots[self.working_channel])
        
        if self.current_cell.lower() == 'all':
            self.SpotContainer.reinitialize(self.current_fov, self.working_channel)
        else:
            cellID = int(self.current_cell) - 1
            canvas = np.zeros((self.CellContainer.h[self.current_fov], self.CellContainer.w[self.current_fov]))
            x,y = self.CellContainer.data[self.current_fov][cellID]
            canvas[y,x] = 1
            self.SpotContainer.remove_spots_from_area(self.current_fov, self.working_channel, canvas)
            
        self.current_spot_indices = self.SpotContainer.load_new_spots(self.current_fov, self.working_channel,
                                                                      coordinates = np.concatenate([(y+self.ymin).reshape(-1,1),
                                                                                                    (x+self.xmin).reshape(-1,1)], axis=1))
        print(self.current_spot_indices)
        self.SpotAnalysisWidget.SpotAnalysisPanel_TransientSpotContainerTestBrowser.setText(f'Current FOV: {self.current_fov}\n' +
                                                                                            f''.join([f'Channel {ch}: {len(v)} spots\n' for (f,ch),v in self.SpotContainer.data.items()]))

    def _updateManualSpotModification(self):
        if self.SpotAnalysisWidget.spotVisualization_TurnONManualModePushButton.isEnabled():
            self.spots[self.working_channel]._setImageSize(self.current_image.shape)
            self.spots[self.working_channel]._turnONManualModification()
            self.SpotAnalysisWidget.spotVisualization_TurnONManualModePushButton.setEnabled(False)
            self.SpotAnalysisWidget.spotVisualization_TurnOFFManualModePushButton.setEnabled(True)
            print(f'Turning on manual spot modification on channel {self.working_channel}...')
            
        elif self.SpotAnalysisWidget.spotVisualization_TurnOFFManualModePushButton.isEnabled():
            self.spots[self.working_channel]._turnOFFManualModification()
            self.SpotAnalysisWidget.spotVisualization_TurnONManualModePushButton.setEnabled(True)
            self.SpotAnalysisWidget.spotVisualization_TurnOFFManualModePushButton.setEnabled(False)
            print(f'Turning off manual spot modification on channel {self.working_channel}...')

    def _runSaveSpots(self):
        reply = QtWidgets.QMessageBox.warning(self, 'Save spots', 'This will overwrite current spot container.\n Are you sure to proceed?',
                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.No: return
        
        print(f'Save {self.working_channel} spots...')
        y,x = np.array(self.spots[self.working_channel].data['x']) - .5, np.array(self.spots[self.working_channel].data['y']) - .5
        self._checkCellRegulation()
        self.SpotContainer.data[(self.current_fov, self.working_channel)] = np.concatenate([(y+self.ymin).reshape(-1,1),
                                                                                            (x+self.xmin).reshape(-1,1)], axis=1)
        self.save_new_localized_spot_signal.emit()
        channels = []
        for key in self.SpotContainer.data.keys():
            f,ch = key
            if (f == self.current_fov) and (ch not in channels): channels.append(ch)
            
        self.SpotAnalysisWidget.spotLocalization_channelListWidget.clear()
        for i,ch in enumerate(channels):
            if ch == self.working_channel:
                currentID = i    
            self.SpotAnalysisWidget.spotLocalization_channelListWidget.addItem(str(ch))
        
        self.SpotAnalysisWidget.spotLocalization_channelListWidget.setCurrentRow(currentID)
        self.SpotAnalysisWidget.SpotAnalysisPanel_TransientSpotContainerTestBrowser.setText(f'Current FOV: {self.current_fov}\n' +
                                                                                            f''.join([f'Channel {ch}: {len(v)} spots\n' for (f,ch),v in self.SpotContainer.data.items()]))

    def _runInitializeCurrentSpots(self):
        self.SpotContainer.reinitialize(self.current_fov,self.working_channel)
        print(f'reset {self.current_fov}, {self.working_channel}')
        self.current_spot_indices = []
        self.SpotAnalysisWidget.SpotAnalysisPanel_TransientSpotContainerTestBrowser.setText(f'Current FOV: {self.current_fov}\n' +
                                                                                            f''.join([f'Channel {ch}: {len(v)} spots\n' for (f,ch),v in self.SpotContainer.data.items()]))

    def _runClearAllCurrentSpots(self):
        print(f'reset {self.current_fov}, all channels')
        for ch in self.channel_list:
            self.SpotContainer.reinitialize(self.current_fov,ch)
        self.current_spot_indices = []
        self.SpotAnalysisWidget.SpotAnalysisPanel_TransientSpotContainerTestBrowser.setText(f'Current FOV: {self.current_fov}\n' +
                                                                                            f''.join([f'Channel {ch}: {len(v)} spots\n' for (f,ch),v in self.SpotContainer.data.items()]))

    def _showSpots(self):
        self._hideSpots()
        selected_channels = self.SpotAnalysisWidget.spotLocalization_channelListWidget.selectedItems()
        self._checkCellRegulation()
        if len(selected_channels) == 0:
            channels = [self.working_channel]
            spot_pens = [pg.mkPen(255,255,0,255)]
        elif len(selected_channels) == 1:
            channels = [ch.text() for ch in selected_channels]
            spot_pens = [pg.mkPen(255,255,0,255)]
        else:
            channels, spot_pens = [],[]
            for i,ch in enumerate(selected_channels):
                channels.append(ch.text())
                r,g,b = hex2rgb(colors[i])
                spot_pens.append(pg.mkPen(r,g,b,255))
            
        for j,ch in enumerate(channels):
            coords = self.SpotContainer.data[(self.current_fov,ch)]
            target = (self.ymin <= coords[:,0]) & (coords[:,0] < self.ymax) & (self.xmin <= coords[:,1]) & (coords[:,1] < self.xmax)
            y,x = coords[target,0] - self.ymin, coords[target,1] - self.xmin

            spots = [{'pos': (y[i] + 1/2,x[i] + 1/2),
                      'data': 255, 'size':10,
                      'brush':pg.mkBrush(255,255,0,0),
                      'pen': spot_pens[j]} for i in range(len(x))]
            self.spots[ch] = removable_spot_item()
            self.spots[ch].addPoints(spots)
            self.view.addItem(self.spots[ch])

    def _hideSpots(self):
        for spots in self.spots.values(): self.view.removeItem(spots)
        self.spots = {ch: removable_spot_item() for ch in self.channel_list}
        self.SpotAnalysisWidget.spotVisualization_TurnOFFManualModePushButton.setEnabled(False)
        self.SpotAnalysisWidget.spotVisualization_TurnONManualModePushButton.setEnabled(True)
        for spots in self.spots.values(): self.view.addItem(spots)

    def _sendPermanentSpotContainerToTransient(self):
        self.request_stored_spots_signal.emit()
        self.SpotAnalysisWidget.SpotAnalysisPanel_TransientSpotContainerTestBrowser.setText(f'Current FOV: {self.current_fov}\n' +
                                                                                            f''.join([f'Channel {ch}: {len(v)} spots\n' for (f,ch),v in self.SpotContainer.data.items()]))
    
    def _runSpotQualityControl(self):
        """
        run spot quality control
        
        """
        self._updateMinimumIntensityAbsoluteValue_FromSlider(self.SpotAnalysisWidget.spotSizeMeasurementParameters_intensityThresholdSlider,
                                                             self.SpotAnalysisWidget.spotSizeMeasurementParameters_intensityThreshold_doubleSpinBox,
                                                             self.SpotAnalysisWidget.spotSizeMeasurementParameters_intensityThresholdAbsoluteLineEdit)
        
        intensity_cutoff = int(self.SpotAnalysisWidget.spotSizeMeasurementParameters_intensityThresholdAbsoluteLineEdit.value())
        min_size = int(self.SpotAnalysisWidget.spotQualityControl_minimumSpotSizeSpinBox.value())
        max_size = int(self.SpotAnalysisWidget.spotQualityControl_maximumSpotSizeSpinBox.value())
        self._checkCellRegulation()
        print(f'run spot quality control with min_spot_size: {min_size} and max_spot_size: {max_size}')
        
        indices = []
        sizes = []
        
        coords = self.SpotContainer.data[(self.current_fov, self.working_channel)]
        target = (self.ymin <= coords[:,0]) & (coords[:,0] < self.ymax) & (self.xmin <= coords[:,1]) & (coords[:,1] < self.xmax)
        brightness = self.current_image[coords[target,0] - self.ymin,coords[target,1] - self.xmin]
        print(brightness.max())
        binary = np.zeros_like(self.current_image)
        binary[self.current_image >= intensity_cutoff] = 1
        labels = skimage.measure.label(binary)
        
        for i in range(len(coords)):
            if target[i]:
                l = labels[int(coords[i,0] - self.ymin), int(coords[i,1] - self.xmin)]
                if l > 0:
                    size = (labels == l).sum()
                    if (size < min_size) | (size > max_size):
                        indices.append(i)
                    else: sizes.append(size)
                else: indices.append(i)
                
        print(f'{len(indices)} cells are filtered out.\nAverage size: {np.mean(np.array(sizes))}, Standard deviation: {np.std(np.array(sizes))}')
            
        self.SpotContainer.remove_spots_from_indices(self.current_fov, self.working_channel, indices)
        self._showSpots()
    
    def _set_lb_hb(self, lb, hb):
        self.lb = lb
        self.hb = hb
        #print(self.lb, self.hb)
  
    def _hideAuxiliary(self):
        self._hideBoundaries()
        self._hideSpots()
            
    def _loadImages(self, images):
        self.stored_images = images
        
    def _update_current_fov_channel_and_cell(self, fov, channel, channels, cell):
        self.current_fov = fov
        self.working_channel = channel
        self.current_channel = channels
        self.current_cell = cell
        
    def _AnalysisPanelConnection(self):
        """
        set-up connections in AnalysisPanel
        
        List:
        """
        self.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotDistanceAddConditionPushButton.clicked.connect(self._addCondition)
        self.AnalysisPanelWidget.AnalysisPanelFrame_ConditionRemoveConditionPushButton.clicked.connect(self._removeCondition)
        self.AnalysisPanelWidget.AnalysisPanelFrame_ShowSelectedCellPushButton.clicked.connect(self._showSelectedCell)
        self.AnalysisPanelWidget.AnalysisPanelFrame_CalculateSpotSizesPushButton.clicked.connect(self._calculateSpotSize)
        self.AnalysisPanelWidget.AnalysisPanelFrame_CountSpotsPerCellPushButton.clicked.connect(self._countSpotsPerCell)
        self.AnalysisPanelWidget.AnalysisPanelFrame_MeasureSpotDistancePushButton.clicked.connect(self._measureSpotDistance)
        self.AnalysisPanelWidget.AnalysisPanelWorkBox_LinkSpotsAndCellsPushButton.clicked.connect(self._linkSpotsAndCells)
        self.AnalysisPanelWidget.AnalysisPanelFrame_ExportCellsAndSpotsPushButton.clicked.connect(self._exportSelectedCellsAndSpots)
        self.AnalysisPanelWidget.AnalysisPanelFrame_InitializePushButton.clicked.connect(self._initializeSpotSize)
        self.AnalysisPanelWidget.AnalysisPanelFrame_ExportSelectedCellPushButton.clicked.connect(self._exportSelectedCell)
        self.AnalysisPanelWidget.AnalysisPanelFrame_showSpotBrightnessShowPushButton.clicked.connect(self._showSpotBrightness)
        
        _listWidgetValueToLabel(self.AnalysisPanelWidget.AnalysisPanelWorkBox_FieldOfViewsListWidget,
                                self.AnalysisPanelWidget.AnalysisPanelWorkBox_FieldOfViewsLabel)
        _listWidgetValueToLabel(self.AnalysisPanelWidget.AnalysisPanelWorkBox_CellTypesListWidget,
                                self.AnalysisPanelWidget.label_8)

    def _linkSpotsAndCells(self):
        """
        link permanent SpotContainers and CellContainer
        """
        self.SpotMetaDataAnalyzer = SpotMetaDataAnalyzer()
        self.SpotMetaDataAnalyzer.link_cell_and_spot(fov_list = self.fov_list,
                                                     channel_list = self.channel_list,
                                                     cellContainer = self.CellContainer,
                                                     spotContainer = self.SpotContainer,
                                                     images = self.stored_images,
                                                     celltype_mode = self.celltype_mode,
                                                     celltype_determination = self.celltype_determination)
        self.link_spot_cell_signal.emit(True)
        
    def _exportSelectedCellsAndSpots(self):
        save_file = QtWidgets.QFileDialog.getSaveFileName(self, 'Save selected cells and spots', '.', 'selected_cells_spots (*cs)')[0]
        if save_file == '':
            print(f'Save failed. No file selected')
        else:
            with open(save_file, 'wb') as f:
                cells = [cell.save() for cell in self.selected_cells]
                spots = [spot.save() for spot in self.selected_spots]
                dump((cells, spots), f)
            print(f'Finished Save: {save_file}')

    def _exportSelectedCell(self):
        cid = int(self.AnalysisPanelWidget.AnalysisPanelFrame_targetCellsListWidget.currentItem().text().split('Cell')[1].split()[0]) - 1
        cell = self.selected_cells[cid]
        save_file = QtWidgets.QFileDialog.getSaveFileName(self, 'Save selected cell image', '.', 'selected_cell (*tiff)')[0]
        if save_file == '':
            print(f'Save failed, No file selected.')
        else:
            save_file_name = save_file.split('.tiff')[0]
            for ch in cell.channels:
                img = Image.fromarray(cell.image[ch])
                img.save(save_file_name + f'{ch}.tiff', 'tiff')

    def _initializeSpotSize(self):
        for spot in self.selected_spots:
            spot.size = 0
    
    def _addCondition(self):
        """
        add new condition followed by:
        
            AnalysisPanelWorkBox_CellTypesListWidget
            AnalysisPanelWorkBox_FieldOfViewsListWidget
            AnalysisPanelWorkBox_SpotSizesListWidget
            AnalysisPanelWorkBox_SpotSizesMinimumSpinBox
            AnalysisPanelWorkBox_SpotSizesMaximumSpinBox
            AnalysisPanelWorkBox_NumberOfSpotsListWidget
            AnalysisPanelWorkBox_NumberOfSpotsMinimumSpinBox
            AnalysisPanelWorkBox_NumberOfSpotsMaximumSpinBox
            AnalysisPanelWorkBox_SpotDistanceChannel1ListWidget
            AnalysisPanelWorkBox_SpotDistanceChannel2ListWidget
            AnalysisPanelWorkBox_SpotDistanceMindistanceLineEdit
            AnalysisPanelWorkBox_SpotDistanceMaxdistanceLineEdit
            AnalysisPanelWorkBox_SpotDistanceCenterToCenterRadioButton
            AnalysisPanelWorkBox_BrightnessListWidget
            AnalysisPanelWorkBox_BrightnessMinimumSpinBox
            AnalysisPanelWorkBox_BrightnessMaximumSpinBox
            
        and add condition text on the panel
            AnalysisPanelFrame_ConditionTextBrowser
            
        with style
            Index: 0
            Celltype: All
            FOV: All
            Brightness: {Channel: All, Min: 0, Max: 1,000,000}
            SpotSizes: {Channel: All, Min: 0, Max: 1,000,000}
            Number of Spots: {Channel: All, Min: 0, Max: 1,000,000}
            Spot Distance: {Channel1: None, Channel2: None, Min: 0, Max: 1,000,000}
            ---------------------------------------------------------------------------
        """
        if self.SpotMetaDataAnalyzer is None: return
        self.AnalysisPanelWidget.AnalysisPanelFrame_ConditionTextBrowser.clear()
        self.AnalysisPanelWidget.AnalysisPanelFrame_targetCellsListWidget.clear()
        print(
            self.AnalysisPanelWidget.AnalysisPanelWorkBox_CellTypesListWidget.selectedItems(),
            self.AnalysisPanelWidget.AnalysisPanelWorkBox_FieldOfViewsListWidget.selectedItems(),
            self.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotSizesListWidget.selectedItems(),
            self.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotSizesMinimumSpinBox.value(),
            self.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotSizesMaximumSpinBox.value(),
            self.AnalysisPanelWidget.AnalysisPanelWorkBox_NumberOfSpotsListWidget.selectedItems(),
            self.AnalysisPanelWidget.AnalysisPanelWorkBox_NumberOfSpotsMinimumSpinBox.value(),
            self.AnalysisPanelWidget.AnalysisPanelWorkBox_NumberOfSpotsMaximumSpinBox.value(),
            self.AnalysisPanelWidget.AnalysisPanelWorkBox_BrightnessListWidget.selectedItems(),
            self.AnalysisPanelWidget.AnalysisPanelWorkBox_BrightnessMinimumSpinBox.value(),
            self.AnalysisPanelWidget.AnalysisPanelWorkBox_BrightnessMaximumSpinBox.value(),
            self.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotDistanceChannel1ListWidget.currentItem(),
            self.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotDistanceChannel2ListWidget.currentItem(),
            self.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotDistanceMindistanceDoubleSpinBox.value(),
            self.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotDistanceMaxdistanceDoubleSpinBox.value(),
            self.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotDistanceCenterToCenterRadioButton.isChecked()
        )
        celltypes = [ct.text() for ct in self.AnalysisPanelWidget.AnalysisPanelWorkBox_CellTypesListWidget.selectedItems()]   
        if 'All' in celltypes: celltypes = ['All']
        field_of_views = [fov.text() for fov in self.AnalysisPanelWidget.AnalysisPanelWorkBox_FieldOfViewsListWidget.selectedItems()]
        if 'All' in field_of_views: field_of_views = ['All']
        spot_size_channel = [ch.text() for ch in self.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotSizesListWidget.selectedItems()]
        spot_size_minimum = self.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotSizesMinimumSpinBox.value()
        spot_size_maximum = self.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotSizesMaximumSpinBox.value()
        num_of_spot_channel = [ch.text() for ch in self.AnalysisPanelWidget.AnalysisPanelWorkBox_NumberOfSpotsListWidget.selectedItems()]
        num_of_spot_minimum = self.AnalysisPanelWidget.AnalysisPanelWorkBox_NumberOfSpotsMinimumSpinBox.value()
        num_of_spot_maximum = self.AnalysisPanelWidget.AnalysisPanelWorkBox_NumberOfSpotsMaximumSpinBox.value()
        spot_brightness_channel = [ch.text() for ch in self.AnalysisPanelWidget.AnalysisPanelWorkBox_BrightnessListWidget.selectedItems()]
        spot_brightness_minimum = self.AnalysisPanelWidget.AnalysisPanelWorkBox_BrightnessMinimumSpinBox.value()
        spot_brightness_maximum = self.AnalysisPanelWidget.AnalysisPanelWorkBox_BrightnessMaximumSpinBox.value()
        spot_distance_ch1 = self.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotDistanceChannel1ListWidget.currentItem().text() if self.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotDistanceChannel1ListWidget.currentItem() is not None else 'None'
        spot_distance_ch2 = self.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotDistanceChannel2ListWidget.currentItem().text() if self.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotDistanceChannel2ListWidget.currentItem() is not None else 'None'
        spot_distance_minimum = self.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotDistanceMindistanceDoubleSpinBox.value()
        spot_distance_maximum = self.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotDistanceMaxdistanceDoubleSpinBox.value()
        spot_distance_center2center = self.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotDistanceCenterToCenterRadioButton.isChecked()
        new_style = self._empty_style()
        new_style['Index'] = len(self.conditions)
        new_style['Celltype'] = celltypes
        new_style['FOV'] = field_of_views
        new_style['Brightness'] = {'Channel': spot_brightness_channel,
                                   'Min': spot_brightness_minimum,
                                   'Max': spot_brightness_maximum}
        new_style['SpotSizes'] = {'Channel': spot_size_channel,
                                  'Min': spot_size_minimum,
                                  'Max': spot_size_maximum}
        new_style['Number of Spots'] = {'Channel': num_of_spot_channel,
                                        'Min': num_of_spot_minimum,
                                        'Max': num_of_spot_maximum}
        new_style['Spot Distance'] = {'Channel1': spot_distance_ch1,
                                      'Channel2': spot_distance_ch2,
                                      'Min': spot_distance_minimum,
                                      'Max': spot_distance_maximum,
                                      'CenterToCenter': spot_distance_center2center}
        new_style['Closing'] = '---------------------------------------------------------------------------'
        self.conditions.append(new_style)
        for style in self.conditions: self.AnalysisPanelWidget.AnalysisPanelFrame_ConditionTextBrowser.append(self._style_dict_to_str(style))
        self.selected_cells, self.selected_spots = self._convert_styles_to_conditions()
        self.AnalysisPanelWidget.AnalysisPanelFrame_ConditionTextBrowser.append(f'Total selected cell: {len(self.selected_cells)}\nTotal selected spots: {len(self.selected_spots)}')
        for i in range(len(self.selected_cells)): 
            self.AnalysisPanelWidget.AnalysisPanelFrame_targetCellsListWidget.addItem(f'Cell {i+1} (fov: {self.selected_cells[i].fov}, id: {self.selected_cells[i].id})')

    def _empty_style(self):
        return {'Index': 0,
                'Celltype': 'All',
                'FOV': 'All',
                'Brightness': {'Channel': 'All',
                               'Min': 0,
                               'Max': 1000000},
                'SpotSizes': {'Channel': 'All',
                              'Min': 0,
                              'Max': 1000000},
                'Number of Spots': {'Channel': 'All',
                                    'Min': 0,
                                    'Max': 1000000},
                'Spot Distance': {'Channel1': 'None',
                                  'Channel2': 'None',
                                  'Min': 0,
                                  'Max': 1000000,
                                  'CenterToCenter': True},
                'Closing': '---------------------------------------------------------------------------'}

    def _style_dict_to_str(self, style_dict):
        result_str = []
        for key,value in style_dict.items():
            if key.lower() != 'closing': result_str.append(key+': ')
            result_str.append(f'{value}')
            result_str.append('\n')
        return ''.join(result_str)

    def _convert_styles_to_conditions(self):
        selected_cells = []
        selected_spots = []
        
        for isfirst,style in enumerate(self.conditions):
            celltypes = style['Celltype']
            field_of_views = style['FOV']
            spot_size_channel = style['SpotSizes']['Channel']
            spot_size_minimum = style['SpotSizes']['Min']
            spot_size_maximum = style['SpotSizes']['Max']
            num_of_spot_channel = style['Number of Spots']['Channel']
            num_of_spot_minimum = style['Number of Spots']['Min']
            num_of_spot_maximum = style['Number of Spots']['Max']
            spot_brightness_channel = style['Brightness']['Channel']
            spot_brightness_minimum = style['Brightness']['Min']
            spot_brightness_maximum = style['Brightness']['Max']
            spot_distance_ch1 = style['Spot Distance']['Channel1']
            spot_distance_ch2 = style['Spot Distance']['Channel2']
            spot_distance_minimum = style['Spot Distance']['Min']
            spot_distance_maximum = style['Spot Distance']['Max']
            spot_distance_center2center = style['Spot Distance']['CenterToCenter']
            if 'All' in celltypes or len(celltypes) == 0: celltypes = self.celltypes + ['']
            if 'All' in field_of_views or len(field_of_views) == 0: field_of_views = self.SpotMetaDataAnalyzer.fov_list
            if 'All' in spot_size_channel or len(spot_size_channel) == 0: spot_size_channel = self.SpotMetaDataAnalyzer.channel_list
            if 'All' in num_of_spot_channel or len(num_of_spot_channel) == 0: num_of_spot_channel = self.SpotMetaDataAnalyzer.channel_list
            if 'All' in spot_brightness_channel or len(spot_brightness_channel) == 0: spot_brightness_channel = self.SpotMetaDataAnalyzer.channel_list
            
            print(celltypes, field_of_views, spot_size_channel,
                  spot_size_minimum, spot_size_maximum, num_of_spot_channel,
                  num_of_spot_minimum, num_of_spot_maximum, spot_brightness_channel,
                  spot_brightness_minimum, spot_brightness_maximum, spot_distance_ch1,
                  spot_distance_ch2, spot_distance_minimum,spot_distance_maximum,
                  spot_distance_center2center)
            
            field_of_views = [int(fov) for fov in field_of_views]
            fovs = []
            if self.celltype_mode == 'fov':
                for ct in celltypes:
                    if ct in self.celltype_determination['fov']['fov_from_celltype'].keys():
                        for fov in self.celltype_determination['fov']['fov_from_celltype'][ct]:
                            if int(fov) in field_of_views: fovs.append(int(fov))
            elif self.celltype_mode == 'barcode':
                fovs = field_of_views
                        
            if isfirst == 0:
                for fov in fovs:
                    for cell in self.SpotMetaDataAnalyzer.cells[fov]:
                        current_selected_spots_len = len(selected_spots)
                        not_a_good_cell = False
                        for ch in self.SpotMetaDataAnalyzer.channel_list:
                            if (ch in spot_size_channel) & (ch in num_of_spot_channel) & (ch in spot_brightness_channel):
                                if (((cell.num_spots[ch] < int(num_of_spot_minimum)) or (cell.num_spots[ch] > int(num_of_spot_maximum))) or
                                    (cell.celltype not in celltypes)):
                                    not_a_good_cell = True
                                
                        if not not_a_good_cell and ((spot_distance_ch1 == 'None') or (spot_distance_ch2 == 'None')):
                            for spot in cell.spots:
                                if ((spot.channel in spot_size_channel) & (spot.channel in spot_brightness_channel) &
                                    (spot.size >= int(spot_size_minimum)) & (spot.size <= int(spot_size_maximum)) &
                                    (spot.brightness >= float(spot_brightness_minimum)) & (spot.brightness <= float(spot_brightness_maximum)) &
                                    (spot.celltype in celltypes)):
                                    selected_spots.append(spot)
                                    
                        elif not not_a_good_cell:
                            target1 = [spot.channel == spot_distance_ch1 for spot in cell.spots]
                            target2 = [spot.channel == spot_distance_ch2 for spot in cell.spots]
                            
                            for si1 in np.arange(cell.total_num_spots)[target1]:
                                spot1 = cell.spots[si1]
                                if ((spot1.channel in spot_size_channel) & (spot1.channel in spot_brightness_channel) &
                                    (spot1.size >= int(spot_size_minimum)) & (spot1.size <= int(spot_size_maximum)) &
                                    (spot1.brightness >= float(spot_brightness_minimum)) & (spot1.brightness <= float(spot_brightness_maximum)) &
                                    (spot1.celltype in celltypes)):
                                    
                                    for si2 in np.arange(cell.total_num_spots)[target2]:
                                        spot2 = cell.spots[si2]
                                        if ((spot2.channel in spot_size_channel) & (spot2.channel in spot_brightness_channel) &
                                            (spot2.size >= int(spot_size_minimum)) & (spot2.size <= int(spot_size_maximum)) &
                                            (spot2.brightness >= float(spot_brightness_minimum)) & (spot2.brightness <= float(spot_brightness_maximum)) &
                                            (spot2.celltype in celltypes)):
                                            
                                            if spot_distance_center2center:
                                                if (cell.distmap[si1,si2] >= float(spot_distance_minimum)) & (cell.distmap[si1,si2] <= float(spot_distance_maximum)):
                                                    selected_spots.append(spot1)
                                                    selected_spots.append(spot2)
                                            else:
                                                if ((cell.distmap[si1,si2] - cell.spots[si1].size - cell.spots[si2].size >= float(spot_distance_minimum)) &
                                                    (cell.distmap[si1,si2] - cell.spots[si1].size - cell.spots[si2].size <= float(spot_distance_maximum))):
                                                    selected_spots.append(spot1)
                                                    selected_spots.append(spot2)
                                                    
                        if len(selected_spots) > current_selected_spots_len: selected_cells.append(cell)
                        
            elif isfirst > 0:
                new_selected_cells = []
                new_selected_spots = []
                for cell in selected_cells:
                    current_selected_spots_len = len(new_selected_spots)
                    not_a_good_cell = False
                    for ch in self.SpotMetaDataAnalyzer.channel_list:
                        if (ch in spot_size_channel) & (ch in num_of_spot_channel) & (ch in spot_brightness_channel):
                            if (((cell.num_spots[ch] < int(num_of_spot_minimum)) or (cell.num_spots[ch] > int(num_of_spot_maximum))) or
                                (cell.fov not in fovs) or (cell.celltype not in celltypes)):
                                not_a_good_cell = True
                            
                    if not not_a_good_cell and ((spot_distance_ch1 == 'None') or (spot_distance_ch2 == 'None')):
                        for spot in cell.spots:
                            if ((spot.channel in spot_size_channel) & (spot.channel in spot_brightness_channel) &
                                (spot.size >= int(spot_size_minimum)) & (spot.size <= int(spot_size_maximum)) &
                                (spot.brightness >= float(spot_brightness_minimum)) & (spot.brightness <= float(spot_brightness_maximum)) &
                                (spot.celltype in celltypes)):
                                new_selected_spots.append(spot)
                                
                    elif not not_a_good_cell:
                        target1 = [spot.channel == spot_distance_ch1 for spot in cell.spots]
                        target2 = [spot.channel == spot_distance_ch2 for spot in cell.spots]
                        
                        for si1 in np.arange(cell.total_num_spots)[target1]:
                            spot1 = cell.spots[si1]
                            if ((spot1.channel in spot_size_channel) & (spot1.channel in spot_brightness_channel) &
                                (spot1.size >= int(spot_size_minimum)) & (spot1.size <= int(spot_size_maximum)) &
                                (spot1.brightness >= float(spot_brightness_minimum)) & (spot1.brightness <= float(spot_brightness_maximum)) &
                                (spot1.celltype in celltypes)):
                                
                                for si2 in np.arange(cell.total_num_spots)[target2]:
                                    spot2 = cell.spots[si2]
                                    if ((spot2.channel in spot_size_channel) & (spot2.channel in spot_brightness_channel) &
                                        (spot2.size >= int(spot_size_minimum)) & (spot2.size <= int(spot_size_maximum)) &
                                        (spot2.brightness >= float(spot_brightness_minimum)) & (spot2.brightness <= float(spot_brightness_maximum)) &
                                        (spot2.celltype in celltypes)):
                                        
                                        if spot_distance_center2center:
                                            if (cell.distmap[si1,si2] >= float(spot_distance_minimum)) & (cell.distmap[si1,si2] <= float(spot_distance_maximum)):
                                                new_selected_spots.append(spot1)
                                                new_selected_spots.append(spot2)
                                        else:
                                            if ((cell.distmap[si1,si2] - cell.spots[si1].size - cell.spots[si2].size >= float(spot_distance_minimum)) &
                                                (cell.distmap[si1,si2] - cell.spots[si1].size - cell.spots[si2].size <= float(spot_distance_maximum))):
                                                new_selected_spots.append(spot1)
                                                new_selected_spots.append(spot2)
                    if len(new_selected_spots) > current_selected_spots_len: new_selected_cells.append(cell)
                    
                selected_cells = new_selected_cells
                selected_spots = new_selected_spots
                
            print('spots',len(selected_spots))
            print('cells',len(selected_cells))  
        return selected_cells, selected_spots

    def _showSelectedCell(self):
        cid = int(self.AnalysisPanelWidget.AnalysisPanelFrame_targetCellsListWidget.currentItem().text().split('Cell')[1].split()[0]) - 1
        cell = self.selected_cells[cid]
        fig = plt.figure(figsize=(6*len(cell.channels)+6,6))
        for i in range(1,len(cell.channels)+1):
            ax = plt.subplot(1,len(cell.channels),i)
            ax.imshow(cell.image[cell.channels[i-1]], cmap=cm.Greys_r)
            ax.set_title(f'{cell.channels[i-1]}')
            ax.set_xticks([],[])
            ax.set_yticks([],[])
        plt.tight_layout()
        fig.show()

    def _removeCondition(self):
        """
        remove condition from the condition index in
            AnalysisPanelFrame_ConditionLineEdit
        
        removed condition text is added on the panel
            AnalysisPanelFrame_ConditionTextBrowser
        """
        index_list = self.AnalysisPanelWidget.AnalysisPanelFrame_ConditionLineEdit.text().strip().split()
        removed_index = [int(index) for index in index_list]
        new_conditions = []
        for condition in self.conditions:
            if int(condition['Index']) not in removed_index:
                new_conditions.append(condition)
                new_conditions[-1]['Index'] = len(new_conditions) - 1
        self.conditions = new_conditions
        self.AnalysisPanelWidget.AnalysisPanelFrame_ConditionTextBrowser.clear()
        self.AnalysisPanelWidget.AnalysisPanelFrame_targetCellsListWidget.clear()
        for style in self.conditions: self.AnalysisPanelWidget.AnalysisPanelFrame_ConditionTextBrowser.append(self._style_dict_to_str(style))
        self.selected_cells, self.selected_spots = self._convert_styles_to_conditions()
        self.AnalysisPanelWidget.AnalysisPanelFrame_ConditionTextBrowser.append(f'Total selected cell: {len(self.selected_cells)}\nTotal selected spots: {len(self.selected_spots)}')
        for i in range(len(self.selected_cells)): 
            self.AnalysisPanelWidget.AnalysisPanelFrame_targetCellsListWidget.addItem(f'Cell {i+1} (fov: {self.selected_cells[i].fov}, id: {self.selected_cells[i].id})')

    def _calculateSpotSize(self):
        """
        calculate spot size of the channel
            AnalysisPanelFrame_CalculateSpotSizesChannelListWidget
            AnalysisPanelFrame_CalculateSpotSizesThresholdDoubleSpinBox
        """
        channels = [ch.text() for ch in self.AnalysisPanelWidget.AnalysisPanelFrame_CalculateSpotSizesChannelListWidget.selectedItems()]
        if 'All' in channels: channels = self.SpotMetaDataAnalyzer.channel_list
        threshold = float(self.AnalysisPanelWidget.AnalysisPanelFrame_CalculateSpotSizesThresholdDoubleSpinBox.value()) / 100
        result = {ct:{ch: [] for ch in channels} for ct in self.celltypes}
        
        cutoff = {channel:{} for channel in channels}
        for ch in channels:
            for cell in self.selected_cells:
                minimal_brightness = float(np.min([spot.brightness for spot in cell.spots]))
                cutoff[ch][cell.id] = np.min((float(cell.image[ch].max() * threshold), minimal_brightness))

        for spot in self.selected_spots:
            sx,sy = spot.coordinate
            ch = spot.channel
            if (spot.size == 0) and (ch in channels):
                corresponding_cell = self.SpotMetaDataAnalyzer.get_cell_from_id(spot.cell)
                x,y = corresponding_cell.area
                xmin,ymin = x.min(), y.min()
                binary = (corresponding_cell.image[ch] >= cutoff[ch][corresponding_cell.id]).astype(int)
                label = skimage.measure.label(binary)
                if label[int(sy-ymin),int(sx-xmin)] != 0: spot.size = int((label == label[int(sy - ymin),int(sx - xmin)]).sum())
                else: spot.size = 1
            print(sy,sx)
            print(f'id: {spot.id}, cell: {spot.cell}, fov: {spot.fov}\ncelltype: {spot.celltype} channel: {spot.channel}\nsize: {spot.size}')
            if ch in channels: result[spot.celltype][ch].append(spot.size)
            
        fig,ax = plt.subplots(len(channels),3,figsize=(20,5 * len(channels)))
        if len(channels) == 1: ax = [ax]
        for i,ch in enumerate(channels):
            for j,ct in enumerate(self.celltypes):
                x = np.array(result[ct][ch])
                if len(x) > 0:
                    p,q = np.histogram(x, bins = np.arange(np.min(x),np.max(x)+1))
                    ax[i][0].bar(1/2*(q[1:]+q[:-1]),p,.9, label=f'{ct} (#{len(x)})', alpha = .2, color = colors[j])
                    ax[i][0].plot(1/2*(q[1:]+q[:-1]),p,linewidth=2.5, color = colors[j])
                    ax[i][0].set_title(f'Spot size of channel {ch}')
                    boot_gen = bootstrap(x)
                    boot_result = np.array([boot_gen.__next__() for k in range(1000)])
                    p = boot_result.mean(1)
                    ax[i][1].bar(j,p.mean(),.5,yerr=p.std(), color = colors[j], label=f'{ct} (#{len(x)})')
                    pp = []
                    for boot in boot_result:
                        p,q = np.histogram(boot, bins = np.arange(np.min(x),np.max(x)))
                        pp.append(1/2 * (q[1:] + q[:-1])[p.argmax()])
                    pp = np.array(pp)
                    ax[i][2].bar(j,pp.mean(),0.5,yerr=pp.std(), color = colors[j], label=f'{ct} (#{len(x)})')
                    ax[i][0].legend()
                    ax[i][1].legend()
                    ax[i][2].legend()
            ax[i][1].set_xlabel('cell type')
            ax[i][1].set_xticks(np.arange(len(self.celltypes)), self.celltypes)
            ax[i][1].set_title(f'Spot size Average\nbootstrapped 1000 steps')
            ax[i][2].set_xlabel('cell type')
            ax[i][2].set_xticks(np.arange(len(self.celltypes)), self.celltypes)
            ax[i][2].set_title(f'Spot size Mode\nbootstrapped 1000 steps')
        plt.tight_layout()
        fig.show()
    
    def _countSpotsPerCell(self):
        """
        count spot size of the channel
            AnalysisPanelFrame_CountSpotsPerCellChannelListWidget
        """
        channels = [ch.text() for ch in self.AnalysisPanelWidget.AnalysisPanelFrame_CountSpotsPerCellChannelListWidget.selectedItems()]
        if 'All' in channels: channels = self.SpotMetaDataAnalyzer.channel_list
        result = {ct:{ch:[] for ch in channels} for ct in self.celltypes}
        for cell in self.selected_cells:
            n = {ch:0 for ch in channels}
            for spot in cell.spots:
                if spot.channel in channels:
                    n[spot.channel] += 1
            for ch in channels:
                result[cell.celltype][ch].append(n[ch])
        
        fig, ax = plt.subplots(len(channels),1,figsize = (5,5 * len(channels)))
        if len(channels) == 1: ax = [ax]
        for i,ch in enumerate(channels):
            for j,ct in enumerate(self.celltypes):
                x = np.array(result[ct][ch])
                if len(x) > 0:
                    p,q = np.histogram(x, bins = np.arange(np.min(x),np.max(x)+1))
                    ax[i].bar(q[:-1],p,.9, alpha = .2, color = colors[j])
                    ax[i].plot(q[:-1],p,linewidth=2.5, color = colors[j],label=f'{ct} (#{len(x)})')
                    ax[i].set_xticks(np.arange(np.min(x),np.max(x)+1),np.arange(np.min(x),np.max(x)+1))
            ax[i].set_title(f'Number of spot per cell of channel {ch}')
            ax[i].legend()
        plt.tight_layout()
        fig.show()
    
    def _showSpotBrightness(self):
        """
        show spot brightness of the channel
            AnalysisPanelFrame_ShowSpotBrightnessListWidget
        """
        channels = [ch.text() for ch in self.AnalysisPanelWidget.AnalysisPanelFrame_ShowSpotBrightnessListWidget.selectedItems()]
        if 'All' in channels: channels = self.SpotMetaDataAnalyzer.channel_list
        result = {ct:{ch:[] for ch in channels} for ct in self.celltypes}
        for spot in self.selected_spots:
            if spot.channel in channels:
                result[spot.celltype][spot.channel].append(spot.brightness)
                
        fig, ax = plt.subplots(len(channels),1,figsize = (5,5 * len(channels)))
        if len(channels) == 1: ax = [ax]
        for i,ch in enumerate(channels):
            for j,ct in enumerate(self.celltypes):
                x = np.array(result[ct][ch])
                if len(x) > 0:
                    p,q = np.histogram(x, bins = np.linspace(np.min(x),np.max(x),8))
                    ax[i].bar(q[:-1],p,.9, alpha = .2, color = colors[j])
                    ax[i].plot(q[:-1],p,linewidth=2.5, color = colors[j],label=f'{ct} (#{len(x)})')
                    ax[i].set_xticks(np.linspace(np.min(x),np.max(x),8).astype(int),np.linspace(np.min(x),np.max(x),8).astype(int))
            ax[i].set_title(f'Spot brightness of channel {ch}')
            ax[i].legend()
        plt.tight_layout()
        fig.show()
    
    def _measureSpotDistance(self):
        """
        measure distance from spots of channel:
            AnalysisPanelFrame_MeasureSpotDistanceChannel1ListWidget
            AnalysisPanelFrame_MeasureSpotDistanceChannel2ListWidget
            AnalysisPanelFrame_MeasureSpotDistanceCenterToCenterRadioButton
        """
        ch1 = self.AnalysisPanelWidget.AnalysisPanelFrame_MeasureSpotDistanceChannel1ListWidget.currentItem().text()
        ch2 = self.AnalysisPanelWidget.AnalysisPanelFrame_MeasureSpotDistanceChannel2ListWidget.currentItem().text()
        center2center = self.AnalysisPanelWidget.AnalysisPanelFrame_MeasureSpotDistanceCenterToCenterRadioButton.isChecked()
        print(ch1, ch2, center2center)
        if ch1 == ch2: return
        selected_spotids = [spot.id for spot in self.selected_spots]
        result = {ct:[] for ct in self.celltypes}
        for cell in self.selected_cells:
            ct = cell.celltype
            for i,spot1 in enumerate(cell.spots):
                if (spot1.id in selected_spotids) & (spot1.channel == ch1):
                    temp = []
                    for j,spot2 in enumerate(cell.spots):
                        if (spot2.id in selected_spotids) & (spot2.channel == ch2):
                            if center2center: temp.append(cell.distmap[i,j])
                            else: temp.append(cell.distmap[i,j] - cell.spots[i].size - cell.spots[j].size)
                    if len(temp) > 0:
                        result[ct].append(np.min(temp))
        fig, ax = plt.subplots(1,3,figsize = (20,5))
        for i,ct in enumerate(self.celltypes):
            x = np.array(result[ct])
            if len(x) > 0:
                p,q = np.histogram(x, bins = np.arange(np.min(x), np.max(x))+1)
                ax[0].bar(1/2 * (q[1:]+q[:-1]),p,.9,label=f'{ct} (#{len(x)})', alpha=.2, color=colors[i])
                ax[0].plot(1/2 * (q[1:]+q[:-1]),p,linewidth=2.5, color = colors[i])
                ax[0].set_title(f'Spot distance\n{ch1} - {ch2}\ncenter-to-center: {center2center}')
                boot_gen = bootstrap(x)
                boot_result = np.array([boot_gen.__next__() for k in range(1000)])
                p = boot_result.mean(1)
                ax[1].bar(i, p.mean(), .5, yerr=p.std(), color=colors[i], label=f'{ct} (#{len(x)})')
                pp = []
                for boot in boot_result:
                    p,q = np.histogram(boot, bins = np.arange(np.min(x), np.max(x)))
                    pp.append(1/2 * (q[1:] + q[:-1])[p.argmax()])
                pp = np.array(pp)
                ax[2].bar(i,pp.mean(),.5,yerr=pp.std(), color=colors[i],label=f'{ct} (#{len(x)})')
                ax[0].legend()
                ax[1].legend()
                ax[2].legend()

            ax[1].set_xlabel('cell type')
            ax[1].set_xticks(np.arange(len(self.celltypes)), self.celltypes)
            ax[1].set_title(f'Distance Average\nbootstrapped 1000 steps')
            ax[2].set_xlabel('cell type')
            ax[2].set_xticks(np.arange(len(self.celltypes)), self.celltypes)
            ax[2].set_title(f'Distance Mode\nbootstrapped 1000 steps')        
        plt.tight_layout()
        fig.show()

    def _CelltypePanelConnection(self):
        """
        set-up celltype determination panel connection
        
        """
        def __changeLineEdit(listWidget, lineedit):
            selectedItems = [item.text() for item in listWidget.selectedItems()]
            if 'All' in selectedItems: selectedItems = deepcopy(self.fov_list)
            text = ' '.join(selectedItems)
            lineedit.setText(text)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CelltypeRemoveSelectedPushButton.clicked.connect(self._removeSelectedCelltype)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CelltypeAddnewcelltypePushButton.clicked.connect(self._addNewCelltype)
        self.CelltypePanelWidget.CellTypeDeterminePanel_FOVModulateSelectedCellTypePushButton.clicked.connect(self._modulateSelectedCelltype)
        self.CelltypePanelWidget.CellTypeDeterminePanel_FOVCelltypesListWidget.itemClicked.connect(self._updateFOVListWidget)
        re = QtCore.QRegExp('[0-9 ]+')
        self.CelltypePanelWidget.CellTypeDeterminePanel_FOVLineEdit.setValidator(QtGui.QRegExpValidator(re))
        self.CelltypePanelWidget.CellTypeDeterminePanel_FOVListWidget.itemSelectionChanged.connect(lambda:__changeLineEdit(self.CelltypePanelWidget.CellTypeDeterminePanel_FOVListWidget,
                                                                                                                    self.CelltypePanelWidget.CellTypeDeterminePanel_FOVLineEdit))
        
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeUpdatePushButton.clicked.connect(self._updateCellBarcodeChannel)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeSetupPushButton.clicked.connect(self._setupCellBarcode)    
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeChannelListWidget.itemClicked.connect(self._updateIntensityDistribution)
        
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVLineEdit.setValidator(QtGui.QRegExpValidator(re))
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVListWidget.itemSelectionChanged.connect(lambda:__changeLineEdit(self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVListWidget,
                                                                                                                                                            self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVLineEdit))
        re = QtCore.QRegExp('\d+\.\d{4}')
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationScaleLineEdit.setValidator(QtGui.QRegExpValidator(re))
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracLineEdit.setValidator(QtGui.QRegExpValidator(re))
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundAbsLineEdit.setValidator(QtGui.QRegExpValidator(re))
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracLineEdit.setValidator(QtGui.QRegExpValidator(re))
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundAbsLineEdit.setValidator(QtGui.QRegExpValidator(re))
        
        ButtonGroup1 = QtWidgets.QButtonGroup(self.CelltypePanelCentralWidget)
        ButtonGroup2 = QtWidgets.QButtonGroup(self.CelltypePanelCentralWidget)
        ButtonGroup1.addButton(self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracRadioButton)
        ButtonGroup1.addButton(self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundAbsRadioButton)
        ButtonGroup2.addButton(self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracRadioButton)
        ButtonGroup2.addButton(self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundAbsRadioButton)
        
        self.CelltypePanelWidget.CellTypeDeterminePanel_ShowCellBarcodePushButton.clicked.connect(self._showCellBarcode)
        self.CelltypePanelWidget.CellTypeDeterminePanel_AlphaHorizontalSlider.valueChanged.connect(self._changeAlphaDoubleSpinBoxValueFromSlider)
        self.CelltypePanelWidget.CellTypeDeterminePanel_AlphaDoubleSpinBox.valueChanged.connect(self._changeHorizontalSliderValueFromSpinBox)
    
        self.CelltypePanelWidget.CellTypeDeterminePanel_FOVPushButton.clicked.connect(self._turnONFOVCelltypeMode)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodePushButton.clicked.connect(self._turnONCellBarcodeCelltypeMode)    
        
    def _removeSelectedCelltype(self):
        """
        """
        id = self.CelltypePanelWidget.CellTypeDeterminePanel_CelltypeListWidget.currentRow()
        text = self.CelltypePanelWidget.CellTypeDeterminePanel_CelltypeListWidget.item(id).text()
        self.CelltypePanelWidget.CellTypeDeterminePanel_CelltypeListWidget.takeItem(id)
        self.celltypes.remove(text)
        self.CelltypePanelWidget.CellTypeDeterminePanel_FOVCelltypesListWidget.clear()
        self.CelltypePanelWidget.CellTypeDeterminePanel_CelltypeListWidget.clear()
        
        for ct in self.celltypes:
            self.CelltypePanelWidget.CellTypeDeterminePanel_CelltypeListWidget.addItem(ct)
            self.CelltypePanelWidget.CellTypeDeterminePanel_FOVCelltypesListWidget.addItem(str(ct))
        
        self.celltype_determination['fov']['fov_from_celltype'][text] = []
        for k,v in self.celltype_determination['fov']['celltype_from_fov'].items():
            if str(v) == text:
                self.celltype_determination['fov']['celltype_from_fov'][k] = ''
        self.celltype_determination['barcode']['barcode_celltype'] = self.celltypes
        print(self.celltype_determination)
        
    def _addNewCelltype(self):
        """
        """
        text = self.CelltypePanelWidget.CellTypeDeterminePanel_CelltypeAddTextEdit.toPlainText().strip()
        if text not in self.celltypes: 
            self.celltypes.append(text)
            self.celltype_determination['fov']['fov_from_celltype'][text] = []
        self.CelltypePanelWidget.CellTypeDeterminePanel_FOVCelltypesListWidget.clear()
        self.CelltypePanelWidget.CellTypeDeterminePanel_CelltypeListWidget.clear()
        for ct in self.celltypes:
            self.CelltypePanelWidget.CellTypeDeterminePanel_CelltypeListWidget.addItem(ct)
            self.CelltypePanelWidget.CellTypeDeterminePanel_FOVCelltypesListWidget.addItem(str(ct))
        self.celltype_determination['barcode']['barcode_celltype'] = self.celltypes
        print(self.celltype_determination)
        
    def _turnONFOVCelltypeMode(self):
        """
        """
        self.celltype_mode = 'fov'
        self.CelltypePanelWidget.CellTypeDeterminePanel_FOVPushButton.setEnabled(False)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodePushButton.setEnabled(True)
        
        self.CelltypePanelWidget.CellTypeDeterminePanel_FOVCelltypesListWidget.setEnabled(True)
        self.CelltypePanelWidget.CellTypeDeterminePanel_FOVModulateSelectedCellTypePushButton.setEnabled(True)
        self.CelltypePanelWidget.CellTypeDeterminePanel_FOVListWidget.setEnabled(True)
        self.CelltypePanelWidget.CellTypeDeterminePanel_FOVLineEdit.setEnabled(True)
        
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeChannelListWidget.setEnabled(False)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeTotalChannelListWidget.setEnabled(False)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeChannelListWidget.setEnabled(False)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeUpdatePushButton.setEnabled(False)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeSetupPushButton.setEnabled(False)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVListWidget.setEnabled(False)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVLineEdit.setEnabled(False)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracLineEdit.setEnabled(False)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundAbsLineEdit.setEnabled(False)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracLineEdit.setEnabled(False)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundAbsLineEdit.setEnabled(False)
        self.CelltypePanelWidget.CellTypeDeterminePanel_ShowCellBarcodePushButton.setEnabled(False)
        
        self.CelltypePanelWidget.CellTypeDeterminePanel_FOVCelltypesListWidget.clear()
        for ct in self.celltypes:
            self.CelltypePanelWidget.CellTypeDeterminePanel_FOVCelltypesListWidget.addItem(str(ct))
    
    def _turnONCellBarcodeCelltypeMode(self):
        """
        """
        self.celltype_mode = 'barcode'
        self.CelltypePanelWidget.CellTypeDeterminePanel_FOVPushButton.setEnabled(True)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodePushButton.setEnabled(False)
        
        self.CelltypePanelWidget.CellTypeDeterminePanel_FOVCelltypesListWidget.setEnabled(False)
        self.CelltypePanelWidget.CellTypeDeterminePanel_FOVModulateSelectedCellTypePushButton.setEnabled(False)
        self.CelltypePanelWidget.CellTypeDeterminePanel_FOVListWidget.setEnabled(False)
        self.CelltypePanelWidget.CellTypeDeterminePanel_FOVLineEdit.setEnabled(False)
        
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeChannelListWidget.setEnabled(True)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeTotalChannelListWidget.setEnabled(True)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeChannelListWidget.setEnabled(True)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeUpdatePushButton.setEnabled(True)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeSetupPushButton.setEnabled(True)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVListWidget.setEnabled(True)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVLineEdit.setEnabled(True)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracLineEdit.setEnabled(True)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundAbsLineEdit.setEnabled(True)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracLineEdit.setEnabled(True)
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundAbsLineEdit.setEnabled(True)
        self.CelltypePanelWidget.CellTypeDeterminePanel_ShowCellBarcodePushButton.setEnabled(True)
        
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeTotalChannelListWidget.clear()
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeChannelListWidget.clear()
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVListWidget.clear()
        
        for ch in self.channel_list:
            self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeTotalChannelListWidget.addItem(str(ch))
        for ch in self.celltype_determination['barcode']['barcode_channel']:
            self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeChannelListWidget.addItem(str(ch))
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVListWidget.addItem(str('All'))
        for fov in self.fov_list:
            self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVListWidget.addItem(str(fov))
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVListWidget.setCurrentRow(0)
        
        for ch in self.celltype_determination['barcode']['barcode_channel']:
            for fov in self.fov_list:
                self.celltype_determination['barcode']['scale'][ch][int(fov)] = float(self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationScaleLineEdit.text())
                img = self.stored_images[fov][ch]
                if self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracRadioButton.isChecked():
                    lb = min(float(self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracLineEdit.text()),1)
                    self.celltype_determination['barcode']['lower_bound'][ch][int(fov)] = np.quantile(img, lb)
                else:
                    lb = min(float(self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundAbsLineEdit.text()),
                             img.max())
                    self.celltype_determination['barcode']['lower_bound'][ch][int(fov)] = lb
                if self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracRadioButton.isChecked():
                    hb = min(float(self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracLineEdit.text()),1)
                    self.celltype_determination['barcode']['upper_bound'][ch][int(fov)] = np.quantile(img,hb)
                else:
                    hb = min(float(self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracLineEdit.text()),
                             img.max())
                    self.celltype_determination['barcode']['upper_bound'][ch][int(fov)] = hb
                
    def _updateFOVListWidget(self):
        """
        """
        item = self.CelltypePanelWidget.CellTypeDeterminePanel_FOVCelltypesListWidget.currentItem()
        ct = item.text()
        fovs = self.celltype_determination['fov']['fov_from_celltype'][ct]
        self.CelltypePanelWidget.CellTypeDeterminePanel_FOVListWidget.clear()
        for i,f in enumerate(self.fov_list):
            self.CelltypePanelWidget.CellTypeDeterminePanel_FOVListWidget.addItem(str(f))
            if int(f) in fovs:
                self.CelltypePanelWidget.CellTypeDeterminePanel_FOVListWidget.item(i).setSelected(True)
        
    def _modulateSelectedCelltype(self):
        """
        """
        selected_ct = self.CelltypePanelWidget.CellTypeDeterminePanel_FOVCelltypesListWidget.currentItem().text()
        selected_fovs = [int(item.text()) for item in self.CelltypePanelWidget.CellTypeDeterminePanel_FOVListWidget.selectedItems()]
        self.celltype_determination['fov']['fov_from_celltype'][selected_ct] = selected_fovs
        for f in selected_fovs:
            self.celltype_determination['fov']['celltype_from_fov'][int(f)] = selected_ct
        print(self.celltype_determination)
        
    def _updateCellBarcodeChannel(self):
        """
        """
        barcode_channel = [item.text() for item in self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeTotalChannelListWidget.selectedItems()]
        if len(barcode_channel) > len(self.celltypes): barcode_channel = barcode_channel[:len(self.celltypes)]
        self.celltype_determination['barcode']['barcode_channel'] = barcode_channel
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeTotalChannelListWidget.clear()
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeChannelListWidget.clear()
        for ch in self.channel_list:
            self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeTotalChannelListWidget.addItem(str(ch))
        for ch in self.celltype_determination['barcode']['barcode_channel']:
            self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeChannelListWidget.addItem(str(ch))
        self._generateCelltypeDeterminationImage()

    def _updateIntensityDistribution(self):
        """
        """
        selectedBarcodeChannel = self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeChannelListWidget.currentItem().text()
        lower_bounds = self.celltype_determination['barcode']['lower_bound'][selectedBarcodeChannel]
        upper_bounds = self.celltype_determination['barcode']['upper_bound'][selectedBarcodeChannel]
        scale = self.celltype_determination['barcode']['scale'][selectedBarcodeChannel]
        self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeChannelDistributionTextBrowser.clear()
        for fov in self.fov_list:
            text = f'FOV: {fov}\nscale: {scale[int(fov)]}\tminimum: {int(lower_bounds[int(fov)])}\tmaximum: {int(upper_bounds[int(fov)])}\n'
            self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeChannelDistributionTextBrowser.append(text)
        
    def _setupCellBarcode(self):
        """
        """
        ch = self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeChannelListWidget.currentItem().text()
        fovs = self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationFOVLineEdit.text().strip().split()
        for fov in fovs:
            self.celltype_determination['barcode']['scale'][ch][int(fov)] = float(self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationScaleLineEdit.text())
            img = self.stored_images[fov][ch]
            if self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracRadioButton.isChecked():
                lb = min(float(self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundFracLineEdit.text()),1)
                self.celltype_determination['barcode']['lower_bound'][ch][int(fov)] = np.quantile(img, lb)
            else:
                lb = min(float(self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationLowerBoundAbsLineEdit.text()),
                            img.max())
                self.celltype_determination['barcode']['lower_bound'][ch][int(fov)] = lb
            if self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracRadioButton.isChecked():
                hb = min(float(self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundFracLineEdit.text()),1)
                self.celltype_determination['barcode']['upper_bound'][ch][int(fov)] = np.quantile(img,hb)
            else:
                hb = min(float(self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeCelltypeClassificationUpperBoundAbsLineEdit.text()),
                            img.max())
                self.celltype_determination['barcode']['upper_bound'][ch][int(fov)] = hb
        
        self._updateIntensityDistribution()
    
    def _showCellBarcode(self):
        """
        """
        self._updateImage()
        barcode_channels = self.celltype_determination['barcode']['barcode_channel']
        lower_bounds = self.celltype_determination['barcode']['lower_bound']
        upper_bounds = self.celltype_determination['barcode']['upper_bound']
        scales = self.celltype_determination['barcode']['scale']
        super_image = np.empty((len(barcode_channels),self.ymax-self.ymin,self.xmax-self.xmin))
        
        for i,ch in enumerate(barcode_channels):
            single_image_original = scales[ch][int(self.current_fov)] * self.stored_images[self.current_fov][ch][self.ymin:self.ymax,self.xmin:self.xmax]
            super_image[i,:,:] = (single_image_original - lower_bounds[ch][int(self.current_fov)]) / (upper_bounds[ch][int(self.current_fov)] - lower_bounds[ch][int(self.current_fov)]) 
            r,g,b = hex2rgb(colors[i])
            brush = QtGui.QBrush(QtGui.QColor(r,g,b))
            self.CelltypePanelWidget.CellTypeDeterminePanel_CellBarcodeChannelListWidget.item(i).setForeground(brush)
            
        index_image = super_image.argmax(0)
        max_image = super_image.max(0)
        image = np.zeros((self.ymax-self.ymin,self.xmax-self.xmin,3))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):        
                image[i,j,:] = hex2rgb(colors[index_image[i,j]]) if max_image[i,j] > 0 else (0,0,0)
        alpha = float(self.CelltypePanelWidget.CellTypeDeterminePanel_AlphaDoubleSpinBox.value())
        if len(self.image.shape) == 2:
            current_image = np.zeros_like(image)
            for i in range(3):
                current_image[:,:,i] = imadjust(self.image, self.lb[self.working_channel], self.hb[self.working_channel])
        elif len(self.image.shape) == 3:
            current_image = self.image[:,:,:3]
        new_image = image * alpha + (1 - alpha) * current_image
        self.setImage(new_image)
    
    def _changeAlphaDoubleSpinBoxValueFromSlider(self):
        slidervalue = float(self.CelltypePanelWidget.CellTypeDeterminePanel_AlphaHorizontalSlider.value()) * .01
        self.CelltypePanelWidget.CellTypeDeterminePanel_AlphaDoubleSpinBox.setValue(float(slidervalue))
        
    def _changeHorizontalSliderValueFromSpinBox(self):
        spinboxvalue = float(self.CelltypePanelWidget.CellTypeDeterminePanel_AlphaDoubleSpinBox.value()) * 100
        self.CelltypePanelWidget.CellTypeDeterminePanel_AlphaHorizontalSlider.setValue(int(spinboxvalue))

class SpotContainer():
    """
    Spot container

    Two-key data container with keys:
        fov, channel
    spotcontainer does not include cell key, for cells are changable.
    
    spots: coordinates
    
    """
    def __init__(self, fov_list, channel_list):
        if not isinstance(fov_list,list): self.fov_list = [fov_list]
        if not isinstance(channel_list,list): self.channel_list = [channel_list]

        self.fov_list = fov_list
        self.channel_list = channel_list
        
        self.data = {(f,ch): np.array([], dtype=int).reshape(-1,2)
                     for f,ch in product(self.fov_list, self.channel_list)}
    
    def reinitialize(self, fov_list, channel_list):
        if not isinstance(fov_list, list): fov_list = [fov_list]
        if not isinstance(channel_list, list): channel_list = [channel_list]

        for f,ch in product(fov_list, channel_list): self.data[(f,ch)] = np.array([], dtype=int).reshape(-1,2)
    
    def load_new_spots(self, fov, channel, coordinates):
        if len(coordinates) == 0: return
            
        current_max_index = len(self.data[(fov,channel)])
        
        if current_max_index == 0:
            self.data[(fov,channel)] = np.array([], dtype=int).reshape(-1,2)
            
        self.data[(fov,channel)] = np.concatenate((self.data[(fov,channel)],
                                                           coordinates.astype(int)), axis=0)

        indices = np.arange(len(coordinates)) + current_max_index
        return indices
    
    def remove_spots_from_indices(self, fov, channel, indices):
        if len(indices) == 0: return
        
        self.data[(fov,channel)] = self.data[(fov,channel)].astype(float)
        self.data[(fov,channel)][indices] = np.nan
        remain = np.isfinite(self.data[(fov,channel)][:,0])
        self.data[(fov,channel)] = self.data[(fov,channel)][remain]
        self.data[(fov,channel)] = self.data[(fov,channel)].astype(int)
        
        current_max_index = len(self.data[(fov,channel)])
        if current_max_index == 0:
            self.data[(fov,channel)] = np.array([], dtype=int).reshape(-1,2)
    
    def remove_spots_from_area(self, fov, channel, area):
        self.data[(fov,channel)] = self.data[(fov,channel)].astype(float)
        for i,(y,x) in enumerate(self.data[(fov,channel)]):
            if area[int(y),int(x)]: self.data[(fov,channel)][i] = np.nan
            
        remain = np.isfinite(self.data[(fov,channel)][:,0])
        self.data[(fov,channel)] = self.data[(fov,channel)][remain]
        current_max_index = len(self.data[(fov,channel)])
        
        self.data[(fov,channel)] = self.data[(fov,channel)].astype(int)
        if current_max_index == 0:
            self.data[(fov,channel)] = np.array([], dtype=int).reshape(-1,2)

class CellContainer():
    """
    Cell Container
    
    One key data container with key:
        fov
    
    cell: cell_shadow, boundaries
    """
    def __init__(self, fov_list):
        if (len(fov_list) == 0): raise ValueError('Make container with positive-length fovs')
        self.fov_list = fov_list
        self.four_connect = scind.generate_binary_structure(2,1)
        self.data = {f:[] for f in self.fov_list}
        self.h = {f:0 for f in self.fov_list}
        self.w = {f:0 for f in self.fov_list}
        self.num_cells = {f:0 for f in self.fov_list}
        
    def load_new_cells(self, fov, shadows):
        h,w = shadows.shape
        num_cells = shadows.max()
        self.initialize(fov)
        self.h[fov], self.w[fov], self.num_cells[fov] = h,w,num_cells
        
        for i in range(1,num_cells+1):
            y,x = np.where(shadows == i)
            self.data[fov].append((x,y))
            
    def add_new_cell(self, fov, coords):
        self.num_cells[fov] = len(self.data[fov])
        self.num_cells[fov] += 1
        self.data[fov].append(coords)

    def update_cell(self,fov,cellID,shadow = None):
        if shadow is None:
            self.data[fov].pop(cellID)
            self.num_cells[fov] -= 1
            return
        y,x = np.where(shadow == 1)
        self.data[fov][cellID] = (x,y)      
      
    def make_boundaries(self, fov, cellIDs):
        if isinstance(cellIDs, int): cellIDs = [cellIDs]
        
        bx,by = np.array([]).reshape(-1,1),np.array([]).reshape(-1,1)
        for id in cellIDs:
            x,y = self.data[fov][id]
            ymin, ymax, xmin, xmax = max(0,y.min()-1), min(self.h[fov],y.max()+1), max(0,x.min()-1), min(self.w[fov],x.max()+1)
            mini_canvas = np.zeros((ymax-ymin+1, xmax-xmin+1))
            mini_canvas[y-ymin,x-xmin] = 1
            boundaryline = mini_canvas - skimage.morphology.erosion(mini_canvas, self.four_connect).astype(int)
            y,x = np.where(boundaryline == 1)
            bx = np.concatenate((bx,x[:,None]+xmin), axis=0)
            by = np.concatenate((by,y[:,None]+ymin), axis=0)
        
        return (bx,by)
    
    def initialize(self, fov):
        self.data[fov] = []
        self.h[fov], self.w[fov], self.num_cells[fov] = 0,0,0

class SpotMetaDataAnalyzer():
    """
    meta
    fov_list | list | int(fov)
    channel_list | list | str(ch)
    cellContainer | CellContainer
    spotContainer | SpotContainer
    cells | dict | {int(f):list(ACell) for f in fov_list}
    spots | dict | {int(f):list(ASpot) for f in fov_list}
    images | dict | {fov: {ch: np.ndarray}}
    max_cell_id, max_spot_id | int, int |
    cell_from_id | dict | {int(f): ACell}
    spot_from_id | dict | {int(f): ASpot}
    
    ACell
    id | int
    fov | int
    area | tuple | (ndarray, ndarray)
    spots | list | [ACell]
    total_num_spots = 0
    distmap | ndarray
    image | dict | {ch: ndarray}
    channels | list | [str(ch)]
    num_spots | dict | {str(ch): int}
    
    spot
    id | int
    coordinate | tuple
    cell | int
    size | int
    brightness | float
    neighbors | tuple
    channel | str
     
    homeless spots have cellID -1
    
    self.stored_images[self.current_fov][self.current_channel[0]]
    """
    def __init__(self):
        self.fov_list = []#| list | int(fov)
        self.channel_list = []#| list | str(ch)
        self.cellContainer = None#| CellContainer
        self.spotContainer = None#| SpotContainer
        self.cells = {}#| dict | {int(f):list(ACell) for f in fov_list}
        self.spots = {}#| dict | {int(f):list(ASpot) for f in fov_list}
        self.cell_from_id = {}#| dict | {int(f): ACell}
        self.spot_from_id = {}#| dict | {int(f): ASpot}
        self.celltype_mode = 'fov'
        self.celltype_determination = {}

    def set_metadata(self, **kwargs):
        if 'fov_list' in kwargs.keys(): self.fov_list = kwargs['fov_list']
        if 'channel_list' in kwargs.keys(): self.channel_list = kwargs['channel_list']
        if 'CellContainer' in kwargs.keys(): self.cellContainer = kwargs['CellContainer']
        if 'SpotContainer' in kwargs.keys(): self.spotContainer = kwargs['SpotContainer']
        if 'cells' in kwargs.keys(): self.cells = kwargs['cells']
        if 'spots' in kwargs.keys(): self.spots = kwargs['spots']
        if 'cell_from_id' in kwargs.keys(): self.cell_from_id = kwargs['cell_from_id']
        if 'spot_from_id' in kwargs.keys(): self.spot_from_id = kwargs['spot_from_id']
        if 'celltype_mode' in kwargs.keys(): self.celltype_mode = kwargs['celltype_mode']
        if 'celltype_determination' in kwargs.keys(): self.celltype_determination = kwargs['celltype_determination']

    def load_linked_cell_and_spot(self, cells, spots):
        self.cells = cells
        self.spots = spots
        self.update_total_dict()
        
    def save(self):
        return ({key: [cell.save() for cell in value] for key, value in self.cells.items()},
                {key: [spot.save() for spot in value] for key, value in self.spots.items()},
                self.channel_list)

    def link_cell_and_spot(self, fov_list, channel_list, cellContainer, spotContainer,
                           images, celltype_mode, celltype_determination):
        self.fov_list = fov_list
        self.channel_list = channel_list
        self.cellContainer = cellContainer
        self.spotContainer = spotContainer
        self.cells = {int(f):[] for f in self.fov_list}
        self.spots = {int(f):[] for f in self.fov_list}
        max_cell_id, max_spot_id = 0,0
        self.cell_from_id = {}
        self.spot_from_id = {}
        self.celltype_mode = celltype_mode
        self.celltype_determination = celltype_determination
        
        for f,v in self.cellContainer.data.items():
            h = self.cellContainer.h[str(f)]
            w = self.cellContainer.w[str(f)]
            for c in v:
                cell = ACell()
                x,y = c
                x[x < 0] = 0
                x[x >= w] = w-1
                y[y < 0] = 0
                y[y >= h] = h-1
                xmin,xmax,ymin,ymax = x.min(), x.max(), y.min(), y.max()
                
                if self.celltype_mode == 'fov':
                    ct = self.celltype_determination['fov']['celltype_from_fov'][int(f)]
                    
                elif self.celltype_mode == 'barcode':
                    barcode_channel = self.celltype_determination['barcode']['barcode_channel']
                    barcode_celltype = self.celltype_determination['barcode']['barcode_celltype']
                    cts = []
                    
                    rand_ids = np.random.choice(np.arange(len(x)),200)
                    for id in rand_ids:
                        values = []
                        for bch in barcode_channel:
                            v = self.celltype_determination['barcode']['scale'][bch][int(f)] * images[str(f)][bch][y[id],x[id]]
                            
                            values.append((v - self.celltype_determination['barcode']['lower_bound'][bch][int(f)]) / (self.celltype_determination['barcode']['upper_bound'][bch][int(f)] - self.celltype_determination['barcode']['lower_bound'][bch][int(f)]))
                            
                        if np.max(values) > 0: cts.append(barcode_celltype[np.argmax(values)])
                        else: cts.append('')    
                    cts = np.array(cts)
                    vals,cnts = np.unique(cts, return_counts=True)
                    ct = vals[np.argmax(cnts)]
                    
                cell.set_metadata(id = max_cell_id, fov = int(f), area = c, celltype = ct,
                                  image = {ch: images[f][ch][ymin:ymax+1,xmin:xmax+1] for ch in self.channel_list},
                                  num_spots = {ch: 0 for ch in self.channel_list},
                                  channels = self.channel_list)
                max_cell_id += 1
                self.cells[int(f)].append(cell)

        for (f,ch), v in self.spotContainer.data.items():
            for s in v:
                y,x = s
                cid = -1
                
                if self.celltype_mode == 'fov':
                    ct = self.celltype_determination['fov']['celltype_from_fov'][int(f)]
                    print(self.celltype_mode, ct)
                elif self.celltype_mode == 'barcode':
                    barcode_channel = self.celltype_determination['barcode']['barcode_channel']
                    barcode_celltype = self.celltype_determination['barcode']['barcode_celltype']
                    values = []
                    for bch in barcode_channel:
                        v = self.celltype_determination['barcode']['scale'][bch][int(f)] * images[str(f)][bch][int(y),int(x)]
                        
                        values.append((v - self.celltype_determination['barcode']['lower_bound'][bch][int(f)]) / (self.celltype_determination['barcode']['upper_bound'][bch][int(f)] - self.celltype_determination['barcode']['lower_bound'][bch][int(f)]))
                        
                    if np.max(values) > 0:
                        bid = np.argmax(values)
                        ct = barcode_celltype[bid]
                    else:
                        ct = ''
                        
                for cell in self.cells[int(f)]:
                    if ((int(x) == cell.area[0]) & (int(y) == cell.area[1])).any():
                        cid = cell.id
                        spot = ASpot()
                        spot.set_metadata(id = max_spot_id, fov = cell.fov, coordinate = (x,y), celltype = ct,
                                          cell = cid, brightness = images[f][ch][int(y),int(x)], channel = ch)
                        max_spot_id += 1
                        self.spots[int(f)].append(spot)
                        
                if cid == -1:
                    spot = ASpot()
                    spot.set_metadata(id = max_spot_id, fov = -1, coordinate = (x,y), celltype = ct,
                                      cell = cid, brightness = images[f][ch][int(y),int(x)], channel = ch)
                    max_spot_id += 1
                    self.spots[int(f)].append(spot)
                    
        for f in self.fov_list:
            for cell in self.cells[int(f)]:
                for spot in self.spots[int(f)]:
                    if spot.cell == cell.id: 
                        cell.spots.append(spot)
                        cell.num_spots[spot.channel] += 1
                        cell.total_num_spots += 1
                cell.calculate_distmap()
                
        self.update_total_dict()
        print(f'Sucessfully Linked. {max_cell_id} cells and {max_spot_id} spots are linked.')
        
    def update_total_dict(self):
        for cell_list in self.cells.values():
            for cell in cell_list:
                self.cell_from_id[cell.id] = cell
        for spot_list in self.spots.values():
            for spot in spot_list:
                self.spot_from_id[spot.id] = spot
        
    def get_cell_from_id(self,id):
        return self.cell_from_id[id]

    def get_spot_from_id(self,id):
        return self.spot_from_id[id]
                        
class ACell():
    """
    a cell class
    
    each cell has attributes:
     id: int: int id
     fov: int: int fov
     celltype: str: str celltype
     area: tuple: (ndarray x, ndarray y)
     spots: list: [class ASpot, class ASpot]
     num_spots: int: int num_spots
     distmap: ndarray (num_spots x num_spots) dtype float
     image: dict: key: str channel, value: ndarray dtype float
    """
    def __init__(self):
        self.id = 0
        self.fov = 0
        self.celltype = ''
        self.area = (np.array([]).reshape(-1,1), np.array([]).reshape(-1,1))
        self.spots = []
        self.total_num_spots = 0
        self.distmap = np.array([])
        self.image = {}#np.array([])
        self.channels = []
        self.num_spots = {}
        
    def set_metadata(self, **kwargs):
        if 'id' in kwargs.keys(): self.id = int(kwargs['id'])
        if 'fov' in kwargs.keys(): self.fov = int(kwargs['fov'])
        if 'celltype' in kwargs.keys(): self.celltype = str(kwargs['celltype'])
        if 'area' in kwargs.keys(): self.area = tuple(kwargs['area'])
        if 'spots' in kwargs.keys(): self.spots = list(kwargs['spots'])
        if 'total_num_spots' in kwargs.keys(): self.total_num_spots = int(kwargs['total_num_spots'])
        if 'distmap' in kwargs.keys(): self.distmap = np.array(kwargs['distmap'])
        if 'image' in kwargs.keys(): self.image = dict(kwargs['image'])
        if 'channels' in kwargs.keys(): self.channels = list(kwargs['channels'])
        if 'num_spots' in kwargs.keys(): self.num_spots = dict(kwargs['num_spots'])
        
    def calculate_distmap(self):
        if self.total_num_spots > 0:
            pos = np.array([spot.coordinate for spot in self.spots])
            self.distmap = ssd.squareform(ssd.pdist(pos))
    
    def save(self):
        return {'id': int(self.id),
                'fov': int(self.fov),
                'celltype': str(self.celltype),
                'area': tuple(self.area),
                'spots': [spot.save() for spot in self.spots],
                'total_num_spots': int(self.total_num_spots),
                'distmap': np.array(self.distmap),
                'image': dict(self.image),
                'channels': list(self.channels),
                'num_spots': dict(self.num_spots)}
    
class ASpot():
    """
    a spot class

    each spot has attributes:
     id: int: int id
     fov: int: int fov
     celltype: str: str ct
     coordinate: tuple: (float x,float y)
     cell: int: int cid
     size: int: int size
     brightness: float
     neighbors: tuple (variable size): (int id1, int id2, ...)
    """
    def __init__(self):
        self.id = 0
        self.fov = 0
        self.celltype = ''
        self.coordinate = (0,0)
        self.cell = -1
        self.size = 0
        self.brightness = 0.0
        self.neighbors = ()
        self.channel = ''
    
    def set_metadata(self, **kwargs):
        if 'id' in kwargs: self.id = int(kwargs['id'])
        if 'fov' in kwargs: self.fov = int(kwargs['fov'])
        if 'celltype' in kwargs: self.celltype = str(kwargs['celltype'])
        if 'coordinate' in kwargs:self.coordinate = tuple(kwargs['coordinate'])
        if 'cell' in kwargs:self.cell = int(kwargs['cell'])
        if 'size' in kwargs:self.size = int(kwargs['size'])
        if 'brightness' in kwargs:self.brightness = float(kwargs['brightness'])
        if 'neighbors' in kwargs:self.neighbors = tuple(kwargs['neighbors'])
        if 'channel' in kwargs:self.channel = str(kwargs['channel'])
        
    def save(self):
        return {'id': int(self.id),
                'fov': int(self.fov),
                'celltype': str(self.celltype),
                'coordinate': tuple(self.coordinate),
                'cell': int(self.cell),
                'size': int(self.size),
                'brightness': float(self.brightness),
                'neighbors': tuple(self.neighbors),
                'channel': str(self.channel)}

class MainWindow(QtWidgets.QMainWindow):
    """
    mainwindow for analysis pipeline
    """
    def __init__(self,
             parameter_xml_path = ''):
        super().__init__()
        
        xml = eT.parse(parameter_xml_path).getroot()
        self.parameters = dict()
        for elem in list(xml):
            self.parameters[elem.tag] = elem.text.strip()
        
        self.celltypes = [p.lstrip().rstrip() for p in self.parameters.get('celltypes').split(',')]
        self.celltypes_fov = [p.lstrip().rstrip() for p in self.parameters.get('celltypes_fov').split(',')]
        self.celltypes_barcode = [p.lstrip().rstrip() for p in self.parameters.get('celltypes_barcode').split(',')]
        self.channel_list = [p.lstrip().rstrip() for p in self.parameters.get('spotNames').split(',')]
        self.celltype_mode = self.parameters.get('celltype_mode')
        self.mosaic_path = self.parameters.get('mosaic_path')
        self.mosaic_name = self.parameters.get('mosaic_name')
        self.save_path = self.parameters.get('save_path')
        self.celltype_determination = {'fov': {'raw':self.celltypes_fov},
                                       'barcode': {'raw':self.celltypes_barcode}}
        
        self.fov_list = ['1']
        self.cell_list = ['All']
        self.current_fov = self.fov_list[0]
        self.current_cell = self.cell_list[0]
        self.current_channel = self.channel_list[0]
        self.current_channels = [self.channel_list[0]]
        
        print(f'Celltypes: {self.celltypes}\nChannel: {self.channel_list}\nCelltype mode: {self.celltype_mode}')
        
        self.exist_image = False
        self.exist_fov_tiles = False
        self.lb = {ch:.3 for ch in self.channel_list}
        self.hb = {ch:.9999 for ch in self.channel_list}
        
        self.createGUI()
        ### load mosaic tiles and files
        self.mosaic_tile_name = self.mosaic_path + self.mosaic_name + 'tileULs.csv'
        self.mosaic_file_names = [self.mosaic_path + self.mosaic_name + f'_c{i+1:03}.tif' for i in range(len(self.channel_list))]
        self._update_mosaic_tile(self.mosaic_tile_name)
        self._update_mosaic_file(self.mosaic_file_names)
        self._synchronizeCanvasAndMain()
        self.MainImageCanvas._load_celltype_determination(self.celltypes, self.celltype_determination)
        if self.celltype_mode == 'fov':
            self.MainImageCanvas._turnONFOVCelltypeMode()
        elif self.celltype_mode == 'barcode':
            self.MainImageCanvas._turnONCellBarcodeCelltypeMode()
        
    def createGUI(self):
        ### GUI pannel
        ## Setting up the panels
        self.setWindowTitle('Cell Classifier')
        self.MainImageCanvas = MainImageCanvas()
        self.MainImageCanvas.celltype_mode = self.celltype_mode
        self.MainWindowUI = MainWindowUI()
        self.MainWindowUI.setupUi(self)
        self.MainImageCanvas.setParent(self.MainWindowUI.GraphicWidget)
        self.MainImageCanvas.setGeometry(self.MainWindowUI.GraphicWidget.geometry())
        self.MainWindowConnection()
        
        for ct in self.celltypes: 
            self.MainImageCanvas.CelltypePanelWidget.CellTypeDeterminePanel_CelltypeListWidget.addItem(str(ct))
            self.MainImageCanvas.CelltypePanelWidget.CellTypeDeterminePanel_FOVCelltypesListWidget.addItem(str(ct))
        
    def MainWindowConnection(self):
        self.MainWindowUI.SelectFOV_ChangeFOVPushButton.clicked.connect(self._change_fov)
        self.MainWindowUI.SelectCell_ChangeCellPushButton.clicked.connect(self._change_cell)
        self.MainWindowUI.ChannelSelection_ChangeChannelPushButton.clicked.connect(self._change_channel)
        self.MainWindowUI.ChannelSelection_MinimumSlider.valueChanged.connect(self._minimumSliderValueChanged)
        self.MainWindowUI.ChannelSelection_MaximumSlider.valueChanged.connect(self._maximumSliderValueChanged)
        re = QtCore.QRegExp('\d+\.\d{4}')
        self.MainWindowUI.ChannelSelection_MinimumLineEdit.setValidator(QtGui.QRegExpValidator(re))
        self.MainWindowUI.ChannelSelection_MaximumLineEdit.setValidator(QtGui.QRegExpValidator(re))
        self.MainWindowUI.ChannelSelection_MinimumLineEdit.textChanged.connect(self._minimumfracValueChanged)
        self.MainWindowUI.ChannelSelection_MaximumLineEdit.textChanged.connect(self._maximumfracValueChanged)
        self.MainWindowUI.ChannelSelection_CurrentChannelListWidget.itemSelectionChanged.connect(self._update_current_channel)

        for ch in self.channel_list:
            self.MainWindowUI.ChannelSelection_ChannelListWidget.addItem(str(ch))
            self.MainWindowUI.ChannelSelection_ChannelListWidgetColor.addItem(str(ch))
            
        self.MainWindowUI.ChannelSelection_ChannelListWidget.item(0).setSelected(True)
        self.MainWindowUI.actionLoad_Mosaic_Tile.triggered.connect(self._load_mosaic_tile)
        self.MainWindowUI.actionLoad_Mosaic_File.triggered.connect(self._load_mosaic_file)
        self.MainWindowUI.actionExit_Cntrl_Q.setShortcut('Ctrl+Q')
        self.MainWindowUI.actionExit_Cntrl_Q.triggered.connect(self.close)
        self.MainWindowUI.actionSave_Cells.triggered.connect(self._save_cells)
        self.MainWindowUI.actionSave_Spots.triggered.connect(self._save_spots)
        self.MainWindowUI.actionSave_Metadata.triggered.connect(self._save_spotmeta)
        self.MainWindowUI.actionLoad_Cells.triggered.connect(self._load_cells)
        self.MainWindowUI.actionLoad_Spots.triggered.connect(self._load_spots)
        self.MainWindowUI.actionLoad_Spot_Metadata.triggered.connect(self._load_spotmeta)
        
        self.MainWindowUI.FixViewPushButton.clicked.connect(self._updateAutoRange)
        self.MainWindowUI.InitializeViewPushButton.clicked.connect(self._updateAutoRange)
        
        ## Setting up the signals
        self.MainImageCanvas._set_lb_hb(self.lb,self.hb)
        self.MainImageCanvas.save_new_classified_cell_signal.connect(self._update_cell)
        self.MainImageCanvas.request_stored_cell_signal.connect(self._send_stored_cell)
        self.MainImageCanvas.save_new_localized_spot_signal.connect(self._update_spots)
        self.MainImageCanvas.request_stored_spots_signal.connect(self._send_stored_spots)
        self.MainImageCanvas.link_spot_cell_signal.connect(self._link_spot_cell)
        self.MainImageCanvas.ui.histogram.sigLevelsChanged.connect(self._updateChannelSlider)
        self.MainImageCanvas.ui.histogram.sigLevelChangeFinished.connect(self._updateChannelSlider)
        
    def _updateAutoRange(self):
        if self.MainWindowUI.FixViewPushButton.isEnabled():
            self.MainWindowUI.FixViewPushButton.setEnabled(False)
            self.MainWindowUI.InitializeViewPushButton.setEnabled(True)
            self.MainImageCanvas._autoRange = False
            print('autoRange disabled')
        elif not self.MainWindowUI.FixViewPushButton.isEnabled():
            self.MainWindowUI.FixViewPushButton.setEnabled(True)
            self.MainWindowUI.InitializeViewPushButton.setEnabled(False)
            self.MainImageCanvas._autoRange = True
            print('autoRange enabled')

    def _minimumfracValueChanged(self):
        value = float(self.MainWindowUI.ChannelSelection_MinimumLineEdit.text())
        self.lb[self.current_channel] = value
        self.MainImageCanvas._set_lb_hb(self.lb, self.hb)
        img_value = int(self.MainImageCanvas.stored_images[self.current_fov][self.current_channel].max() * value)
        self.MainWindowUI.ChannelSelection_MinimumAbsLineEdit.setText(f'{img_value}')
    
    def _maximumfracValueChanged(self):
        value = float(self.MainWindowUI.ChannelSelection_MaximumLineEdit.text())
        self.hb[self.current_channel] = value
        self.MainImageCanvas._set_lb_hb(self.lb, self.hb)
        img_value = int(self.MainImageCanvas.stored_images[self.current_fov][self.current_channel].max() * value)
        self.MainWindowUI.ChannelSelection_MaximumAbsLineEdit.setText(f'{img_value}')
        
    def _minimumSliderValueChanged(self):
        slidervalue = round(float(self.MainWindowUI.ChannelSelection_MinimumSlider.value()) * .0001,4)
        self.MainWindowUI.ChannelSelection_MinimumLineEdit.setText(f'{slidervalue}')
        self._minimumfracValueChanged()
        if len(self.current_channels) == 1:
            mn = self.MainImageCanvas.current_image.max() * self.lb[self.current_channel]
            mx = self.MainImageCanvas.current_image.max() * self.hb[self.current_channel]
            self.MainImageCanvas.ui.histogram.setLevels(mn,mx)
        
    def _maximumSliderValueChanged(self):
        slidervalue = round(float(self.MainWindowUI.ChannelSelection_MaximumSlider.value()) * .0001,4)
        self.MainWindowUI.ChannelSelection_MaximumLineEdit.setText(f'{slidervalue}')
        self._maximumfracValueChanged()
        if len(self.current_channels) == 1:
            mn = self.MainImageCanvas.current_image.max() * self.lb[self.current_channel]
            mx = self.MainImageCanvas.current_image.max() * self.hb[self.current_channel]
            self.MainImageCanvas.ui.histogram.setLevels(mn,mx)
        
    def _updateChannelSlider(self):
        if (self.MainImageCanvas.ui.histogram.imageItem() is not None) and len(self.current_channels) == 1:
            mn,mx = self.MainImageCanvas.ui.histogram.getLevels()
            img_max = self.MainImageCanvas.current_image.max()
            self.lb[self.current_channel] = round(mn / img_max,4)
            self.hb[self.current_channel] = round(mx / img_max,4)
            self.MainImageCanvas._set_lb_hb(self.lb, self.hb)
            self.MainWindowUI.ChannelSelection_MinimumSlider.setValue(int(mn/img_max * 10000))
            self.MainWindowUI.ChannelSelection_MaximumSlider.setValue(int(mx/img_max * 10000))
            self.MainWindowUI.ChannelSelection_MinimumLineEdit.setText(f'{round(mn / img_max,4)}')
            self.MainWindowUI.ChannelSelection_MaximumLineEdit.setText(f'{round(mx / img_max,4)}')
            
    def _change_fov(self):
        item = self.MainWindowUI.SelectFOV_ListWidget.currentItem()
        if item is not None: self.current_fov = item.text() 
        self._updateImage()
        
    def _change_cell(self):
        item = self.MainWindowUI.SelectCell_ListWidget.currentItem()
        if item is not None: self.current_cell = item.text()
        self._updateImage()
        
    def _change_channel(self):
        self.current_channels = [item.text() for item in self.MainWindowUI.ChannelSelection_ChannelListWidget.selectedItems()]
        
        self.MainWindowUI.ChannelSelection_CurrentChannelListWidget.clear()
        self.MainWindowUI.ChannelSelection_ChannelListWidgetColor.clear()
        for i,ch in enumerate(self.current_channels):
            self.MainWindowUI.ChannelSelection_CurrentChannelListWidget.addItem(str(ch))
            self.MainWindowUI.ChannelSelection_ChannelListWidgetColor.addItem(str(ch))
            r,g,b = hex2rgb(colors[i])
            brush = QtGui.QBrush(QtGui.QColor(r,g,b)) if len(self.current_channels) > 1 else QtGui.QBrush(QtGui.QColor(0,0,0))
            self.MainWindowUI.ChannelSelection_ChannelListWidgetColor.item(i).setForeground(brush)
        self.MainWindowUI.ChannelSelection_CurrentChannelListWidget.setCurrentRow(i)
        self.current_channel = self.MainWindowUI.ChannelSelection_CurrentChannelListWidget.item(i).text()
        print(self.current_channel)
        self._updateImage()
        
    def _update_current_channel(self):
        self.current_channel = self.MainWindowUI.ChannelSelection_CurrentChannelListWidget.currentItem().text()
        self.MainImageCanvas.working_channel = self.current_channel
        print(self.current_channel)
    
    def _updateImage(self):
        if not self.exist_image or not self.exist_fov_tiles: return
        self.MainImageCanvas._update_current_fov_channel_and_cell(self.current_fov, self.current_channel,
                                                                    self.current_channels, self.current_cell)
        self.MainImageCanvas._updateImage()
        
    def _load_mosaic_tile(self):
        self.mosaic_tile_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', self.save_path)[0]
        self._update_mosaic_tile(self.mosaic_tile_path)
        
    def _load_mosaic_file(self):
        self.mosaic_file_path = QtWidgets.QFileDialog.getOpenFileNames(self, 'Open File', self.save_path)[0]
        self._update_mosaic_file(self.mosaic_file_path)

    def _update_mosaic_tile(self, mosaic_tile_name):
        with open(mosaic_tile_name,'r') as f:
            self.fov_tile_info = read_csv(f)
            f.flush()
            
        self.exist_fov_tiles = True
        self.fov_list = [str(i) for i in np.arange(1, len(self.fov_tile_info)+1)]
        for fov in self.fov_list:
            self.MainWindowUI.SelectFOV_ListWidget.addItem(str(fov))
        self._divideImage()
        self.MainImageCanvas.fov_list = deepcopy(self.fov_list)
        self.SpotContainer = SpotContainer(self.fov_list, self.channel_list)
        self.CellContainer = CellContainer(self.fov_list)
        
    def _update_mosaic_file(self, mosaic_file_names):
        self.mosaic_images = dict()
        
        for i, ch in enumerate(self.channel_list):
            self.mosaic_images[ch] = plt.imread(mosaic_file_names[i])
            print(mosaic_file_names[i])
            
        self.exist_image = True            
        self._divideImage()
    
    def _divideImage(self):
        if (not self.exist_image) or (not self.exist_fov_tiles): return
        self.images = {f:{ch:{} for ch in self.channel_list} for f in self.fov_list}
        
        for f in self.fov_list:
            f = int(f)
            x,y = self.fov_tile_info.iloc[f-1]['x'], self.fov_tile_info.iloc[f-1]['y']
            w,h = self.fov_tile_info.iloc[f-1]['w'], self.fov_tile_info.iloc[f-1]['h']
            
            for ch in self.channel_list:
                self.images[str(f)][ch] = self.mosaic_images[ch][y-1:y+w-1,x-1:x+h-1]
                
        self.MainImageCanvas._loadImages(self.images)
        self._updateImage()
        
        for f in self.fov_list:
            for ch in self.channel_list:
                img_max = self.images[str(f)][ch].max()
                mn = round(np.quantile(self.images[str(f)][ch], .5) / img_max, 4)
                mx = round(np.quantile(self.images[str(f)][ch], .9999) / img_max, 4)
                self.lb[ch] = mn
                self.hb[ch] = mx
                self.MainImageCanvas._set_lb_hb(self.lb, self.hb)
                self.MainWindowUI.ChannelSelection_MinimumSlider.setValue(int(mn * 10000))
                self.MainWindowUI.ChannelSelection_MaximumSlider.setValue(int(mx * 10000))
                self.MainWindowUI.ChannelSelection_MinimumLineEdit.setText(f'{mn}')
                self.MainWindowUI.ChannelSelection_MaximumLineEdit.setText(f'{mx}')
        
    def _synchronizeCanvasAndMain(self):
        self.MainImageCanvas.fov_list = deepcopy(self.fov_list)
        self.MainImageCanvas.channel_list = deepcopy(self.channel_list)
        self.MainImageCanvas.cell_list = deepcopy(self.cell_list)
        self._send_stored_cell()
        self._send_stored_spots()
        
    def _send_stored_cell(self):
        print(f'current cell list in container: {self.cell_list}')
        self.MainImageCanvas.CellContainer_permanent = self.CellContainer
        self.MainImageCanvas.CellContainer = deepcopy(self.CellContainer)
        self.MainImageCanvas.cell_list = deepcopy(self.cell_list)

        self.MainImageCanvas.SegmentCellsWidget.transientCellContainer_textBrowser.setText(f'Cell-containing FOVs: {[k for k,v in self.CellContainer.num_cells.items() if v > 0]}\n' +
                                                                           f'Total cells: {np.sum([v for v in self.CellContainer.num_cells.values()])}\n' +
                                                                           ''.join([f'FOV {k}: {v} cells\n' for k,v in self.CellContainer.num_cells.items()]))
        self.MainImageCanvas.SegmentCellsWidget.permanentCellContainer_textBrowser.setText(f'Cell-containing FOVs: {[k for k,v in self.CellContainer.num_cells.items() if v > 0]}\n' +
                                                                           f'Total cells: {np.sum([v for v in self.CellContainer.num_cells.values()])}\n' +
                                                                           ''.join([f'FOV {k}: {v} cells\n' for k,v in self.CellContainer.num_cells.items()]))
    
    def _send_stored_spots(self):
        self.MainImageCanvas.SpotContainer_permanent = self.SpotContainer
        self.MainImageCanvas.SpotContainer = deepcopy(self.SpotContainer)
        self.MainImageCanvas.SpotAnalysisWidget.spotLocalization_channelListWidget.clear()
        channels = []
        
        for key in self.SpotContainer.data.keys():
            f,ch = key
            if (f == self.current_fov) and (ch not in channels): channels.append(ch)
            
        for ch in channels: self.MainImageCanvas.SpotAnalysisWidget.spotLocalization_channelListWidget.addItem(str(ch))
    
    def _save_cells(self):
        save_file = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Cells', self.save_path, 'cell files (*.cells)')[0]
        if save_file == '':
            print(f'Save failed. No file selected')
        else:
            with open(save_file, 'wb') as f:
                total_data = {'fov_list': self.CellContainer.fov_list,
                              'h': self.CellContainer.h,
                              'w': self.CellContainer.w,
                              'data': self.CellContainer.data,
                              'num_cells': self.CellContainer.num_cells}
                dump(total_data, f)
            print(f'Finished Save: {save_file}')
    
    def _save_spots(self):
        save_file = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Spots', self.save_path, 'spot files (*.spots)')[0]
        if save_file == '':
            print(f'Save failed. No file selected')
        else:
            with open(save_file, 'wb') as f:
                total_data = {'fov_list': self.SpotContainer.fov_list,
                              'channel_list': self.SpotContainer.channel_list,
                              'data': self.SpotContainer.data}
                dump(total_data, f)
            print(f'Finished Save: {save_file}')
    
    def _save_spotmeta(self):
        save_spot_metadata = QtWidgets.QFileDialog.getSaveFileName(self, 'Save spot metadata', self.save_path, 'spot metadata file (.*smeta)')[0]
        if save_spot_metadata == '':
            print(f'Save failed. No file selected')
        else:
            with open(save_spot_metadata, 'wb') as f:
                cells, spots, channel_list = self.MainImageCanvas.SpotMetaDataAnalyzer.save()
                dump((cells, spots, channel_list), f)
            print(f'Finished Save: {save_spot_metadata}')
    
    def _load_cells(self):
        load_file = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Cells', self.save_path, 'cell files (*.cells)')[0]
        if load_file == '':
            print(f'Load failed. No file selected')
        else:
            with open(load_file, 'rb') as f:
                cell_data = load(f)
            fov_list = cell_data['fov_list']
            h,w = cell_data['h'], cell_data['w']
            data = cell_data['data']
            num_cells = cell_data['num_cells']
            self.CellContainer = CellContainer(fov_list)
            self.CellContainer.data = data
            self.CellContainer.h = h
            self.CellContainer.w = w
            self.CellContainer.num_cells = num_cells
            
            self._send_stored_cell()
            self._update_cell()
            self.MainImageCanvas._reindexCells()
    
    def _load_spots(self):
        load_file = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Spots', self.save_path, 'spot files (*.spots)')[0]
        if load_file =='':
            print(f'Load failed. No file selected')
        else:
            with open(load_file, 'rb') as f:
                spot_data = load(f)
            fovs = spot_data['fov_list']
            channels = spot_data['channel_list']
            data = spot_data['data']
            self.SpotContainer = SpotContainer(fovs, channels)
            self.SpotContainer.data = data
            self._send_stored_spots()

    def _load_spotmeta(self):
        load_file = QtWidgets.QFileDialog.getOpenFileName(self, 'Load spot metadata', self.save_path, 'spot metadata file (.*smeta)')[0]
        if load_file == '':
            print('Load failed. No file selected')
        else:
            with open(load_file, 'rb') as file:
                cells_dict, spots_dict, channel_list = load(file)
                fov_list = list(cells_dict.keys())
                cells = {int(f):[] for f in fov_list}
                for f in fov_list:
                    for i in range(len(cells_dict[f])):
                        spots_list = []
                        for j in range(len(cells_dict[f]['spots'])):
                            Aspot = ASpot()
                            Aspot.set_metadata(id = cells_dict[f][i]['spots'][j]['id'], fov = cells_dict[f][i]['spots'][j]['fov'],
                                               coordinate = cells_dict[f][i]['spots'][j]['coordinate'], celltype = cells_dict[f][i]['spots'][j]['celltype'],
                                               cell = cells_dict[f][i]['spots'][j]['cell'], size = cells_dict[f][i]['spots'][j]['size'],
                                               brightness = cells_dict[f][i]['spots'][j]['brightness'], neighbors = cells_dict[f][i]['spots'][j]['neighbors'],
                                               channel = cells_dict[f][i]['spots'][j]['channel'])
                            spots_list.append(Aspot)
                        Acell = ACell()
                        Acell.set_metadata(id = cells_dict[f][i]['id'], fov = cells_dict[f][i]['fov'],
                                        area = cells_dict[f][i]['area'], spots = spots_list, celltype = cells_dict[f][i]['celltype'],
                                        total_num_spots = cells_dict[f][i]['total_num_spots'],
                                        distmap = cells_dict[f][i]['distmap'], image = cells_dict[f][i]['image'],
                                        channels = cells_dict[f][i]['channels'], num_spots = cells_dict[f][i]['num_spots'])
                    cells[f].append(Acell)
                    
                spots = {int(f):[] for f in fov_list}
                for f in fov_list:
                    for i in range(len(spots_dict[f])):
                        Aspot = ASpot()
                        Aspot.set_metadata(id = spots_dict[f][i]['id'], fov = cells_dict[f][i]['spots'][j]['fov'], celltype = spots_dict[f][i]['celltype'],
                                           coordinate = spots_dict[f][i]['coordinate'], cell = spots_dict[f][i]['cell'],
                                           size = spots_dict[f][i]['size'], brightness = spots_dict[f][i]['brightness'], neighbors = spots_dict[f][i]['neighbors'],
                                           channel = spots_dict[f][i]['channel'])
                    spots[f].append(Aspot)
                
                self.MainImageCanvas.SpotMetaDataAnalyzer = SpotMetaDataAnalyzer()
                self.MainImageCanvas.SpotMetaDataAnalyzer.set_metadata(fov_list = fov_list, channel_list = channel_list)
                self.MainImageCanvas.SpotMetaDataAnalyzer.load_linked_cell_and_spot(cells = cells, spots = spots)

    def _link_spot_cell(self, b):
        if b:
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_LinkSpotsAndCellsPushButton.setEnabled(False)
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_CellTypesListWidget.clear()
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_FieldOfViewsListWidget.clear()
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotSizesListWidget.clear()
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_BrightnessListWidget.clear()
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_NumberOfSpotsListWidget.clear()
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotDistanceChannel1ListWidget.clear()
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotDistanceChannel2ListWidget.clear()
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelFrame_CalculateSpotSizesChannelListWidget.clear()
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelFrame_CountSpotsPerCellChannelListWidget.clear()
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelFrame_ShowSpotBrightnessListWidget.clear()
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelFrame_MeasureSpotDistanceChannel1ListWidget.clear()
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelFrame_MeasureSpotDistanceChannel2ListWidget.clear()
            
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_CellTypesListWidget.addItem('All')
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_FieldOfViewsListWidget.addItem('All')
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotSizesListWidget.addItem('All')
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_BrightnessListWidget.addItem('All')
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_NumberOfSpotsListWidget.addItem('All')
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelFrame_CalculateSpotSizesChannelListWidget.addItem('All')
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelFrame_ShowSpotBrightnessListWidget.addItem('All')
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelFrame_CountSpotsPerCellChannelListWidget.addItem('All')
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotDistanceChannel1ListWidget.addItem('None')
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotDistanceChannel2ListWidget.addItem('None')
            
            for ct in self.celltypes:
                self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_CellTypesListWidget.addItem(str(ct))
            for fov in self.fov_list:
                self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_FieldOfViewsListWidget.addItem(str(fov))
            for ch in self.channel_list:
                self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotSizesListWidget.addItem(str(ch))
                self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_NumberOfSpotsListWidget.addItem(str(ch))
                self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_BrightnessListWidget.addItem(str(ch))
                self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotDistanceChannel1ListWidget.addItem(str(ch))
                self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotDistanceChannel2ListWidget.addItem(str(ch))
                self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelFrame_CalculateSpotSizesChannelListWidget.addItem(str(ch))
                self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelFrame_ShowSpotBrightnessListWidget.addItem(str(ch))
                self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelFrame_CountSpotsPerCellChannelListWidget.addItem(str(ch))
                self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelFrame_MeasureSpotDistanceChannel1ListWidget.addItem(str(ch))
                self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelFrame_MeasureSpotDistanceChannel2ListWidget.addItem(str(ch))
        else:
            self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_LinkSpotsAndCellsPushButton.setEnabled(True)

    def _update_cell(self):
        self.cell_list = ['All'] + [str(e) for e in list(range(1,self.MainImageCanvas.CellContainer.num_cells[self.current_fov]+1))]
        self.MainWindowUI.SelectCell_ListWidget.clear()
        for cell in self.cell_list: self.MainWindowUI.SelectCell_ListWidget.addItem(str(cell))
        
        self.CellContainer = deepcopy(self.MainImageCanvas.CellContainer)
        self.MainImageCanvas.CellContainer_permanent = self.CellContainer
        self.MainImageCanvas.cell_list = deepcopy(self.cell_list)
        self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_CellContainerStateTextBrowser.setText(f'Cell-containing FOVs: {[k for k,v in self.CellContainer.num_cells.items() if v > 0]}\n' +
                                                                                                            f'Total cells: {np.sum([v for v in self.CellContainer.num_cells.values()])}\n' +
                                                                                                            ''.join([f'FOV {k}: {v} cells\n' for k,v in self.CellContainer.num_cells.items()]))
        self.MainImageCanvas.SegmentCellsWidget.transientCellContainer_textBrowser.setText(f'Cell-containing FOVs: {[k for k,v in self.CellContainer.num_cells.items() if v > 0]}\n' +
                                                                           f'Total cells: {np.sum([v for v in self.CellContainer.num_cells.values()])}\n' +
                                                                           ''.join([f'FOV {k}: {v} cells\n' for k,v in self.CellContainer.num_cells.items()]))
        self.MainImageCanvas.SegmentCellsWidget.permanentCellContainer_textBrowser.setText(f'Cell-containing FOVs: {[k for k,v in self.CellContainer.num_cells.items() if v > 0]}\n' +
                                                                           f'Total cells: {np.sum([v for v in self.CellContainer.num_cells.values()])}\n' +
                                                                           ''.join([f'FOV {k}: {v} cells\n' for k,v in self.CellContainer.num_cells.items()]))
        self._link_spot_cell(False)
        
    def _update_spots(self):
        self.SpotContainer = deepcopy(self.MainImageCanvas.SpotContainer)
        self.MainImageCanvas.SpotContainer_permanent = self.SpotContainer
        spot_containing_fovs = []
        total_spots = {f: 0 for f in self.fov_list}
        for (f,ch), v in self.SpotContainer.data.items():
            if len(v) > 0 and f not in spot_containing_fovs: spot_containing_fovs.append(f)
            total_spots[f] += len(v)
            
        self.MainImageCanvas.AnalysisPanelWidget.AnalysisPanelWorkBox_SpotContainerStateTextBrowser.setText(f'Spot-containing FOVs: {spot_containing_fovs}\n' + 
                                                                                                            f'Total spots: {np.sum([v for v in total_spots.values()])}\n' +
                                                                                                            f''.join([f'FOV {f} Channel {ch}: {len(v)} spots\n' for (f,ch), v in self.SpotContainer.data.items()]))
        self.MainImageCanvas.SpotAnalysisWidget.SpotAnalysisPanel_PermanentSpotContainerTestBrowser.setText(f'Current FOV: {self.MainImageCanvas.current_fov}\n' +
                                                                                                            f'Number of spots in the channel: {np.sum(total_spots[self.MainImageCanvas.current_fov])}\n' + 
                                                                                                            f''.join([f'Channel {ch}: {len(v)} spots\n' for (f,ch),v in self.SpotContainer.data.items()]))
        self._link_spot_cell(False)
        
if __name__ == '__main__':
    question_app = QtWidgets.QApplication(sys.argv)
    question_window = QtWidgets.QMainWindow()
    question_window.show()
    config_file = QtWidgets.QFileDialog.getOpenFileName(question_window, 'Load configuration file', path, 'configuration file (*.xml)')[0]
    if config_file == '':
        question_window.close()
    else:
        question_window.close()
        print(config_file)
        app = QtWidgets.QApplication(sys.argv)
        window = MainWindow(config_file)#(path + config_name)
        window.show()
        sys.exit(app.exec_())