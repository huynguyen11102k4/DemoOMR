import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class SheetLayout:
    def __init__(self, 
                 ansLeft = 0.27, 
                 ansRight = 0.73, 
                 ansTop = 0.36, 
                 ansBottom = 0.94, 
                 splitCol = 0.5, 
                 questionsPerCol = 25, 
                 optionsPerRow = 5, 
                 roiShrink = 0.18):
        self.ansLeft = ansLeft
        self.ansRight = ansRight
        self.ansTop = ansTop
        self.ansBottom = ansBottom
        self.splitCol = splitCol
        self.questionsPerCol = questionsPerCol
        self.optionsPerRow = optionsPerRow
        self.roiShrink = roiShrink
    
    def buildRois(self, W, H):
        x0 = int(W * self.ansLeft)
        x1 = int(W * self.ansRight)
        y0 = int(H * self.ansTop)
        y1 = int(H * self.ansBottom)
        
        xMid = x0 + int(self.splitCol * (x1 - x0))
        colBoxes = [(x0, xMid), (xMid, x1)]
        
        rois = []
        rowHeight = (y1 - y0) / self.questionsPerCol
        
        for colIdx, (colX0, colX1) in enumerate(colBoxes):
            colWidth = colX1 - colX0
            optionWidth = colWidth / self.optionsPerRow
            for r in range(self.questionsPerCol):
                quesIdx = r + 1 + colIdx * self.questionsPerCol
                yTop = int(y0 + r * rowHeight)
                yBottom = int(y0 + (r + 1) * rowHeight)
                dy = int((yBottom - yTop) * self.roiShrink)
                for o in range(self.optionsPerRow):
                    xLeft = int(colX0 + o * optionWidth)
                    xRight = int(colX0 + (o + 1) * optionWidth)
                    dx = int((xRight - xLeft) * self.roiShrink)
                    roi = (yTop + dy, yBottom - dy, xLeft + dx, xRight - dx, quesIdx, o)
                    rois.append(roi)
        return rois
