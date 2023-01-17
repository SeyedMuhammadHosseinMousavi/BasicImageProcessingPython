# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 09:25:47 2022
@author: S. M. Hossein Mousavi
"""
import numpy as np
import cv2 as cv
import glob
from skimage.feature import hog
import warnings
import sklearn.model_selection as ms
import sklearn.neighbors as ne
import sklearn.naive_bayes as nb
import sklearn.tree as tr
import sklearn.linear_model as lm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# Suppressing All Warnings
warnings.filterwarnings("ignore")

# Three DIM loading image
ColorImg = []
for img2 in glob.glob("data/*.jpg"):
    n= cv.imread(img2)
    ColorImg.append(n)

# Two DIM Loading Images
img=cv.imread('tst.jpg')
imggray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
images = [cv.imread(file) for file in glob.glob("data/*.jpg")]

# Dataset Size
DataSize=len(images) 

# Converting to Gray
Gray=images
for i in range(DataSize):
    Gray[i]=cv.cvtColor(images[i],cv.COLOR_BGR2GRAY)
    print('Convert to Gray Image',i)
    
    
# Resize Images
Resized=Gray
width = 512
height = 512
dim = (width, height)
for i in range(DataSize):
    Resized[i] = cv.resize(Gray[i], dim, interpolation = cv.INTER_AREA)
    print('Resize Image',i)

# Image Sizes
SampSize=Resized[1].shape 

# Hist EQ
HistEQ=Resized
for i in range(DataSize):
    HistEQ[i]=cv.equalizeHist(Resized[i])
    print('HistEQ Image',i)

# Edge Detection
CannyEdge=HistEQ
for i in range(DataSize):
    CannyEdge[i]=cv.Canny(HistEQ[i],100,200)
    print('Canny Edges for Image',i)
    
# Extracting SIFT Features     
sift = cv.SIFT_create()
SiftF=HistEQ
des=HistEQ
kp=HistEQ
DesSift=HistEQ
for i in range(DataSize):
    kp[i], des[i] = sift.detectAndCompute(HistEQ[i],None)
    DesSift[i]=sum(des[i])
    print('SIFT Features for Image',i)
    #pts = cv.KeyPoint_convert(kp)
    #pts2 = [p.pt for p in kp]

# Extracting HOG Features
HogF=ColorImg
for i in range(DataSize):
    HogF[i], hog_image = hog(ColorImg[i], orientations=8, pixels_per_cell=(64, 64),
                        cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    print('HOF Features for Image',i)

# List to matrix conversion
F1 = np.array(DesSift)
F2 = np.array(HogF)
F2 = np.float32(F2)

# Feature Fusion
FinalF = np.hstack((F1, F2))

# Labeling for Classification
ClassLabel = np.arange(DataSize)
ClassLabel[:20]=0
ClassLabel[20:40]=1
ClassLabel[40:60]=2

# Data Assign to X and Y
X=FinalF
Y=ClassLabel

# Data Split
Xtr, Xte, Ytr, Yte = ms.train_test_split(X, Y, train_size = 0.8)

# KNN Classifier
trAcc=[]
teAcc=[]
Ks=[]
for i in range(1,5):
    KNN = ne.KNeighborsClassifier(n_neighbors = i)
    KNN.fit(Xtr, Ytr)
    trAcc.append(KNN.score(Xtr, Ytr))
    teAcc.append(KNN.score(Xte, Yte))
    Ks.append(i)

# Logistic Regression Classifier
LR = lm.LogisticRegression(max_iter = 100)
LR.fit(Xtr, Ytr)
PredTrainLR= LR.predict(Xtr) # for train confusion matrix
PredTestLR= LR.predict(Xte) # for test confusion matrix
LRtrAcc = LR.score(Xtr, Ytr)
LRteAcc = LR.score(Xte, Yte)
        
# Naive Bayes Classifier
NB = nb.GaussianNB()
NB.fit(Xtr, Ytr)
NBtrAcc = NB.score(Xtr, Ytr)
NBteAcc = NB.score(Xte, Yte)

# Decision Tree Classifier
DTtrAcc = []
DTteAcc = []
MD = []
for i in range(2, 12):
    DT = tr.DecisionTreeClassifier(max_depth = i)
    DT.fit(Xtr, Ytr)
    DTtrAcc.append(DT.score(Xtr, Ytr))
    DTteAcc.append(DT.score(Xte, Yte))
    MD.append(i)

# Train and Test Results
print ('KNN Train Accuracy is :')
print (trAcc[-1])
print ('KNN Test Accuracy is :')
print (teAcc[-1])

print('Logestic Regression Train Accuracy is : ')
print (LRtrAcc)
print('Logestic Regression Test Accuracy is :')
print (LRteAcc)

print('Naive Bayes Train Accuracy is :')
print (NBtrAcc)
print('Naive Bayes Test Accuracy is :')
print (NBteAcc)

print('Decision Tree Train Accuracy is :')
print (DTtrAcc[-1])
print('Decision Tree Test Accuracy is :')
print (DTteAcc[-1])

# Plot Confusion Matrix for Logistic Regression
# Train LR
cm = confusion_matrix(Ytr, PredTrainLR)
cm_display = ConfusionMatrixDisplay(cm).plot()
# Test LR
cm2 = confusion_matrix(Yte, PredTestLR)
cm_display2 = ConfusionMatrixDisplay(cm2).plot()
