# import templateMatch tM
import numpy as np
import argparse
import imutils
import glob
import cv2
from matplotlib import pyplot as plt
import sys
import os
import time
# import wda
import math
import random
import itertools
import os
from PIL import Image, ImageDraw
import shutil
import csv

import pandas as pd
from sklearn.linear_model import LinearRegression

a=pd.read_csv('train2.csv')
columns1 = ['perX',	'perY',	'topLeftX',	'topLeftY',	'botRightX','botRightY','matchedIndex']

predictors =  pd.DataFrame(a, columns=columns1)
labeledOutcome = a['timePress']
model1 = LinearRegression()
model1.fit(predictors, labeledOutcome)



# def loadPicture(absPath):
# 	# path = 'User/Desktop/jumpImg/'
# 	fullPath = absPath
# 	im = Image.open(fullPath)
# 	pixName=im.load()
# 	return pixName


# def findRestart(newImgPix):
# 	restartName = '/Users/JL/Desktop/weJump/randomPic/templateRestart.png'
# 	(personTopX,personTopY,personBotX,peronBotY,_,_)=matchTemplate(restartName,newImgPix)

def findPerson(newImgPix):
	# define the paths
	personImgName = '/Users/JL/Desktop/weJump/randomPic/person.png'
# 1 download the image from phone - and save it?
# -- already downloaded.
# 2 check state:
	# pix1=loadPicture(newImgName)
	# state1=checkState(pix1)  # assumes it is in the correct state
	# if state is false -- continue
	# look for person's position.
	(personTopX,personTopY,personBotX,peronBotY,_,_)=matchTemplate(personImgName,newImgPix)
	personCoorX = personTopX/2 + personBotX/2
	personCoorY = peronBotY
	return (personCoorX,personCoorY)
# base on the position of the person, I want to crop the new image: the new platform will only
# show up to the left/right of the person.
	# newImgName = '/Users/JL/Desktop/weJump/randomPic/testCircle.png'
def findFigCenter(newImgPix,personCoorX,personCoorY):
	pix2=newImgPix
	# crop_img = pix1[y:y+h, x:x+w]
	ybegin = 200
	yend = int(personCoorY)
	approach = 1
	if personCoorX < pix2.shape[1]/2: # person on the left, jump toward right
	    xbegin = int(personCoorX)
	    xend = int(pix2.shape[1]-1)
		# print('method1',xbegin)
	else:  # person on the right, jump toward left
		approach = 2
		xbegin = 0
		xend = int(personCoorX)
		# print('method2',xbegin)
	crop_pix2= pix2[ybegin:yend, xbegin:xend]
	ind1,coordinates = LoopThroughTemplate(crop_pix2)
	figTopX = coordinates[0]
	figTopY = coordinates[1]
	figBotX = coordinates[2]
	figBotY = coordinates[3]
	# (figTopX,figTopY,figBotX,figBotY,_,_) =

	if approach == 1:
		figTopX = (figTopX + personCoorX)*1.2
		figBotX = (figBotX + personCoorX)*1.2
		figTopY = (figTopY + 200)
		figBotY = (figBotY + 200)
	return (figTopX,figTopY,figBotX,figBotY,ind1)  # return the index of the shape, and the corners
# plt.imshow(crop_pix2,cmap = 'gray') ## to view the picture
# plt.show()

def loadPicture(absPath):
	# path = 'User/Desktop/jumpImg/'
	# fullPath = absPath
	# im = Image.open(fullPath)
	pixName=cv2.imread(absPath)
	return pixName


def calculatePixDistance(personC,boardC):
	euDistance = (personC[0]-boardC[0])*(personC[0]-boardC[0]) + (personC[1]-boardC[1])*(personC[1]-boardC[1])
	return np.sqrt(euDistance)

def takeScreenShot(fileName):
	os.system("adb shell screencap -p /sdcard/screen.png")
	saveStr = "adb pull /sdcard/screen.png "+fileName # absolute path
	os.system(saveStr)


def SquaredError(p1,p2):
	if len(p1) != len(p2):
		return -1
		print ("not same length")
	else:
		sum = 0
		for i in range(len(p1)):
			sum = sum+(p1[i]-p2[i])*(p1[i]-p2[i])
		return np.sqrt(sum)


def checkState(newImgName):
	# restartImage = '/Users/JL/Desktop/weJump/randomPic/templateRestart.png'
	# a=SquaredError(newImgName[600,1000],[51, 49, 34])
	a=SquaredError(newImgName[200,600],[227, 216, 212])
	return (a<20)


def checkStateV2(newImgName):
	index1,_=LoopThroughTemplate(newImgName)
	if index1 == 2:
		return True
	else:
		return False

# check if the one particular pixel is a certain color.
def matchTemplate(templatePath,imagePathPix):
    template = cv2.imread(templatePath)
    gray1=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    (tH,tW) = template.shape[:2]  # the person is 239 in height and 97 in width
    edges1 = cv2.Canny(template,tH,tW)

    # image = cv2.imread(imagePath)
    image = imagePathPix
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None
# loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
    # resize the image according to the scale, and keep track
    # of the ratio of the resizing
        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
    # if the resized image is smaller than the template, then break
    # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            print ("size become too small")
            break
    	# detect edges in the resized, grayscale image and apply template
    	# matching to find the template in the image
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, edges1, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)#
    # check to see if the iteration should be visualized
    # if args.get("visualize", False):
    	# draw a bounding box around the detected region
    # clone = np.dstack([edged, edged, edged])
    # cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
    # 	(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
    # cv2.imshow("Visualize", clone)
    # cv2.waitKey(0)
    # if we have found a new maximum correlation value, then ipdate
    # the bookkeeping variable
        if found is None or maxVal > found[0]:
			# print('none')
            found = (maxVal,maxLoc,r)

	# unpack the bookkeeping varaible and compute the (x, y) coordinates
	# of the bounding box based on the resized ratio
    (maxVal, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    return (startX,startY,endX,endY,r,maxVal)



# then find the block position:
def LoopThroughTemplate(imagePix):
    evalScore= 0
    indexSelected = 0
    tempateImgs = ['/Users/JL/Desktop/weJump/randomPic/templateCircle.png',
	'/Users/JL/Desktop/weJump/randomPic/templateSquare.png',
	'/Users/JL/Desktop/weJump/randomPic/templateRestart.png']
	# matchingResult= ()
    matchingResult = (0,0,0,0,0,0)
    for i in range(len(tempateImgs)):
        print('fitting %s' %tempateImgs[i] )
        # indexSelected = -1
        try:
            a=matchTemplate(tempateImgs[i],imagePix)
            print(a)
            if a[-1] > evalScore:
                evalScore=a[-1]
                matchingResult = a
                indexSelected=i
        except TypeError:
            pass
        # print(matchingResult)
    return [indexSelected,matchingResult]


def calculatePixDistance(personC,boardC):
	euDistance = (personC[0]-boardC[0])*(personC[0]-boardC[0]) + (personC[1]-boardC[1])*(personC[1]-boardC[1])
	return np.sqrt(euDistance)

def longPress(timeMs):
	xcor=random.randint(300,900)
	ycor = random.randint(1000,1200)
	str1="adb shell input swipe %d %d %d %d %d" %(xcor, ycor, xcor+5, ycor+5, timeMs)
	os.system(str1)



def OneAction(ProjectName,model=model1):
	testReName = ProjectName+'_record.csv'
	fullPath = os.getcwd()+'/' + ProjectName
	picName = ProjectName+'_'+str(int(time.time()))+'.png'
	i=0
	# i=i+1
	absolutePath = fullPath+'/'+picName
	print ('here is ' + absolutePath)
	takeScreenShot(absolutePath)
	print ('loaded img ')
	pix1=loadPicture(absolutePath)
	state1=checkStateV2(pix1)
	# state0=checkState(pix1)

	if state1 == True:  # true means need to restart
		print (state1)
		time.sleep(5)  # wait for 10 seconds
		os.system("adb shell input tap 555 1555")
		os.system('mv '+absolutePath+' /Users/JL/Desktop/weJump/jumpImg/backup/restart.png')
		# jumpRecord[-1][6]=0
		tempRecord = [0,'end',0,0,0,0,0,0,0,0,99]
		with open(testReName,'a') as f:
			writer = csv.writer(f)
			writer.writerow(tempRecord)
		i = -1
	else:
		print (state1)
		pX,pY = findPerson(pix1)
		topLeftX,topLeftY,botRightX,botRightY,ind1=findFigCenter(pix1,pX,pY)

		distance = calculatePixDistance([pX,pY],[topLeftX/2+botRightX/2,topLeftY/2+botRightY/2])
		timePress = distance*1.2
		timePress = model1.predict([[pX,pY,topLeftX,topLeftY,botRightX,botRightY,ind1]])[0]
		if pX < pix1.shape[1]/2:
			leftRight = 1  # person on left, jump right
		else:
			leftRight = 2 #person on right, jump left
		tempRecord = [i,picName,pX,pY,topLeftX,topLeftY,botRightX,botRightY,distance,timePress,ind1]
		jumpRecord.append(tempRecord)
		print (jumpRecord)
		with open(testReName,'a') as f:
			writer = csv.writer(f)
			writer.writerow(tempRecord)
		longPress(timePress)
	return i



# The process :
# create a new project folder:
ProjectName = sys.argv[1]  # ex: Test1, Trial1Folder

jumpByHand = []
# ProjectName = 'Test2'
if '/User/Desktop' not in ProjectName:
    fullPath = os.getcwd()+'/' + ProjectName
else:
    fullPath = ProjectName
#Just to confirm the way files are transfered is correct.

os.mkdir(fullPath)
jumpRecord = []
item = 5
i=0
state1 = False
testReName = ProjectName+'_record.csv'
tempRecord = ['index','imageName','perX','perY','topLeftX','topLeftY','botRightX','botRightY','distance','timePress','matchedIndex']
with open(testReName,'a') as f:
	writer = csv.writer(f)
	writer.writerow(tempRecord)

J = 0
while(J !=-1):
	J = OneAction(ProjectName)
	i=i+1
	time.sleep(2)
