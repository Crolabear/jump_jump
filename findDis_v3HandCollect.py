import time
# import wda
import math
import random
import itertools
import os
from PIL import Image, ImageDraw
import shutil
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from pylab import *

#adb shell input tap 1 2
#adb shell screencap -p /sdcard/screencap.png
#adb pull /sdcard/screen.mp4


def SquaredError(p1,p2):
	if len(p1) != len(p2):
		return -1
		print "not same length"
	else:
		sum = 0
		for i in range(len(p1)):
			sum = sum+(p1[i]-p2[i])*(p1[i]-p2[i])
		return np.sqrt(sum)




# obtain person coordinate and obtain all non-background coordinate
# aiming for 2 loop instead of 3 loops over the image

def personCoor3(pix,w1,h0,h1,personOfColor,err1):
	x=[]
	y=[]
	for j in range(h0,h1):
		for i in range(0,w1):
			if SquaredError(pix[i,j],personOfColor) < err1:
				x.append(i)
				y.append(j)
# 				count = count + 1
	xBase = np.median(x)
	yBase = max(y) - (max(y)-min(y)) * 0.1
	return (int(xBase),int(yBase))

def findNonBgColor(pix,w0,w1,h0,h1,bgGradient,err1):
	x=[]
	y=[]
	for j in range(h0,h1):
		for i in range(w0,w1):
			if SquaredError(pix[i,j],bgGradient[j]) > err1:	
				x.append(i)
				y.append(j)
	return (int(np.median(x)),int(np.median(y)))
	

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def findBoardposition(picName):
# 	fullPath = './jumpImg/'+picName
# 	op2=Image.open(picName)
# 	imshow(op2)
# 	x=540
# 	y=960
# 	plot(x,y,'r*')
# 	# plot(x,y,'r*',xlim((-1080,0)),ylim((-1920,0)))
# 	show()
# 	var1 = raw_input("Please enter Xcor,Ycor: ")
# 	xy =var1.split(',')
# 	return (int(xy[0]),int(xy[1]))


def calculatePixDistance(personC,boardC):
	euDistance = (personC[0]-boardC[0])*(personC[0]-boardC[0]) + (personC[1]-boardC[1])*(personC[1]-boardC[1])
	return np.sqrt(euDistance)
	
def takeScreenShot(fileName):
	os.system("adb shell screencap -p /sdcard/screen.png")
	saveStr = "adb pull /sdcard/screen.png "+'./jumpImg/'+fileName
	os.system(saveStr)
	


def checkState(newImgName):
	a=SquaredError(newImgName[1000,600],[51, 49, 34, 255])
	return (a<20)
# 		os.system("adb shell input tap 555 1555")


def longPress(timeMs):
	xcor=random.randint(300,900)
	ycor = random.randint(1000,1200)
	str1="adb shell input swipe %d %d %d %d %d" %(xcor, ycor, xcor+5, ycor+5, timeMs)
	os.system(str1)


def loadPicture(pictureName):
	path = './jumpImg/'
	fullPath = path+pictureName
	im = Image.open(fullPath)
	pixName=im.load()
	return pixName



def findBoardposition(picName):
# 	fullPath = './jumpImg/'+picName
	fig = plt.figure()
# 	plt.ion()
	ax = fig.add_subplot(111)
	op2=Image.open(fullPath)
	imshow(op2)
	x=540
	y=960
	ax.plot(x,y,'r*')
	cid = fig.canvas.mpl_connect('button_press_event', onclick)
	show()
	

def processPicture(pxl,b2,pictureName):
	path = './jumpImg/'
	fullPath = path+pictureName
# 	im = Image.open(fullPath)
# 	pxl=im.load()
	purple = [54, 51, 90, 255]#  purple
	widthPix = 1080  # index 0, so 0 to 1079   
	lengthPix = 1920 # index 0, so 0 to 1919
	actualHeight1 = 290
	actualHeight2 = 1400
	totalPix = 1080*(1400-290)
	
# 	restart = checkState(pxl)
# 	if restart == True:
# 		os.system("adb shell input tap 555 1555")
# 		os.system('mv '+fullPath+' '+path+'restart.png')
# 		cont1 = 0
# 		return [0,0,0,0,0,0]
# 	else:
	backgroundGradient2 = []
	gstep = range(1920)
	for item in gstep:
		backgroundGradient2.append(pxl[1070,item])

	t2=personCoor3(pxl,1080,290,1400,purple,100)
# 	b2=findBoardposition(fullPath)
	pixDistance=calculatePixDistance(t2,b2)
# 	cont1=1
# 	eps=random.randint(epsLow,epsHigh)
	return [t2[0],t2[1],b2[0],b2[1],pixDistance,pixDistance,1]
# steps are:
# take picture

# load picture and determine if the background is normal or black
# if black, restart
# if normal, calculate distance -- > jump

def writeJumpRecord(jumpList, fileName):
    """Write the list to csv file."""
    with open(fileName, "wb") as outfile:
    	writer = csv.writer(outfile)
        for item in jumpList:
            writer.writerow(item)


def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print 'x = %d, y = %d'%(
        ix, iy)
    global coords
    coords.append((ix, iy))
    if len(coords) == 1:
        fig.canvas.mpl_disconnect(cid)
    return coords

# will use human input to get 
jumpRecord = []

#~~~~~~

state1=False
i=0
ProjectName = str(3)
while(state1==False):
	time.sleep(5)
	picName = str(int(time.time()))+'.png'
	i=i+1
	takeScreenShot(picName)
	coords = []
	findBoardposition('./jumpImg/'+picName)
	pix1=loadPicture(picName)
	state1=checkState(pix1)
	if state1 == True:
		os.system("adb shell input tap 555 1555")
		os.system('mv ./jumpImg/'+picName+' ./jumpImg/restart.png')
		jumpRecord[-1][6]=0
	else:
		record=processPicture(pix1,coords[0][0],coords[0][1],picName)
		record.append(picName)
		print record
		longPress(record[5])
		jumpRecord.append(record)

print "Simulation finished"

writeJumpRecord(jumpRecord,'j4.csv')





# 	var1 = raw_input("Please enter Xcor,Ycor: ")
# 	xy =var1.split(',')
# 	print coords
# 	return (int(coords[0]),int(coords[1]))



# imshow(op2)
# op2=Image.open('./jumpImg/1517045099.png')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# imshow(op2)
# x=540
# y=960
# ax.plot(x,y,'r*')
# # plot(x,y,'r*',xlim((-1080,0)),ylim((-1920,0)))
# show()


    
cid = fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_disconnect(cid)


# cid = fig.canvas.mpl_connect('button_press_event', onclick)

