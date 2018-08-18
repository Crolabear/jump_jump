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


#distance


# we know the starting image is always the same. Use it to hard code the color of the person.
im = Image.open("./jumpImg/autojump.png")
pxl=im.load()
# startingPosition=pxl[330,1100]
# startColor = pxl[330,1100]
purple = [54, 51, 90, 255]#  purple

widthPix = 1080  # index 0, so 0 to 1079   
lengthPix = 1920 # index 0, so 0 to 1919
actualHeight1 = 290
actualHeight2 = 1400

totalPix = 1080*(1400-290)

# instead of taking the dot product, maybe we should take the diff then square it
dpArray = []
for i in range(widthPix):
	for j in range(actualHeight1,actualHeight2):
		dpArray.append(np.dot(purple,pxl[i,j]))	

s=np.unique(dpArray)

# look at hte color
steps = range(0,len(s),50)
proportion=[]
for item in steps:
	proportion.append(sum(dpArray<s[item]))

proportionD={}
for i in range(len(steps)):
	proportionD[steps[i]] = proportion[i]


step = range(0,1920,5)
backgroundGradient1 = []
for item in gstep:
	backgroundGradient1.append(pxl[540,item])

# identify the boundary of the next 
backgroundGradient2 = []
gstep = range(1920)
for item in gstep:
	backgroundGradient2.append(pxl[1070,item])


def SquaredError(p1,p2):
	if len(p1) != len(p2):
		return -1
		print "not same length"
	else:
		sum = 0
		for i in range(len(p1)):
			sum = sum+(p1[i]-p2[i])*(p1[i]-p2[i])
		return np.sqrt(sum)


def findDiffColor(px1,px2):
	count = (abs(px1[0] - px2[0]) > 3) +(abs(px1[1] - px2[1]) > 3)+(abs(px1[2] - px2[2]) > 3)		
	return count				

# to test: use findPeak(pxl,1070,290,1400,backgroundGradient2)
def findPeak(pix,w1,h1,h2,bgGradient):
# look for the first pixel that is very different from the image. use rms. return an coordinate
# 	m=[]
	for j in range(h1,h2):
		for i in range(w1):	
			err = findDiffColor(bgGradient[j],pix[i,j])
			if err > 1:
# 			if pix[i,j] != bgGradient[j]:
# 				m.append((i,j))
				m=(i,j)
				break
				
	return m	



	
# testing to see if i can filter out the background.		
newImg = []	
for j in range(lengthPix):
	for i in range(widthPix):
		if pxl[i,j] == backgroundGradient2[j]:
			newImg.append((0,0,0,255))
		else:
			newImg.append(pxl[i,j])
			
im2=Image.new(im.mode,im.size)
im2.putdata(newImg)
im2.save('newImg.png')



# isolate figure:
personColor =  (63, 52, 79, 255)
person1 = []	
count = 0

person1 = list(itertools.repeat((255,255,255,255),im.size[0]*im.size[1]))
for j in range(actualHeight1,actualHeight2):
	for i in range(0,widthPix):
		if SquaredError(pxl[i,j],personColor) < 100:
			person1[i+widthPix*j]=pxl[i,j]
			count = count + 1
im3=Image.new(im.mode,im.size)
im3.putdata(person1)
im3.save('person.png')
# BWPerson = im2.load()

def makeBlackAndWhite(pix,bgcolor,w1,height1,actualHeight1,actualHeight2,err1):
	whiteCanvas = list(itertools.repeat((255,255,255,255),w1*height1))
	for j in range(actualHeight1,actualHeight2):
		for i in range(0,w1):	
			if SquaredError(pxl[i,j],bgcolor[j]) > err1:
				whiteCanvas[i+w1*j]=pxl[i,j]
# 				count = count + 1	
	return whiteCanvas
	
def makeBinary(pix,bgcolor,w1,height1,actualHeight1,actualHeight2,err1):
	whiteCanvas = list(itertools.repeat(0,w1*height1))
	for j in range(actualHeight1,actualHeight2):
		for i in range(0,w1):
			if SquaredError(pxl[i,j],bgcolor[j]) > err1:
				whiteCanvas[i+w1*j]=1
# 				count = count + 1	
	return whiteCanvas
	

# def identifyPerson(pix,w0,w1,h0,h1,personOfColor,err1):
# 	m=[]
# 	for j in range(h0,h1):
# 		for i in range(w0,w1):
# 			if SquaredError(pix[i,j],personColor) < err1:
# 				m.append((i,j))
# 	return m

def identifyPersonV2(pix,w1,h0,h1,personOfColor,err1):
	count = 0
	person1 = list(itertools.repeat((255,255,255,255),im.size[0]*im.size[1]))
	for j in range(actualHeight1,actualHeight2):
		for i in range(0,w1):
			if SquaredError(pxl[i,j],personColor) < err1:
				person1[i+w1*j]=pxl[i,j]
				count = count + 1
	return person1



def personCoor2(pix,w1,h0,h1,personOfColor,err1):
	count = 0
	x=[]
	y=[]
# 	person1 = list(itertools.repeat((255,255,255,255),im.size[0]*im.size[1]))
	for j in range(actualHeight1,actualHeight2):
		for i in range(0,w1):
			if SquaredError(pxl[i,j],personColor) < err1:
				x.append(i)
				y.append(j)
# 				count = count + 1
	return [x,y]

def makeStringPix2Img(stringPix,imgName,imWidth,imLen):
	newImg=Image.new('RGBA',(imWidth,imLen))
	newImg.putdata(stringPix)
	newImg.save(imgName)
	

# t1=identifyPerson(pxl,personColor,100)  # t1 has the indice of the person
t2=identifyPersonV2(pxl,widthPix,290,1400,personColor,100)
t3=personCoor2(pxl,widthPix,290,1400,personColor,100)
Xs = map(lambda x:x[0],t1)
min(Xs)
max(Xs)
Ys = map(lambda y:y[1],t1)
min(Ys)
max(Ys)


bwPic = makeBlackAndWhite(pxl,backgroundGradient2,1080,1920,290,1400,10)
makeStringPix2Img(bwPic,'bwpic2.png',1080,1920)


def findPeak2(pix,w0,w1,h1,h2,bgGradient):
# look for the first pixel that is very different from the image. use rms. return an coordinate
	m=0
	for j in range(h1,h2):
		for i in range(w0,w1):		
			err = SquaredError(bgGradient[j],pix[i,j])
			if err > 100:
# 			if pix[i,j] != bgGradient[j]:
# 				m.append((i,j))
				m=(i,j)
				break
				
	return m	



imGO = Image.open("./jumpImg/restart.png")
GO2=imGO.load()

imYellow = Image.open("./jumpImg/restartYellow.png")
imYellow2=imYellow.load()
		