import numpy as np
import argparse
import imutils
import glob
import cv2
from matplotlib import pyplot as plt
import sys
import os

# argument parser:
# ap = argparse.ArgumentParser()  # declare an argument parser
# # add the arguments, and offer a help text
# ap.add_argument("-t","--template",required=True,help="path to the template")
# ap.add_argument("-i","--images",required=True,help="path to image to be analyzied")
# ap.add_argument("-v", "--visualize",
# 	help="Flag indicating whether or not to visualize each iteration")
# args = vars(ap.parse_args()) # not sure yet

# template=cv2.imread(args["template"])
#
# personPath = sys.argv[1]
# # personPath = '/Users/JL/Desktop/weJump/randomPic/person.png'
# # personPath = '/Users/JL/Desktop/weJump/randomPic/circleTop.png'
#
# template = cv2.imread(personPath)
# gray1=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
# (tH,tW) = template.shape[:2]  # the person is 239 in height and 97 in width
# edges1 = cv2.Canny(template,tH,tW)

# to check how the pic look like
# plt.subplot(121),plt.imshow(template,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges1,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()


# imagePath = '/Users/JL/Desktop/weJump/randomPic/bwpic2.png'
# imagePath = '/Users/JL/Desktop/weJump/randomPic/testCircle.png'
# image = cv2.imread(imagePath)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# found = None
def matchTemplate(templatePath,imagePath):
    template = cv2.imread(templatePath)
    gray1=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    (tH,tW) = template.shape[:2]  # the person is 239 in height and 97 in width
    edges1 = cv2.Canny(template,tH,tW)

    image = cv2.imread(imagePath)
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
            found = (maxVal,maxLoc,r)

	# unpack the bookkeeping varaible and compute the (x, y) coordinates
	# of the bounding box based on the resized ratio
    (maxVal, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    return (startX,startY,endX,endY,r,maxVal)
	# draw a bounding box around the detected result and display the image
	# cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
	# cv2.imshow("Image", image)


def LoopThroughTemplate(tempateImgs,imagePath):
    # a = list(map(lambda x:matchTemplate(x,'/Users/JL/Desktop/weJump/randomPic/bwpic2.png'),tempateImgs))
    # print(a)
    evalScore= 0
    indexSelected = 0
    # matchingResult= ()
    for i in range(len(tempateImgs)):
        print('fitting %s' %tempateImgs[i] )
        matchingResult = (0,0,0,0,0,0)
        indexSelected = -1
        try:
            a=matchTemplate(tempateImgs[i],imagePath)
            print(a)
            if a[-1] > evalScore:
                evalScore=a[-1]
                matchingResult = a
                indexSelected=i
        except TypeError:
            pass
        # print(matchingResult)
    return matchingResult




def SquaredError(p1,p2):
	if len(p1) != len(p2):
		return -1
		print ("not same length")
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
def findBoardposition(picName):
	fullPath = './jumpImg/'+picName
	op2=Image.open(fullPath)
	fig, ax = plt.subplots()
	ax.imshow(op2)
	x=540
	y=960
# 	ax.plot(x,y,'r*')
	# plot(x,y,'r*',xlim((-1080,0)),ylim((-1920,0)))
	plt.show()
	ax = raw_input("Please enter Xcor,Ycor: ")
	xy =var1.split(',')
	return (int(xy[0]),int(xy[1]))


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
	path = 'User/Desktop/jumpImg/'
	fullPath = path+pictureName
	im = Image.open(fullPath)
	pixName=im.load()
	return pixName

def processPicture(pxl,epsLow,epsHigh,picName):
# 	path = './jumpImg/'
# 	fullPath = path+picName
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
	b2=findBoardposition(picName)
	pixDistance=calculatePixDistance(t2,b2)
	cont1=1
	eps=random.randint(epsLow,epsHigh)
	return [t2[0],t2[1],b2[0],b2[1],pixDistance,pixDistance+eps,1,picName]
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
    print 'x = %d, y = %d' %(ix, iy)
    global coords
    coords = [ix,iy]
    if len(coords) == 1:
        fig.canvas.mpl_disconnect(cid)
    return coords



def findBoardposition(picName):
	fullPath = './jumpImg/'+picName
	op2=Image.open(fullPath)
	fig, ax = plt.subplots()
	ax.imshow(op2)
	cid= fig.canvas.mpl_connect('button_press_event', onclick)
	plt.show()
	fig.canvas.mpl_disconnect(cid)
# 	ax = raw_input("Please enter Xcor,Ycor: ")
# 	xy =var1.split(',')
# 	return (int(xy[0]),int(xy[1]))
	return (coords[0],coords[1])




if __name__ == "__main__":
    imgPath = '/Users/JL/Desktop/weJump/randomPic/testCircle.png'
    tempateImgs = []
    for img in os.listdir('/Users/JL/Desktop/weJump/randomPic'):
        if img.startswith('template'):
            tempateImgs.append('/Users/JL/Desktop/weJump/randomPic'+'/'+img)
    a=LoopThroughTemplate(tempateImgs,imgPath)
    print(a)
