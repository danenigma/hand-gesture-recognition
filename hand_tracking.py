#import serial
#import win32api, win32con
#from visual import *
import cv2
import numpy as np
#from visual.controls import *
#ser = serial.Serial('COM6',9600)
#scene.width=600
#scene.height=600
#ball = sphere(pos=(0,0,0), radius=2)
xMax = 1200
yMax = 1000
cx1 = 0
cy1 = 0
cz1 = 0
z_cali = True
KNOWN_DISTANCE = 10.0
KNOWN_WIDTH = 9.0
focalLength = 0
inches = 0 
cap  = cv2.VideoCapture(0)
enable = False
"""theta1 = pi/4
theta2 = pi/2
theta3 = pi/4
"""
# draw board
def calibrateZ(event,x,y,flags,param):
    global z_cali,enable
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if z_cali is  True:
             z_cali = False
             if enable is True:
                 enable = False
             else:
                 enable = True
             print "not calebrating....."
             #print enable
def hand(img):
    global z_cali
    global inches
    global KNOWN_DISTANCE 
    global KNOWN_WIDTH 
    global focalLength
    origImage = img.copy()
    drawing = np.zeros(img.shape,np.uint8)  
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #cv2.imshow("hsv",hsv)
    """    l1 = cv2. getTrackbarPos('lvalue1', 'tracker' )
    l2 = cv2. getTrackbarPos('lvalue2', 'tracker' )
    l3 = cv2. getTrackbarPos('lvalue3', 'tracker' )
    u1 = cv2. getTrackbarPos('uvalue4', 'tracker' )
    u2 = cv2. getTrackbarPos('uvalue5', 'tracker' )
    u3 = cv2. getTrackbarPos('uvalue6', 'tracker' )
    """ #print l1 , l2,l3,u1,u2,u3
    lower = np. array([0, 38, 28])
    upper = np. array([30, 241, 250])
    mask = cv2.inRange(hsv,lower,upper)
    res = cv2.bitwise_and(img,img,mask = mask)
    gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)  
    ret ,thresh = cv2.threshold(gray,30,255,0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #res = cv2.drawContours(res, contours, -1, (0,255,0), 3)
    #cv2.imshow("res",res)
    max_area = 0
    ci = 0
    index_val = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > max_area:
            max_area = area
            ci = i
    if ci==0:
          return    
    cnt  = contours[ci]
    moments = cv2.moments(cnt)
    if moments['m00']!=0:
          cx = int(moments['m10']/moments['m00'])
          cy = int(moments['m01']/moments['m00'])

    hull1 = cv2.convexHull(cnt)
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)
    rect = cv2.minAreaRect(cnt)
    """
    if rect is not None:    
        if z_cali is True:
            focalLength = (rect[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
        if z_cali is False:
            inches = distance_to_camera(KNOWN_WIDTH, focalLength, rect[1][0])
    """      
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    if defects is None:
            return
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        #print d
        if d>15000:
            far = tuple(cnt[f][0])
            dist = cv2.pointPolygonTest(box,far,True)
            #print dist
            if dist > 50:
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])                
                cv2.line(origImage,start,end,[0,255,255],2)
                cv2.circle(origImage,far,10,[255,0,0],-1)
                index_val += 1
                
    font = cv2.FONT_HERSHEY_SIMPLEX
    fingers = index_val+1
    cv2.drawContours(origImage,[box],0,(0,0,255),2)
    cv2.drawContours(origImage,[hull1],0,(0,255,0),2)
    cv2.drawContours(origImage,[cnt],0,(0,255,0),2)

    cv2.putText(origImage,str(fingers),(40,150), font, 6 ,(255,0,255),2)
    #cv2.putText(origImage,"x= "+ str(cx)+ " y= "+ str(cy), font, 0.5 ,(255,0,0),2)
    cv2.circle(origImage,(cx-10,cy+10),10,[0,0,255],-1)
    #cv2.setMouseCallback("image",calibrateZ)
    cv2.imshow("image",origImage)
   
    return index_val+1,cx,cy#,inches

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth
################################################################################
#
################################################################################
while 1:
   # rate(50)
    ret,img  =  cap.read()
    if img is None:
          continue
    if hand(img) is not None:
           fin,camx,camy = hand(img)
     
           x  = max(min(xMax,camx+300),0)
           y  = max(min(yMax,camy+300),0)
           #if enable is True:    
               #win32api.SetCursorPos((int(x),int(y)))

    key = cv2.waitKey(10)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
 
