import cv2 as cv
import numpy as np

cap = cv.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

def findContours(inFrame):
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    grayFrame = cv.bilateralFilter(grayFrame, 10, 5, 5)
    outFrame = cv.adaptiveThreshold(grayFrame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 10)
    outFrame = cv.bitwise_not(outFrame)
    im2, cnts, something = cv.findContours(outFrame.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:10]
    screenCnt = np.zeros((4,2))
    for c in cnts:
	    # approximate the contour
	    peri = cv.arcLength(c, True)
	    approx = cv.approxPolyDP(c, 0.01 * peri, True)

	    # if our approximated contour has four points, then
	    # we can assume that we have found our screen
	    if len(approx) == 4:
		    screenCnt = approx
		    break


    return screenCnt

def perspectivewarp(screenCnt):
    if np.sum(screenCnt) == 0:
        return frame
    cv.drawContours(frame, screenCnt, -1, (0, 255, 0), 3)
    pts = screenCnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    mainFrame = np.float32([[0,0],[640,0],[0,480],[640,480]])

    # the top-left point has the smallest sum whereas the
    #  bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]

    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]

    # multiply the rectangle by the original ratio
    M = cv.getPerspectiveTransform(rect, mainFrame)
    warpFrame = cv.warpPerspective(frame,M,(640,480))
    return warpFrame

while(1):

    # Take each frame
    ret, frame = cap.read()
    cap.set(11, 35)  # Brightness
    cap.set(12, 30)  # Contrast

    screenCnt = findContours(frame)
    warpFrame = perspectivewarp(screenCnt)

    cv.imshow('frame', frame)
    cv.imshow('frame Warped', warpFrame)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
    if k == 112 or k == 80:
        print('Print image')
        cv.imwrite('tileMask.png', outFrameFinal)

cv.destroyAllWindows()
