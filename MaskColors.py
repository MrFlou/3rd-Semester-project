import cv2 as cv
import numpy as np

cap = cv.VideoCapture(1)
ret = cap.set(3, 640)
ret = cap.set(4, 480)


def findContours(inFrame):
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    outFrame = cv.adaptiveThreshold(grayFrame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 10)
    outFrame = cv.bitwise_not(outFrame)
    im2, cnts, something = cv.findContours(outFrame.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:10]
    screenCnt = np.zeros((4, 2))

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
# End of findContours


def perspectivewarp(screenCnt):
    if np.sum(screenCnt) == 0:
        return frame
    cv.drawContours(frame, screenCnt, -1, (0, 255, 0), 3)
    pts = screenCnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    mainFrame = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]

    M = cv.getPerspectiveTransform(rect, mainFrame)
    warpFrame = cv.warpPerspective(frame, M, (640, 480))
    return warpFrame
# End of perspectivewarp


def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mn = min(r, g, b)
    mx = max(r, g, b)
    df = mx-mn

    if mx == mn:
        h = 0
    elif mx == r and g >= b:
        h = (60 * ((g - b)/df))
    elif g == mx:
        h = (60 * (((b - r)/df) + 2))
    elif b == mx:
        h = (60 * (((r - g)/df) + 4))
    elif mx == r and g < b:
        h = 60 * (((r - b)/df) + 5)

    if mx == 0:
        s = 0
    else:
        s = df/mx

    v = mx

    h = h/2
    s = s*255
    v = v*255

    return h, s, v
# End of rgb2hsv


def colorThreshold(inFrame):
    height, width, channel = inFrame.shape
    hsv = np.zeros((height, width, channel), dtype=np.uint8)

    # Convert BGR to HSV with OpenCV
    hsv = cv.cvtColor(inFrame, cv.COLOR_BGR2HSV)

    # Ouer own BGR(RGB) to HSV conversion
    # for x in range(0, width):
    #     for y in range(0, height):
    #         b = frame.item(y, x, 0)
    #         g = frame.item(y, x, 1)
    #         r = frame.item(y, x, 2)
    #         h, s, v = rgb2hsv(r, g, b)
    #         hsv.itemset((y, x, 0), h)
    #         hsv.itemset((y, x, 1), s)
    #         hsv.itemset((y, x, 2), v)

    # define range of red color in HSV
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([5, 255, 255])
    lower_red2 = np.array([160, 40, 40])
    upper_red2 = np.array([180, 255, 255])

    maskR1 = cv.inRange(hsv, lower_red1, upper_red1)
    maskR2 = cv.inRange(hsv, lower_red2, upper_red2)
    maskR = maskR1 + maskR2

    # define range of black color in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 40])

    maskB = cv.inRange(hsv, lower_black, upper_black)

    # define range of green color in HSV
    lower_green = np.array([40, 35, 35])
    upper_green = np.array([90, 255, 255])

    maskG = cv.inRange(hsv, lower_green, upper_green)

    return maskR, maskG, maskB
# End of colorThreshold


def maskSizeReduction(arr):
    h, w = arr.shape
    return (arr.reshape(h//20, 20, -1, 20)
               .swapaxes(1, 2)
               .reshape(-1, 20, 20))
# End of maskSizeReduction


def maskReduction(nmask, sense):
    mask = maskSizeReduction(nmask)
    maskOut = np.zeros((24, 32), dtype=np.uint8)
    h1, w1 = maskOut.shape
    h, w, c = mask.shape

    h2 = 0
    for x in range(h1):
        for y in range(w1):
            maskOut.itemset((x, y), np.average(mask[h2]))
            h2 = h2+1

    for x in range(len(maskOut)):
        for y in range(len(maskOut[x])):
            if maskOut[x, y] > sense:
                maskOut.itemset((x, y), 255)
            else:
                maskOut.itemset((x, y), 0)

    return maskOut
# End of maskReduction

# Main Process(Loop)
while(1):

    # Take each frame
    ret, frame = cap.read()
    cap.set(11, 40)  # Brightness
    cap.set(12, 40)  # Contrast

    screenCnt = findContours(frame)
    warpFrame = perspectivewarp(screenCnt)

    maskR, maskG, maskB = colorThreshold(warpFrame)

    # Displaying diffrent stages of the process the frame goes though, to help debug any faults
    cv.imshow('frame', frame)
    cv.imshow('warpFrame', warpFrame)
    cv.imshow('Mask-Red', maskR)
    cv.imshow('Mask-Black', maskB)
    cv.imshow('Mask-Green', maskG)

    k = cv.waitKey(5) & 0xFF
    # Esc to quit the python code
    if k == 27:
        break
    # (Numpad P and p) to Print the processed image
    if k == 112 or k == 80:
        print('Print image')
        maskR2 = maskReduction(maskR, 30)
        maskB2 = maskReduction(maskB, 30)
        maskG2 = maskReduction(maskG, 30)
        maskRGB = cv.merge((maskB2, maskG2, maskR2))
        cv.imshow('blockMask', maskRGB)
        cv.imwrite('tileMask.png', maskRGB)

cv.destroyAllWindows()
