import cv2 as cv
import numpy as np

sense = 0
cap = cv.VideoCapture(1)
ret = cap.set(3, 640)
ret = cap.set(4, 480)


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


def colorThreshold():
    height, width, channel = frame.shape
    hsv = np.zeros((height, width, channel), dtype=np.uint8)

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

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
    lower_red2 = np.array([150, 40, 40])
    upper_red2 = np.array([180, 255, 255])

    maskR1 = cv.inRange(hsv, lower_red1, upper_red1)
    maskR2 = cv.inRange(hsv, lower_red2, upper_red2)
    maskR = maskR1 + maskR2

    # define range of black color in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 35])

    maskB = cv.inRange(hsv, lower_black, upper_black)

    # define range of green color in HSV
    lower_green = np.array([10, 20, 20])
    upper_green = np.array([100, 255, 255])

    maskG = cv.inRange(hsv, lower_green, upper_green)

    return maskR, maskG, maskB


def blockout(mask):
    MaskX, MaskY = mask.shape
    blackMask = cv.resize(mask, (int(MaskY/20), int(MaskX/20)))
    blackMaskXL = cv.resize(blackMask, (MaskY, MaskX), interpolation=cv.INTER_NEAREST)

    for x in range(len(blackMask)):
        for y in range(len(blackMask[x])):
            if blackMask[x, y] > 0:
                blackMask.itemset((x, y), 255)
            else:
                blackMask.itemset((x, y), 0)

    for x in range(len(blackMaskXL)):
        for y in range(len(blackMaskXL[x])):
            if blackMaskXL[x, y] > 1:
                blackMaskXL.itemset((x, y), 255)
            else:
                blackMaskXL.itemset((x, y), 0)

    return blackMask, blackMaskXL

    # height, width, channels = mask.shape
    # tileSize = 30
    # tileX = width/tileSize
    # tileY = height/tileSize
    # tileMap = np.zeros((tileY, tileX, channels), dtype=np.uint8)
    #
    # nr1 = 0
    # for xTile in range(0, tileX):
    #     nr2 = 0
    #     for yTile in range(0, tileY):
    #         sum = np.average(mask[nr1:nr1+tileSize, nr2:nr2+tileSize])
    #
    #         if sum >= 127:
    #             for x in range(0)
    #             tilemap.itemset((y, x, 0), 255)
    #         else:
    #             tilemap.itemset((y, x, 0), 0)

#Main Process
while(1):
    # Take each frame
    _, frame = cap.read()

    maskR, maskG, maskB = colorThreshold()
    #blockMask, blockMaskXL = blockout(maskB)

    cv.imshow('frame', frame)
    #cv.imshow('maskR', maskR)
    cv.imshow('maskG', maskG)
    cv.imshow('maskB', maskB)
    #cv.imshow('blockMask', blockMask)
    #cv.imshow('blockMaskXL', blockMaskXL)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
    if k == 43:
        sense = sense + 1
        print(sense)
    if k == 45:
        sense = sense - 1
        print(sense)
    if k == 112 or k == 80:
        print('Print image')
        maskRGB = cv.merge((maskB, maskG, maskR))
        cv.imwrite('tileMask.png', maskRGB)


cv.destroyAllWindows()
