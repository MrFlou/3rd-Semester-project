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


def colorThreshold(color):
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

    if color == 0:
        # define range of blue color in HSV
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])

        mask = cv.inRange(hsv, lower_blue, upper_blue)
        res = cv.bitwise_and(frame, frame, mask=mask)

    elif color == 1:
        # define range of green color in HSV
        lower_green = np.array([60 - sense, 40, 40])
        upper_green = np.array([100 + sense, 255, 255])

        mask = cv.inRange(hsv, lower_green, upper_green)
        res = cv.bitwise_and(frame, frame, mask=mask)

    return mask, res


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


while(1):
    # Take each frame
    _, frame = cap.read()

    mask, res = colorThreshold(1)
    blockMask, blockMaskXL = blockout(mask)

    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('blockMask', blockMask)
    cv.imshow('blockMaskXL', blockMaskXL)

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
        cv.imwrite('Grid.png', blockMask)
        print('Print image')


cv.destroyAllWindows()
