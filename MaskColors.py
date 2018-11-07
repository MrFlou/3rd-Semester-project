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
    lower_red2 = np.array([150, 40, 40])
    upper_red2 = np.array([180, 255, 255])

    maskR1 = cv.inRange(hsv, lower_red1, upper_red1)
    maskR2 = cv.inRange(hsv, lower_red2, upper_red2)
    maskR = maskR1 + maskR2

    # define range of black color in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])

    maskB = cv.inRange(hsv, lower_black, upper_black)

    # define range of green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])

    maskG = cv.inRange(hsv, lower_green, upper_green)

    return maskR, maskG, maskB

# def perspectiveMatch(inFrame):
#
#
#     return outFrame, matchOne, matchTwo


def blockout(mask):
    MaskX, MaskY = mask.shape
    blackMask = cv.resize(mask, (int(MaskY/20), int(MaskX/20)))

    for x in range(len(blackMask)):
        for y in range(len(blackMask[x])):
            if blackMask[x, y] > sense:
                blackMask.itemset((x, y), 255)
            else:
                blackMask.itemset((x, y), 0)

    return blackMask


def maskSizeReduction(arr):
    h, w = arr.shape
    return (arr.reshape(h//20, 20, -1, 20)
               .swapaxes(1, 2)
               .reshape(-1, 20, 20))


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

# Main Process
while(1):

    # Take each frame
    _, frame = cap.read()

    maskR, maskG, maskB = colorThreshold()
    # blockMask, blockMaskBXL = blockout(maskB)
    # blockMask, blockMaskRXL = blockout(maskR)
    # blockMask, blockMaskGXL = blockout(maskG)
    maskR2 = maskReduction(maskR, 20)
    maskB2 = maskReduction(maskB, 20)
    maskG2 = maskReduction(maskG, 20)
    maskRGB = cv.merge((maskB2, maskG2, maskR2))

    cv.imshow('frame', frame)
    # cv.imshow('maskR', maskR)
    # cv.imshow('maskG', maskG)
    # cv.imshow('maskB', maskB)
    cv.imshow('blockMask', maskRGB)
    # cv.imshow('blockMaskBXL', blockMaskBXL)
    # cv.imshow('blockMaskGXL', blockMaskGXL)
    # cv.imshow('blockMaskRXL', blockMaskRXL)

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
        maskR2 = maskReduction(maskR, 20)
        maskB2 = maskReduction(maskB, 20)
        maskG2 = maskReduction(maskG, 20)
        maskRGB = cv.merge((maskB2, maskG2, maskR2))
        cv.imwrite('tileMask.png', maskRGB)
        # cv.imwrite('tileMaskR.png', maskR)
        # cv.imwrite('tileMaskG.png', maskG)
        # cv.imwrite('tileMaskB.png', maskB)


cv.destroyAllWindows()
