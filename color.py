import cv2
import numpy as np


def enhance_mycode(img):
    imgHeight, imgWidth, imgDeep = img.shape
    dst1 = np.zeros((imgHeight, imgWidth, 3), np.uint8)

    for i in range(0, imgHeight):
        for j in range(0, imgWidth):
            (b, g, r) = map(int, img[i, j])
            b += 40
            g += 40
            r += 40
            if b > 255:
                b = 255
            if g > 255:
                g = 255
            if r > 255:
                r = 255
            dst1[i, j] = (b, g, r)
    return dst1


def enhance_api(img):
    dst2 = np.uint8(np.clip((1.5 * img), 0, 255))
    return dst2


img = cv2.imread('oemyner.jpg', 1)
dst1 = enhance_mycode(img)
dst2 = enhance_mycode(img)

cv2.imshow('src', img)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()

