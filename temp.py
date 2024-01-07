import cv2

# Read image
img = cv2.imread('ddd.png')
img = cv2.resize(img, (64, 64))
cv2.imwrite('resize.png', img)

