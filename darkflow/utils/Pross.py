import cv2


img = cv2.imread("/Users/marceloimamura/Desktop/darkflow-master/test/placa/placa608.jpg")

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(img_hsv)

_, img_v = cv2.threshold(v, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow('img_v',img_v)

# findcontours
_, contours, _ = cv2.findContours(img_v, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
main_contour = sorted(contours, key=cv2.contourArea, reverse=True)[3]

#x, y, w, h = cv2.boundingRect(main_contour)

cv2.drawContours(img, main_contour, -1, (0, 255, 0), 2)

cv2.imshow('img_v2',img)

cv2.waitKey(0)