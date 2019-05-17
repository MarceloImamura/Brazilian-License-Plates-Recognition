import cv2


def processar_img(i):
        img = cv2.imread("/Users/marceloimamura/Desktop/darkflow-master/test/Placa/placa"+ str(i)+".jpg")

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(img_hsv)

        _, img_v = cv2.threshold(v, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


        # findcontours
        _, contours, _ = cv2.findContours(img_v, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        main_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]


        for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)

                if(h>20):
                        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                        cv2.imwrite("/Users/marceloimamura/Desktop/darkflow-master/test/placa/placa"+str(i)+"-1.jpg", img_v)
                        cv2.imwrite("/Users/marceloimamura/Desktop/darkflow-master/test/placa/placa"+str(i)+"-2.jpg", img)



#x, y, w, h = cv2.boundingRect(main_contour)

#cv2.drawContours(img, main_contour, -1, (0, 255, 0), 2)



for i in range(0,100):
        processar_img(i)