import cv2
import numpy as np



def rec_digitor(dir, index, exibe):
    print('Dir: ',dir)
    
    aux = cv2.imread(dir)
    img = np.ones((aux.shape[0]+2, aux.shape[1]+2, 3), np.uint8) * 255
    
    x_offset=y_offset=1
    img[y_offset:y_offset+aux.shape[0], x_offset:x_offset+aux.shape[1]] = aux   
    
    if(exibe):
        cv2.imshow('Original - 1',img)
        cv2.waitKey(0)
    
    print('Altura: ',img.shape[0])
    maxH = img.shape[0]
    print('Larugra: ',img.shape[1])
    maxW = img.shape[1]
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaus = cv2.GaussianBlur(img_gray, (3, 3), 0)
    
    if(exibe):
        cv2.imshow('Gaussian - 2',img_gaus)
    
    img_bilateral = cv2.bilateralFilter(img_gaus, 13, 70, 50)
    
    if(exibe):
        cv2.imshow('Equalizacao - 3',img_bilateral)


    #laplaciano
    # -1 -1 -1
    # -1  9 -1
    # -1 -1 -1
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    laplaciano = cv2.filter2D(img_bilateral, -1, kernel)
    
    if(exibe):
        cv2.imshow('Laplaciano - 4',laplaciano)
        
    
    equ = cv2.equalizeHist(laplaciano)
        
    if(exibe):       
        cv2.imshow('Equalize - 5',equ)
    
    ''' 
    h, s, v = cv2.split(img_hsv)
    
    
    '''
    _, img_l = cv2.threshold(img_bilateral, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
 
    if(exibe):
        cv2.imshow('threshold - 6',img_l)
    

        
    
    # findcontours
    _, contours, _ = cv2.findContours(img_l, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if(not(exibe)):
        cv2.imwrite("test/Pross/placa%d-Pross.jpg" %index, img_l) 
        
    i = 0
    print('Index: ',str(index))
    for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            porcH = (100*h)/maxH
            porcW = (100*w)/maxW
            #img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)        
            
            if(porcH>25 and porcW>2 and porcH<60 and porcW<20):
                #img_rec = img[y:y+h, x:x+w]
                #placa = imageData[top:bottom, left:right]
                #cv2.imwrite("/Users/marceloimamura/Desktop/darkflow-master/test/Pross/t"+str(i)+".jpg", img_rec)
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                print('Pos',i,'Altura: ',h,' Largura: ',w, ' % altura: ',int(porcH), ' % largura',int(porcW))
            '''
            elif(porcH>25 and porcW>2 and porcH<60 and porcW<35):
                #img_rec = img[y:y+h, x:x+w]
                #cv2.imshow('img_v2',img_rec)
                #cv2.imwrite("/Users/marceloimamura/Desktop/darkflow-master/test/Pross/t"+str(i)+".jpg", img_rec)
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                print('Arr',i,'Altura: ',h,' Largura: ',w, ' % altura: ',int(porcH), ' % largura',int(porcW))
            '''
            i=i+1
            
    if(not(exibe)):
        cv2.imwrite("test/Pross/placa%d.jpg" %index, img) 
    print('\n')
    
    if(exibe):
        cv2.imshow('Resul ',img)
        cv2.waitKey(0)

def test():    
    
    aux = cv2.imread('/Users/marceloimamura/Desktop/darkflow-master/test/Recorte/p-190.jpg')
    
    img = np.ones((aux.shape[0]+20, aux.shape[1]+20, 3), np.uint8) * 255
    
    x_offset=y_offset=10
    img[y_offset:y_offset+aux.shape[0], x_offset:x_offset+aux.shape[1]] = aux     

    cv2.imshow('Resul ',img)
    cv2.waitKey(0)

    
def main():
    #for i in range(1,5371):
    #    rec_digitor('/Users/marceloimamura/Desktop/darkflow-master/test/Recorte/p-'+str(i)+'.jpg',i,False)
    rec_digitor("/Users/marceloimamura/Desktop/darkflow-master/test/Recorte/p-900.jpg",0,True) 
    #test()


if __name__ == '__main__':
    main()