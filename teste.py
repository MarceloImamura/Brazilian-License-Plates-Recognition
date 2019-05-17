import cv2
import numpy as np
from matplotlib import pyplot as plt


class Letra(object):
    
    def __init__(self,x,w,img):
        self.x = x
        self.w = w
        self.img = img
    
    def getImg(self):
        return self.img
    
    def getW(self):
        return self.w
    
    def getX(self):
        return self.x

def letra_dentro(x,letras):
    flag = True
    
    if(len(letras)>0):
        for letra in letras:
            xl = letra.x
            wl = letra.w
            if(xl<x and x-1<xl+wl-10):
                flag = False
            
    return flag


def conta_preto_ini(img):
    metade = img.shape[0]*0.4
    
    print('Metade: ',metade)
    i = 0
    flag = True
    while(i<img.shape[1]/3 and flag):
        k = 0
        pixel_p = 0
        
        for k in range(img.shape[0]):
            if(img[k,i]==0):
                pixel_p +=1
                
        if(metade<pixel_p):
            print('Ini Pixel: ',pixel_p)
            flag = False
            i-=1
            
        i+=1
    return i

def conta_preto_fim(img):
    metade = img.shape[0]*0.4
    i = img.shape[1]-1
    flag = True
    while(i>img.shape[1]/3 and flag):
        k = 0
        pixel_p = 0
        
        for k in range(img.shape[0]):
            if(img[k,i]==0):
                pixel_p +=1
                
        if(metade<pixel_p):
            print('FIm Pixel: ',pixel_p)
            flag = False
            
        i-=1
    i-=1
    return i
            
def conta_preto_tudo(img):
    
    pixel = 0
    for i in range(img.shape[0]-1):
        for k in range(img.shape[1]-1):
            if(img[i,k] == 0):
                pixel+=1
    return pixel


def test(dir):    
    
    aux = cv2.imread(dir)
    
    img = np.ones((aux.shape[0]+20, aux.shape[1]+20, 3), np.uint8)
    
    _, img = cv2.threshold(aux, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(aux, contours, -1, (0, 255, 0), 2)
    
    print('Tam: ',len(contours))
    cv2.imshow('Resul ',aux)
    cv2.waitKey(0)

def test_thre(dir):
    img = cv2.imread(dir)
    grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, img_s = cv2.threshold(grayscaled, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 3)
    
    cv2.imshow('original',img)
    cv2.imshow('threshold Ots',img_s)
    
    cv2.imshow('Adaptive threshold',th)
    
    
    hist = cv2.calcHist([th],[0],None,[256],[0,256])
    print('Hist: ',hist)
    print('Tam: ',len(hist))
    print('Largura: ',img.shape[0])
    plt.show()
    cv2.waitKey(0)
    
def test_cor(dir): 
    #17
    
    print(dir)
    img = cv2.imread(dir)
    print('Altura: ',img.shape[0])
    print('Larugra: ',img.shape[1])
    
    grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 3)
    teste = grayscaled[x1:x2, y1:y2]
    _, s1 = cv2.threshold(teste, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, s2 = cv2.threshold(grayscaled, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    cv2.imshow('Origem ',img)
    cv2.imshow('Adaptativo ',th)
    cv2.imshow('Ori sem corte ',s1)
    cv2.imshow('Ori com corte',s2)
    cv2.waitKey(0)

def testArray():
    aux = cv2.imread('/Users/marceloimamura/Desktop/darkflow-master/test/Recorte/p-190.jpg')
    aux2 = cv2.imread('/Users/marceloimamura/Desktop/darkflow-master/test/Recorte/p-191.jpg')

    l1 = Letra(9,aux2)
    l2 = Letra(1,aux)

    lista = []
    lista.append(l1)
    lista.append(l2)
    
    lista_ordenada = sorted(lista, key = Letra.getLargura)
    
    l = lista_ordenada[0]
    
    cv2.imshow('Resul ',l.img)
    cv2.waitKey(0)
'''    
def rec_digitor(dir, index, exibe):
    print('Dir: ',dir)
    
    img = cv2.imread(dir)
    
    if(exibe):
        cv2.imshow('Original - 1',img)

    
    print('Altura: ',img.shape[0])
    maxH = img.shape[0]
    print('Larugra: ',img.shape[1])
    maxW = img.shape[1]
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaus = cv2.GaussianBlur(img_gray, (3, 3), 0)
    
    if(exibe):
        cv2.imshow('Gaussian - 2',img_gaus)
    

        
    #laplaciano
    # -1 -1 -1
    # -1  9 -1
    # -1 -1 -1
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    laplaciano = cv2.filter2D(img_gaus, -1, kernel)
    
    if(exibe):
        cv2.imshow('Laplaciano - 3',laplaciano)
    
    #equ = cv2.equalizeHist(laplaciano)
    img_bilateral = cv2.bilateralFilter(laplaciano, 13, 70, 50)
    
    if(exibe):
        #cv2.imshow('Equalizacao - 4',equ)
        cv2.imshow('bilateral - 5',img_bilateral)
        img_l = cv2.adaptiveThreshold(img_bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 3)
        cv2.imshow('Adaptativo',img_l)

    _, img_l = cv2.threshold(img_bilateral, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #img_l = cv2.adaptiveThreshold(img_bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 3)
    
    if(exibe):
        cv2.imshow('Otsu',img_l)
        
    
    aux = cv2.cvtColor(img_l,cv2.COLOR_GRAY2RGB)
    
    img_fundo = np.ones((maxH+20, maxW+20, 3), np.uint8) * 255
    x_offset=y_offset=10
    img_fundo[y_offset:y_offset+maxH, x_offset:x_offset+maxW] = aux 
    img_fundo = cv2.cvtColor(img_fundo, cv2.COLOR_BGR2GRAY)
  
    
    if(exibe):
        cv2.imshow('threshold com listra - 7',img_fundo)
    
    i=0
    _, contours, _ = cv2.findContours(img_fundo, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
     
    
    
    
    
    aux_fundo = np.ones((maxH+20, maxW+20, 3), np.uint8) * 255
    x_offset=y_offset=10
    aux_fundo[y_offset:y_offset+maxH, x_offset:x_offset+maxW] = img 
    
    letras = []
    i = 0
    _, contours, _ = cv2.findContours(img_fundo, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            porcH = (100*h)/maxH
            porcW = (100*w)/maxW
            #img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)        
            
            if(porcH>20 and porcW>2 and porcH<60 and porcW<20 and i<1):
                img = cv2.rectangle(img,(x-10,y-10),(x+w-10,y+h-10),(0,255,0),2)
                l = aux_fundo[y:y+h, x:x+w]
                letra = Letra(x,l)
                letras.append(letra)
                i+=1
            
              
    lista_ordenada = letras
    lista_ordenada = sorted(letras, key = Letra.getX)
    
    if(len(letras)==7):
        print('Correto ! ',len(lista_ordenada))
    else:
        print('Errado !',len(lista_ordenada))
            
    if(not(exibe)):
        cv2.imwrite("test/Pross/placa%d.jpg" %index, img) 
    print('\n')
    
    
    if(exibe):
        cv2.imshow('Resul ',img)
        k = cv2.waitKey(0)
        
        if(k == 32): # espaço
            i = 0
            print('tamanho: ',len(lista_ordenada))
            
            for letra in lista_ordenada:
                l = letra.img
                cv2.imshow('Mostrar letra',l)
                k = cv2.waitKey(0)
                
                if(k != 27):
                    cv2.imwrite('test/arroz/'+chr(k)+'-'+ str(index)+'-'+str(i)+'.jpg', l)
                    print('Salvo !')
                else:
                    print('Nãooooo')
                i+=1
                
            cv2.destroyAllWindows()
            
'''
def rec_digitor2(dir, index, exibe):
    print('Dir: ',dir)
    
    
    img = cv2.imread(dir)
    img_copy = img.copy()
    
    if(exibe):
        cv2.imshow('Original - 1',img)

    
    print('Altura: ',img.shape[0])
    maxH = img.shape[0]
    print('Larugra: ',img.shape[1])
    maxW = img.shape[1]
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaus = cv2.GaussianBlur(img_gray, (3, 3), 0)
    
    
    if(exibe):
        cv2.imshow('Gaussian - 2',img_gaus)

    #laplaciano
    # -1 -1 -1
    # -1  9 -1
    # -1 -1 -1
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    laplaciano = cv2.filter2D(img_gaus, -1, kernel)    



        
        
    adaptive = cv2.adaptiveThreshold(laplaciano, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 3)
    _, otsu = cv2.threshold(laplaciano, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    if(exibe):
        cv2.imshow('Laplaciano - 4',laplaciano)        
        cv2.imshow('Adap',adaptive)        
        cv2.imshow('Otsu - 4',otsu)        

    iAdaptive = conta_preto_tudo(adaptive)
    iOtsu = conta_preto_tudo(otsu)
    
    print('Otsu: ',iOtsu, 'iAdaptive: ',iAdaptive)
    if(iAdaptive<iOtsu):
        img_l = adaptive
    else:
        img_l = otsu

        
    
    aux = cv2.cvtColor(img_l,cv2.COLOR_GRAY2RGB)
    
    img_fundo = np.ones((maxH+20, maxW+20, 3), np.uint8) * 255
    x_offset=y_offset=10
    img_fundo[y_offset:y_offset+maxH, x_offset:x_offset+maxW] = aux 
    img_fundo = cv2.cvtColor(img_fundo, cv2.COLOR_BGR2GRAY)
  
    # findcontours
    _, contours, _ = cv2.findContours(img_l, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if(not(exibe)):
        cv2.imwrite("test/Pross/placa%d-1.jpg" %index, img_l) 
    
    if(exibe):
        cv2.imshow('threshold com listra - 7',img_fundo)
    
    i=0
    _, contours, _ = cv2.findContours(img_fundo, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    
    aux_fundo = np.ones((maxH+20, maxW+20, 3), np.uint8) * 255
    x_offset=y_offset=10
    aux_fundo[y_offset:y_offset+maxH, x_offset:x_offset+maxW] = img 
    
    i = 0

    for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            porcH = (100*h)/maxH
            porcW = (100*w)/maxW
            #img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)        
            
            if(porcH>20 and porcW>2 and porcH<60 and porcW<20 and i<1):
                x1 = x
                y1 = y
                w1 = w
                h1 = h
                #print('X1:',x1, 'y1: ',y1,' w: ',w1,' h: ',h1)
                img = cv2.rectangle(img,(x-10,y-10),(x+w-10,y+h-10),(0,255,0),2)
                i+=1
                
    letras = []
    if(i>0):      
        x1 = 0
        x2 = maxW
                
        y1 -=10
        y2 = y1+h1
        
        recorte = laplaciano[y1:y2, x1:x2]
        x1 = conta_preto_ini(recorte)
        x2 = conta_preto_fim(recorte)
        
        print('X1: ',x1, 'X2:',x2)
        
        if(exibe):
            cv2.imshow('recorte 1',recorte)
    
        recorte = laplaciano[y1:y2, x1:x2]
        rec2 = img[y1:y2, x1:x2]
        rec_copy = img_copy[y1:y2, x1:x2]        
        
        _, img_l = cv2.threshold(recorte, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        if(exibe):
            cv2.imshow('Saida1',img_l)
            
        aux = cv2.cvtColor(img_l,cv2.COLOR_GRAY2RGB)
        

        maxH = img_l.shape[0]
        maxW = img_l.shape[1]
    
        img_fundo = np.ones((maxH+20, maxW+20, 3), np.uint8) * 255
        x_offset=y_offset=10
        img_fundo[y_offset:y_offset+maxH, x_offset:x_offset+maxW] = aux 
        img_fundo = cv2.cvtColor(img_fundo, cv2.COLOR_BGR2GRAY)
        
        img_copy = np.ones((maxH+20, maxW+20, 3), np.uint8) * 255
        img_copy[y_offset:y_offset+maxH, x_offset:x_offset+maxW] = rec_copy
      
        # findcontours
        _, contours, _ = cv2.findContours(img_fundo, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        maxH = img_fundo.shape[0]
        maxW = img_fundo.shape[1]
        
        i=0
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            porcH = (100*h)/maxH
            porcW = (100*w)/maxW
            #img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)   
            
            if(porcH>30 and porcW>2 and porcW<40):
                if(letra_dentro(x,letras)):
                    img_fundo = cv2.rectangle(img_fundo,(x,y),(x+w,y+h),(0,255,0),2)
                    rec2 = cv2.rectangle(rec2,(x-10,y-10),(x+w-10,y+h-10),(0,255,0),2)
                    l = img_copy[y:y+h, x:x+w]
                    letra = Letra(x,w,l)
                    letras.append(letra)
                    i+=1

        
        if(exibe):
            cv2.imshow('Saida2',rec2)
            

    lista_ordenada = letras
    lista_ordenada = sorted(letras, key = Letra.getX)
    
    if(len(letras)==7):
        print('Correto ! ',len(lista_ordenada))
    else:
        print('Errado !',len(lista_ordenada))
            
    if(not(exibe)):
        cv2.imwrite("test/Pross/placa%d.jpg" %index, img) 
    print('\n')
    
    
    if(exibe):
        cv2.imshow('Resul ',img)
        k = cv2.waitKey(0)
        
        if(k == 32): # espaço
            i = 0
            print('tamanho: ',len(lista_ordenada))
            
            for letra in lista_ordenada:
                l = letra.img
                cv2.imshow('Mostrar letra',l)
                k = cv2.waitKey(0)
                
                if(k != 27):
                    cv2.imwrite('test/arroz/'+chr(k)+'-'+ str(index)+'-'+str(i)+'.jpg', l)
                    print('Salvo !')
                else:
                    print('Nãooooo')
                i+=1
                
            cv2.destroyAllWindows()
            

            

    
    
def main():
    #for i in range(1,5371):
    for i in range(104,2000):
        rec_digitor2('/Users/marceloimamura/Desktop/darkflow-master/test/Placa/p'+str(i)+'.jpg',i,True)
       #test_thre('/Users/marceloimamura/Desktop/darkflow-master/test/Placa/p'+str(i)+'.jpg')
 
    #test_thre('/Users/marceloimamura/Desktop/darkflow-master/test/placa.png')
    
    #for i in range(1,414):
#    test_cor('/Users/marceloimamura/Desktop/darkflow-master/test/Placa/p27.jpg')
    
    #for i in range(59,2000):
    #   test_cor('/Users/marceloimamura/Desktop/darkflow-master/test/Placa/p'+str(i)+'.jpg')
    
    #rec_digitor('/Users/marceloimamura/Desktop/darkflow-master/test/Placa/p99.jpg',0,True)     
    
    #testArray()
    


if __name__ == '__main__':
    main()