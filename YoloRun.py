from darkflow.net.build import TFNet
import cv2
import time
import numpy as np

class Letra(object):
    
        def __init__(self,x,img):
                self.x = x
                self.img = img

        def getImg(self):
                return self.img

        def getX(self):
                return self.x
                
def letra_dentro(x,letras,media):
        flag = True
        
        if(len(letras)>0):
                for letra in letras:
                        xl = letra.x

                        if(xl<x and x-1<xl+media):
                                flag = False
                                
                
        return flag

def desenhaBox(imageData, inferenceResults,i):
        
        for res in inferenceResults:
                left = res['topleft']['x']
                top = res['topleft']['y']
                right = res['bottomright']['x']
                bottom = res['bottomright']['y']
                #label = res['label']
                confidence = res['confidence']
                imgHeight, imgWidth, _ = imageData.shape
                thick = int((imgHeight + imgWidth) // 300)
                w = right - left
                h = bottom - top

                if(w>145):
                    #print('W: '+str(w)+' H: '+str(h))
                    placa = imageData[top:bottom, left:right]
                    cv2.imwrite('test/Placa/placa'+str(i)+'.jpg', placa) 
                    cv2.rectangle(imageData,(left, top), (right, bottom), (255,0,0), thick)

                    rec_digitor(placa,i)                        
                    #cv2.putText(imageData, label, (left, top - 12), 0, 1e-3 * imgHeight, (255,0,0), thick//3)
                    cv2.putText(imageData, str(confidence), (left, top - 12), 0, 1e-3 * imgHeight, (255,0,0), thick//3)
                else:
                    cv2.rectangle(imageData,(left, top), (right, bottom), (0,0,255), thick)
        #cv2.imwrite(imageOutputPath, imageData)
        return imageData

def rec_digitor(placa, i):

        maxH = placa.shape[0]
        maxW = placa.shape[1]

        #gray
        img_gray = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
        #gaussain blur
        img_gaus = cv2.GaussianBlur(img_gray, (3, 3), 0)

        #laplaciano
        # -1 -1 -1
        # -1  9 -1
        # -1 -1 -1
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        laplaciano = cv2.filter2D(img_gaus, -1, kernel)

        #equalizee Histograma
        equ = cv2.equalizeHist(laplaciano)

        #remoção de ruidos
        img_bilateral = cv2.bilateralFilter(equ, 13, 70, 50)

        #OTSU
        _, img_l = cv2.threshold(img_bilateral, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        #Colocar fundo
        img_l = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)

        aux_fundo = np.ones((maxH+20, maxW+20, 3), np.uint8) * 255
        aux_fundo2 = np.ones((maxH+20, maxW+20, 3), np.uint8) * 255
        x_offset=y_offset=10
        
        aux_fundo[y_offset:y_offset+maxH, x_offset:x_offset+maxW] = img_l 
        img_fundo = cv2.cvtColor(aux_fundo, cv2.COLOR_BGR2GRAY)
        
        aux_fundo[y_offset:y_offset+maxH, x_offset:x_offset+maxW] = placa 

        # findcontours
        _, contours, _ = cv2.findContours(img_fundo, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
        
        #media
        mediaW = 0
        i=0
        for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)
                porcH = (100*h)/maxH
                porcW = (100*w)/maxW
                
                if(porcH>20 and porcW>2 and porcH<60 and porcW<20 and i<7):
                        i+=1
                        mediaW += porcW
        
        if(i!=0):
                mediaW = mediaW/i
        else:
                mediaW = 0

        
        
        letras = []
        i = 0
        
        _, contours, _ = cv2.findContours(img_fundo, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)
                porcH = (100*h)/maxH
                porcW = (100*w)/maxW

                if(porcH>20 and porcW>2 and porcH<60 and porcW<20 and i<7):
                        if(porcW>mediaW+5):
                                meio = int(w/2)
                                placa = cv2.rectangle(placa,(x-10,y-10),(x+meio-10,y+h-10),(0,255,0),2)
                                placa = cv2.rectangle(placa,(x+meio-10,y-10),(x+w-10,y+h-10),(0,255,0),2)
                                
                                l1 = aux_fundo[y:y+h, x:x+meio-1]
                                l2 = aux_fundo[y:y+h, x+meio:x+w]
                                
                                letra1 = Letra(x,l1)
                                letra2 = Letra(x+meio,l2)
                                
                                
                                letras.append(letra1)
                                letras.append(letra2)
                                i+=2
                        
                        else: 
                                if(letra_dentro(x,letras,mediaW)):
                                        img = cv2.rectangle(placa,(x-10,y-10),(x+w-10,y+h-10),(0,255,0),2)
                                        l = aux_fundo[y:y+h, x:x+w]
                                        letra = Letra(x,l)
                                        letras.append(letra)
                                        i+=1
              
        lista_ordenada = letras
        lista_ordenada = sorted(letras, key = Letra.getX)

        return lista_ordenada

def recortarImg(imageData, inferenceResults,i):
        for res in inferenceResults:
                left = res['topleft']['x']
                top = res['topleft']['y']
                right = res['bottomright']['x']
                bottom = res['bottomright']['y']

                placa = imageData[top:bottom, left:right]
                cv2.imwrite("test/Placa/placa%d.jpg" %i, placa)    
                i= i +1
        return i



def detectar_img(dir,options):
        tfnet = TFNet(options)
        img = cv2.imread(dir)
        results = tfnet.return_predict(img)
        img = desenhaBox(img,results)
        cv2.imshow("YOLO", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def exec_deteccao(dir,options):
        tfnet = TFNet(options)
        i = 0
        iframe = 0
        cap = cv2.VideoCapture(dir)
        if not cap.isOpened(): 
                 print("could not open :")
        else:
            while (cap.isOpened()):
                    
                start_time = time.time()
                
                if cap.grab():
                    flag, frame = cap.retrieve()
                    
                    if not flag:
                        continue
                    else:
                        if iframe % 8 == 0:
                            result = tfnet.return_predict(frame)
                            #i = recortarImg(frame,aux,i)
                            frame = desenhaBox(frame,result,i)
                            
                            print('FPS {:.1f}'.format(1 / (time.time() - start_time)))
                            cv2.imshow('video', frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                                
                            i+=1
                        
                iframe += 1
            
def test_img(dir,options):
        tfnet = TFNet(options)
        arq = open(dir,'r')
        i = 1615
        for linha in arq:
                imgdir  = linha.rstrip()
                
                img = cv2.imread(imgdir)
                result = tfnet.return_predict(img)
                img = desenhaBox(img,result)
                cv2.imwrite('/Users/marceloimamura/Desktop/darkflow-master/train/test/'+str(i)+'.jpg',img)
                i+=1
                print(i)
                
        arq.close()
        
def main():
    options = {"model": "cfg/yolov2-tiny-voc-1c.cfg", "load": -1, "threshold": 0.5}
    exec_deteccao('/Users/marceloimamura/Desktop/darkflow-master/test/test4.mp4', options)


if __name__ == '__main__':
    main()
        