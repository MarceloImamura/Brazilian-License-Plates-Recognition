from darkflow.net.build import TFNet
import cv2

def recortarImg(imageData, inferenceResults,i):
        for res in inferenceResults:
                left = res['topleft']['x']
                top = res['topleft']['y']
                right = res['bottomright']['x']
                bottom = res['bottomright']['y']

                placa = imageData[top:bottom, left:right]
                cv2.imwrite("test/Recorte/placa%d.jpg" %i, placa)    
                i= i +1
        return i
    
def test_img(dir,i):
    img = cv2.imread(dir)
    result = tfnet.return_predict(img)
    recortarImg(img, result, i)
    print('Save: '+str(i))
    
        
        
options = {"model": "cfg/yolov2-tiny-voc-1c.cfg", "load": -1, "threshold": 0.5}
tfnet = TFNet(options)

for i in range(1,10):
    test_img('/Users/marceloimamura/Desktop/darkflow-master/train/ImgPlates/0000'+str(i)+'.jpg',i)
    
for i in range(10,100):
    test_img('/Users/marceloimamura/Desktop/darkflow-master/train/ImgPlates/000'+str(i)+'.jpg',i)
    
for i in range(100,1000):
    test_img('/Users/marceloimamura/Desktop/darkflow-master/train/ImgPlates/00'+str(i)+'.jpg',i)
    
for i in range(1000,5729):
    test_img('/Users/marceloimamura/Desktop/darkflow-master/train/ImgPlates/0'+str(i)+'.jpg',i)
    
    


