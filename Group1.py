import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import time
import os
from keras.models import load_model
from urllib.request import urlopen
from urllib.request import urlopen as uReq
import webbrowser


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model=load_model('my_model.h5')
model.load_weights('my_model_weights.h5')

f = open('1.jpg','wb')
f.write(uReq('https://www.spd.hk/pub/ai/uploads/1.png').read())
f.close()

while(True):
    
    tstart=time.time()
    img2=cv2.imread("1.jpg")
    img2=cv2.resize(img2,(1024,1024))

    cv2.imshow('image',img2)

    img2_gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    img3=255-img2_gray
    img3=img3.astype('float32')
    img3_min=np.amin(img3)
    img4=img3-np.amin(img3)
    img5=255*img4/(np.amax(img4)+1)
    kernel=np.ones((5,5),np.uint8)
    img6=cv2.dilate(img5,kernel,iterations=3)
    img7=cv2.resize(img6,(28,28),1)
    img8=img6.astype('uint8')
    cv2.imshow('image',img8)

    x_test_image=np.reshape(img7,(1,28,28))
    
    x_Test4d=x_test_image.reshape(x_test_image.shape[0],28,28,1).astype('float32')
    x_Test4d_normalize=(x_Test4d/np.amax(x_test_image))
    prediction=model.predict_classes(x_Test4d_normalize)
    result = prediction[0]
    webbrowser.open('https://spd.hk/pub/ai/update.php?g=1&r=&l='+str(result))
    tend=time.time()
    
    print (prediction[0])
    
    break

cv2.destroyAllWindows()

exit()
