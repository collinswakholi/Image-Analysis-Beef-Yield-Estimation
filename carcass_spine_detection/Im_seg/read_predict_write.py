import os
import time

import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import image

from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers


# model directory
mdlDir = r'F:\Collins_ops\Deep_learning\weights_ribline_only.hdf5'

## load model
model = models.load_model(mdlDir, custom_objects=None)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy','mean_squared_error'])
print('Model loaded')


dirpath = os.getcwd() #Get current directory
#print("current directory is : " + dirpath)

imageName = '/Im_1.png' # name of image to be read

test_image_dir = dirpath + imageName

img_shape = (256, 256) #input image size

countt = 1


while (countt>0): # repeat loop pepetually
    try:
        aa = os.path.exists(test_image_dir)
    except OSError as e:
        aa = 'False'
    time.sleep(0.001) #pause execution for 0.001 seconds
    print(aa)
    
    if aa is True:
        try:
            test_image = image.load_img(test_image_dir, target_size=img_shape)
        except:
            continue
        
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255
        test_image.shape
        
        pred = model.predict(test_image)
        outName = dirpath + '/predicted.png'
        plt.imsave(outName, pred[0,:,:,0], cmap='gray')
        
        print('Done...')
        
        os.remove(test_image_dir)
        countt = 1
        print('Reset count...')
        time.sleep(0.0001) #pause execution for 0.00001 seconds
    else:
        countt = countt+1
        print(countt)