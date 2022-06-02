import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random

from keras.preprocessing import image

import tensorflow as tf
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers


## select folder that contains imageData (UI)

img_dir = r'F:\Collins_ops\Deep_learning\Deep_201906\Im_data'

## OR
#im_folder = r'F:\Collins_ops\Deep_learning\Deep_201906'
#img_dir = os.path.join(im_folder, "Im_data")

print(img_dir)

img_names = os.listdir(img_dir)
print(img_names[:5])



def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = losses.mean_squared_error(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


model_path = r'F:\Collins_ops\Deep_learning\weights3.hdf5' #model directory
## load model
#model = models.load_model(model_path, custom_objects=None)
model = models.load_model(model_path, custom_objects={'bce_dice_loss': bce_dice_loss,
                                                          'dice_loss': dice_loss})

#model.compile(optimizer='adam',
#              loss='binary_crossentropy',
#              metrics=['accuracy','mean_squared_error'])
    
model.compile(optimizer='adam',
              loss = bce_dice_loss,
              metrics=['accuracy','mean_squared_error',dice_loss])

print('Model loaded...')


img_shape = (256, 256)

im_filenames = []
for i in img_names:
    im_filenames.append(img_dir + "/" + format(i))
print(im_filenames[:5])
print("Number of images loaded: {}".format(len(im_filenames)))

length = len(im_filenames)
num = 10


T = []
plt.figure(num=1,figsize=(10, 20))
for z in range(num):
    ran_num = random.randint(0,length)
    test_image = image.load_img(im_filenames[ran_num], target_size=img_shape)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image /= 255.
    test_image.shape
    

    
    t1 = 1000*time.time()
    pred = model.predict(test_image)
    t2 = 1000*time.time()-t1
    T.append(t2)
    print(str(t2)+'ms')
    blend = test_image[0,:,:,:]

    bin_pred = (pred[0,:,:,0])>0.5
    blend[:,:,1] = blend[:,:,1]+(bin_pred)
    blend[:,:,2] = blend[:,:,2]-(bin_pred)
    
    #plt.subplot(2,2,2)
    plt.imshow(blend) 
    plt.title("Predicted Spine")
    
    plt.show()
    time.sleep(.0000300)
    plt.close()
print(T)