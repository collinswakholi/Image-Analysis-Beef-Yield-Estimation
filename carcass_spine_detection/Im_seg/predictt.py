import os
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from keras.preprocessing import image
from matplotlib import pyplot as plt

def predict_deep( test_image_dir, model_path, outputFolder ):

    model = models.load_model(model_path, custom_objects=None)

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy','mean_squared_error'])
    
    
    img_shape = (256, 256) #input image size
    
    test_image = image.load_img(test_image_dir, target_size=img_shape)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image /= 255.
    test_image.shape
    
    pred = model.predict(test_image)
    
    #blend = test_image[0,:,:,:]

    #bin_pred = (pred[0,:,:,0])>0.5
    #blend[:,:,1] = blend[:,:,1]+(bin_pred)
    #blend[:,:,2] = blend[:,:,2]-(bin_pred)
    
    #plt.imshow(blend) 
    #plt.title("Predicted Spine")
    #plt.show()
    
  
    if not os.path.exists(outputFolder):
      os.makedirs(outputFolder)
      
    outName = outputFolder + '\predicted.png'
    
    plt.imsave(outName, pred[0,:,:,0], cmap='gray')

    return;