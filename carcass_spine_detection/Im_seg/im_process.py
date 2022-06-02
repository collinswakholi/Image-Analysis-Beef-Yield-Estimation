import os
#import numpy as np
#from matplotlib import pyplot as plt
#from keras.preprocessing import image
from predictt import predict_deep as pred1

my_image_dir = r'F:\Collins_ops\Deep_learning\Deep_201906\new_img\im_0001.png'# image directory

model_path = r'F:\Collins_ops\Deep_learning\weights_ribline_only.hdf5' # model directory

outputFolder = r'F:\Collins_ops\Deep_learning\Deep_201906\new_predict'

#img_shape = (256, 256) #input image size

#image_T = image.load_img(my_image_dir, target_size=img_shape)
#test_image = image.img_to_array(image_T)
#test_image /= 255.
#print(test_image.shape)

pred1 ( my_image_dir, model_path, outputFolder );

print("Done...")

#plt.imshow(test_image) 
#plt.title("Image")
#plt.show()