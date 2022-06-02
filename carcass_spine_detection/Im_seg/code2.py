# Very Important Dependancies
from tensorflow.keras import optimizers, losses
from tensorflow.keras import backend as K
from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow_addons as tfa
import tensorflow as tf
from PIL import Image
import pandas as pd
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import os
import glob
import zipfile
import functools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12, 12)


#import tensorflow.contrib as tfcontrib


# from keras import layers, losses, models, optimizers


# load image directories
# im_folder = r'F:\Collins_ops\Deep_learning\Deep_201906'
#im_folder = r'F:\Collins_ops\Deep_learning\my_deepL\train'
im_folder = r'X:\collins\Beef carcass\Collins_ops\Deep_learning\Deep_201906'
model_folder = os.path.join(im_folder, "model")

img_dir = os.path.join(im_folder, "Im_data")
label_dir = os.path.join(im_folder, "new_lab")
#label_dir = os.path.join(im_folder, "masks_341_rgb")

img_names = os.listdir(img_dir)
lab_names = os.listdir(label_dir)

print(len(img_names), 'Images found')

x_train_filenames = []
y_train_filenames = []
for img_id in img_names:  # image directories
    x_train_filenames.append(os.path.join(img_dir, format(img_id)))
print(x_train_filenames[:5])

for imgL_id in lab_names:  # Label directories
    y_train_filenames.append(os.path.join(label_dir, format(imgL_id)))
print(y_train_filenames[:5])


# Data split (Training:70, Validation:30)
x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = \
    train_test_split(x_train_filenames, y_train_filenames,
                     test_size=0.3, random_state=42, shuffle=True)

num_train_examples = len(x_train_filenames)
num_val_examples = len(x_val_filenames)

print("Number of training examples: {}".format(num_train_examples))
print("Number of validation examples: {}".format(num_val_examples))


# Display some images
display_num = 3

r_choices = np.random.choice(num_train_examples, display_num)

plt.figure(figsize=(10, 15))
for i in range(0, display_num * 2, 2):
    img_num = r_choices[i // 2]
    x_pathname = x_train_filenames[img_num]
    y_pathname = y_train_filenames[img_num]

    plt.subplot(display_num, 2, i + 1)
    plt.imshow(mpimg.imread(x_pathname))
    plt.title("Original Image")

    example_labels = Image.open(y_pathname)
    label_vals = np.unique(example_labels)

    plt.subplot(display_num, 2, i + 2)
    plt.imshow(example_labels)
    plt.title("Masked Image")

plt.suptitle("Examples of Images and their Masks")
plt.show()


# Image setup
img_shape = (256, 256, 3)
batch_size = 3
epochs = 5

out_ch = 3  # output channels
block_type = 2  # 1 --> simple convulutional block;   2 --> resnet ID block


# Data Augmentation Functions

def _process_pathnames(fname, label_path):
    # We map this function onto each pathname pair
    #img_str = tf.read_file(fname)
    img_str = tf.io.read_file(fname)
    img = tf.image.decode_png(img_str, channels=3)

    #label_img_str = tf.read_file(label_path)
    label_img_str = tf.io.read_file(label_path)
    # These are gif images so they return as (num_frames, h, w, c)
    # label_img = tf.image.decode_png(label_img_str)
    # *************************************************** watch
    label_img = tf.image.decode_png(label_img_str, channels=3)
    # The label image should only have values of 1 or 0, indicating pixel wise
    # object (car) or not (background). We take the first channel only.
    # label_img = label_img[:, :, 0]
    label_img = tf.expand_dims(label_img, axis=-1)
    return img, label_img


def shift_img(output_img, label_img, width_shift_range, height_shift_range):
    """This fn will perform the horizontal or vertical shift"""
    if width_shift_range or height_shift_range:
        if width_shift_range:
            width_shift_range = tf.random.uniform([],
                                                  -width_shift_range *
                                                  img_shape[1],
                                                  width_shift_range * img_shape[1])
        if height_shift_range:
            height_shift_range = tf.random.uniform([],
                                                   -height_shift_range *
                                                   img_shape[0],
                                                   height_shift_range * img_shape[0])
        # Translate both
        output_img = tfa.image.translate(output_img,
                                         [width_shift_range, height_shift_range])
        label_img = tfa.image.translate(label_img,
                                        [width_shift_range, height_shift_range])
    return output_img, label_img


def flip_img(horizontal_flip, tr_img, label_img):
    if horizontal_flip:
        flip_prob = tf.random.uniform([], 0.0, 1.0)
        tr_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
                                    lambda: (tf.image.flip_left_right(
                                        tr_img), tf.image.flip_left_right(label_img)),
                                    lambda: (tr_img, label_img))
    return tr_img, label_img


def _augment(img,
             label_img,
             resize=None,  # Resize the image to some size e.g. [256, 256]
             scale=1,  # Scale image e.g. 1 / 255.
             hue_delta=0,  # Adjust the hue of an RGB image by random factor
             horizontal_flip=False,  # Random left right flip,
             width_shift_range=0,  # Randomly translate the image horizontally
             height_shift_range=0):  # Randomly translate the image vertically
    if resize is not None:
        # Resize both images
        label_img = tf.image.resize(label_img, resize)
        img = tf.image.resize(img, resize)

    if hue_delta:
        img = tf.image.random_hue(img, hue_delta)

    img, label_img = flip_img(horizontal_flip, img, label_img)
    img, label_img = shift_img(
        img, label_img, width_shift_range, height_shift_range)
    label_img = tf.cast(label_img, dtype=tf.float64) * scale
    img = tf.cast(img, dtype=tf.float64) * scale
    return img, label_img


def get_baseline_dataset(filenames,
                         labels,
                         preproc_fn=functools.partial(_augment),
                         threads=5,
                         batch_size=batch_size,
                         shuffle=True):
    num_x = len(filenames)
    # Create a dataset from the filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Map our preprocessing function to every element in our dataset, taking
    # advantage of multithreading
    dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)
    if preproc_fn.keywords is not None and 'resize' not in preproc_fn.keywords:
        assert batch_size == 1, "Batching images must be of the same size"

    dataset = dataset.map(preproc_fn, num_parallel_calls=threads)

    if shuffle:
        dataset = dataset.shuffle(num_x)

    # It's necessary to repeat our data for all epochs
    dataset = dataset.repeat().batch(batch_size)
    return dataset


# setup Training and validation datasets
tr_cfg = {
    'resize': [img_shape[0], img_shape[1]],
    'scale': 1 / 255.,
    'hue_delta': 0.1,
    'horizontal_flip': True,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1
}
tr_preprocessing_fn = functools.partial(_augment, **tr_cfg)

val_cfg = {
    'resize': [img_shape[0], img_shape[1]],
    'scale': 1 / 255.,
}
val_preprocessing_fn = functools.partial(_augment, **val_cfg)


train_ds = get_baseline_dataset(x_train_filenames,
                                y_train_filenames,
                                preproc_fn=tr_preprocessing_fn,
                                batch_size=batch_size)
val_ds = get_baseline_dataset(x_val_filenames,
                              y_val_filenames,
                              preproc_fn=val_preprocessing_fn,
                              batch_size=batch_size)


# Preview Augmented data
# temp_ds = get_baseline_dataset(x_train_filenames,
#                                y_train_filenames,
#                                preproc_fn=tr_preprocessing_fn,
#                                batch_size=1,
#                                shuffle=False)
# # Examine some of these augmented images
# #data_aug_iter = temp_ds.make_one_shot_iterator()
# data_aug_iter = tf.compat.v1.data.make_one_shot_iterator(temp_ds)
# next_element = data_aug_iter.get_next()

# tf.compat.v1.disable_eager_execution()
# with tf.compat.v1.Session() as sess:
#   batch_of_imgs, label = sess.run(next_element)

#   # Running next element in our graph will produce a batch of images
#   plt.figure(figsize=(10, 10))
#   img = batch_of_imgs[0]

#   plt.subplot(1, 2, 1)
#   plt.imshow(img)

#   plt.subplot(1, 2, 2)
#   plt.imshow(label[0, :, :, 0])
#   plt.show()


# build model using Keras API

def conv_block(input_tensor, num_filters, batchnorm=True):
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    if batchnorm:
        encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)

    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    if batchnorm:
        encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)

    # encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    # if batchnorm:
    #       encoder = layers.BatchNormalization()(encoder)
    # encoder = layers.Activation('relu')(encoder)

    return encoder


def resnet_block(input_tensor, num_filters, batchnorm=True):

    out_size = input_tensor.shape[3]
    if tf.keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    # Component 1
    x = layers.Conv2D(filters=num_filters, kernel_size=(
        1, 1), kernel_initializer="he_normal")(input_tensor)
    if batchnorm:
        x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation("relu")(x)

    # Component 2
    x = layers.Conv2D(filters=num_filters, kernel_size=(
        3, 3), padding="same", kernel_initializer="he_normal")(x)
    if batchnorm:
        x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation("relu")(x)

    # Component 3
    x = layers.Conv2D(filters=out_size, kernel_size=(1, 1),
                      kernel_initializer="he_normal")(x)
    if batchnorm:
        x = layers.BatchNormalization(axis=bn_axis)(x)

    # Addition and short circuit top input tensor
    x = layers.add([x, input_tensor])
    x = layers.Activation("relu")(x)

    return x


def encoder_block(input_tensor, num_filters):
    if block_type == 1:
        encoder = conv_block(input_tensor, num_filters)
    elif block_type == 2:
        encoder = resnet_block(input_tensor, num_filters)

    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)

    return encoder_pool, encoder


def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(
        num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)

    if block_type == 1:
        decoder = conv_block(decoder, num_filters)
    elif block_type == 2:
        decoder = resnet_block(decoder, num_filters)

    if block_type == 1:
        decoder = conv_block(decoder, num_filters)
    elif block_type == 2:
        decoder = resnet_block(decoder, num_filters)

    # decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    # decoder = layers.BatchNormalization()(decoder)
    # decoder = layers.Activation('relu')(decoder)
    # decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    # decoder = layers.BatchNormalization()(decoder)
    # decoder = layers.Activation('relu')(decoder)
    return decoder


inputs = layers.Input(shape=img_shape)
# 256

encoder0_pool, encoder0 = encoder_block(inputs, 16)
# 128

encoder1_pool, encoder1 = encoder_block(encoder0_pool, 32)
# 64

encoder2_pool, encoder2 = encoder_block(encoder1_pool, 64)
# 32

encoder3_pool, encoder3 = encoder_block(encoder2_pool, 128)
# 16

encoder4_pool, encoder4 = encoder_block(encoder3_pool, 256)
# 8

center = conv_block(encoder4_pool, 512)
# center

decoder4 = decoder_block(center, encoder4, 256)
# 16

decoder3 = decoder_block(decoder4, encoder3, 128)
# 32

decoder2 = decoder_block(decoder3, encoder2, 64)
# 64

decoder1 = decoder_block(decoder2, encoder1, 32)
# 128

decoder0 = decoder_block(decoder1, encoder0, 16)
# 256

outputs = layers.Conv2D(out_ch, (1, 1), activation='sigmoid')(decoder0)


model = models.Model(inputs=[inputs], outputs=[outputs])


# define metrics and loss function

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / \
        (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(
        y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


##sgd = SGD(lr=0.1, decay=1e-3, momentum=0.9, nesterov=True)
#rmsprop1 = optimizers.RMSprop(lr=0.0007, rho=0.9, epsilon=None, decay=0.0)
adam1 = optimizers.Adam(lr=0.0008, beta_1=0.9, beta_2=0.995,
                        epsilon=None, decay=0.0, amsgrad=False)

# compile model
model.compile(optimizer=adam1, loss=bce_dice_loss, metrics=[
              'accuracy', 'mean_squared_error', dice_loss])
# model.compile(optimizer=adam1, loss='binary_crossentropy', metrics=['accuracy','mean_squared_error'])
model.summary()


# train model

save_model_path = os.path.join(model_folder, "weights.h5")
cp = tf.keras.callbacks.ModelCheckpoint(
    filepath=save_model_path, monitor='val_dice_loss', save_best_only=True, verbose=1)
# cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_loss', save_best_only=True, verbose=1)

# specify model callback
history = model.fit(train_ds,
                    steps_per_epoch=int(
                        np.ceil(num_train_examples / float(batch_size))),
                    epochs=epochs,
                    validation_data=val_ds,
                    validation_steps=int(
                        np.ceil(num_val_examples / float(batch_size))),
                    callbacks=[cp])


# visualize training process

loss = history.history['dice_loss']
val_loss = history.history['val_dice_loss']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

#ms_err = history.history['mean_squared_error']
#val_ms_err = history.history['val_mean_squared_error']

#loss = history.history['loss']
#val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
#plt.plot(epochs_range, ms_err, label='Training MSE')
#plt.plot(epochs_range, val_ms_err, label='Validation MSE')
plt.legend(loc='upper right')
plt.title('Training and Validation Accuracy')
#plt.title('Training and Validation MSE')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()

# Visualize actual performance on validation set

# Alternatively, load the weights directly: model.load_weights(save_model_path)
# model = models.load_model(save_model_path, custom_objects={'bce_dice_loss': bce_dice_loss,
#                                                           'dice_loss': dice_loss})
# model = models.load_model(save_model_path, custom_objects={'bce_dice_loss': bce_dice_loss,
#                                                           'dice_loss': dice_loss})
model = models.load_model(save_model_path, custom_objects=None)

# Let's visualize some of the outputs
data_aug_iter = val_ds.make_one_shot_iterator()
next_element = data_aug_iter.get_next()

# Running next element in our graph will produce a batch of images
plt.figure(figsize=(10, 20))
for i in range(5):
    batch_of_imgs, label = tf.compat.v1.keras.backend.get_session().run(next_element)
    img = batch_of_imgs[0]
    predicted_label = model.predict(batch_of_imgs)[0]

    plt.subplot(5, 3, 3 * i + 1)
    plt.imshow(img)
    plt.title("Input image")

    plt.subplot(5, 3, 3 * i + 2)
    plt.imshow(label[0, :, :, 0])
    plt.title("Actual Mask")
    plt.subplot(5, 3, 3 * i + 3)
    plt.imshow(predicted_label[:, :, 0])
    plt.title("Predicted Mask")
plt.suptitle("Examples of Input Image, Label, and Prediction")
plt.show()
