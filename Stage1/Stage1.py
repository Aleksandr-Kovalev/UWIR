import tensorflow as tf
import numpy as np
import keras
import os
import shutil
import matplotlib.pyplot as plt

from os import path
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import models
from keras import optimizers
from time import time
from keras.callbacks import TensorBoard
from keras.layers import Input, Conv2D, Conv2DTranspose, Dropout, concatenate
from keras.models import Model
from keras import backend as K
from PIL import Image
from keras.applications.vgg16 import preprocess_input
from keras.regularizers import l2

#This is a U-net with VGG16 as the encoder, the goal is to train the network
#to restores underwater images that suffer from underwater image degradation.
#
#STAGE 1 (basic setup to learn and to establish the foundation, contains 2 images)
#
#needed packages: keras, tensorflow, numpy, pillow, matplotlib, scikit-image
#By: Aleksandr Kovalev
#Date: 9.23.2019

#size of images for model input (make sure divisible by 8)
IMG_HEIGHT = 240
IMG_WIDTH = 240

def deprocess_image(x):
    """utility function to convert a float array into a valid uint8 image.
    # Arguments
        x: A numpy-array representing the generated image.
    # Returns
        A processed numpy-array, which could be used in e.g. imshow.
    """
    # normalize tensor: center on 0., ensure std is 0.25

    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

print("tensorflow version: " + tf. __version__)
print("numpy version: " + np. __version__)
print("keras version: " + keras. __version__)

#dataset percentage splits, used to split up the data into different folders
train_split_pct = 1
val_split_pct = 0.0

#Dataset Directory SetUp
#Path where labeled data is collected
dataset_dir = 'pool'
dataset_dir_org = 'pool/org'
dataset_dir_edit = 'pool/edit'

#count of dataset
org_image_count = len(os.listdir(dataset_dir_org))
edit_image_count = len(os.listdir(dataset_dir_edit))

print('Orginal dataset orginal image count: %d' % org_image_count)
print('Orginal dataset edited image count: %d ' % edit_image_count)

#Directory to store train, test, and validation
base_dir = 'pool'

#Directory locations
train_org_dir = os.path.join(base_dir, 'org')
train_edit_dir = os.path.join(base_dir, 'edit')

#Image Summary
print('total training org images:', len(os.listdir(train_org_dir)))
print('total training edit images:', len(os.listdir(train_edit_dir)))

#function to return a tensor to RGB format
import statistics

#Image Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    data_format='channels_last',
    fill_mode='reflect') #set to insure the image stability sense this NN is learning to post-process

#Two generator needed to feed the network and to train it.
train_org_generator = train_datagen.flow_from_directory(
    train_org_dir,
    target_size=(IMG_HEIGHT, IMG_HEIGHT),
    #batch_size=10,
    shuffle=False,
    class_mode=None,
    seed=18) # seed must match, this insures that the augumentation is identical

#second generator
train_edit_generator = train_datagen.flow_from_directory(
    train_edit_dir,
    target_size=(IMG_HEIGHT, IMG_HEIGHT),
    #batch_size=10,
    shuffle=False,
    class_mode=None,
    seed=18)

#combine the generators into a tuple with iteration for the input
train_generator = zip(train_org_generator, train_edit_generator)

#check generator and the photos generated. Needs to match
x_batch, y_batch = next(train_generator)

#loop to show the augmentation, range can be set to a lower number
for i in range(0, org_image_count):
    #iterate through batch to get the photo
    x = x_batch[i]
    y = y_batch[i]

    #np.squeeze is needed if dealing with more then one photo in dataset
    #x = np.squeeze(x, axis=0)
    print(x.shape)

    #y = np.squeeze(y, axis=0)
    print(y.shape)

    imgs = plt.figure(figsize=(12, 6))
    imgs.add_subplot(1, 2, 1)
    plt.imshow(x)
    imgs.add_subplot(1, 2, 2)
    plt.imshow(y)
    plt.show()
    #refresh the plot and variables
    plt.clf()
    plt.cla()
    plt.close()

#VGG16 model base, this is the encoder of the U-NET
input_b = Input(batch_shape=(None, IMG_WIDTH,IMG_HEIGHT,3))
print(input_b.shape)

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_tensor=input_b)

conv_base.summary()

#New Top model U-net

#instantiate regularizer
reg = l2(0.001)
# activity_regularizer=reg <- place in layers you want to apply

#the VGG16 model
vgg_top = conv_base.get_layer('block5_conv2').output

block1_conv2 = conv_base.get_layer('block1_conv2').output
block2_conv2 = conv_base.get_layer('block2_conv2').output
block3_conv3 = conv_base.get_layer('block3_conv3').output
block4_conv3 = conv_base.get_layer('block4_conv3').output

start_neurons = 32 #mulipler for the neurons per layer

# 8 to 16
deconv4 = Conv2DTranspose(start_neurons * 8, (3,3), strides=(2, 2), padding='same')(vgg_top)
uconv4 = concatenate([deconv4, block4_conv3])
#uconv4 = Dropout(0.1)(uconv4)
uconv4 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(uconv4)
uconv4 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(uconv4)
uconv4 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(uconv4)

# 16 -> 32
deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
uconv3 = concatenate([deconv3, block3_conv3])
#uconv3 = Dropout(0.1)(uconv3)
uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

# 32 -> 64
deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
uconv2 = concatenate([deconv2, block2_conv2])
#uconv2 = Dropout(0.1)(uconv2)
uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

# 64 -> 128
deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
uconv1 = concatenate([deconv1, block1_conv2])
#uconv1 = Dropout(0.1)(uconv1)
uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

out = Conv2D(3,(1,1), padding='same', activation='sigmoid')(uconv1) #the ouput image

model = Model(inputs=input_b, outputs=out)

#for layer in model.layers[:17]:
#    layer.trainable = False

for layer in conv_base.layers:
    layer.trainable = False

#print network architecture
model.summary()

#Custom Loss function area below. Structural similarity, MSE, Euclidean distance
def SSIM(x,y):

    #Tensor calculation for SSIM function
    mean_x = K.mean(x, axis=-1)
    mean_y = K.mean(y, axis=-1)
    var_x = K.var(x, axis=-1)
    var_y = K.var(y, axis=-1)
    std_x = K.sqrt(var_x)
    std_y = K.sqrt(var_y)
    k1 = 0.01 ** 2
    k2 = 0.03 ** 2
    ssim = (2 * mean_x * mean_y + k1) * (2 * std_y * std_x + k2)
    denom = (mean_x ** 2 + mean_y ** 2 + k1) * (var_y + var_x + k2)

    ssim /= K.clip(denom, K.epsilon(), np.inf)
    #ssim = tf.select(tf.is_nan(ssim), K.zeros_like(ssim), ssim)

    return ssim

#SSIM for red channel only
def loss_SSIM_Red(y_true, y_pred):

    # extract red channels from tensors
    x = y_true[:,:,0]
    y = y_pred[:,:,0]

    ssim = SSIM(x, y)

    return ssim

#Green channel only
def loss_SSIM_Green(y_true, y_pred):

    # extract Green channels from tensors
    x = y_true[:,:,1]
    y = y_pred[:,:,1]

    ssim = SSIM(x, y)

    return ssim

#Blue channel only
def loss_SSIM_Blue(y_true, y_pred):

    # extract Blue channels from tensors
    x = y_true[:,:,2]
    y = y_pred[:,:,2]

    ssim = SSIM(x, y)

    return ssim

#Calculate the DSSIM for the image as a whole, does not consider colors, works as grey-scale
def DSSIM(y_true, y_pred):

    patches_true = y_true
    patches_pred = y_pred

    u_true = K.mean(patches_true, axis=-1)
    u_pred = K.mean(patches_pred, axis=-1)
    var_true = K.var(patches_true, axis=-1)
    var_pred = K.var(patches_pred, axis=-1)
    std_true = K.sqrt(var_true)
    std_pred = K.sqrt(var_pred)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)

    ssim /= K.clip(denom, K.epsilon(), np.inf)
    #ssim = tf.select(tf.is_nan(ssim), K.zeros_like(ssim), ssim)

    #returns an error number where 0 is a perfect match
    return K.mean((1.0 - ssim) / 2.0)

#custom loss were functions can be tested
def custom_loss(y_true, y_pred):

    channels = 3 #number of channels to calculate ssim over
    red_ssim = loss_SSIM_Red(y_true, y_pred)
    green_ssim = loss_SSIM_Green(y_true, y_pred)
    blue_ssim = loss_SSIM_Blue(y_true, y_pred)

    total_ssim = (red_ssim + green_ssim + blue_ssim)
    mean_ssim  = total_ssim / channels

    #loss functions variations
    #dssim = K.mean((channels - total_ssim) / 2.0)
    dssim = (channels - total_ssim)
    eucli_dis = K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
    mse_and_dssim = keras.losses.mean_squared_error(y_true, y_pred) + DSSIM(y_true, y_pred)

    #return loss chosen
    return (eucli_dis)

model.compile(optimizer='adam', loss=custom_loss)

#training area
history = model.fit_generator(train_generator,
                              steps_per_epoch=1, # TotalTrainingSamples / TrainingBatchSize
                              #validation_steps=10, # TotalvalidationSamples / ValidationBatchSize
                              #validation_data=validation_generator,
                              verbose=1,
                              epochs=1000)

#Area to see how well the model performs on an image
org_img = load_img('pool/org/x/org.0.jpg', target_size=(IMG_WIDTH,IMG_HEIGHT))
org_img = img_to_array(org_img)
org_img /= 255
org_img = np.expand_dims(org_img, axis=0)

output = model.predict(org_img)
output = np.squeeze(output, axis=0)
print(output.shape)
output = deprocess_image(output)

#make a sound when done and chart is ready
import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)

#show comparison orginal to networks generated
org_img = load_img('pool/org/x/org.0.jpg', target_size=(IMG_WIDTH,IMG_HEIGHT))
edit_img = load_img('pool/edit/y/edit.0.jpg', target_size=(IMG_WIDTH,IMG_HEIGHT))
imgs = plt.figure(figsize=(18, 6))
imgs.add_subplot(1, 3, 1)
plt.imshow(org_img)
imgs.add_subplot(1, 3, 2)
plt.imshow(edit_img)
imgs.add_subplot(1, 3, 3)
plt.imshow(output)
plt.show()

print("END")
