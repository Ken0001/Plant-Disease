# DenseNet practice
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv3D, DepthwiseConv2D, SeparableConv2D, Conv3DTranspose
from keras.layers import Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, UpSampling2D, BatchNormalization
from keras.layers import Concatenate, Add, Dropout, ReLU, Lambda, Activation, LeakyReLU, PReLU
from keras.optimizers import Adam, SGD

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from time import time
import numpy as np

from data import getData, getTrainData

def densenet(img_shape, n_classes, f=12):
  repetitions = 6, 12, 24, 16
  
  def bn_rl_conv(x, f, k=1, s=1, p='same'):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(f, k, strides=s, padding=p)(x)
    return x
  
  
  def dense_block(tensor, r):
    for _ in range(r):
      #x = bn_rl_conv(tensor, 4*f)
      x = bn_rl_conv(tensor, f, 3)
      tensor = Concatenate()([tensor, x])
    return tensor
  
  
  def transition_block(x):
    x = bn_rl_conv(x, K.int_shape(x)[-1] // 2)
    x = AvgPool2D(2, strides=2, padding='same')(x)
    return x
  
  
  input = Input(img_shape)
  
  x = Conv2D(64, 7, strides=2, padding='same')(input)
  x = MaxPool2D(3, strides=2, padding='same')(x)
  
  for r in repetitions:
    d = dense_block(x, r)
    x = transition_block(d)
  
  x = GlobalAvgPool2D()(d)
  
  output = Dense(n_classes, activation='softmax')(x)
  
  model = Model(input, output)
  return model

input_shape = 224, 224, 3
n_classes = 5

model = densenet(input_shape, n_classes)
model.summary()
#opt = Adam(lr=0.3)
opt = SGD(lr=0.1)#, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["categorical_accuracy"])

# Parameter
EP = 90
BS = 64
LR = 1e-2
width = 224
height = 224
# Data location
LOC = "clean"

x_train, y_train = getTrainData(LOC, width, height)

history = model.fit(x_train, y_train, 
                    batch_size=BS, 
                    epochs=EP,
                    validation_split=0.2,
                    #validation_data = (x_val, y_val),
                    #callbacks=[reduce_lr],
                    verbose=1)

model.save("densenet.h5")