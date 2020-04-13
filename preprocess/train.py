import cv2, os, glob
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Input, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from skimage.transform import pyramid_expand
from Subpixel import Subpixel
from DataGenerator import DataGenerator

base_path = '/dataset‘

x_train_list = sorted(glob.glob(os.path.join(base_path, 'x_train', '*.npy')))
x_val_list = sorted(glob.glob(os.path.join(base_path, 'x_val', '*.npy')))

#train_set의 사진들은 현재 전처리를 통해 176*176 크기의 사진을 pyramid_reduce를 통해 44*44의 크기로 만든 상태.

train_gen = DataGenerator(list_IDs=x_train_list, labels=None, batch_size=16,
dim=(44,44), n_channels=3, n_classes=None, shuffle=True)

val_gen = DataGenerator(list_IDs=x_val_list, labels=None, batch_size=16, dim=(44,44),
n_channels=3, n_classes=None, shuffle=False)

upscale_factor = 4

inputs = Input(shape=(44, 44, 3))

def residual_block(x):
  y = Conv2D(filters=128, kernel_size=5, strides=1, padding='same',
  activation='relu')(x)
  y = Conv2D(filters=128, kernel_size=5, strides=1, padding='same',
  activation='relu')(y)
  y = Conv2D(filters=128, kernel_size=5, strides=1, padding='same',
  activation='relu')(y)
  return add([x, y])
  
net = Conv2D(filters=128, kernel_size=5, strides=1, padding='same',
activation='relu')(inputs)
net1 = residual_block(net)
result1 = Conv2D(filters=128, kernel_size=5, strides=1, padding='same',
activation='relu')(net1)
net2 = BatchNormalization()(net1)
net2 = residual_block(net2)
result2 = Conv2D(filters=128, kernel_size=5, strides=1, padding='same',
activation='relu')(net2)
net3 = BatchNormalization()(net2)
net3 = residual_block(net2)
result3 = Conv2D(filters=128, kernel_size=5, strides=1, padding='same',
activation='relu')(net3)
net4 = BatchNormalization()(net3)
net4 = residual_block(net4)
result4 = Conv2D(filters=128, kernel_size=5, strides=1, padding='same',
activation='relu')(net4)
net5 = BatchNormalization()(net4)
net5 = residual_block(net5)
result5 = Conv2D(filters=128, kernel_size=5, strides=1, padding='same',
activation='relu')(net5)

result1_2 = add([result1, result2])
result1_2_3 = add([result1_2, result3])
result1_2_3_4 = add([result1_2_3, result4])
result1_2_3_4_5 = add([result1_2_3_4, result5])

outputs = Subpixel(filters=3, kernel_size=3, r=upscale_factor,
padding=’same’)(result1_2_3_4_5)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')
model.summary()
