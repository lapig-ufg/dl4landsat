import os

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D, Cropping2D, ZeroPadding2D
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.callbacks import TensorBoard

import rasterio
import numpy as np
from rasterio import windows
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score

tensorboard = TensorBoard(log_dir='logs',histogram_freq=10)

def get_chips(path_image, size = 100, pad_x = 0, pad_y = 0):
	indexes = None
	dataset = rasterio.open(path_image)
	height = dataset.height
	width = dataset.width
	n_grid_height = math.ceil(dataset.height/float(size)) - 1
	n_grid_width = math.ceil(dataset.width/float(size)) - 1

	chipped = []

	for i in range(n_grid_height):
	    for j in range(n_grid_width):
	        row_start = i*n_grid_height + pad_y
	        col_start = j*n_grid_width + pad_x
	        window = windows.Window(col_start, row_start, size, size)
	        data = dataset.read(indexes, window=window, masked=False, boundless=True)
	        chipped.append(data)

	        if (row_start + size*2) > height or (col_start + size*2) > width:
	          break
	return chipped

def get_chips_padding(path_image, size = 100, lista_porc = [(0,0)], n_rotate =0):
	lista_resultado = []
    #get list of chips
	for porc_x, porc_y in lista_porc:
		pad_x = int(size * (porc_x / 100.0))
		pad_y = int(size * (porc_y / 100.0))
		lista_temp = get_chips(path_image, pad_x = pad_x, pad_y = pad_y, size = size)
		lista_resultado = lista_resultado + lista_temp

	       
    #rotate chips
	for i in range(n_rotate):
        
		lista_resultado = lista_resultado + [np.rot90(m,k = i + 1,axes=(1,2)) for m in lista_resultado]


	return np.transpose(np.stack(lista_resultado), [0,2,3,1])

path = ""
name = "Unet"
batch_size = 2
chip_size = 256
weights_path = path + name + '.h5'
path_image_data = 'SF-23-Y-C_jan18_clipped.tif'
path_image_label = 'SF-23-Y-C_jan18_ref_clipped.tif'

padding_list = [(0,0), (0,30), (30, 0), (30,30)]
#padding_list = [(0,0), (0,30)]

x_train = get_chips_padding(path_image_data, chip_size, padding_list)
y_train = get_chips_padding(path_image_label, chip_size, padding_list)

nsamples, _, _, channels = x_train.shape
img_size = chip_size

print(x_train.shape)
print(y_train.shape)

X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.2)

print('Split train: ', len(X_train), len(Y_train))
print('Split valid: ', len(X_val), len(Y_val))


def get_crop_shape(target, refer):
	# width, the 3rd dimension
	cw = (target.get_shape()[2] - refer.get_shape()[2]).value
	assert (cw >= 0)
	if cw % 2 != 0:
		cw1, cw2 = int(cw/2), int(cw/2) + 1
	else:
		cw1, cw2 = int(cw/2), int(cw/2)
	# height, the 2nd dimension
	ch = (target.get_shape()[1] - refer.get_shape()[1]).value
	assert (ch >= 0)
	if ch % 2 != 0:
		ch1, ch2 = int(ch/2), int(ch/2) + 1
	else:
		ch1, ch2 = int(ch/2), int(ch/2)

	return (ch1, ch2), (cw1, cw2)

def get_unet(n_ch,patch_height,patch_width):
	concat_axis = 3

	inputs = Input((patch_height, patch_width, n_ch))

	conv1 = Conv2D(32, (3, 3), padding="same", name="conv1_1", activation="relu", data_format="channels_last")(inputs)
	conv1 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv1)
	conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool1)
	conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv2)

	conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool2)
	conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv3)

	conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool3)
	conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv4)

	conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool4)
	conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv5)

	up_conv5 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv5)
	ch, cw = get_crop_shape(conv4, up_conv5)
	crop_conv4 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv4)
	up6   = concatenate([up_conv5, crop_conv4], axis=concat_axis)
	conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(up6)
	conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv6)

	up_conv6 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv6)
	ch, cw = get_crop_shape(conv3, up_conv6)
	crop_conv3 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv3)
	up7   = concatenate([up_conv6, crop_conv3], axis=concat_axis)
	conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(up7)
	conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv7)

	up_conv7 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv7)
	ch, cw = get_crop_shape(conv2, up_conv7)
	crop_conv2 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv2)
	up8   = concatenate([up_conv7, crop_conv2], axis=concat_axis)
	conv8 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(up8)
	conv8 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv8)

	up_conv8 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv8)
	ch, cw = get_crop_shape(conv1, up_conv8)
	crop_conv1 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv1)
	up9   = concatenate([up_conv8, crop_conv1], axis=concat_axis)
	conv9 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(up9)
	conv9 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv9)

	ch, cw = get_crop_shape(inputs, conv9)
	conv9  = ZeroPadding2D(padding=(ch[0],cw[0]), data_format="channels_last")(conv9)
	conv10 = Conv2D(1, (1, 1), data_format="channels_last", activation="sigmoid")(conv9)

	model = Model(inputs=inputs, outputs=conv10)

	return model

model = get_unet(channels, img_size, img_size)
tensorboard.set_model(model)

epochs_arr  = [   20,      5,      5]
learn_rates = [0.001, 0.0003, 0.0001]

tensorboard.on_train_begin()
for learn_rate, epochs in zip(learn_rates, epochs_arr):
	tensorboard.on_epoch_begin(epochs)
	if os.path.isfile(weights_path):
		print("loading existing weight for training")
		model.load_weights(weights_path)

	opt  = optimizers.Adam(lr=learn_rate)
	model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
								optimizer=opt,
								metrics=['accuracy'])
	callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=1),
							 ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=2), tensorboard]

	model.fit(x = X_train, y= Y_train, validation_data=(X_val, Y_val),
		batch_size=batch_size, verbose=2, epochs=epochs, callbacks=callbacks, shuffle=True)
	tensorboard.on_epoch_end(epochs)

tensorboard.on_train_end()

if os.path.isfile(weights_path):
		model.load_weights(weights_path)

p_val = model.predict(X_val, batch_size = batch_size, verbose=1)
p_test = model.predict(x_test, batch_size = batch_size, verbose=1)
print(fbeta_score(Y_val, np.array(p_val) > 0.2, beta=2, average='samples'))
