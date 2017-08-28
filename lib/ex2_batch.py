import numpy as np
from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input


def print_feature(img_path):
	#img_path = '1360x.jpeg'
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	features = model.predict(x)
	print features
	


import os

model = VGG16(weights='imagenet', include_top=False)

path = 'img/euro'

for filename in os.listdir(path):
	print filename
	img_path = path + '/' + filename
	print_feature(img_path)
