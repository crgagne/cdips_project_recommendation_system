import numpy as np
from vgg19 import VGG19
from resnet50 import ResNet50
from xception import Xception
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.models import Model
import itertools


def get_middle_layer(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	block4_pool_features = model.predict(x)
	flattened_list  = list(itertools.chain(*block4_pool_features.tolist()[0][0]))
	return flattened_list

def dot(K, L):
   if len(K) != len(L):
      return 0
   return sum(i[0] * i[1] for i in zip(K, L))

def similarity(item_1, item_2): 
	return dot(item_1, item_2) / np.sqrt(dot(item_1, item_1) * dot(item_2, item_2))



import os
import sys

base_model = ResNet50(include_top=False, weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
#base_model = VGG19(include_top=False, weights='imagenet')
#model = Model(input=base_model.input, output=base_model.get_layer('block5_pool').output)
#base_model = Xception(include_top=False, weights='imagenet')
#model = Model(input=base_model.input, output=base_model.get_layer('block14_sepconv2_act').output)

#path = '/Users/ninghuawang/Downloads/deep-learning-models-master/img/' + sys.argv[1]
path = sys.argv[1]

features = dict()
for filename in os.listdir(path):
	img_path = path + '/' + filename
	features[filename] = get_middle_layer(img_path)

import itertools
similarities = {item: similarity(features[item[0]], features[item[1]]) for item in itertools.product(features,features)}
for key, item in similarities.items():
	print key[0] + '|' + key[1] + '|' + str(item)
