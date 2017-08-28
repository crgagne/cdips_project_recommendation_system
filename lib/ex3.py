import numpy as np
from vgg19 import VGG19
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.models import Model

base_model = VGG19(include_top=False, weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('block5_pool').output)

#img_path = '1360x.jpeg'
img_path = 'img/euro/EUROMODA-U125256-39-5.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block4_pool_features = model.predict(x)

import itertools
flattened_list  = list(itertools.chain(*block4_pool_features.tolist()[0][0]))

for item in flattened_list: print item
