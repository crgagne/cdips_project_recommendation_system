import numpy as np
from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions

def classify_image(img_path):
	#model = ResNet50(weights='imagenet')
	#img_path = '1360x.jpeg'
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	preds = model.predict(x)
	print('Predicted:', decode_predictions(preds))
	# print: [[u'n02504458', u'African_elephant']]



import os

model = ResNet50(weights='imagenet')

path = 'img/euro'

for filename in os.listdir(path):
	print filename
	img_path = path + '/' + filename
	classify_image(img_path)
