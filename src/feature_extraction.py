from keras.preprocessing import image
#import cv2
import glob
import sys
import os
import numpy as np
from lib.resnet50 import ResNet50
from lib.imagenet_utils import preprocess_input
from keras.models import Model
import itertools
import pandas as pd
import numpy as np
from IPython.core.debugger import Tracer

def file_loader(folder_path, file_type,filter=None):
    '''
    Parameters:
    -----------
    folder_path: str
        path of the input folder
    file_type: str
        type of files of interest

    Return:
    -------
    file_list: list
        List of absolute path of files within input folder
    '''
    curwd = os.getcwd()
    ab_path = curwd + '/' + folder_path
    if filter is not None:
        ab_path = ab_path+filter
        folder = '{}*.{}'.format(ab_path, file_type)
    else:

        folder = '{}/*.{}'.format(ab_path, file_type)
    print(folder)
    file_list = glob.glob(folder)
    return file_list

def layer_feature_extraction(input_folder='br_img',
                             model_name='ResNet50', layer_name='avg_pool',filter=None):
    '''
    Parameters:
    -----------
    input_folder: str
        The path of input folder
    model: str
        The name of pre-trained CNN model
    layer_name: str
        The name of layer which we want to extract features from

    Returns:
    --------
    feature_list: dict
        key: The name of image file
        value: The list of extracted features of images within input folder
    '''
    if model_name == 'ResNet50':
        base_model = ResNet50(include_top=False, weights='imagenet')
    elif model_name == 'VGG16':
        base_model = VGG16(include_top=False, weights='imagenet')
        target_size
    else:
        base_model = VGG19(include_top=False, weights='imagenet')
    model = Model(input=base_model.input,
                  output=base_model.get_layer(layer_name).output)

    #feature_list = {}
    import collections
    feature_list = collections.OrderedDict()

    file_in_list = file_loader(input_folder, 'jpg',filter)
    for img_in_path in file_in_list:
        img = image.load_img(img_in_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        layer_features = model.predict(x)
        flattened_list  = list(itertools.chain(*layer_features.tolist()[0][0]))
        feature_list[img_in_path] = flattened_list
    return feature_list

def main():

    input_folder = sys.argv[1]
    model_name = sys.argv[2]
    feature_list = layer_feature_extraction(input_folder,
                                 model_name, layer_name='avg_pool')
    result = pd.DataFrame(feature_list)
    result = result.transpose()
    result.to_pickle(sys.argv[3])


if __name__ == '__main__':
    main()
