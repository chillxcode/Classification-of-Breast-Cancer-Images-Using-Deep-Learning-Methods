import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.nasnet import NASNetLarge
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.densenet import DenseNet201
from keras.applications.inception_v3 import InceptionV3
from keras.utils.np_utils import to_categorical
from keras.applications.resnet_v2 import ResNet152V2
import math


#Save BACH Dataset
def save_bach_bottleneck(glob_variables):
    model = ResNet152V2(include_top=False, weights='imagenet')

    # datagen = ImageDataGenerator(rescale=1. / 255, 
    #                              rotation_range=90)
    datagen = ImageDataGenerator(rescale=1. / 255)
  
    generator = datagen.flow_from_directory(
        glob_variables['bach_patch_images_data_dir'] + str(glob_variables['img_width']),
        target_size=(glob_variables['img_width'],
        glob_variables['img_height']),
        batch_size=glob_variables['batch_size'],
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)
    print("nb_validation_samples", nb_validation_samples)
  
    predict_size_validation = int(math.ceil(nb_validation_samples / glob_variables['batch_size']))
    print("predict_size_validation", predict_size_validation)
  
    bottleneck_features_validation = model.predict(generator, 
                                                 steps = predict_size_validation)
  
    np.save('bach/bottleneck_features_train_bach_' + glob_variables['model'] +'_' + str(glob_variables['img_width']) +'.npy',
            bottleneck_features_validation)

  
def load_bach_bottleneck(glob_variables):  
    print('bach/bottleneck_features_train_bach_' + glob_variables['model'] +'_' + str(glob_variables['img_width']) +'.npy')
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    num_classes = 4
    generator_top = datagen_top.flow_from_directory(
        glob_variables['bach_patch_images_data_dir'] + str(glob_variables['img_width']),
        target_size=(glob_variables['img_width'], glob_variables['img_height']),
        batch_size=glob_variables['batch_size'],
        class_mode=None,
        shuffle=False)

#  nb_validation_samples = len(generator_top.filenames)

    validation_data = np.load('bach/bottleneck_features_train_bach_' + glob_variables['model'] +'_' + str(glob_variables['img_width']) +'.npy')

    validation_labels = generator_top.classes
    # print(validation_labels.shape)
    validation_labels = to_categorical(validation_labels, 
                                     num_classes=num_classes)
  
    return validation_data,validation_labels



#Save Bioimaging Dataset
def save_bioimaging_bottleneck(glob_variables):
    model = ResNet152V2(include_top=False, weights='imagenet')
    
    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(glob_variables['bioimaging_patch_images_data_dir'] + str(glob_variables['img_width']),
                                            target_size=(glob_variables['img_width'], glob_variables['img_height']),
                                            batch_size=glob_variables['batch_size'],
                                            class_mode=None,
                                            shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(
        math.ceil(nb_validation_samples / glob_variables['batch_size']))
    
    bottleneck_features_validation = model.predict(
        generator, predict_size_validation)
    
    np.save('bioimaging/bottleneck_features_validation_bioimaging_' + glob_variables['model'] + '_' + str(glob_variables['img_width']) + '.npy',
            bottleneck_features_validation)
    
    
#Load Bioimaging Dataset

def load_bioimaging_bottleneck(glob_variables):
    print('bioimaging/bottleneck_features_validation_bioimaging_' + glob_variables['model'] + '_' + str(glob_variables['img_width']) + '.npy')
    datagen = ImageDataGenerator(rescale=1. / 255)
    num_classes = 4
    generator_top = datagen.flow_from_directory(
        glob_variables['bioimaging_patch_images_data_dir'] + str(glob_variables['img_width']),                    
        target_size=(glob_variables['img_width'], glob_variables['img_height']),
        batch_size=glob_variables['batch_size'],
        class_mode=None,
        shuffle=False)

#    nb_validation_samples = len(generator_top.filenames)

    validation_data = np.load('bioimaging/bottleneck_features_validation_bioimaging_' + glob_variables['model'] + '_' + str(glob_variables['img_width']) + '.npy')

    validation_labels = generator_top.classes
    validation_labels = to_categorical(
        validation_labels, num_classes=num_classes)

    return validation_data,validation_labels


def get_prediction_bottleneck(glob_variables):
    model = None
    
    if glob_variables['model'] == 'Xception':
        model = Xception(include_top=False, weights='imagenet')
        
    elif glob_variables['model'] == 'NASNetLarge':
        model = NASNetLarge(include_top=False, weights='imagenet')
        
    elif glob_variables['model'] == 'DenseNet201':
        model = DenseNet201(include_top=False, weights='imagenet')
        
    elif glob_variables['model'] == 'VGG16':
        model = VGG16(include_top=False, weights='imagenet')
        
    elif glob_variables['model'] == 'VGG19':
        model = VGG19(include_top=False, weights='imagenet')
        
    elif glob_variables['model'] == 'InceptionV3':
        model = InceptionV3(include_top=False, weights='imagenet')
        
    elif glob_variables['model'] == 'InceptionResNetV2':
        model = InceptionResNetV2(include_top=False, weights='imagenet')
        
    elif glob_variables['model'] == 'ResNet152V2':
        model = ResNet152V2(include_top=False, weights='imagenet')

    
    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(glob_variables['predict_patch_images_data_dir'],
                                            target_size=(glob_variables['img_width'], glob_variables['img_height']),
                                            batch_size=glob_variables['batch_size'],
                                            class_mode=None,
                                            shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(
        math.ceil(nb_validation_samples / glob_variables['batch_size']))
    
    bottleneck_features_validation = model.predict(
        generator, predict_size_validation)
    
    return bottleneck_features_validation
    
    # np.save('bioimaging/bottleneck_features_validation_bioimaging_' + glob_variables['model'] + '_' + str(glob_variables['img_width']) + '.npy',
            # bottleneck_features_validation)
    
def predict_single(glob_variables):
    model = None
    
    if glob_variables['model'] == 'Xception':
        model = Xception(include_top=False, weights='imagenet')
        
    elif glob_variables['model'] == 'NASNetLarge':
        model = NASNetLarge(include_top=False, weights='imagenet')
        
    elif glob_variables['model'] == 'DenseNet201':
        model = DenseNet201(include_top=False, weights='imagenet')
        
    elif glob_variables['model'] == 'VGG16':
        model = VGG16(include_top=False, weights='imagenet')
        
    elif glob_variables['model'] == 'VGG19':
        model = VGG19(include_top=False, weights='imagenet')
        
    elif glob_variables['model'] == 'InceptionV3':
        model = InceptionV3(include_top=False, weights='imagenet')
        
    elif glob_variables['model'] == 'InceptionResNetV2':
        model = InceptionResNetV2(include_top=False, weights='imagenet')
        
    elif glob_variables['model'] == 'ResNet152V2':
        model = ResNet152V2(include_top=False, weights='imagenet')


    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory('predict_single/',
                                            target_size=(glob_variables['img_width'], glob_variables['img_height']),
                                            batch_size=glob_variables['batch_size'],
                                            class_mode=None,
                                            shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(
        math.ceil(nb_validation_samples / glob_variables['batch_size']))
    
    bottleneck_features_validation = model.predict(
        generator, predict_size_validation)
    
    return bottleneck_features_validation

# python -m PyQt5.uic.pyuic -x final.ui -o final.py








