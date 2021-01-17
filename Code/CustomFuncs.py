import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import itertools
import glob
import random

import NormalizeStaining
import RollingWindow

#pre-processing
def myFunc(e):
  return e[0]


def get_patch_ratio(patch:np.ndarray, patch_size:int,cell_label = 0 ):
   return np.count_nonzero(patch == cell_label) / (patch_size ** 2)

def get_patch_ratio_for_image(patch:np.ndarray,cell_label = 0 ):
   return np.count_nonzero(patch == cell_label) / (1536*2048)

def find_patches_from_patch(list_tensorr,patch_size,step):
#  cell_label = 0
  csv_file = []

  for index in range(len(list_tensorr)):
    patch_ratio = get_patch_ratio(list_tensorr[index][0],patch_size)
    img_arr = []
    if ( patch_ratio>=0.35 and patch_ratio<=0.65):
      img_arr.append(patch_ratio)
      img_arr.append(list_tensorr[index][1])
      img_arr.append(list_tensorr[index][2])
     
      csv_file.append(img_arr)
      
  csv_file.sort(reverse=True, key=myFunc)
  return csv_file



#pre-processing (Custom Funcs) Patch
def patch_images(glob_variables, image_data_dir, patch_data_dir):
    class_labels= ['Benign','InSitu','Invasive','Normal']
    class_short_label = ['Bn','Is','Iv','Nr']
    patch_size = glob_variables['img_width']

    step = 8
    
    print("patch_images called for " + image_data_dir)
    for index, fold_dir in enumerate(class_labels):
        print(fold_dir)
        
        train_tiffs = []
        train_tiffs = sorted(os.listdir(image_data_dir+fold_dir+'/'))
        print(train_tiffs)

        save_image_size = 1
        for image_dir in train_tiffs:
            
            
            directory = image_data_dir + class_labels[index] + '/' + image_dir
            savedir = patch_data_dir +  class_labels[index] + '/' + class_short_label[index]
            img = cv2.imread(directory)
            print(directory)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            normal_image = NormalizeStaining.normalizeStaining(image)
            gray = cv2.cvtColor(normal_image, cv2.COLOR_BGR2GRAY)
            (thresh, blackAndWhiteImage) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            list_patched = RollingWindow.rolling_window(blackAndWhiteImage, patch_size, step, print_dims = False)
    
            list_images = find_patches_from_patch(list_patched,patch_size,step)
    
            img = np.array(image)
            if ( len( list_images ) > glob_variables['patch_image_count'] ):
              for indis in range( glob_variables['patch_image_count'] ):
                start_row = list_images[indis][1]
                start_col = list_images[indis][2]
                temp= img[start_row:start_row+patch_size ,start_col:start_col+patch_size]
                saveimagedir = savedir + str(save_image_size)
                Image.fromarray(temp).save(saveimagedir+'.png')
                save_image_size +=1
                
            else:
              for indis in range( len(list_images) ):
                start_row = list_images[indis][1]
                start_col = list_images[indis][2]
                temp= img[start_row:start_row+patch_size ,start_col:start_col+patch_size]
                saveimagedir = savedir + str(save_image_size)
                Image.fromarray(temp).save(saveimagedir+'.png')
                save_image_size +=1




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.close()
    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt.savefig("confusion_matrix.png")

def Average(lst): 
    return sum(lst) / len(lst)


def get_random_image_name_from_directory(path):
    files = os.listdir(path)
    d = random.choice(files)
    return d


def clear_predict_images_from_folder(glob_variables):
    path = glob_variables['predict_patch_images_data_dir']
    files = glob.glob(path + 'Benign')
    
    for f in files:
        os.chmod(f, 0o777)
        os.remove(f)



# conda install keras=2.0.5