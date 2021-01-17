import CustomFuncs
import DatasetFuncs
import AI

#Global Variables
glob_variables = {
        'img_width' : 256,
        'img_height': 256,
        
        'bach_data_dir': 'bach/',                                   #path to training images 
        'bioimaging_data_dir': 'bioimaging/',                        #path to testing images 
        'top_model_weights_path': 'models/',
        
        'bach_patch_images_data_dir': 'train/',
        'bioimaging_patch_images_data_dir': 'test/',
        'predict_patch_images_data_dir': 'predict/',
        
        'num_classes': 4,
        'epochs': 16,
        'batch_size': 32,                                           #batch size used by flow_from_directory and predict_generator
        'patch_image_count': 20,
        
        'model': 'VGG16',
        'activation': 'softmax',
        'optimizer': 'SGD'
}

# Creating BACH patches
CustomFuncs.patch_images(glob_variables, glob_variables['bach_data_dir'], glob_variables['bach_patch_images_data_dir'])

#Saving patch images as bottleneck
DatasetFuncs.save_bach_bottleneck(glob_variables)

#Creating Bioimaging patches
CustomFuncs.patch_images(glob_variables, glob_variables['bioimaging_data_dir'], glob_variables['bioimaging_patch_images_data_dir'])

#Saving patch images as bottleneck
DatasetFuncs.save_bioimaging_bottleneck(glob_variables)




# Test with K-Fold Cross-Validation
AI.train_bach(glob_variables)

# Test with bioimaging dataset
AI.test_bioimaging(glob_variables)


# Save Model
bottleneck = CustomFuncs.predict_single(glob_variables)
AI.predict_bioimaging(glob_variables, bottleneck)






