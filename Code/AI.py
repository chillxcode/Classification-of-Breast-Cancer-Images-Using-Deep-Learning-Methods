from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support


import DatasetFuncs
import CustomFuncs

# Test with K-Fold Cross-Validation
def train_bach(glob_variables):
    print("BACH TRAIN")
    validation_data,validation_labels = DatasetFuncs.load_bach_bottleneck(glob_variables)
    class_labels= ['Benign','InSitu','Invasive','Normal']
    acc_per_fold = []
    loss_per_fold = []
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    fold_no = 0
    
    acc_arr = []
    recall_arr = []
    precision_arr = []
    f1_arr = []
    
    for train, test in kf.split(validation_data,validation_labels):
        # print(validation_data[train].shape, validation_data[test].shape,validation_labels[train].shape,validation_labels[test].shape)
        model = Sequential()
        model.add(Flatten(input_shape=validation_data[train].shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(glob_variables['num_classes'], 
                        activation=glob_variables['activation']))
    
        model.compile(optimizer=glob_variables['optimizer'],
                      loss='poisson', 
                      metrics=['accuracy'])
      
        print('Training for fold {fold_no} ...',fold_no)
      
        history = model.fit(validation_data[train], 
                            validation_labels[train],
                            epochs=glob_variables['epochs'],
                            batch_size=glob_variables['batch_size'])
    
        #model.save_weights(top_model_weights_path)
    
        (eval_loss, eval_accuracy) = model.evaluate(validation_data[test], 
                                                    validation_labels[test], 
                                                    batch_size=glob_variables['batch_size'], 
                                                    verbose=1)
    
        print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
        print("[INFO] Loss: {}".format(eval_loss))
        fold_no = fold_no + 1
        acc_per_fold.append(eval_accuracy)
        loss_per_fold.append(eval_loss)
        # model = None
        
        pred = model.predict(validation_data, verbose = 0)
        y_pred = np.argmax(pred, axis = 1)
        y_test = np.argmax(validation_labels, axis = 1)
        cnf_matrix = confusion_matrix(y_test, y_pred)
        CustomFuncs.plot_confusion_matrix(cnf_matrix,classes=class_labels,
                                          normalize=False,
                                          title='Confusion matrix', 
                                          cmap=plt.cm.Blues)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average = "weighted")
        print("Recall: ", recall)
        print("Precision: ", precision)
        print("F1: ", f1)
        
        acc_arr.append(eval_accuracy)
        recall_arr.append(recall)
        precision_arr.append(precision)
        f1_arr.append(f1)
        
    print("Average Accuracy: ", CustomFuncs.Average(acc_arr))
    print("Average Recall: ", CustomFuncs.Average(recall_arr))
    print("Average Precision: ", CustomFuncs.Average(precision_arr))
    print("Average F1: ", CustomFuncs.Average(f1_arr))
    
# Test with bioimaging dataset
def test_bioimaging(glob_variables):

    print("BIOIMAING TEST")
    validation_data,validation_labels = DatasetFuncs.load_bach_bottleneck(glob_variables)
    
    class_labels= ['Benign','InSitu','Invasive','Normal']
    
    model = Sequential()
    model.add(Flatten(input_shape=validation_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(glob_variables['num_classes'], 
                    activation=glob_variables['activation']))
    
    model.compile(optimizer=glob_variables['optimizer'],
                  loss='poisson', 
                  metrics=['accuracy'])
    
    print('Training for fold {fold_no} ...')
    
    history = model.fit(validation_data, 
                        validation_labels ,
                        epochs=glob_variables['epochs'],
                        batch_size=glob_variables['batch_size'])
    
    #model.save_weights(top_model_weights_path)
    
    validation_data = None
    validation_labels = None
    bioimaging_data, bioimaging_labels = DatasetFuncs.load_bioimaging_bottleneck(glob_variables)
    
    (eval_loss, eval_accuracy) = model.evaluate(bioimaging_data, 
                                                bioimaging_labels, 
                                                batch_size=glob_variables['batch_size'], 
                                                verbose=1)
    
    pred = model.predict(bioimaging_data, verbose = 0)
    y_pred = np.argmax(pred, axis = 1)
    y_test = np.argmax(bioimaging_labels, axis = 1)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    CustomFuncs.plot_confusion_matrix(cnf_matrix,classes=class_labels,
                                      normalize=False,
                                      title='Confusion matrix', 
                                      cmap=plt.cm.Blues)
    
    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average = "weighted")
    
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("F1: ", f1)
    
    bioimaging_data = None
    bioimaging_labels = None
    
    
def predict_bioimaging(glob_variables, predict_bottleneck):
    print("BIOIMAING PREDICTION")
    validation_data,validation_labels = DatasetFuncs.load_bach_bottleneck(glob_variables)
    
    class_labels= ['Benign','InSitu','Invasive','Normal']
    
    model = Sequential()
    model.add(Flatten(input_shape=validation_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(glob_variables['num_classes'], 
                    activation=glob_variables['activation']))
    
    model.compile(optimizer=glob_variables['optimizer'],
                  loss='poisson', 
                  metrics=['accuracy'])
    
    print('Training for fold {fold_no} ...')
    
    history = model.fit(validation_data, 
                        validation_labels ,
                        epochs=glob_variables['epochs'],
                        batch_size=glob_variables['batch_size'])
    
    #model.save_weights(top_model_weights_path)
    
    validation_data = None
    validation_labels = None
    bioimaging_data, bioimaging_labels = DatasetFuncs.load_bioimaging_bottleneck(glob_variables)
    
    print(bioimaging_data.shape)
    
    (eval_loss, eval_accuracy) = model.evaluate(bioimaging_data, 
                                                bioimaging_labels, 
                                                batch_size=glob_variables['batch_size'], 
                                                verbose=1)
    
    model.save(glob_variables['top_model_weights_path'] + '/' + glob_variables['model'] + '/' + str(glob_variables['img_width']) + '/')
    
    # new_model = tf.keras.models.load_model(glob_variables['top_model_weights_path'])
    # print(new_model.summary())
    
    # pred = model.predict_classes(predict_bottleneck, batch_size=10)
    # print(pred)
    
    
    
    
    # y_pred = np.argmax(pred, axis = 1)
    # y_test = np.argmax(bioimaging_labels, axis = 1)
    # cnf_matrix = confusion_matrix(y_test, y_pred)
    # CustomFuncs.plot_confusion_matrix(cnf_matrix,classes=class_labels,
                                      # normalize=False,
                                      # title='Confusion matrix', 
                                      # cmap=plt.cm.Blues)
    
    # print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    # print("[INFO] Loss: {}".format(eval_loss))
    
    # precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average = "weighted")
    
    # print("Recall: ", recall)
    # print("Precision: ", precision)
    # print("F1: ", f1)
    
    # bioimaging_data = None
    # bioimaging_labels = None
    
    
    
    
    
    
    
    
    
    
    