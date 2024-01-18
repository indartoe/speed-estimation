from model6 import CNNModel
# from model5 import Pruning
# from model4 import CNNModel

# import os, time, sys, shutil
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/bin")
# os.add_dll_directory("C:/Program Files/cudnn-windows-x86_64-8.4.0.27_cuda11.6-archive/bin")
# os.add_dll_directory("C:/Program Files/cudnn-windows-x86_64-8.4.0.27_cuda11.6-archive/lib")
import cv2  
import numpy as np
import os, sys
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
import tensorflow as tf

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout, Reshape, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras import backend

from PIL import Image
# from frames_to_opticalFlow import convertToOptical
from frames_to_opticalFlow import convertToOptical

import tempfile
import tensorflow_model_optimization as tfmot

import time
from tensorflow.keras.callbacks import TensorBoard
import visualkeras
from PIL import ImageFont

import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_contour

from scipy.stats import zscore
import seaborn as sns

PATH_DATA_FOLDER = './data/'
PATH_TRAIN_LABEL = PATH_DATA_FOLDER +  'train.txt'
PATH_VELOCITY = PATH_DATA_FOLDER + 'kitti_velocity/'
# PATH_TRAIN_LABEL = PATH_VELOCITY + '10.txt'
PATH_TRAIN_IMAGES_FOLDER = PATH_DATA_FOLDER +  'train_images/'
PATH_TRAIN_IMAGES_FLOW_FOLDER = PATH_DATA_FOLDER +  'train_images_flow/'

TYPE_FLOW_PRECOMPUTED = 0
TYPE_ORIGINAL = 1

# BATCH_SIZE = 128
BATCH_SIZE = 64
EPOCH = 200

MODEL_NAME = 'CNNModel_flow'
BEST_MODEL_NAME = 'best'+MODEL_NAME+'.h5'
# MODEL_NAME = 'CNNModel_combined'

type_ = TYPE_FLOW_PRECOMPUTED   ## optical flow pre computed
# type = TYPE_ORIGINAL

# trainFolder = PATH_TRAIN_IMAGES_FOLDER
trainFolder = None
trainFolderFlow = PATH_TRAIN_IMAGES_FLOW_FOLDER
velocityFolder = PATH_VELOCITY
testFolder = PATH_DATA_FOLDER + 'test_images/'
testFolderFlow = PATH_DATA_FOLDER + 'test_images_flow/'

suggested_param = []


def prepareData(labels_path, images_path, flow_images_path, train_images_pair_paths_param = None, train_labels_param = None, type=TYPE_FLOW_PRECOMPUTED):
    num_train_labels = 0
    train_labels = []
    train_images_pair_paths = []
    if train_images_pair_paths_param is not None and train_labels_param is not None:
        train_images_pair_paths = train_images_pair_paths_param
        train_labels = train_labels_param

    with open(labels_path) as txt_file:
        labels_string = txt_file.read().split()

        for i in range(4, len(labels_string)):
            speed = float(labels_string[i])
            train_labels.append(speed)

            if type == TYPE_FLOW_PRECOMPUTED:
                # Combine original and pre computed optical flow
                train_images_pair_paths.append( ( os.getcwd() + images_path[1:] + str(i)+ '.jpg',  os.getcwd() + flow_images_path[1:] + str(i-3) + '.jpg',   os.getcwd() + flow_images_path[1:] + str(i-2) + '.jpg',   os.getcwd() + flow_images_path[1:] + str(i-1) + '.jpg',  os.getcwd() + flow_images_path[1:] + str(i) + '.jpg') )
            else:
                # Combine 2 consecutive frames and calculate optical flow
                train_images_pair_paths.append( ( os.getcwd() + images_path[1:] + str(i-1)+ '.jpg',  os.getcwd() + images_path[1:] + str(i) + '.jpg') )

    return train_images_pair_paths, train_labels


def generatorData(samples, batch_size=32, type=TYPE_FLOW_PRECOMPUTED):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:

                combined_image = None
                flow_image_bgr = None

                if type == TYPE_FLOW_PRECOMPUTED:

                    # curr_image_path, flow_image_path = imagePath
                    # flow_image_bgr = cv2.imread(flow_image_path)
                    curr_image_path, flow_image_path1, flow_image_path2,flow_image_path3, flow_image_path4 = imagePath
                    flow_image_bgr = (cv2.imread(flow_image_path1) +cv2.imread(flow_image_path2) +cv2.imread(flow_image_path3) +cv2.imread(flow_image_path4) )/4

                    curr_image = cv2.imread(curr_image_path)
                    curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)

                else:
                    prev_image_path, curr_image_path = imagePath
                    prev_image = cv2.imread(prev_image_path)
                    curr_image = cv2.imread(curr_image_path)
                    flow_image_bgr = convertToOptical(prev_image, curr_image)
                    curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)


                # flow_image_bgr = cv2.resize(flow_image_bgr, (640, 480))
                combined_image = 0.1*curr_image + flow_image_bgr
                #CHOOSE IF WE WANT TO TEST WITH ONLY OPTICAL FLOW OR A COMBINATION OF VIDEO AND OPTICAL FLOW
                combined_image = flow_image_bgr
                combined_image = cv2.normalize(combined_image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # combined_image = cv2.resize(combined_image, (80, 80), fx=0.5, fy=0.5)
                combined_image = cv2.resize(combined_image, (256,256), fx=0.5, fy=0.5)
                combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2GRAY)
                # combined_image = cv2.resize(combined_image, (0,0), fx=0.5, fy=0.5)

                # im = Image.fromarray(combined_image)
                # plt.imshow(im)
                # plt.show()

                images.append(combined_image)
                angles.append(measurement)

                # AUGMENTING DATA
                # Flipping image, correcting measurement and  measuerement

                images.append(cv2.flip(combined_image,1))
                angles.append(measurement)

            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)
            # return (np.array(images), np.array(angles))

"""
find best Learning rate method
"""
class LRFind(Callback): 
    def __init__(self, min_lr, max_lr, n_rounds): 
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step_up = (max_lr / min_lr) ** (1 / n_rounds)
        self.lrs = []
        self.losses = []
     
    def on_train_begin(self, logs=None):
        self.weights = self.model.get_weights()
        self.model.optimizer.lr = self.min_lr

    def on_train_batch_end(self, batch, logs=None):
        self.lrs.append(self.model.optimizer.lr.numpy())
        self.losses.append(logs["loss"])
        self.model.optimizer.lr = self.model.optimizer.lr * self.step_up
        if self.model.optimizer.lr > self.max_lr:
            self.model.stop_training = True
        
    def on_train_end(self, logs=None):
        self.model.set_weights(self.weights)

suggested_param = []
def objective(trial):
    print('START::FLOW TRAINING BASED ON FOLDER')
    start_time = time.time()
    
    model, parameterSuggestion = CNNModel(trial)
    if parameterSuggestion in suggested_param:
        print("Trial Pruned:Skipped")
        raise optuna.exceptions.TrialPruned()
    suggested_param.append(parameterSuggestion)
    print("add "+ parameterSuggestion)     
    train_images_pair_paths, train_labels = [], []
    #get all directory  in that directory
    for root, dirs, files in os.walk(trainFolder):
        for dir in dirs:
            #directory name should be the same with velocity file name
            trainFolderSub = os.path.join(trainFolder, dir + "/")
            print(trainFolderSub)
            trainFolderFlowSub = os.path.join(trainFolderFlow, dir + "/")
            velocityFile = os.path.join(velocityFolder, dir + ".txt")
            print(velocityFile)
            train_images_pair_paths, train_labels =  prepareData(velocityFile, trainFolderSub, trainFolderFlowSub, train_images_pair_paths, train_labels, type=type_)
            

    samples = list(zip(train_images_pair_paths, train_labels))
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    print('Total Images: {}'.format( len(train_images_pair_paths)))
    print('Train samples: {}'.format(len(train_samples)))
    print('Validation samples: {}'.format(len(validation_samples)))

    training_generator = generatorData(train_samples, batch_size=BATCH_SIZE, type=type_)
    validation_generator = generatorData(validation_samples, batch_size=BATCH_SIZE, type=type_)

    print('Training model...')

    start_time_training = time.time()

    model.summary()
    font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is strictly prohibited!
    visualkeras.layered_view(model, to_file='output_model.png', legend=True, font=font, spacing=100)  # font is optional!

    lr_finder_steps = 400
    ##find best learning rate
    lr_find = LRFind(1e-6, 1e1, lr_finder_steps)
    callbacks = [EarlyStopping(monitor='val_loss', patience=3),
        ModelCheckpoint(filepath = BEST_MODEL_NAME, monitor='loss', save_best_only=True)]
    # callbacks = [ModelCheckpoint(filepath='tmp/weights{epoch:02d}'+ MODEL_NAME+'.h5', monitor='val_loss', save_best_only=False)]

    history_object = model.fit(training_generator, steps_per_epoch= \
            len(train_samples)//BATCH_SIZE, validation_data=validation_generator, \
            validation_steps=len(validation_samples)//BATCH_SIZE, callbacks=[callbacks], epochs=EPOCH, verbose=1)
    
    validation_loss = np.min(history_object.history['val_loss'])
    
    tf.keras.backend.clear_session()
    print("--- %s seconds ---" % (time.time() - start_time_training))
    print("temporary end")
    print("END::FLOW TRAINING BASED ON FOLDER")
    return validation_loss

if __name__ == '__main__':
    # initialize the cluster of TPU
    # tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 
    # tf.config.experimental_connect_to_cluster(tpu)
    # tf.tpu.experimental.initialize_tpu_system(tpu)
    # strategy = tf.distribute.TPUStrategy(tpu)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # #trying optuna
    start_time_total = time.time()
    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    # study = optuna.create_study(direction='minimize')
    # study.optimize(objective, n_trials= 1, n_jobs=1, show_progress_bar=True)
    
    # # save results
    # df_results = study.trials_dataframe()
    # df_results.to_pickle(PATH_DATA_FOLDER + 'df_optuna_results.pkl')
    # df_results.to_csv(PATH_DATA_FOLDER + 'df_optuna_results.csv')
    # elapsed_time_total = (time.time()-start_time_total)/60
    # print('\n\ntotal elapsed time =',elapsed_time_total,' minutes')

    if trainFolder != None:
        print('START::FLOW TRAINING BASED ON FOLDER')

        train_images_pair_paths, train_labels = [], []
        #get all directory  in that directory
        for root, dirs, files in os.walk(trainFolder):
            for dir in dirs:
                #directory name should be the same with velocity file name
                trainFolderSub = os.path.join(trainFolder, dir + "/")
                print(trainFolderSub)
                trainFolderFlowSub = os.path.join(trainFolderFlow, dir + "/")
                velocityFile = os.path.join(velocityFolder, dir + ".txt")
                print(velocityFile)
                train_images_pair_paths, train_labels =  prepareData(velocityFile, trainFolderSub, trainFolderFlowSub, train_images_pair_paths, train_labels, type=type_)
                

        samples = list(zip(train_images_pair_paths, train_labels))
        train_samples, validation_samples = train_test_split(samples, test_size=0.2, random_state=42)

        speed_z_scores = zscore(train_labels)

        trainingSpeedArr = []
        validationSpeedArr = []
        for train_sample in train_samples:
            trainingSpeedArr.append(train_sample[1])

        for validation_sample in validation_samples:
            validationSpeedArr.append(validation_sample[1])
        sns.boxplot(x=np.array(train_labels))
        plt.savefig('visual_inspection_dataset.png')
        plt.clf()

        sns.boxplot(x=np.array(trainingSpeedArr))
        plt.savefig('visual_inspection_train.png')
        plt.clf()

        sns.boxplot(x=np.array(validationSpeedArr))
        plt.savefig('visual_inspection_validation.png')
        plt.clf()
        plt.close()
        # plt.show()

        print('Total Images: {}'.format( len(train_images_pair_paths)))
        print('Train samples: {}'.format(len(train_samples)))
        print('Validation samples: {}'.format(len(validation_samples)))

        training_generator = generatorData(train_samples, batch_size=BATCH_SIZE, type=type_)
        validation_generator = generatorData(validation_samples, batch_size=BATCH_SIZE, type=type_)

        print('Training model...')

        model = CNNModel()
        start_time_training = time.time()

        model.summary()
        font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is strictly prohibited!
        visualkeras.layered_view(model, to_file='output_model.png', legend=True, font=font, spacing=100)  # font is optional!

        lr_finder_steps = 400
        ##find best learning rate
        lr_find = LRFind(1e-6, 1e1, lr_finder_steps)
        callbacks = [EarlyStopping(monitor='val_loss', patience=5),
            ModelCheckpoint(filepath = BEST_MODEL_NAME, monitor='loss', save_best_only=True)]
        # callbacks = [ModelCheckpoint(filepath='tmp/weights{epoch:02d}'+ MODEL_NAME+'.h5', monitor='val_loss', save_best_only=False)]

        history_object = model.fit(training_generator, steps_per_epoch= \
                len(train_samples)//BATCH_SIZE, validation_data=validation_generator, \
                validation_steps=len(validation_samples)//BATCH_SIZE, callbacks=[callbacks], epochs=EPOCH, verbose=1)
        
        tf.keras.backend.clear_session()
        print("--- %s seconds ---" % (time.time() - start_time_total))
        print("temporary end")
        print("END::FLOW TRAINING BASED ON FOLDER")
    else:
        print('START::FLOW TRAINING BASED ON FILE')
        train_images_pair_paths, train_labels =  prepareData(PATH_TRAIN_LABEL, PATH_TRAIN_IMAGES_FOLDER, PATH_TRAIN_IMAGES_FLOW_FOLDER, type=type_)

        samples = list(zip(train_images_pair_paths, train_labels))
        train_samples, validation_samples = train_test_split(samples, test_size=0.2)

        print('Total Images: {}'.format( len(train_images_pair_paths)))
        print('Train samples: {}'.format(len(train_samples)))
        print('Validation samples: {}'.format(len(validation_samples)))

        training_generator = generatorData(train_samples, batch_size=BATCH_SIZE, type=type_)
        validation_generator = generatorData(validation_samples, batch_size=BATCH_SIZE, type=type_)
        # traininputs, trainoutputs = generatorData(train_samples, batch_size=BATCH_SIZE, type=type_)
        # validationinputs, validationoutputs = generatorData(validation_samples, batch_size=BATCH_SIZE, type=type_)

        print('Training model...')

        model = CNNModel()
        model.summary()
        font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is strictly prohibited!
        visualkeras.layered_view(model, to_file='output_model.png', legend=True, font=font, spacing=100)  # font is optional!

        lr_finder_steps = 400
        ##find best learning rate
        lr_find = LRFind(1e-6, 1e1, lr_finder_steps)
        # callbacks = [EarlyStopping(monitor='val_loss', patience=5),
        #     ModelCheckpoint(filepath = BEST_MODEL_NAME, monitor='loss', save_best_only=True)]
        callbacks = [ModelCheckpoint(filepath='tmp/weights{epoch:02d}'+ MODEL_NAME+'.h5', monitor='val_loss', save_best_only=False)]
                # TensorBoard(log_dir='log',
                #              histogram_freq=1,
                #              write_graph=True,
                #              write_images=True,
                #              update_freq='epoch',
                #              profile_batch=2,
                #              embeddings_freq=1)]

        # history_object = model.fit(training_generator,
        #                 batch_size=BATCH_SIZE,
        #                 epochs=EPOCH,  
        #                 callbacks=callbacks,
        #                 validation_steps=len(validation_samples)//BATCH_SIZE)
        # history_object = model.fit(traininputs, trainoutputs, steps_per_epoch= \
        #                  len(train_samples)//BATCH_SIZE, validation_data=(validationinputs, validationoutputs), \
        #                  validation_steps=len(validation_samples)//BATCH_SIZE, callbacks=callbacks, epochs=EPOCH, verbose=1)

        history_object = model.fit(training_generator, steps_per_epoch= \
                        len(train_samples)//BATCH_SIZE, validation_data=validation_generator, \
                        validation_steps=len(validation_samples)//BATCH_SIZE, callbacks=[callbacks], epochs=EPOCH, verbose=1)
                        #  validation_steps=len(validation_samples)//BATCH_SIZE, callbacks=[lr_find], epochs=EPOCH, verbose=1)

        #For LR finder
        # plt.plot(lr_find.lrs, lr_find.losses)
        # plt.xscale('log')
        # plt.show()
        # plt.savefig('lr_find.png')

        # history_object = model.fit_generator(training_generator, steps_per_epoch= \
        #                  len(train_samples)//BATCH_SIZE, validation_data=validation_generator, \
        #                  validation_steps=len(validation_samples)//BATCH_SIZE, callbacks=callbacks, epochs=EPOCH, verbose=1)

    print('Training model complete...')
    # print("--- %s seconds ---" % (time.time() - start_time_training))
    print(history_object.history.keys())
    print('Loss')
    print(history_object.history['loss'])
    print('Validation Loss')
    print(history_object.history['val_loss'])

    plt.figure(figsize=[10,8])
    plt.plot(np.arange(1, len(history_object.history['loss'])+1), history_object.history['loss'],'r',linewidth=3.0)
    plt.plot(np.arange(1, len(history_object.history['val_loss'])+1), history_object.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.show()
    plt.savefig('graph.png')
    print("END::FLOW TRAINING BASED ON FILE")
