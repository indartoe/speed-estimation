import cv2
import os, time, sys, shutil
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
# os.add_dll_directory("C:/Program Files/cudnn-11.2-windows-x64-v8.1.1.33/cuda/bin")
import numpy as np
from frames_to_opticalFlow import convertToOptical #classic optical flow
# from frames_to_raftOpticalFlow import convertToOptical

import tensorflow as tf

from skimage import exposure
import math
import torch
import ffmpeg

# import matplotlib.pyplot as plt
# from PIL import Image

PATH_DATA_FOLDER = './data/'

PATH_TRAIN_LABEL_PREPROCESSED = PATH_DATA_FOLDER +  'train_preprocessed.txt'

PATH_TRAIN_LABEL = PATH_DATA_FOLDER +  'train.txt'
PATH_TRAIN_VIDEO = PATH_DATA_FOLDER + 'train.mp4'
PATH_TRAIN_FLOW_VIDEO = PATH_DATA_FOLDER + 'flow_train.mp4'
PATH_TRAIN_IMAGES_FOLDER = PATH_DATA_FOLDER +  'train_images/'
PATH_TRAIN_IMAGES_FLOW_FOLDER = PATH_DATA_FOLDER +  'train_images_flow/'

PATH_TEST_LABEL = PATH_DATA_FOLDER +  'test.txt'

PATH_TEST_VIDEO = PATH_DATA_FOLDER + 'test.mp4'
PATH_TEST_FLOW_VIDEO = PATH_DATA_FOLDER + 'flow_test.mp4'
PATH_TEST_IMAGES_FOLDER = PATH_DATA_FOLDER +  'test_images/'
PATH_TEST_IMAGES_FLOW_FOLDER = PATH_DATA_FOLDER +  'test_images_flow/'
PATH_VIDEO_INFORMATION = 'difference_frame.txt'

PATH_FOLDER_TRAIN = 'train/'
PATH_FOLDER_TEST = 'test/'
PATH_VELOCITY = PATH_DATA_FOLDER + 'velocity/'
# PATH_TRAIN_LABEL = PATH_DATA_FOLDER +  PATH_VELOCITY + 'train.txt'
# PATH_TRAIN_VIDEO = PATH_DATA_FOLDER + PATH_FOLDER_TRAIN + 'train.mp4'

KITTI_DATASET_PATH = 'E:/Edwin-Data/Thesis/Dataset/Video Dataset/Kitti'



def gammaCorrection(src, gamma):
    invGamma = 1 / gamma
 
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
 
    return cv2.LUT(src, table)

#added 1/6/2022 for brightness
def change_brightness(image, bright_factor):
    """
    Augments the brightness of the image by multiplying the saturation by a uniform random variable
    Input: image (RGB)
    returns: image with brightness augmentation
    """
    # plt.figure()
    # plt.imshow(image)
    # cv2.imwrite('preprocessed_result/ori.jpg', image)

    """
    Gamma Correction
    """
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # compute gamma = log(mid*255)/log(mean)
    # mid = 0.5
    # mean = np.mean(gray)
    # gamma = math.log(mid*255)/math.log(mean)
    # # print(gamma)
    # image_rgb = gammaCorrection(image, 2.2)
    # img_gamma1 = np.power(image, gamma).clip(0,255).astype(np.uint8)

    #HSV
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hue, sat, val = cv2.split(hsv)

    # # compute gamma = log(mid*255)/log(mean)
    # mid = 0.5
    # mean = np.mean(val)
    # gamma = math.log(mid*255)/math.log(mean)
    # print(gamma)

    # # do gamma correction on value channel
    # val_gamma = np.power(val, gamma).clip(0,255).astype(np.uint8)

    """
    Note 6/6/2022
    Need to try implement this gaussian blur
    """
    image_rgb = cv2.GaussianBlur(image, (1,1), 0) 

    """
    Note 8/6/2022
    TODO: Implement resize (scaling 1/2 & 2)
    """
    # frame = cv2.resize(image, None, fx=0.5, fy=0.5) 

    """
    Delta = {0.2, 0.3, 0.4}
    """
    # newImage = tf.image.adjust_brightness(image, delta = bright_factor)
    # image_rgb = tf.keras.preprocessing.image.img_to_array(newImage, dtype=np.uint8)

    """
    Image contrast => LAB
    """
    # newImg = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # l_channel, a, b = cv2.split(newImg)
    # clahe = cv2.createCLAHE(clipLimit=bright_factor, tileGridSize=(8,8))
    # cl = clahe.apply(l_channel)
    # limg = cv2.merge((cl,a,b))
    # image_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # # plt.imshow(frame)
    # hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # # plt.imshow(hsv_image)
    # # cv2.imwrite('preprocessed_result/hsv.jpg', hsv_image)
    # # perform brightness augmentation only on the second channel
    # hsv_image[:,:,2] = hsv_image[:,:,2] * bright_factor
    # # plt.imshow(hsv_image)
    # # cv2.imwrite('preprocessed_result/hsv_afterbright.jpg', hsv_image)
    
    # # change back to RGB
    # image_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    # # plt.imshow(image_rgb)
    # # cv2.imwrite('preprocessed_result/image_afterpreprocess.jpg', image_rgb)
    return image_rgb

# def blurAugmentation(image):
#     rgb_planes = cv2.split(image)

#     result_planes = []
#     result_norm_planes = []
#     for plane in rgb_planes:
#         dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
#         bg_img = cv2.medianBlur(dilated_img, 21)
#         diff_img = 255 - cv2.absdiff(plane, bg_img)
#         norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#         result_planes.append(diff_img)z
#         result_norm_planes.append(norm_img)

#     result_norm = cv2.merge(result_norm_planes)
#     return result_norm
'''
FIRST: VIDEO DATASET COPY TO REPRESENTATIVE FOLDER (TRAIN/TEST)
AND THEN IT WILL BE GENERATED USING THIS CODE
'''
def preprocess_data(video_input_path, flow_video_output_path, image_folder_path, flow_image_folder_path, type, folder):

    if os.path.exists(image_folder_path):
        shutil.rmtree(image_folder_path)
    os.makedirs(image_folder_path)
    if os.path.exists(flow_image_folder_path):
        shutil.rmtree(flow_image_folder_path)
    os.makedirs(flow_image_folder_path)

    print("Converting video to optical flow folder for: ", folder)

    #get 3 datetime
    nextPath = ''
    if type == 'train':
        nextPath = PATH_FOLDER_TRAIN
    else:
        nextPath = PATH_FOLDER_TEST

    #get all folder in train/test folder
    for file in os.listdir(os.path.join(KITTI_DATASET_PATH, nextPath)):
        t1 = time.time()
        prev_frame = None
        count = 1
        if os.path.exists(os.path.join(image_folder_path, file.replace("_sync",''))):
            shutil.rmtree(os.path.join(image_folder_path, file.replace("_sync",'')))
        os.makedirs(os.path.join(image_folder_path, file.replace("_sync",'')))
        if os.path.exists(os.path.join(flow_image_folder_path, file.replace("_sync",''))):
            shutil.rmtree(os.path.join(flow_image_folder_path, file.replace("_sync",'')))
        os.makedirs(os.path.join(flow_image_folder_path, file.replace("_sync",'')))

        for fileImage in os.listdir(os.path.join(KITTI_DATASET_PATH, nextPath, file, 'image_03', 'data')):
            next_frame = cv2.imread(os.path.join(KITTI_DATASET_PATH, nextPath, file, 'image_03', 'data', fileImage))

            if prev_frame is None:
                prev_frame = next_frame
                continue
            
            
            bgr_flow = convertToOptical(prev_frame, next_frame)

            image_path_out = os.path.join(image_folder_path, file.replace("_sync",''), str(count) + '.jpg')
            flow_image_path_out = os.path.join(flow_image_folder_path, file.replace("_sync",''), str(count) + '.jpg')
            cv2.imwrite(image_path_out, next_frame)
            cv2.imwrite(flow_image_path_out, bgr_flow)

            prev_frame = next_frame
            count += 1
            
        print("folder " + file + " has :" + str(count) + " file")
        t2 = time.time()
        print(' Time Taken:', (t2 - t1), 'seconds')


    return

if __name__ == '__main__':

    '''PREPROCESS DATA DOES 3 THINGS:
        1. Convert video to optical flow and save their respective images
        2. Augment image and optical flow data by Inverting them horizontally'''   ## NOW DONE IN TRAIN_MODEL.PY ITSELF IN GENERATOR DATA

    # video_reader = cv2.VideoCapture(os.path.join(PATH_DATA_FOLDER, 'train.mp4'))
    # num_frames = video_reader.get(cv2.CAP_PROP_FRAME_COUNT)
    preprocess_data(PATH_TRAIN_VIDEO, PATH_TRAIN_FLOW_VIDEO, PATH_TRAIN_IMAGES_FOLDER, PATH_TRAIN_IMAGES_FLOW_FOLDER, type='train', folder=PATH_DATA_FOLDER + PATH_FOLDER_TRAIN)
    # preprocess_data(PATH_TRAIN_VIDEO, PATH_TRAIN_FLOW_VIDEO, PATH_TRAIN_IMAGES_FOLDER, PATH_TRAIN_IMAGES_FLOW_FOLDER, type='train', folder=None)
    preprocess_data(PATH_TEST_VIDEO, PATH_TEST_FLOW_VIDEO, PATH_TEST_IMAGES_FOLDER, PATH_TEST_IMAGES_FLOW_FOLDER, type='test', folder=PATH_DATA_FOLDER + PATH_FOLDER_TEST)
    # preprocess_data(PATH_TEST_VIDEO, PATH_TEST_FLOW_VIDEO, PATH_TEST_IMAGES_FOLDER, PATH_TEST_IMAGES_FLOW_FOLDER, type='test', folder=None)
