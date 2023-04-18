import cv2
import os, time, sys, shutil
import numpy as np
from frames_to_opticalFlow import convertToOptical

import tensorflow as tf

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

def preprocess_data(video_input_path, flow_video_output_path, image_folder_path, flow_image_folder_path, type):

    if os.path.exists(image_folder_path):
        shutil.rmtree(image_folder_path)
    os.makedirs(image_folder_path)
    if os.path.exists(flow_image_folder_path):
        shutil.rmtree(flow_image_folder_path)
    os.makedirs(flow_image_folder_path)

    print("Converting video to optical flow for: ", video_input_path)

    video_reader = cv2.VideoCapture(video_input_path)

    num_frames = video_reader.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_size = (int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))

    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fourcc = 0x00000021
    video_writer = cv2.VideoWriter(flow_video_output_path, fourcc, fps, frame_size)

    t1 = time.time()
    prev_frame = cv2.imread('data/train_images_preprocessed/0.jpg')
    hsv = np.zeros_like(prev_frame)

    image_path_out = os.path.join(image_folder_path, str(0) + '.jpg')
    cv2.imwrite(image_path_out, prev_frame)

    count = 1
    while True:
        if count >= 20399:
            break

        pathNextFrame = os.path.join('data/train_images_preprocessed', str(count) + '.jpg')
        next_frame = cv2.imread(pathNextFrame)
        #Note 6/6/2022
        #     """
        #     Need to try implement this gaussian blur
        #     """
        # prevImage = cv2.GaussianBlur(prev_frame, (3,3), 0) 
        # nextImage = cv2.GaussianBlur(next_frame, (3,3), 0) 
        # prevImage = cv2.medianBlur(prev_frame, 3)
        # nextImage = cv2.medianBlur(next_frame, 3) 

        """
        Note 8/6/2022
        TODO: Implement resize (scaling 1/2 & 2)
        """
        # prevImage = cv2.resize(prev_frame, None, fx=0.5, fy=0.5) 
        # nextImage = cv2.resize(next_frame, None, fx=0.5, fy=0.5) 
        
        bgr_flow = convertToOptical(prev_frame, next_frame)

        image_path_out = os.path.join(image_folder_path, str(count) + '.jpg')
        flow_image_path_out = os.path.join(flow_image_folder_path, str(count) + '.jpg')

        cv2.imwrite(image_path_out, next_frame)
        cv2.imwrite(flow_image_path_out, bgr_flow)

        video_writer.write(bgr_flow)

        prev_frame = next_frame
        count += 1

        '''FOR FLIP PREPROCESSING, CURRENTLY HANDLED IN GENERATE DATA IN TRAIN'''
        # if type == 'train':
        #     image_flip = cv2.flip( next_frame, 1 )
        #     bgr_flow_flip = cv2.flip( bgr_flow, 1 )
        #
        #     image_path_out_flip = os.path.join(image_folder_path, str(count) + '.jpg')
        #     flow_image_path_out_flip = os.path.join(flow_image_folder_path, str(count) + '.jpg')
        #
        #     cv2.imwrite(image_path_out_flip, image_flip)
        #     cv2.imwrite(flow_image_path_out_flip, bgr_flow_flip)
        #
        #     sys.stdout.write('\rprocessed frames: %d of %d' % (count//2, num_frames))
        #     count += 1
        # else:
        #     sys.stdout.write('\rprocessed frames: %d of %d' % (count, num_frames))


    t2 = time.time()
    video_reader.release()
    video_writer.release()
    print(' Conversion completed !')
    print(' Time Taken:', (t2 - t1), 'seconds')

    '''FOR FLIP PREPROCESSING, CURRENTLY HANDLED IN GENERATE DATA IN TRAIN'''
    # if type == 'train':
    #     file_r = open(PATH_TRAIN_LABEL, 'r')
    #
    #     if os.path.exists(PATH_TRAIN_LABEL_PREPROCESSED):
    #         os.remove(PATH_TRAIN_LABEL_PREPROCESSED)
    #     file = open(PATH_TRAIN_LABEL_PREPROCESSED, 'w')
    #
    #     speed_list = file_r.read().split()
    #     for i in range(len(speed_list)-1):
    #         speed= speed_list[i]  + speed_list[i+1]/2
    #         file.write(speed+ '\n')
    #         file.write(speed + '\n')
    #
    #     file_r.close()
    #     file.close()
    #
    #     print(' New labels written !')


    return

if __name__ == '__main__':


    '''PREPROCESS DATA DOES 3 THINGS:
        1. Convert video to optical flow and save their respective images
        2. Augment image and optical flow data by Inverting them horizontally'''   ## NOW DONE IN TRAIN_MODEL.PY ITSELF IN GENERATOR DATA

    preprocess_data(PATH_TRAIN_VIDEO, PATH_TRAIN_FLOW_VIDEO, PATH_TRAIN_IMAGES_FOLDER, PATH_TRAIN_IMAGES_FLOW_FOLDER, type='train')
    preprocess_data(PATH_TEST_VIDEO, PATH_TEST_FLOW_VIDEO, PATH_TEST_IMAGES_FOLDER, PATH_TEST_IMAGES_FLOW_FOLDER, type='test')
