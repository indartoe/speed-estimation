from model7 import CNNModel
import cv2, json, os
import sys
import time
import numpy as np
from frames_to_opticalFlow import convertToOptical
import matplotlib.pyplot as plt
import overpy
import time
import ffmpeg
import xlsxwriter 
import math

PATH_DATA_FOLDER = './data/'
PATH_TEST_LABEL = PATH_DATA_FOLDER +  'test.txt'
PATH_TEST_VIDEO = PATH_DATA_FOLDER + 'test.mp4'
PATH_TEST_VIDEO_OUTPUT = PATH_DATA_FOLDER + 'test_output.mp4'
PATH_COMBINED_TEST_VIDEO_OUTPUT = PATH_DATA_FOLDER + 'combined_test_output.mp4'
PATH_TEST_IMAGES_FOLDER = PATH_DATA_FOLDER +  'test_images/'
PATH_TEST_IMAGES_FLOW_FOLDER = PATH_DATA_FOLDER +  'test_images_flow/'

TYPE_FLOW_PRECOMPUTED = 0
TYPE_ORIGINAL = 1


MODEL_NAME = 'CNNModel_flow'
# MODEL_NAME = 'CNNModel_combined'

PRE_TRAINED_WEIGHTS = './best'+MODEL_NAME+'.h5'

PATH_VELOCITY = PATH_DATA_FOLDER + 'kitti_velocity/'
PATH_VIDEO_INFORMATION = 'difference_frame.txt'


def predict_from_video(video_input_path, original_video_output_path, combined_video_output_path):
    print("START::Start predicting speed from video")
    t0 = time.time()
    predicted_labels = []
    
    test_folder = PATH_DATA_FOLDER + 'test/'
    test_folder_output = PATH_DATA_FOLDER + 'test_output/'
    test_folder_output_file = PATH_DATA_FOLDER + 'test_output_file/'


    if test_folder is not None:
        '''
        This Part is normal
        '''
        # for root, dirs, files in os.walk(test_folder):
        #     # with open('data/accuracy.txt', mode="w") as outAccuracy:
        #     book = xlsxwriter.Workbook(PATH_DATA_FOLDER + 'test_accuracy.xlsx')     
        #     for file in files:
        #         predicted_labels = []
        #         video_reader = cv2.VideoCapture(test_folder + file)
        #         num_frames = video_reader.get(cv2.CAP_PROP_FRAME_COUNT)
        #         frame_size = (int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        #         fps = int(video_reader.get(cv2.CAP_PROP_FPS))

        #         fourcc = 0x00000021
        #         video_writer = cv2.VideoWriter(test_folder_output + file, fourcc, fps, frame_size)
        #         video_writer_combined = cv2.VideoWriter(combined_video_output_path, fourcc, fps, frame_size)

        #         t1 = time.time()
        #         ret, prev_frame = video_reader.read()
        #         hsv = np.zeros_like(prev_frame)

        #         video_writer.write(prev_frame)

        #         predicted_labels.append(0.0)

        #         flow_image_bgr_prev1 =  np.zeros_like(prev_frame)
        #         flow_image_bgr_prev2 =  np.zeros_like(prev_frame)
        #         flow_image_bgr_prev3 =  np.zeros_like(prev_frame)
        #         flow_image_bgr_prev4 =  np.zeros_like(prev_frame)

        #         ##-------Get Max speed limit----------------###
        #         api = overpy.Overpass(retry_timeout=10)
        #         ##-------Get Max speed limit----------------###
        #         maxspeed = 0
        #         # for file in os.listdir('data/train'):
        #         #     if(file == '0a3bb2d8-c195d91e.json'):
        #         #         full_filename = "%s/%s" % ('data/train', file)
        #         #         file_found = False
        #         #         with open(full_filename,'r') as fi:
        #         #             jsonfile = json.load(fi)
        #         #             for i in jsonfile['locations']:
        #         #                 result = api.query("""way[maxspeed](around:25.0,""" + str(i.get('latitude')) + """,""" + str(i.get('longitude')) + """);(._;>;);out body;""")
        #         #                 time.sleep(5.0)
        #         #                 if(len(result.ways) > 0):
        #         #                     if (result.ways[0] is not None):
        #         #                         if(result.ways[0].tags.get("maxspeed","n/a") != "n/a"):
        #         #                             result_maxspeed = result.ways[0].tags.get("maxspeed","0")
        #         #                             maxspeed = int(result_maxspeed.split()[0]) * 0.44704
        #         #                             file_found = True
        #         #                             break
        #         #             if(file_found == True):
        #         #                 break
        #             ###-------End of speedlimit-------###
                
        #         font                   = cv2.FONT_HERSHEY_SIMPLEX
        #         place = (50,50)
        #         fontScale              = 1
        #         fontColor              = (255,255,255)
        #         lineType               = 2

        #         count =0
        #         while True:
        #             ret, next_frame = video_reader.read()
        #             if ret is False:
        #                 break

        #             flow_image_bgr_next = convertToOptical(prev_frame, next_frame)
        #             flow_image_bgr = (flow_image_bgr_prev1 + flow_image_bgr_prev2 +flow_image_bgr_prev3 +flow_image_bgr_prev4 + flow_image_bgr_next)/4

        #             curr_image = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)

        #             combined_image_save = 0.1*curr_image + flow_image_bgr

        #             #CHOOSE IF WE WANT TO TEST WITH ONLY OPTICAL FLOW OR A COMBINATION OF VIDEO AND OPTICAL FLOW
        #             combined_image = flow_image_bgr
        #             # combined_image = combined_image_save

        #             combined_image_test = cv2.normalize(combined_image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        #             # plt.imshow(combined_image)
        #             # plt.show()

        #             #CHOOSE IF WE WANT TO TEST WITH ONLY OPTICAL FLOW OR A COMBINATION OF VIDEO AND OPTICAL FLOW
        #             # combined_image_test = cv2.resize(combined_image, (0,0), fx=0.5, fy=0.5)
        #             # combined_image_test = cv2.resize(combined_image_test, (640, 480))
        #             combined_image_test = cv2.resize(combined_image_test, (320, 240), fx=0.5, fy=0.5)

        #             combined_image_test = combined_image_test.reshape(1, combined_image_test.shape[0], combined_image_test.shape[1], combined_image_test.shape[2])

        #             prediction = model.predict(combined_image_test)

                    

        #             predicted_labels.append(prediction[0][0])

        #             # print(combined_image.shape, np.mean(flow_image_bgr), prediction[0][0])
        #             fontColor              = (255,255,255)
        #             ### Check whether the speed is above speed limit
        #             if(maxspeed > 0):
        #                 if(prediction[0][0] > maxspeed):
        #                     fontColor = (136,8,8)
        #                     print("Exceed max speed")

        #             cv2.putText(next_frame, str(prediction[0][0]), place, font, fontScale,fontColor,lineType)
        #             cv2.putText(combined_image_save, str(prediction[0][0]), place, font, fontScale,fontColor,lineType)

        #             video_writer.write(next_frame)
        #             video_writer_combined.write(combined_image_save.astype('uint8'))

        #             prev_frame = next_frame
        #             flow_image_bgr_prev4 = flow_image_bgr_prev3
        #             flow_image_bgr_prev3 = flow_image_bgr_prev2
        #             flow_image_bgr_prev2 = flow_image_bgr_prev1
        #             flow_image_bgr_prev1 = flow_image_bgr_next

        #             count +=1
        #             sys.stdout.write('\rprocessed frames: %d of %d' % (count, num_frames))


        #         t2 = time.time()
        #         video_reader.release()
        #         video_writer.release()
        #         video_writer_combined.release()
        #         print(' Prediction completed !')
        #         print(' Time Taken:', (t2 - t1), 'seconds')

        #         predicted_labels[0] = predicted_labels[1]

        #         #write speed in file
        #         WriteSpeed(test_folder_output_file + file.split(".")[0] + '.txt', predicted_labels)

        #         #calculate the accuracy of the prediction
        #         #Get how many frames need to be removed first to syncronize the calculation
        #         framesRemoved = 0
        #         with open(os.path.join(PATH_DATA_FOLDER, PATH_VIDEO_INFORMATION),'r') as fi:
        #             differenceFrameLines = fi.readlines()
        #             for line in differenceFrameLines:
        #                 if line.split(" ")[0] == file.split(".")[0]:
        #                     framesRemoved = abs(int(line.split(" ")[2]))
        #                     break

        #         #open the velocity file
        #         testPredictedVelocity = predicted_labels[framesRemoved:]
        #         cursor = 0
        #         arrAcuuracy = []
        #         arrErrors = []
        #         arrSquaredErrors = []
        #         # if not os.path.exists(os.path.join(PATH_VELOCITY, file.split(".")[0] + ".txt")):
        #         if not os.path.exists(os.path.join(PATH_VELOCITY, file.split(".")[0] + ".txt")):
        #             continue

        #         sheet = book.add_worksheet(file.split(".")[0])   
        #         sheet.write(0, 0, "Ground Truth")
        #         sheet.write(0, 1, "Predicted")
        #         sheet.write(0, 2, "Error")
        #         sheet.write(0, 3, "Accuracy")
        #         row = 1
        #         with open(os.path.join(PATH_VELOCITY, file.split(".")[0] + ".txt"),'r') as fiVelocity:
        #             groundTruthVelocities = fiVelocity.read().split()
        #             for groundTruthVelocity in groundTruthVelocities:
        #                 predictedVelocity = testPredictedVelocity[cursor]
        #                 cursor += 1 

        #                 accuracy = 0 
        #                 if float(groundTruthVelocity) > 0:
        #                     accuracy = 1 - (abs(float(groundTruthVelocity) - predictedVelocity)/ float(groundTruthVelocity))

        #                 if accuracy < 0:
        #                     accuracy = 0
        #                 sheet.write(row, 0, groundTruthVelocity)
        #                 sheet.write(row, 1, predictedVelocity)
        #                 sheet.write(row, 2, abs(float(groundTruthVelocity) - predictedVelocity))
        #                 sheet.write(row, 3, accuracy)
        #                 row += 1

        #                 arrAcuuracy.append(accuracy)
        #                 arrErrors.append(abs(float(groundTruthVelocity) - predictedVelocity))
        #                 arrSquaredErrors.append(pow(float(groundTruthVelocity) - predictedVelocity, 2))

        #         finalAccuracy = sum(arrAcuuracy)/len(arrAcuuracy)
        #         finalError = sum(arrErrors) / len(arrErrors)
                
        #         sheet.write(row, 2,"Average Accuracy")
        #         sheet.write(row, 3, finalAccuracy)
        #         sheet.write(row, 0,"mean Average Error")
        #         sheet.write(row, 1, finalError)

        #         #NEW:: Calculate MSE
        #         sheet.write(row, 5, "Mean Square Error")
        #         sheet.write(row, 6, sum(arrSquaredErrors) / len(arrSquaredErrors))
        #         sheet.write(row, 7, "Root Mean Square Error")
        #         sheet.write(row, 8, math.sqrt(sum(arrSquaredErrors) / len(arrSquaredErrors)))
                
        #         # outAccuracy.write("{0}:{1};ground truth:{2};predicted velocity:{3}\n".format(file.split(".")[0], str(finalAccuracy), str(groundTruthVelocity), str(predictedVelocity)))
        #         print("finish")
        #     book.close()
        '''
        End of normal testing part
        '''

        '''
        START::KITTI TESTING
        '''
        book = xlsxwriter.Workbook(PATH_DATA_FOLDER + 'test_accuracy.xlsx')  
            
        count =0
        prev_frame = None
        flow_image_bgr_prev1 =  None
        flow_image_bgr_prev2 =  None
        flow_image_bgr_prev3 =  None
        flow_image_bgr_prev4 =  None
        for dir in os.listdir(os.path.join(PATH_TEST_IMAGES_FOLDER)):
            t1 = time.time()
            predicted_labels = []
            for testFile in os.listdir(os.path.join(PATH_TEST_IMAGES_FOLDER, dir)):
            # for file in files:
                
                next_frame = cv2.imread(os.path.join(PATH_TEST_IMAGES_FOLDER, dir, testFile))
                if np.any(prev_frame == None):
                    prev_frame = next_frame
                    hsv = np.zeros_like(prev_frame)

                    predicted_labels.append(0.0)

                    flow_image_bgr_prev1 =  np.zeros_like(prev_frame)
                    flow_image_bgr_prev2 =  np.zeros_like(prev_frame)
                    flow_image_bgr_prev3 =  np.zeros_like(prev_frame)
                    flow_image_bgr_prev4 =  np.zeros_like(prev_frame)
                    continue
                
                

                '''
                Get Speed limit
                ##-------Get Max speed limit----------------###
                api = overpy.Overpass(retry_timeout=10)
                ##-------Get Max speed limit----------------###
                maxspeed = 0
                # for file in os.listdir('data/train'):
                #     if(file == '0a3bb2d8-c195d91e.json'):
                #         full_filename = "%s/%s" % ('data/train', file)
                #         file_found = False
                #         with open(full_filename,'r') as fi:
                #             jsonfile = json.load(fi)
                #             for i in jsonfile['locations']:
                #                 result = api.query("""way[maxspeed](around:25.0,""" + str(i.get('latitude')) + """,""" + str(i.get('longitude')) + """);(._;>;);out body;""")
                #                 time.sleep(5.0)
                #                 if(len(result.ways) > 0):
                #                     if (result.ways[0] is not None):
                #                         if(result.ways[0].tags.get("maxspeed","n/a") != "n/a"):
                #                             result_maxspeed = result.ways[0].tags.get("maxspeed","0")
                #                             maxspeed = int(result_maxspeed.split()[0]) * 0.44704
                #                             file_found = True
                #                             break
                #             if(file_found == True):
                #                 break
                    ###-------End of speedlimit-------###
                '''
                    
                font                   = cv2.FONT_HERSHEY_SIMPLEX
                place = (50,50)
                fontScale              = 1
                fontColor              = (255,255,255)
                lineType               = 2

                flow_image_bgr_next = convertToOptical(prev_frame, next_frame)
                flow_image_bgr = (flow_image_bgr_prev1 + flow_image_bgr_prev2 +flow_image_bgr_prev3 +flow_image_bgr_prev4 + flow_image_bgr_next)/4

                curr_image = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)

                combined_image_save = 0.1*curr_image + flow_image_bgr

                #CHOOSE IF WE WANT TO TEST WITH ONLY OPTICAL FLOW OR A COMBINATION OF VIDEO AND OPTICAL FLOW
                combined_image = flow_image_bgr
                # combined_image = combined_image_save

                combined_image_test = cv2.normalize(combined_image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                # plt.imshow(combined_image)
                # plt.show()

                #CHOOSE IF WE WANT TO TEST WITH ONLY OPTICAL FLOW OR A COMBINATION OF VIDEO AND OPTICAL FLOW
                # combined_image_test = cv2.resize(combined_image, (0,0), fx=0.5, fy=0.5)
                # combined_image_test = cv2.resize(combined_image_test, (640, 480))
                combined_image_test = cv2.resize(combined_image_test, (320, 240), fx=0.5, fy=0.5)

                combined_image_test = combined_image_test.reshape(1, combined_image_test.shape[0], combined_image_test.shape[1], combined_image_test.shape[2])

                prediction = model.predict(combined_image_test)
                predicted_labels.append(prediction[0][0])

                # print(combined_image.shape, np.mean(flow_image_bgr), prediction[0][0])
                # fontColor              = (255,255,255)
                ### Check whether the speed is above speed limit
                # if(maxspeed > 0):
                #     if(prediction[0][0] > maxspeed):
                #         fontColor = (136,8,8)
                #         print("Exceed max speed")

                # cv2.putText(next_frame, str(prediction[0][0]), place, font, fontScale,fontColor,lineType)
                # cv2.putText(combined_image_save, str(prediction[0][0]), place, font, fontScale,fontColor,lineType)

                # video_writer.write(next_frame)
                # video_writer_combined.write(combined_image_save.astype('uint8'))

                prev_frame = next_frame
                flow_image_bgr_prev4 = flow_image_bgr_prev3
                flow_image_bgr_prev3 = flow_image_bgr_prev2
                flow_image_bgr_prev2 = flow_image_bgr_prev1
                flow_image_bgr_prev1 = flow_image_bgr_next

                count +=1
                # sys.stdout.write('\rprocessed frames: %d of %d' % (count, num_frames))

            t2 = time.time()
            # video_reader.release()
            # video_writer.release()
            # video_writer_combined.release()
            print(' Prediction completed !')
            print(' Time Taken:', (t2 - t1), 'seconds')

            predicted_labels[0] = predicted_labels[1]

            #write speed in file
            WriteSpeed(test_folder_output_file + dir + '.txt', predicted_labels)

            #calculate the accuracy of the prediction
            #Get how many frames need to be removed first to syncronize the calculation
            # framesRemoved = 0
            # with open(os.path.join(PATH_DATA_FOLDER, PATH_VIDEO_INFORMATION),'r') as fi:
            #     differenceFrameLines = fi.readlines()
            #     for line in differenceFrameLines:
            #         if line.split(" ")[0] == testFile.split(".")[0]:
            #             framesRemoved = abs(int(line.split(" ")[2]))
            #             break

            #open the velocity file
            # testPredictedVelocity = predicted_labels[]
            cursor = 0
            arrAcuuracy = []
            arrErrors = []
            arrSquaredErrors = []
            # if not os.path.exists(os.path.join(PATH_VELOCITY, file.split(".")[0] + ".txt")):
            if not os.path.exists(os.path.join(PATH_VELOCITY, dir + ".txt")):
                continue

            sheet = book.add_worksheet(dir)   
            sheet.write(0, 0, "Ground Truth")
            sheet.write(0, 1, "Predicted")
            sheet.write(0, 2, "Error")
            sheet.write(0, 3, "Accuracy")
            row = 1
            with open(os.path.join(PATH_VELOCITY, dir + ".txt"),'r') as fiVelocity:
                groundTruthVelocities = fiVelocity.read().split()
                for groundTruthVelocity in groundTruthVelocities:
                    if len(groundTruthVelocities) - 1 <= cursor:
                        break
                    predictedVelocity = predicted_labels[cursor]
                    cursor += 1 

                    accuracy = 0 
                    if float(groundTruthVelocity) > 0:
                        accuracy = 1 - (abs(float(groundTruthVelocity) - predictedVelocity)/ float(groundTruthVelocity))

                    if accuracy < 0:
                        accuracy = 0
                    sheet.write(row, 0, groundTruthVelocity)
                    sheet.write(row, 1, predictedVelocity)
                    sheet.write(row, 2, abs(float(groundTruthVelocity) - predictedVelocity))
                    sheet.write(row, 3, accuracy)
                    row += 1

                    arrAcuuracy.append(accuracy)
                    arrErrors.append(abs(float(groundTruthVelocity) - predictedVelocity))
                    arrSquaredErrors.append(pow(float(groundTruthVelocity) - predictedVelocity, 2))

            finalAccuracy = sum(arrAcuuracy)/len(arrAcuuracy)
            finalError = sum(arrErrors) / len(arrErrors)
            
            sheet.write(row, 2,"Average Accuracy")
            sheet.write(row, 3, finalAccuracy)
            sheet.write(row, 0,"mean Average Error")
            sheet.write(row, 1, finalError)

            #NEW:: Calculate MSE
            sheet.write(row, 5, "Mean Square Error")
            sheet.write(row, 6, sum(arrSquaredErrors) / len(arrSquaredErrors))
            sheet.write(row, 7, "Root Mean Square Error")
            sheet.write(row, 8, math.sqrt(sum(arrSquaredErrors) / len(arrSquaredErrors)))
            
            # outAccuracy.write("{0}:{1};ground truth:{2};predicted velocity:{3}\n".format(file.split(".")[0], str(finalAccuracy), str(groundTruthVelocity), str(predictedVelocity)))
        print("finish")
        book.close()
        '''
        END:: KITTI TESTING
        '''
    else:
        #Changing the fps to 20
        # stream = ffmpeg.input(video_input_path)
        # stream = stream.filter('fps', fps = 20, round = 'up')

        # stream = ffmpeg.output(stream,'tmp/temp.mp4')
        # ffmpeg.run(stream)
        # video_reader = cv2.VideoCapture('tmp/temp.mp4')
        #end of changing fps

        video_reader = cv2.VideoCapture(video_input_path)
        # video_reader.set(cv2.CAP_PROP_FPS, 20)

        num_frames = video_reader.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_size = (int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = int(video_reader.get(cv2.CAP_PROP_FPS))

        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = 0x00000021
        video_writer = cv2.VideoWriter(original_video_output_path, fourcc, fps, frame_size)
        video_writer_combined = cv2.VideoWriter(combined_video_output_path, fourcc, fps, frame_size)

        t1 = time.time()
        ret, prev_frame = video_reader.read()
        hsv = np.zeros_like(prev_frame)

        video_writer.write(prev_frame)

        predicted_labels.append(0.0)

        flow_image_bgr_prev1 =  np.zeros_like(prev_frame)
        flow_image_bgr_prev2 =  np.zeros_like(prev_frame)
        flow_image_bgr_prev3 =  np.zeros_like(prev_frame)
        flow_image_bgr_prev4 =  np.zeros_like(prev_frame)

        ##-------Get Max speed limit----------------###
        api = overpy.Overpass(retry_timeout=10)
        ##-------Get Max speed limit----------------###
        maxspeed = 0
        # for file in os.listdir('data/train'):
        #     if(file == '0a3bb2d8-c195d91e.json'):
        #         full_filename = "%s/%s" % ('data/train', file)
        #         file_found = False
        #         with open(full_filename,'r') as fi:
        #             jsonfile = json.load(fi)
        #             for i in jsonfile['locations']:
        #                 result = api.query("""way[maxspeed](around:25.0,""" + str(i.get('latitude')) + """,""" + str(i.get('longitude')) + """);(._;>;);out body;""")
        #                 time.sleep(5.0)
        #                 if(len(result.ways) > 0):
        #                     if (result.ways[0] is not None):
        #                         if(result.ways[0].tags.get("maxspeed","n/a") != "n/a"):
        #                             result_maxspeed = result.ways[0].tags.get("maxspeed","0")
        #                             maxspeed = int(result_maxspeed.split()[0]) * 0.44704
        #                             file_found = True
        #                             break
        #             if(file_found == True):
        #                 break
            ###-------End of speedlimit-------###
        
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        place = (50,50)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2

        count =0
        while True:
            ret, next_frame = video_reader.read()
            if ret is False:
                break

            flow_image_bgr_next = convertToOptical(prev_frame, next_frame)
            flow_image_bgr = (flow_image_bgr_prev1 + flow_image_bgr_prev2 +flow_image_bgr_prev3 +flow_image_bgr_prev4 + flow_image_bgr_next)/4

            curr_image = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)

            combined_image_save = 0.1*curr_image + flow_image_bgr

            #CHOOSE IF WE WANT TO TEST WITH ONLY OPTICAL FLOW OR A COMBINATION OF VIDEO AND OPTICAL FLOW
            combined_image = flow_image_bgr
            # combined_image = combined_image_save

            combined_image_test = cv2.normalize(combined_image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            # plt.imshow(combined_image)
            # plt.show()

            #CHOOSE IF WE WANT TO TEST WITH ONLY OPTICAL FLOW OR A COMBINATION OF VIDEO AND OPTICAL FLOW
            # combined_image_test = cv2.resize(combined_image, (0,0), fx=0.5, fy=0.5)
            # combined_image_test = cv2.resize(combined_image_test, (640, 480))
            combined_image_test = cv2.resize(combined_image_test, (320, 240), fx=0.5, fy=0.5)

            combined_image_test = combined_image_test.reshape(1, combined_image_test.shape[0], combined_image_test.shape[1], combined_image_test.shape[2])

            prediction = model.predict(combined_image_test)

            

            predicted_labels.append(prediction[0][0])

            # print(combined_image.shape, np.mean(flow_image_bgr), prediction[0][0])
            fontColor              = (255,255,255)
            ### Check whether the speed is above speed limit
            if(maxspeed > 0):
                if(prediction[0][0] > maxspeed):
                    fontColor = (136,8,8)
                    print("Exceed max speed")

            cv2.putText(next_frame, str(prediction[0][0]), place, font, fontScale,fontColor,lineType)
            cv2.putText(combined_image_save, str(prediction[0][0]), place, font, fontScale,fontColor,lineType)

            video_writer.write(next_frame)
            video_writer_combined.write(combined_image_save.astype('uint8'))

            prev_frame = next_frame
            flow_image_bgr_prev4 = flow_image_bgr_prev3
            flow_image_bgr_prev3 = flow_image_bgr_prev2
            flow_image_bgr_prev2 = flow_image_bgr_prev1
            flow_image_bgr_prev1 = flow_image_bgr_next

            count +=1
            sys.stdout.write('\rprocessed frames: %d of %d' % (count, num_frames))


        t2 = time.time()
        video_reader.release()
        video_writer.release()
        video_writer_combined.release()
        print(' Prediction completed !')
        print(' Time Taken:', (t2 - t1), 'seconds')

        predicted_labels[0] = predicted_labels[1]
    print(' Time Taken:', (time.time() - t0), 'seconds')
    print("END::predicting speed video")
    return predicted_labels

def WriteSpeed(testOutput, predictedLabel):
    with open(testOutput, mode="w") as outfile:
        for label in predictedLabel:
            outfile.write("%s\n" % str(label))


if __name__ == '__main__':
    start_time = time.time()
    model = CNNModel()
    model.load_weights(PRE_TRAINED_WEIGHTS)

    print('Testing model...')
    predicted_labels = predict_from_video(PATH_TEST_VIDEO,  PATH_TEST_VIDEO_OUTPUT, PATH_COMBINED_TEST_VIDEO_OUTPUT)
    print('finish predicting videos in')
    print("--- %s seconds ---" % (time.time() - start_time))

    with open(PATH_TEST_LABEL, mode="w") as outfile:
        for label in predicted_labels:
            outfile.write("%s\n" % str(label))

    print("--- %s seconds ---" % (time.time() - start_time))
