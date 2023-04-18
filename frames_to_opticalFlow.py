import cv2
import numpy as np

def convertToOptical(prev_image, curr_image):

    prev_image_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    curr_image_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

    prev_image_gray = cv2.equalizeHist(prev_image_gray)
    curr_image_gray = cv2.equalizeHist(curr_image_gray)

    # hsv = np.zeros(prev_image.shape)
    # # set saturation
    # hsv[:,:,1] = cv2.cvtColor(curr_image, cv2.COLOR_RGB2HSV)[:,:,1]

    flow = cv2.calcOpticalFlowFarneback(prev_image_gray, curr_image_gray, None, 0.5, 3, 10, 5, 3, 1.1, 0)
    # lk_params = dict(winSize = (21, 21),
	# 						  maxLevel = 2,
	# 						  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
    # flow = cv2.calcOpticalFlowPyrLK(prev_image_gray, curr_image_gray, None, None, lk_params)
    # flow = cv2.calcOpticalFlowFarneback(prev_image_gray, curr_image_gray, None, 0.5, 2, 15, 2, 5, 1.2, 0)

    hsv = np.zeros_like(prev_image)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_image_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return flow_image_bgr
