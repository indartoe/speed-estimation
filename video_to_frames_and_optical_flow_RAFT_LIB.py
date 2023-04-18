import sys
print(sys.path)
sys.path.append('raftmaster/core')
sys.path.append('raftmaster')

import torch
import cv2
import os, time
import numpy as np
from raftmaster.core.raft import RAFT
from raftmaster.core.utils import flow_viz
from raftmaster.core.utils.utils import InputPadder
from raftmaster.config import RAFTConfig

from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as T

PATH_DATA_FOLDER = './data/'
PATH_TRAIN_IMAGES_FLOW_FOLDER = PATH_DATA_FOLDER +  'train_images_flow/'
PATH_TEST_IMAGES_FLOW_FOLDER = PATH_DATA_FOLDER +  'test_images_flow/'

config = RAFTConfig(
    dropout=0,
    alternate_corr=False,
    small=False,
    mixed_precision=False
)

model = RAFT(config)
model

video_path_train = PATH_DATA_FOLDER + 'train.mp4'
video_path_test = PATH_DATA_FOLDER + 'test.mp4'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

t1 = time.time()
weights_path = 'raftmaster/raft-things.pth'

cap_train = cv2.VideoCapture(video_path_train)
cap_test = cv2.VideoCapture(video_path_test)

ckpt = torch.load(weights_path, map_location=device)
model.to(device)
model.load_state_dict(ckpt)

frames = []

while True:
    has_frame, image = cap_train.read()
    
    if has_frame:
        image = image[:, :, ::-1] # convert BGR -> RGB
        frames.append(image)
    else:
        break
frames = np.stack(frames, axis=0)

print(f'frame shape: {frames.shape}')    
# plt.imshow(frames[0])

n_vis = 3

# def viz(img1, img2, flo):
#     img1 = img1[0].permute(1,2,0).cpu().numpy()
#     img2 = img2[0].permute(1,2,0).cpu().numpy()
#     flo = flo[0].permute(1,2,0).cpu().numpy()
    
#     # map flow to rgb image
#     flo = flow_viz.flow_to_image(flo)
    
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))
#     ax1.set_title('input image1')
#     ax1.imshow(img1.astype(int))
#     ax2.set_title('input image2')
#     ax2.imshow(img2.astype(int))
#     ax3.set_title('estimated optical flow')
#     ax3.imshow(flo)
#     plt.show()

##---------------------training video
transform = T.ToPILImage()
for i in range(len(frames) - 1 ):
    image1 = torch.from_numpy(frames[i]).permute(2, 0, 1).float().to(device)
    image2 = torch.from_numpy(frames[i+1]).permute(2, 0, 1).float().to(device)
    
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)

    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    
    with torch.no_grad():
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
    
    flow_image_path_out = os.path.join(PATH_TRAIN_IMAGES_FLOW_FOLDER, str(i + 1) + '.jpg')
    flo = flow_up[0].permute(1,2,0).cpu().numpy()
    flo = flow_viz.flow_to_image(flo)
    cv2.imwrite(flow_image_path_out, flo)

t2 = time.time()
print(' Conversion completed !')
print(' Time Taken:', (t2 - t1), 'seconds')

##--------------------test video
t1 = time.time()
frames = []
while True:
    has_frame, image = cap_test.read()
    
    if has_frame:
        image = image[:, :, ::-1] # convert BGR -> RGB
        frames.append(image)
    else:
        break
frames = np.stack(frames, axis=0)

print(f'frame shape: {frames.shape}')  

for i in range(len(frames) - 1):
    image1 = torch.from_numpy(frames[i]).permute(2, 0, 1).float().to(device)
    image2 = torch.from_numpy(frames[i+1]).permute(2, 0, 1).float().to(device)
    
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)

    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    
    with torch.no_grad():
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
    
    flow_image_path_out = os.path.join(PATH_TEST_IMAGES_FLOW_FOLDER, str(i + 1) + '.jpg')
    flo = flow_up[0].permute(1,2,0).cpu().numpy()
    flo = flow_viz.flow_to_image(flo)
    cv2.imwrite(flow_image_path_out, flo)


t2 = time.time()
print(' Training completed !')
print(' Time Taken:', (t2 - t1), 'seconds')