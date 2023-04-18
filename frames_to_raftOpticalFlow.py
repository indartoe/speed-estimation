import torch
from torchvision.models.optical_flow import raft_large
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.utils import flow_to_image
import numpy as np
from PIL import Image

import time

def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=(520, 960)),
        ]
    )
    batch = transforms(batch)
    return batch

def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    img = F.to_pil_image(imgs[0][0])
    newImg = np.asarray(img)
    return newImg
    # index = 0
    # _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    # for row_idx, row in enumerate(imgs):
    #     for col_idx, img in enumerate(row):
    #         # ax = axs[row_idx, col_idx]
    #         img = F.to_pil_image(img.to("cpu"))
    #         return np.asarray(img)

def convertToOptical(prev_image, curr_image):
    start_time = time.time()
    prev_imageTorch = torch.from_numpy(prev_image)
    curr_imageTorch = torch.from_numpy(curr_image)
    prev_imageTorch = np.transpose(prev_imageTorch, (2,0,1))
    curr_imageTorch = np.transpose(curr_imageTorch, (2,0,1))
    imgBatchPrev = torch.stack([prev_imageTorch,prev_imageTorch])
    imgBatchNext = torch.stack([curr_imageTorch,curr_imageTorch])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    imgBatchPrev = preprocess(imgBatchPrev).to(device)
    imgBatchNext = preprocess(imgBatchNext).to(device)

    model = raft_large(pretrained=True, progress=False).to(device)
    model = model.eval()
    
    list_of_flows = model(imgBatchPrev.to(device),imgBatchNext.to(device))
    predicted_flows = list_of_flows[-1]

    flow_imgs = flow_to_image(predicted_flows)
    img1_batch = [(img1 + 1) / 2 for img1 in imgBatchPrev]
    opticalFlowImgs = [[flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
    flow_image_bgr = plot(opticalFlowImgs)
    # print("--- Execution time for converting to optical flow %s seconds ---" % (time.time() - start_time))
    return flow_image_bgr