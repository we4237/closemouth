import argparse
import glob
import time
from pathlib import Path
import sys
import os
from typing import Tuple

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from numpy import random
import copy
import argparse

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path

class YoloImages:  # for inference
    def __init__(self, path, img_size=640):
        p = str(Path(path))  # os-agnostic
        p = os.path.abspath(p)  # absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        ni = len(images)

        self.img_size = img_size
        self.files = images
        self.nf = ni
        self.mode = 'image'


    def __getitem__(self, index):
        path = self.files[index]

        im0 = cv2.imread(path)  # BGR

        # Padded resize
        img = letterbox(im0, new_shape=self.img_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        orgimg = img.transpose(1, 2, 0)
        orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
        img0 = copy.deepcopy(orgimg)
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = max(self.img_size) / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
        img = letterbox(img0, new_shape=self.img_size)[0]
        # Convert from w,h,c to c,w,h
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)

        return img,im0

    def __len__(self):
        return self.nf  # number of files


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model

def detect(model, source, device):
    # Load model
    img_size = 640
    conf_thres = 0.6
    iou_thres = 0.5
    imgsz = (640, 640)


    ans = []
    # Dataloader
    dataset = YoloImages(source, img_size=imgsz)
    bs = 1  # batch_size

    dataloader = DataLoader(dataset,num_workers=1, batch_size=bs,shuffle=False,drop_last=False)
    for n,(ims,im0s) in enumerate(dataloader):
            

        # orgimg = im.transpose(1, 2, 0)
        # orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
        # img0 = copy.deepcopy(orgimg)

        # h0, w0 = orgimg.shape[:2]  # orig hw
        # r = img_size / max(h0, w0)  # resize image to img_size
        # if r != 1:  # always resize down, only resize up if training with augmentation
        #     interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        #     img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
        

        im = (ims / 255.0).to(device)

        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        # Inference
        preds = model(im)[0]
        # Apply NMS
        preds = non_max_suppression_face(preds, conf_thres, iou_thres)

        for i,det in enumerate(preds):

            # Process detections
            # detections per image
            if len(det) == 0:
                ans.append([0,img_size,0,img_size])
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s[0].shape).round()

                # Print results
                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    print('xyxy:', xyxy)
                    x1,y1,x2,y2 = xyxy
                    # 将框扩大1.2
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    new_width = width * 1.5
                    new_height = height * 1.5
                    x1 = x_center - new_width / 2
                    y1 = y_center - new_height / 2
                    x2 = x_center + new_width / 2
                    y2 = y_center + new_height / 2
                    ans.append([int(y1),int(y2),int(x1),int(x2)])
                    # color = (0, 255, 0)  # BGR color (blue in this case)
                    # thickness = 2  # Thickness of the rectangle
                    # im1 = cv2.rectangle(im0s[i].cpu().numpy(), (int(x1),int(y1)), (int(x2),int(y2)), color, thickness)
                    # cv2.imwrite('test.png', im1)

    return ans

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5face/yolov5l-face.pt', help='model.pt path(s)')
    parser.add_argument("--dataset_folder", type=str, default="/mnt/users/chenmuyin/closedmouth/dataset_0824/xiaoya_1/筱雅_1")
    args = parser.parse_args()
    
    dataset_folder = args.dataset_folder
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_model = load_model(args.weights, device)

    video_path = os.path.join(dataset_folder, 'video.mp4')
    yolo_frame_folder = os.path.join(dataset_folder,'yolo_frame')
    if not os.path.exists(yolo_frame_folder):
        os.mkdir(yolo_frame_folder)
    yolo_frame_path = os.path.join(yolo_frame_folder, 'yolo_frame.png')
    if os.path.exists(yolo_frame_path):
        os.remove(yolo_frame_path)
        # continue
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 25)
    _, frame = cap.read()
    cv2.imwrite(yolo_frame_path,frame) 

    test_folder = yolo_frame_path
    face_ranges = detect(yolo_model, test_folder , device)

    # print(face_ranges)
    # 保存人脸范围
    face_ranges = face_ranges[0]
    facerange_path = os.path.join(dataset_folder,'face_range.npy')
    np.save(facerange_path,face_ranges)

