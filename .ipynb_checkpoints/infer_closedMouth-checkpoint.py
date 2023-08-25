import argparse
import glob
from model import BiSeNet
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import time
import sys
import os
import numpy as np
from PIL import Image
from pathlib import Path
import cv2


class CustomDataset:
    def __init__(self, video_path, face_ranges):
        self.video_path = video_path
        self.face_ranges = face_ranges
        self.frames = self.read_frames()  # 存储视频的帧
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, index):
        # 从预先读取的帧列表中获取指定帧
        frame = self.frames[index]
        img = self.to_tensor(frame)
        return img

    def read_frames(self):
        # 使用OpenCV从视频中读取所有帧并存储在列表中
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 进行人脸裁剪和调整大小操作
            cropped_img = frame[face_range[0]:face_range[1], face_range[2]:face_range[3]]
            resized_img = self.resize(cropped_img,512)
            frames.append(resized_img)
        cap.release()
        return frames
    
    def resize(self,cropped_img,desired_size):
        target_height, target_width = desired_size,desired_size
        # 获取原始图像的尺寸
        height, width = cropped_img.shape[:2]
        # 计算缩放比例
        scale = min(target_width / width, target_height / height)
        # 计算调整后的尺寸
        resized_width = int(width * scale)
        resized_height = int(height * scale)
        # 缩放图像
        resized_img = cv2.resize(cropped_img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        # 计算填充尺寸
        delta_w = target_width - resized_width
        delta_h = target_height - resized_height
        top = delta_h // 2
        bottom = delta_h - top
        left = delta_w // 2
        right = delta_w - left
        # 使用cv2.copyMakeBorder()函数进行填充
        resized_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return resized_img    


def improve_infer(video_path,npyout,face_range,face_net,batch_size):   
    dataset = CustomDataset(video_path, face_range)
    time_ = time.time()
    time_ = time_ - start_time
    print(f"dataset存进内存操作执行时间：{time_} 秒")
    dataloader = DataLoader(dataset,num_workers=8, batch_size=batch_size,shuffle=False,drop_last=False)
    mouthareas = []
    teeth = 11
    time_list = []
    for i,batch in enumerate(dataloader):

        # time_list.append(time.time())
        # if i > 0 :
        #     print(time_list[i] - time_list[i-1])

        with torch.no_grad():
            img = batch.cuda()
            out = face_net(img)[0].cpu().numpy()
            parsing = out.argmax(1)
            for idx,pair in enumerate(parsing):
                itern = i * batch_size + idx
                unique = np.unique(pair)
                print(f'{unique}/{itern}')
                # 统计嘴巴区域像素数量
                if teeth not in unique:
                    vis_parsing_anno = pair.copy().astype(np.uint8)
                    face_coord = np.argwhere(np.logical_or(vis_parsing_anno == 12, vis_parsing_anno == 13))
                    mouth_area = len(face_coord)
                    mouthareas.append(mouth_area)
                else:
                    mouthareas.append(0)


    mouthareas = np.array(mouthareas)
    np.save(npy_path, mouthareas)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default="dataset")
    # parser.add_argument("--video_folder", type=str, default="wav/13.mp4")
    # parser.add_argument("--picture_path", type=str, default="closedMouth_pre/0519/2")
    # parser.add_argument("--png_path", type=str, default="closedMouth_pre/0519/2_res")
    # parser.add_argument("--npy_folder", type=str, default="npy/13.mp4")
    parser.add_argument("--cp", type=str, default="cp/79999_iter.pth")
    parser.add_argument("--batch_size", type=int, default="32") 
    args = parser.parse_args()

    # video_folder = args.video_folder
    # picture_path = args.picture_path
    # png_path = args.png_path
    # npy_folder = args.npy_folder
    dataset = args.dataset_folder
    cp = args.cp
    batch_size = args.batch_size


    names = glob.glob(os.path.join(dataset, '*/'))
    # 加载face模型
    n_classes = 19
    face_net = BiSeNet(n_classes=n_classes)
    face_net.cuda()
    save_pth = os.path.join(cp)
    face_net.load_state_dict(torch.load(save_pth))
    face_net.eval()
    for name in names:
        video_path = os.path.join(name,'video.mp4')
        npy_path = os.path.join(name, "mouth_areas.npy")
        if os.path.exists(npy_path):
            continue
        facerange_path = os.path.join(name,'face_range.npy')
        if not os.path.exists(facerange_path):
            print('Please yolo first')
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 25)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _, frame = cap.read()       
            face_range = [0, frame_height, 0, frame_width]
            cap.release()       
        else:
            face_range = np.load(facerange_path)

                    
        # 记录开始时间
        start_time = time.time()

        improve_infer(video_path,npy_path,face_range,face_net,batch_size)

        # 记录结束时间并计算时间差
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"{name}花费时间：{elapsed_time:.3f} 秒")




            