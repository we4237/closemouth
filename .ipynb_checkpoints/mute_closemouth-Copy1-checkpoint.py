import argparse
import os
import shlex
import cv2
import ffmpeg
import numpy as np
import torch
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import glob
import subprocess
import json
import time

def get_video_info(video_path):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout.strip()
    info = json.loads(output)
    return info

class Fairseq:
    def __init__(self, cache_dir: str, device: str = "cuda"):
        """
        初始化Fairseq

        Args:
            cache_dir (str): 用于保存（缓存）wav2vec需要的预训练模型的目录
            device (str): 用于指定模型推理的时候所使用的设备类型。 Note:不保证相同音频在不同设备下得到的vector完全相同
        """
        self.cache_dir = cache_dir
        config = Wav2Vec2Config.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
            cache_dir=self.cache_dir,
            output_hidden_states=True,
        )
        # load model and processor
        # self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft", cache_dir = self.cache_dir)
        model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
            cache_dir=self.cache_dir,
            config=config,
        )
        self.device = device
        self.model = model.to(self.device)

    def get_emission3(self, audio_path):
        """
            从audio_path提取音频特文件的征向量
        Args:
            audio_path (str): 输入的音频

        Returns:
            提取得到的音频特征
            返回的维度为(1,X)，X是单个特征向量维度

        """
        audio_input, sample_rate = librosa.core.load(audio_path, sr=16000)
        wfsize = audio_input.shape[0]
        batch = 1
        windows = (160000 * 10) // batch  # 滑动窗口长160000
        overlap = windows // 10  # overlap为滑动窗口的1/10
        epochs = int(max(round(wfsize / (windows), 0), 1))  # 循环次数
        predicted_list = [0] * epochs
        input_values = (
            (audio_input - audio_input.mean()) / np.sqrt(audio_input.var() + 1e-7)
        )[
            np.newaxis,
        ]
        input_values = torch.from_numpy(input_values).to(self.device)
        with torch.no_grad():
            for i in range(0, epochs):
                start = max(0, i * windows - overlap)  # 音频波形起始位置
                end = (
                    i * windows + windows + overlap if i < epochs - 1 else wfsize
                )  # 音频波形终止位置
                res = self.model(input_values[:, start:end])
                logits = res.logits
                fstart = 0 if (i == 0) else (overlap * 50 // 16000)  # 取特征起始位置
                fend = (
                    logits.size()[1]
                    if (i == epochs - 1)
                    else (fstart + windows * 50 // 16000)
                )  # 取特征终止位置
                predicted_list[i] = torch.argmax(logits[:, fstart:fend], dim=-1)
        predicts = torch.cat([predict for predict in predicted_list], dim=1)
        predicts = predicts.cpu().numpy()
        return predicts
    

# 找到闭嘴且没有声音的帧
def get_closed_pickle(txt_path,video_path,audio_path,model_wav2vec,threshold,num_of_frames):
    if os.path.exists(txt_path):
        os.remove(txt_path)
    if not os.path.exists(audio_path):
        cmd = f'ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ab 350000 {audio_path}'
        subprocess.run(cmd, shell=True)
    moutharea_path = os.path.join(name, "mouth_areas.npy")

    ans = {}
    predicts  = model_wav2vec.get_emission3(audio_path)
    length = predicts.shape[1] // 2

    print(f"process {name}")
    moutharea = np.load(moutharea_path)

    for i in range(min(len(moutharea),length)):
        
        pre = post = i
        #非静音直接跳过
        if(predicts[:, i * 2] != 0 or predicts[:, i * 2 + 1] != 0):
            pass
        #当前帧静音往前遍历
        else:
            for j in range(i - 1, -1, -1):
                if(predicts[:, j * 2] == 0 and predicts[:, j * 2 + 1] == 0):
                    pre = j
                    continue
                else:
                    break
            #当前帧静音往后遍历静音帧
            for k in range(i + 1, length, 1):
                if(predicts[:, k * 2] == 0 and predicts[:, k * 2 + 1] == 0):
                    post = k
                    continue
                else:
                    break
        #判断为静音
        flag = False

        if not flag:
            if moutharea[i] != 0:                   
                if i+1 == min(len(moutharea),length) or moutharea[i+1] == 0:
                    sub_arr = moutharea[pre:post+1] 
                    nonzero_arr = sub_arr[sub_arr != 0] 

                    if len(nonzero_arr) > threshold:
                        # 找到非零数中位数的索引号
                        
                        sorted_values = sorted(nonzero_arr)
                        middle_index = len(nonzero_arr) // 2
                        
                        if len(nonzero_arr) % 2 == 0:  # 非零数个数是偶数
                            median_index = pre + sorted_values.index(sorted_values[middle_index-1])
                        else:
                            median_index = pre + sorted_values.index(sorted_values[middle_index])
                        
                        ans[median_index] = 1
    frame_numbers = list(ans.keys())
    print(frame_numbers)
    if num_of_frames < len(frame_numbers):
        frame_numbers = frame_numbers[:num_of_frames]


    np.savetxt(txt_path,frame_numbers, fmt='%d')

def save_closemouth(dataset):
    names = glob.glob(os.path.join(dataset, '*/'))
    for name in names:
        fps = 25
        video_name = name.split('/')[-2]
        txt_path = os.path.join(name,'frame_numbers.txt')
        video_path = os.path.join(name,'video.mp4')
        frame_indices = np.loadtxt(txt_path) 
        if frame_indices.size >1:
            for index in frame_indices:
                minutes = index // (60 * fps)
                seconds = (index % (60 * fps)) / fps
                # 使用OpenCV从视频中读取指定帧
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                _, frame = cap.read()
                save_path = os.path.join(name,f'{video_name}_{int(index)}frame_{int(minutes)}:{seconds:.2f}.png')
                print(f'{video_name}的第{int(index)}帧被保存')
                cv2.imwrite(save_path, frame)
                cap.release()
        elif frame_indices.size == 1:
            index = frame_indices.item()
            minutes = index // (60 * fps)
            seconds = (index % (60 * fps)) / fps
            # 使用OpenCV从视频中读取指定帧
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            _, frame = cap.read()
            save_path = os.path.join(name,f'{video_name}_{int(index)}frame_{int(minutes)}:{seconds:.2f}.png')
            print(f'{video_name}的第{int(index)}帧被保存')
            cv2.imwrite(save_path, frame)
            cap.release()

        else:
            continue



def extract_and_extend_frames_ffmepg(txt_path,video_path,output_path, extension_duration):

    name = os.path.dirname(txt_path)
    frame_indices = np.loadtxt(txt_path)         
    # 检查同名文件是否存在并删除
    if os.path.exists(output_path):
        os.remove(output_path)          
    if not os.path.exists(os.path.join(name,'cache')):
        os.makedirs(os.path.join(name,'cache'))
    video_info = get_video_info(video_path)
    # 获取视频流的比特率和编码器
    fps = 25
    bit_rate = video_info["streams"][0]["bit_rate"]
    codec_name = video_info["streams"][0]["codec_name"]
    color_range = video_info['streams'][0]['color_range']
    color_space = video_info['streams'][0]['color_space']
    color_transfer = video_info['streams'][0]['color_transfer']
    color_primaries = video_info['streams'][0]['color_primaries']
    pix_fmt = video_info['streams'][0]['pix_fmt']       
    total_frames =  video_info['streams'][0]['nb_frames']    
    # 获取音频流的比特率
    audio_bitrate = video_info['streams'][1]['bit_rate']
    audio_codec = video_info['streams'][1]['codec_name']
    ffmpeg_params = ' '.join([
        '-c:v', codec_name,
        '-c:a', audio_codec,
        '-b:v', bit_rate,
        '-b:a', audio_bitrate,
        '-pix_fmt', pix_fmt,
        '-color_range', color_range,
        '-colorspace', color_space,
        '-color_trc', color_transfer,
        '-color_primaries', color_primaries
    ])     

    # 将视频切开
    pre_frame = 0
    # 切片记录txt
    cache_txt = os.path.join(name,'slice.txt')
    if os.path.exists(cache_txt):
        os.remove(cache_txt)
    for index in frame_indices:
        #将视频cut
        cache_path = os.path.join(name,'cache',f'subset_{index}.mp4')
        cache_path2txt = os.path.join('cache',f'subset_{index}.mp4')
        if os.path.exists(cache_path):
            os.remove(cache_path)
        start = round(pre_frame/fps,4)
        end = round((index-pre_frame)/fps,4)
        command = (
            f'ffmpeg -i {video_path} -ss {start} -t {end} '
            f'{ffmpeg_params} {cache_path}'
        )
        subprocess.run(command,shell=True)
        # 打开文本文件以追加模式写入，如果文件不存在则创建
        with open(cache_txt, 'a+') as f:
            # 将cache_path写入文本文件
            f.write(f"file '{cache_path2txt}'\n")

        # 处理固定延长帧
        extension_path = os.path.join(name,'cache',f'extension_{index}.mp4')
        extension_path2txt = os.path.join('cache',f'extension_{index}.mp4')
        if os.path.exists(extension_path):
            os.remove(extension_path)    
        command =  (
            f'ffmpeg -i {video_path} -vf "select=\'eq(n,{index})\', setpts=\'PTS+{extension_duration}/TB\'" '
            f'-t {extension_duration} {extension_path}'
        )
        subprocess.run(command,shell=True)
        # 打开文本文件以追加模式写入，如果文件不存在则创建
        with open(cache_txt, 'a+') as f:
            # 将cache_path写入文本文件
            f.write(f"file '{extension_path2txt}'\n")


        pre_frame = index + 1
    pre_frame = frame_indices[-1]+1
    cache_path = os.path.join(name,'cache',f'subset_{total_frames}.mp4')
    cache_path2txt = os.path.join('cache',f'subset_{total_frames}.mp4')
    start = round(pre_frame/fps,4)
    end = round((int(total_frames)-pre_frame)/fps,4)
    # 最后一个切片
    command = (
        f'ffmpeg -i {video_path} -ss {start} -t {end} '
        f'{ffmpeg_params} {cache_path}'
    )
    subprocess.run(command,shell=True)
    # 打开文本文件以追加模式写入，如果文件不存在则创建
    with open(cache_txt, 'a+') as f:
        # 将cache_path写入文本文件
        f.write(f"file '{cache_path2txt}'\n")

    # files = sorted(os.listdir('data/2/cache'), key=lambda x: os.path.getmtime(os.path.join('data/2/cache', x)))
    # with open('data/2/slice.txt', 'a+') as f:
    #     for file in files:
    #         file_path = os.path.join('data/2/cache', file)
    #         if os.path.isfile(file_path):  # 确保是文件而不是目录
    #             f.write(f"file '{file_path}'\n")
    
    command = f'ffmpeg -f concat -safe 0 -i {cache_txt} {ffmpeg_params} {output_path}'

    subprocess.run(command, shell=True)

    # file_list = np.loadtxt(cache_txt)
    # for file_path in file_list:
    #     os.remove(file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--mask_folder", type=str, default="mask_folder")
    parser.add_argument("--cache_dir", type=str, default="temp")
    parser.add_argument("--dataset_folder", type=str, default="dataset")
    parser.add_argument("--threshold", type=int, default="5")
    parser.add_argument("--num_of_frames", type=int, default="20")
    args = parser.parse_args()

    # mask_folder = args.mask_folder
    cache_dir = args.cache_dir
    dataset_folder = args.dataset_folder
    threshold = args.threshold
    num_of_frames = args.num_of_frames
    
    # 静音检测模型
    model = Fairseq(cache_dir)

    names = glob.glob(os.path.join(dataset_folder, '*/'))

    # for name in names:
    #     txt_path = os.path.join(name,'frame_numbers.txt')
    #     video_path = os.path.join(name,'video.mp4')
    #     audio_path = os.path.join(name,'audio.wav')
    #     output_path = os.path.join(name,"extended_video.mp4")
    #     # 获取时间戳
    #     get_closed_pickle(txt_path,video_path,audio_path,model,threshold,num_of_frames)
        

        
        # 制作新视频

        # extract_and_extend_frames_ffmepg(txt_path,video_path,output_path,5)



    # 保存闭嘴帧
    save_closemouth(dataset_folder)