import argparse
import os
import random
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
import tqdm

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
    
def _get_silence_frame_index(predicts, mouth_areas, start_idx, fps=25, threshold=3,
                                index_count_per_minute=4):
    ans = {}
    predicts = predicts[start_idx * 2:]
    length = predicts.shape[1] // 2

    baseline = None
    last_zero_idx = -1

    for i in range(min(len(mouth_areas), length)):
        # 声音或者画面中有一个不为0，就不是静音帧，跳过
        if (predicts[:, i * 2] != 0 or predicts[:, i * 2 + 1] != 0) or mouth_areas[i] == 0:
            last_zero_idx = i
            continue
        else:
            # 如果是连续的静音帧，就继续往后找
            if i + 1 != min(len(mouth_areas), length) and mouth_areas[i + 1] != 0:
                continue

            # 如果当前静音片段长度小于阈值，就跳过
            silent_part_start = last_zero_idx + 1 + 1
            silent_part_end = i - 1
            silent_part_length = silent_part_end - silent_part_start + 1
            
            # print('silent_part')
            # for x in range(silent_part_start,silent_part_end+1):
            #     print(f'moutharea:{mouth_areas[x]}\n')
            #     print(f'predicts:{predicts[0][x*2]},{predicts[0][x*2+1]}\n')   

            if silent_part_length < threshold:
                continue

            silent_part = list(mouth_areas[silent_part_start:silent_part_end+1])
            
            # 获取baseline，选择第一个静音片段的中位数作为baseline
            if not baseline:
                middle_index = silent_part_length // 2
                target_index = silent_part_start + silent_part.index(sorted(silent_part)[middle_index])
                baseline = mouth_areas[target_index]
            else:
                target_index = silent_part_start + silent_part.index(sorted(silent_part,key=lambda x:abs(x-baseline))[0])

            target_index_time_location = target_index // (fps * 60)
            ans[target_index] = target_index_time_location

    frame_numbers = list(ans.keys())
    frame_times = list(ans.values())
    print(f'origin_silence_index_list :{frame_numbers}, count: {len(frame_numbers)}')

    filtered_frames = []

    if len(frame_numbers) == 0:
        return filtered_frames

    for minute in range(0, max(frame_times) + 1):
        # 获取当前分钟的帧数列表
        frames_in_minute = [frame for m, frame in zip(frame_times, frame_numbers) if m == minute]

        if len(frames_in_minute) <= index_count_per_minute:
            # 如果帧数不超过4个，则全部记录下来
            filtered_frames.extend(frames_in_minute)
            print(f'第{minute}分钟记录了{len(frames_in_minute)}帧')
        else:
            # 选择最接近baseline大小的四个帧数
            diff = [abs(mouth_areas[frame] - baseline) for frame in frames_in_minute]
            sorted_frames = sorted(zip(frames_in_minute, diff), key=lambda x: x[1])
            closest_frames = [frame for frame, _ in sorted_frames[:4]]
            filtered_frames.extend(sorted(closest_frames))
            print(f'第{minute}分钟记录了4帧')
    filtered_frames = [idx + start_idx for idx in filtered_frames]
    # print(f'filtered_frames: {filtered_frames}')
    return filtered_frames

# 找到闭嘴且没有声音的帧
def get_closed_pickle(txt_path,video_path,audio_path,model_wav2vec,threshold,baseline_frame=None):
    if os.path.exists(txt_path):
        os.remove(txt_path)
    if not os.path.exists(audio_path):
        cmd = f'ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ab 350000 {audio_path}'
        subprocess.run(cmd, shell=True)
    moutharea_path = os.path.join(name, "mouth_areas.npy")
    if not os.path.exists(moutharea_path):
        print(f'{moutharea_path}还没有被创建')
        return
    ans = {}
    predicts  = model_wav2vec.get_emission3(audio_path)
    length = predicts.shape[1] // 2

    print(f"process {name}")
    moutharea = np.load(moutharea_path)

    filter_frames = _get_silence_frame_index(predicts=predicts,mouth_areas=moutharea,start_idx=0,fps=25,threshold=threshold,index_count_per_minute=4)
    np.savetxt(txt_path,filter_frames, fmt='%d')


# def get_closed_pickle(txt_path,video_path,audio_path,model_wav2vec,threshold,baseline_frame=None):
#     if os.path.exists(txt_path):
#         os.remove(txt_path)
#     if not os.path.exists(audio_path):
#         cmd = f'ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ab 350000 {audio_path}'
#         subprocess.run(cmd, shell=True)
#     moutharea_path = os.path.join(name, "mouth_areas.npy")

#     ans = {}
#     predicts  = model_wav2vec.get_emission3(audio_path)
#     length = predicts.shape[1] // 2

#     print(f"process {name}")
#     moutharea = np.load(moutharea_path)
#     count,baseline = 0,0
#     for i in range(min(len(moutharea),length)):
        
#         #非静音直接跳过
#         if(predicts[:, i * 2] != 0 or predicts[:, i * 2 + 1] != 0):
#             continue

#         # else:
#         #     #当前帧静音往前遍历
#         #     for j in range(i - 1, -1, -1):
#         #         if(predicts[:, j * 2] == 0 and predicts[:, j * 2 + 1] == 0):
#         #             pre = j
#         #             continue
#         #         else:
#         #             break
#         #     #当前帧静音往后遍历静音帧
#         #     for k in range(i + 1, length, 1):
#         #         if(predicts[:, k * 2] == 0 and predicts[:, k * 2 + 1] == 0):
#         #             post = k
#         #             continue
#         #         else:
#         #             break
#         #判断为静音
#         if moutharea[i] != 0: 
#             print(f'moutharea:{moutharea[i]}\n')
#             print(f'predicts:{predicts[0][i*2]},{predicts[0][i*2+1]}\n')                  
#             if i+1 == min(len(moutharea),length) or moutharea[i+1] == 0:
#                 sub_arr = moutharea[count:i+1] 
#                 nonzero_arr = sub_arr[sub_arr != 0] 
#                 zero_numbers = len(sub_arr) - len(nonzero_arr)
#                 print(f'nonzeroarea:{nonzero_arr}\n')
#                 if len(nonzero_arr) > threshold:
                    
#                     if not ans:
#                         if not baseline_frame: 
#                             # 找到非零数中位数的索引号
#                             nonzero_arr = list(nonzero_arr) 
#                             sorted_values = sorted(nonzero_arr)
#                             middle_index = len(nonzero_arr) // 2
#                             target_index = count + zero_numbers + nonzero_arr.index(sorted_values[middle_index])
                            
#                             time = target_index // (25 * 60) # 暂时固定每秒钟为25帧
#                             ans[target_index] = time
#                             baseline = sorted_values[middle_index]
#                         else:
#                             time = baseline_frame // (25 * 60)
#                             ans[int(baseline_frame)] = time
#                             baseline = moutharea[baseline_frame]


#                     else:
#                         # 找到最接近第一个baseline的目标
#                         closest_index = np.argmin(np.abs(nonzero_arr - baseline))
#                         # 获取最接近的数在非零数组中的唯一索引位置
#                         target_index = count + zero_numbers + np.where(nonzero_arr == nonzero_arr[closest_index])[0][0]

#                         time = target_index // (25 * 60) # 暂时固定每秒钟为25帧
#                         ans[target_index] = time



#                 count = i + 1

#     frame_numbers = list(ans.keys())
#     frame_times = list(ans.values())
#     print(f'未筛选前符合条件的帧数{frame_numbers},共{len(frame_numbers)}个')

#     filtered_frames = [] 
#     for minute in range(0,max(frame_times)+1):
#         # 获取当前分钟的帧数列表
#         frames_in_minute = [frame for m, frame in zip(frame_times,frame_numbers) if m == minute]
        
#         if len(frames_in_minute) <= 4:
#         # 如果帧数不超过4个，则全部记录下来
#             filtered_frames.extend(frames_in_minute)
#             print(f'第{minute}分钟记录了{len(frames_in_minute)}帧')
#         else:
#             # 选择最接近baseline大小的四个帧数
#             diff = [abs(moutharea[frame] - baseline) for frame in frames_in_minute]
#             sorted_frames = sorted(zip(frames_in_minute, diff), key=lambda x: x[1])
#             closest_frames = [frame for frame, _ in sorted_frames[:4]]
#             filtered_frames.extend(sorted(closest_frames))
#             print(f'第{minute}分钟记录了4帧')

#     np.savetxt(txt_path,filtered_frames, fmt='%d')

def save_closemouth(dataset):
    names = glob.glob(os.path.join(dataset, '*/'))
    for name in names:
        if not os.path.exists(os.path.join(name,'cache')):
            os.makedirs(os.path.join(name,'cache'))
        fps = 25
        video_name = name.split('/')[-2]
        txt_path = os.path.join(name,'frame_numbers.txt')
        video_path = os.path.join(name,'video.mp4')
        frame_indices = np.loadtxt(txt_path) 
        if frame_indices.size >1:
            with tqdm.tqdm(total=len(frame_indices)) as pbar:
                for index in frame_indices:
                    minutes = index // (60 * fps)
                    seconds = (index % (60 * fps)) / fps
                    # 使用OpenCV从视频中读取指定帧
                    cap = cv2.VideoCapture(video_path)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    _, frame = cap.read()
                    save_path = os.path.join(name,'cache',f'{video_name}_{int(index)}frame_{int(minutes)}m{seconds:.2f}s.png')
                    print(f'{video_name}的第{int(index)}帧被保存')
                    cv2.imwrite(save_path, frame)
                    cap.release()
                    pbar.update(1)
        elif frame_indices.size == 1:
            index = frame_indices.item()
            minutes = index // (60 * fps)
            seconds = (index % (60 * fps)) / fps
            # 使用OpenCV从视频中读取指定帧
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            _, frame = cap.read()
            save_path = os.path.join(name,'cache',f'{video_name}_{int(index)}frame_{int(minutes)}:{seconds:.2f}.png')
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
    video_bitrate = video_info["streams"][0]["bit_rate"]
    codec_name = video_info["streams"][0]["codec_name"]
    color_range = video_info['streams'][0]['color_range'] if 'color_range' in video_info['streams'][0] else 'tv'
    color_space = video_info['streams'][0]['color_space'] if 'color_space' in video_info['streams'][0] else 'bt709'
    color_transfer = video_info['streams'][0]['color_transfer'] if 'color_transfer' in video_info['streams'][0] else 'bt709'
    color_primaries = video_info['streams'][0]['color_primaries'] if 'color_primaries' in video_info['streams'][0] else 'bt709'
    pix_fmt = video_info['streams'][0]['pix_fmt']       
    total_frames =  video_info['streams'][0]['nb_frames']    
    # 获取音频流的比特率
    audio_bitrate = video_info['streams'][1]['bit_rate']
    audio_codec = video_info['streams'][1]['codec_name']
    audio_sample = video_info['streams'][1]['sample_rate']
    av_params = ' '.join([        
        '-c:v', codec_name,
        '-c:a', audio_codec,
        '-b:v', video_bitrate,
        '-b:a', audio_bitrate,
        '-ar',audio_sample     
        ])
    color_params = ' '.join([
        '-pix_fmt', pix_fmt,
        '-color_range', color_range,
        '-colorspace', color_space,
        '-color_trc', color_transfer,
        '-color_primaries', color_primaries,
    ])     

    # 将视频切开
    pre_frame = 0
    # 切片记录txt
    cache_txt = os.path.join(name,'slice.txt')
    if os.path.exists(cache_txt):
        os.remove(cache_txt)

    silence_path = os.path.join(name,'cache',f'extension.{audio_codec}')
    silence_cmd = f"ffmpeg -y -f lavfi -i anullsrc=channel_layout=stereo:sample_rate={audio_sample} -t {extension_duration} {silence_path} -loglevel quiet"
    subprocess.call(silence_cmd, shell=True)

    with tqdm.tqdm(total=len(frame_indices)) as pbar:

        for index in frame_indices:
            #将视频cut
            cache_path = os.path.join(name,'cache',f'subset_{index}.mp4')
            cache_path2txt = os.path.join('cache',f'subset_{index}.mp4')
            if os.path.exists(cache_path):
                os.remove(cache_path)
            start = round(pre_frame/fps,4)
            end = round((index-pre_frame)/fps,4)
            command = (
                f'ffmpeg -y -i {video_path} -ss {start} -t {end} '
                f'{av_params} {color_params} {cache_path} -loglevel quiet'
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

            extension_path_temp = os.path.join(name,'cache',f'extension_{index}_temp.mp4')
            command =  (
                f'ffmpeg -y -i {video_path} -vf "select=\'eq(n,{index})\', setpts=\'PTS+{extension_duration}/TB\'" '
                f'-t {extension_duration} -an {extension_path_temp} -loglevel quiet'
            )
            subprocess.run(command,shell=True)

            # 音频加到视频上
            command = f'ffmpeg -y -i {extension_path_temp} -i {silence_path} -c copy {extension_path} -loglevel quiet'
            subprocess.run(command,shell=True)
            
            # 打开文本文件以追加模式写入，如果文件不存在则创建
            with open(cache_txt, 'a+') as f:
                # 将cache_path写入文本文件
                f.write(f"file '{extension_path2txt}'\n")


            pre_frame = index + 1

            pbar.update(1)

    pre_frame = frame_indices[-1]+1
    cache_path = os.path.join(name,'cache',f'subset_{total_frames}.mp4')
    cache_path2txt = os.path.join('cache',f'subset_{total_frames}.mp4')
    start = round(pre_frame/fps,4)
    end = round((int(total_frames)-pre_frame)/fps,4)
    # 最后一个切片
    command = (
        f'ffmpeg -y -i {video_path} -ss {start} -t {end} '
        f'{av_params} {color_params} {cache_path} -loglevel quiet'
    )
    subprocess.run(command,shell=True)
    # 打开文本文件以追加模式写入，如果文件不存在则创建
    with open(cache_txt, 'a+') as f:
        # 将cache_path写入文本文件
        f.write(f"file '{cache_path2txt}'\n")

    
    command = f'ffmpeg -y -f concat -safe 0 -i {cache_txt} {av_params} {color_params} {output_path}'

    subprocess.run(command, shell=True)

    # file_list = np.loadtxt(cache_txt)
    # for file_path in file_list:
    #     os.remove(file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="temp")
    parser.add_argument("--dataset_folder", type=str, default="/mnt/users/chenmuyin/closedmouth/dataset_0824/xiaoya_1")
    parser.add_argument("--threshold", type=int, default="3")
    parser.add_argument("--baseline_frame", type=int, default=None)  
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    cache_dir = args.cache_dir
    dataset_folder = args.dataset_folder
    threshold = args.threshold
    baseline_frame = args.baseline_frame
    
    # 静音检测模型
    model = Fairseq(cache_dir)

    names = glob.glob(os.path.join(dataset_folder, '*/'))


    for name in names:
        txt_path = os.path.join(name,'frame_numbers.txt')
        video_path = os.path.join(name,'video.mp4')
        audio_path = os.path.join(name,'audio.wav')
        output_path = os.path.join(name,"extended_video.mp4")
        time1 = time.time()
        # 获取时间戳
        get_closed_pickle(txt_path,video_path,audio_path,model,threshold,baseline_frame)
        
        # 制作新视频
        # extract_and_extend_frames_ffmepg(txt_path,video_path,output_path,extension_duration=5)

    # 保存闭嘴帧
    save_closemouth(dataset_folder)
    time2 = time.time()
    elapsed_time = time2-time1
    print(f"{name}花费时间：{elapsed_time:.3f}秒")