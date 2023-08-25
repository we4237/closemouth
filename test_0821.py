# 找到闭嘴且没有声音的帧
import argparse
import glob
import os
import subprocess
import librosa
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
import torch

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
    

def get_closed_pickle(txt_path,video_path,audio_path,model_wav2vec,threshold,baseline_frame=None):
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
    count,baseline = 0,0
    for i in range(min(len(moutharea),length)):
        
        pre = post = i
        #非静音直接跳过
        if(predicts[:, i * 2] != 0 or predicts[:, i * 2 + 1] != 0):
            pass

        else:
            #当前帧静音往前遍历
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
        if moutharea[i] != 0:                   
            if i+1 == min(len(moutharea),length) or moutharea[i+1] == 0:
                sub_arr = moutharea[count:i+1] 
                nonzero_arr = sub_arr[sub_arr != 0] 
                zero_numbers = len(sub_arr) - len(nonzero_arr)
                if len(nonzero_arr) > threshold:
                    
                    if not ans:
                        if not baseline_frame: 
                            # 找到非零数中位数的索引号
                            nonzero_arr = list(nonzero_arr) 
                            sorted_values = sorted(nonzero_arr)
                            middle_index = len(nonzero_arr) // 2
                            target_index = count + zero_numbers + nonzero_arr.index(sorted_values[middle_index])
                            
                            time = target_index // (25 * 60) # 暂时固定每秒钟为25帧
                            ans[target_index] = time
                            baseline = sorted_values[middle_index]
                        else:
                            time = baseline_frame // (25 * 60)
                            ans[int(baseline_frame)] = time
                            baseline = moutharea[baseline_frame]


                    else:
                        # 找到最接近第一个baseline的目标
                        closest_index = np.argmin(np.abs(nonzero_arr - baseline))
                        # 获取最接近的数在非零数组中的唯一索引位置
                        target_index = count + zero_numbers + np.where(nonzero_arr == nonzero_arr[closest_index])[0][0]

                        time = target_index // (25 * 60) # 暂时固定每秒钟为25帧
                        ans[target_index] = time



                count = i + 1

    frame_numbers = list(ans.keys())
    frame_times = list(ans.values())
    print(f'未筛选前符合条件的帧数{frame_numbers},共{len(frame_numbers)}个')

    filtered_frames = [] 
    for minute in range(0,max(frame_times)+1):
        # 获取当前分钟的帧数列表
        frames_in_minute = [frame for m, frame in zip(frame_times,frame_numbers) if m == minute]
        
        if len(frames_in_minute) <= 4:
        # 如果帧数不超过4个，则全部记录下来
            filtered_frames.extend(frames_in_minute)
            print(f'第{minute}分钟记录了{len(frames_in_minute)}帧')
        else:
            # 选择最接近baseline大小的四个帧数
            diff = [abs(moutharea[frame] - baseline) for frame in frames_in_minute]
            sorted_frames = sorted(zip(frames_in_minute, diff), key=lambda x: x[1])
            closest_frames = [frame for frame, _ in sorted_frames[:4]]
            filtered_frames.extend(sorted(closest_frames))
            print(f'第{minute}分钟记录了4帧')

    np.savetxt(txt_path,filtered_frames, fmt='%d')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="temp")
    parser.add_argument("--dataset_folder", type=str, default="dataset_0821")
    parser.add_argument("--threshold", type=int, default="4")
    parser.add_argument("--baseline_frame", type=int, default=None)  
    args = parser.parse_args()


    cache_dir = args.cache_dir
    dataset_folder = args.dataset_folder
    threshold = args.threshold
    baseline_frame = args.baseline_frame
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # 静音检测模型
    model = Fairseq(cache_dir)

    names = glob.glob(os.path.join(dataset_folder, '*/'))


    for name in names:
        txt_path = os.path.join(name,'frame_numbers.txt')
        video_path = os.path.join(name,'video.mp4')
        audio_path = os.path.join(name,'audio.wav')
        output_path = os.path.join(name,"extended_video.mp4")
        get_closed_pickle(txt_path,video_path,audio_path,model,threshold,baseline_frame)
