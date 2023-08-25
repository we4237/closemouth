import numpy as np

b = np.load('yongpeng.npy')
a = np.load('/mnt/users/chenmuyin/closedmouth/dataset_0822_1/1/mouth_areas.npy')
print(a.max())

# frame_list = np.loadtxt('dataset_0821/1/frame_numbers.txt')
# fps = 25  # 帧率为 25
# output_file_path = "time_list.txt"  # 指定保存的文本文件路径

# # 将帧数转换为时间（分钟和秒，包括小数部分）
# time_list = [frame / fps for frame in frame_list]

# # 保存时间值到文本文件
# with open(output_file_path, 'w') as f:
#     for time in time_list:
#         minutes = int(time // 60)  # 计算分钟部分
#         seconds = time % 60  # 计算秒部分
#         formatted_time = f"{minutes} 分 {seconds:.2f} 秒"  # 格式化时间字符串
#         f.write(formatted_time + '\n')

