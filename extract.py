import os
from pathlib import Path
from torch.autograd import Variable
from PIL import Image
import torch
from torchvision import transforms
from X3D import create_x3d
from I3D import i3_res50
import time
import numpy as np

import warnings
warnings.filterwarnings("ignore")

mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]

def generate(datasetpath, outputpath, pretrainedpath, frequency, batch_size, sample_mode, type):
    outputdir = os.path.join(outputpath, type) # Features\train_abnormal
    Path(outputdir).mkdir(parents = True, exist_ok = True)
    list_names = os.listdir(datasetpath)
    
    # model = create_x3d(input_clip_length = 16, input_crop_size = 256, depth_factor = 2.2) # X3D_M
    model = i3_res50(400, "i3d_r50_kinetics.pth")
    # model.load_state_dict(torch.load(pretrainedpath))
    # print("Load pretrained weight successfully!!!")
    model.cuda()
    model.train(False)

    total_clips = 0
    total_time = 0

    for name in list_names:
        if name == "01_0054":
            print("Found")
            frames_dir = os.path.join(datasetpath, name) # F:\ShanghaiTech\output_frames_train_abnormal\01_0014
            features, time_per_video, num_clips_per_video = run(model, frequency, outputdir, frames_dir, batch_size, sample_mode)
            total_clips += num_clips_per_video
            total_time += time_per_video
            # np.save(os.path.join(outputdir, name), features)
            break

        # frames_dir = os.path.join(datasetpath, name) # F:\ShanghaiTech\output_frames_train_abnormal\01_0014
        # features, time_per_video, num_clips_per_video = run(model, frequency, outputdir, frames_dir, batch_size, sample_mode)
        # # np.save(os.path.join(outputdir, name), features)
        # total_time += time_per_video
        # total_clips += num_clips_per_video
        # print(name)
        # del features

    time_avg = total_time / total_clips
    print(total_clips)
    print(total_time)
    print(time_avg)

    print("Process Done !")

def run(model, frequency, outputdir, frames_dir, batch_size, sample_mode):
    # Takes 01_0014 as an example
    chunk_size = 16

    def forward_batch(batch_data):
        batch_data = batch_data.transpose([0, 4, 1, 2, 3]) # (16, 3, 16, 256, 340)
        batch_data = torch.from_numpy(batch_data)
        with torch.no_grad():
            batch_data = Variable(batch_data.cuda()).float()
            
            # X3D
            # features = model(batch_data)

            # I3D
            inp = {"frames": batch_data}
            features = model(inp)
        return features.cpu().numpy() # torch.Size([16, 2048, 1, 1, 1])

    rgb_files = [i for i in os.listdir(frames_dir)]
    rgb_files.sort()
    frames_cnt = len(rgb_files) # 256

    clipped_length = ((frames_cnt - chunk_size) // frequency) * frequency # 240
    frames_indices = []

    for i in range(clipped_length // frequency + 1): # 16
        frames_indices.append([j for j in range(i * frequency, i * frequency + chunk_size)])
    
    frames_indices = np.array(frames_indices) # (16, 16)
    chunk_num = frames_indices.shape[0] # 16 (clips) # number of clips
    batch_num = int(np.ceil(chunk_num / batch_size)) # 1 (How many batches)
    frames_indices = np.array_split(frames_indices, batch_num, axis = 0) # (1, 16, 16) (batch_num, batch_size, num frames)
    
    # print(frames_indices)

    if sample_mode == "oversample":
        full_features = [[] for i in range(10)]
    else:
        full_features = [[]]
    
    total_time_preprocessing = 0
    for batch_id in range(batch_num):
        # print(frames_indices[batch_id].shape) # (16, 16)
        # scale = 1
        scale = 256/224
        
        batch_data, time_preprocessing = load_rgb_batch(frames_dir, rgb_files, frames_indices[batch_id], scale) # (16, 16, 256, 340, 3)
        total_time_preprocessing += time_preprocessing
        start_time = time.time()
        
        if sample_mode == "oversample":
            batch_data_ten_crop = oversample_data(batch_data, scale) # len = 10, batch_data_ten_crop[0].shape = (16, 16, 224, 224, 3)
            for i in range(10):
                assert (batch_data_ten_crop[i].shape[-2] == 256)
                assert (batch_data_ten_crop[i].shape[-3] == 256)
                temp = forward_batch(batch_data_ten_crop[i])
                full_features[i].append(temp)
        
        elif sample_mode == "center_crop":
            batch_data = batch_data[:, :, 16:240, 58:282, :]
            assert (batch_data.shape[-2] == 256)
            assert (batch_data.shape[-3] == 256)
            temp = forward_batch(batch_data)
            full_features[0].append(temp)
                
    full_features = [np.concatenate(feature, axis = 0) for feature in full_features]
    full_features = [np.expand_dims(feature, axis = 0) for feature in full_features]
    # print(full_features[0].shape) # (1, 16, 2048, 1, 1, 1)
    # print(len(full_features)) # 10
    
    full_features = np.concatenate(full_features, axis = 0)
    full_features = full_features[:, :, :, 0, 0, 0]
    full_features = np.array(full_features).transpose([1, 0, 2]) # (16, 10, 2048)
    
    # print(full_features.shape)
    total_time = total_time_preprocessing + (time.time() - start_time)
    return full_features, total_time, chunk_num
    
def load_rgb_batch(frames_dir, rgb_files, frames_indices, scale):
    batch_data = np.zeros(frames_indices.shape + (int(256 * scale), int(340 * scale), 3)) # (height, width)
    
    total_time_preprocessing = 0
    
    for i in range(frames_indices.shape[0]):
        for j in range(frames_indices.shape[1]):
            data, time_preprocessing = load_frame(os.path.join(frames_dir, rgb_files[frames_indices[i][j]]), scale)
            batch_data[i, j, :, :, :] = data
            total_time_preprocessing += time_preprocessing
    
    return batch_data, total_time_preprocessing

def load_frame(frame_file, scale):
    data = Image.open(frame_file)

    start_time = time.time()

    data = data.resize((int(340 * scale), int(256 * scale)), Image.ANTIALIAS) # (width, height)
    data = np.array(data)
    data = data.astype(float)
    data = data / 255

    # transform_norm = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std)
    # ])
    # data = transform_norm(data)
    
    # data = np.array(torch.permute(data, (1, 2, 0)))
    
    assert (data.max() <= 1.0)
    assert (data.min() >= 0.0)
    time_preprocessing = time.time() - start_time
    return data, time_preprocessing

def oversample_data(data, scale):
    data_flip = np.array(data[:,:,:,::-1,:])

    data_1 = np.array(data[:, :, :int(224 * scale), :int(224 * scale), :])
    data_2 = np.array(data[:, :, :int(224 * scale), int(-224 * scale):, :])
    data_3 = np.array(data[:, :, int(16 * scale):int(240 * scale), int(58 * scale):int(282 * scale), :])
    data_4 = np.array(data[:, :, int(-224 * scale):, :int(224 * scale), :])
    data_5 = np.array(data[:, :, int(-224 * scale):, int(-224 * scale):, :])

    data_f_1 = np.array(data_flip[:, :, :int(224 * scale), :int(224 * scale), :])
    data_f_2 = np.array(data_flip[:, :, :int(224 * scale), -int(224 * scale):, :])
    data_f_3 = np.array(data_flip[:, :, int(16 * scale):int(240 * scale), int(58 * scale):int(282 * scale), :])
    data_f_4 = np.array(data_flip[:, :, -int(224 * scale):, :int(224 * scale), :])
    data_f_5 = np.array(data_flip[:, :, int(-224 * scale):, int(-224 * scale):, :])

    return [data_1, data_2, data_3, data_4, data_5,
            data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]

if __name__ == "__main__":
    datasetpath = "F:\ShanghaiTech\output_frames_train_abnormal"
    outputpath = "Features"
    pretrainedpath = "X3D_M_extract_features.pth"
    frequency = 16
    batch_size = 1
    sample_mode = "oversample"
    type = "train_abnormal"
    
    generate(datasetpath, outputpath, pretrainedpath, frequency, batch_size, sample_mode, type)