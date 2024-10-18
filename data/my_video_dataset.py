from __future__ import print_function, absolute_import
import torch
from torch.utils.data import Dataset
import os
from skimage import io
from torchvision import transforms
# import numpy as np
import pandas as pd
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, frame_root, csv_in, clip_len, overlap, use_cuda=True, transform=None):
        self.frame_root = frame_root
        self.clip_len = int(clip_len)
        self.overlap = int(overlap)
        self.frame_df = self.generate_lists(csv_in)  # Directly store the dataframe
        self.use_cuda = use_cuda
        self.transform = transform

    def __len__(self):
        return len(self.frame_df)

    def __getitem__(self, item):  
        video_id = self.frame_df.iloc[item]['video_id']
        frame_indices = self.frame_df.iloc[item]['frame_indices']

        v_dir = os.path.join(self.frame_root, video_id)
        vid_frames = []
        for idx in frame_indices:
            frame_file_name = f"{idx}.jpg"
            frame_path = os.path.join(v_dir, frame_file_name)
            tmp_frame = io.imread(frame_path)
            tmp_frame = Image.fromarray(tmp_frame)  # Convert np.array to PIL Image
            vid_frames.append(tmp_frame)

        # Apply transformations
        if self.transform:
            frames = torch.cat([self.transform(frame).unsqueeze(0) for frame in vid_frames], dim=0)
        else:
            tmp_frame_trans = transforms.ToTensor()
            frames = torch.cat([tmp_frame_trans(frame).unsqueeze(0) for frame in vid_frames], dim=0)

        return video_id, frames # TODO: change here if causing problems during next training run!

    def generate_lists(self, csv_in, split=[0], frame_skip=4): # don't change frame_skip, would require extracting new frames for all videos
        df = pd.read_csv(csv_in)
        frame_data = []
        for _, row in df.iterrows():
            if row['label'] in split: 
                start = int(row['start'] * 60) # convert time to frame index
                end = int(row['end'] * 60)
                video_id = row['video_id']
                # print(video_id, start, end)
                length = end - start
                num_clips = (length - self.overlap) // ((self.clip_len - self.overlap) * frame_skip) # accounting for overlap and frame rate
                # print(num_clips) # check if value is correct
                next_index = start
                for _ in range(int(num_clips) + 1):
                    sub_list = []
                    while len(sub_list) < self.clip_len and next_index <= end:
                        sub_list.append(next_index)
                        next_index += frame_skip # 4
                    remaining_frames = end - next_index + frame_skip  # Check how many frames remain
                    if remaining_frames < self.clip_len and len(sub_list) < self.clip_len: # If not enough frames are left, increase overlap to fill the last list
                        while len(sub_list) < self.clip_len:
                            sub_list.insert(0, sub_list[0] - frame_skip)

                    frame_data.append([video_id, sub_list])
                    next_index -= self.overlap * frame_skip # index + 4 - 20 -> index - 16 -> overlap of 4
        frame_df = pd.DataFrame(frame_data, columns=['video_id', 'frame_indices'])
        return frame_df
