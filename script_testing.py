from __future__ import absolute_import, print_function
import os
from data.merge_csvs_val import assign_labels_to_frames
import utils
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import data
from options.testing_options import TestOptions
import utils
from models import AutoEncoderCov3DMem


opt_parser = TestOptions()
opt = opt_parser.parse(is_print=True)
use_cuda = opt.UseCUDA
device = torch.device("cuda" if use_cuda else "cpu")
batch_size_in = opt.BatchSize  # 1
chnum_in_ = opt.ImgChnNum  
framenum_in_ = opt.FrameNum  
mem_dim_in = opt.MemDim  
sparse_shrink_thres = opt.ShrinkThres
img_crop_size = 0
model_setting = utils.get_model_setting(opt)

# model path
model_root = opt.ModelRoot
if opt.ModelFilePath:
    model_path = opt.ModelFilePath
else:
    model_path = os.path.join(model_root, model_setting + ".pt")

# test result path
te_res_root = opt.OutRoot  # ./results/1/
test_results_path = te_res_root + "/" + "res_" + model_setting
utils.mkdir(test_results_path)

# loading trained model
if opt.ModelName == "MemAE":
    model = AutoEncoderCov3DMem(chnum_in_, mem_dim_in, shrink_thres=sparse_shrink_thres)
else:
    raise ValueError("Wrong model name.")

model_para = torch.load(model_path)
model.load_state_dict(model_para)
model.to(device)
model.eval()

# Frame transformations & data normalisation
if chnum_in_ == 1:
    norm_mean = [0.5]
    norm_std = [0.5]
    frame_trans = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),  # seems to be necessary. Why not train on 3 channels though?
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )
elif chnum_in_ == 3:
    norm_mean = (0.5, 0.5, 0.5)
    norm_std = (0.5, 0.5, 0.5)
    frame_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])

height = width = 128

# Data
frame_root = "/local/scratch/hendrik/cataract_frames_downsized/"
test_csv = "/local/scratch/hendrik/test_set.csv"
overlap_ratio = opt.Overlap
overlap_len = framenum_in_ * overlap_ratio
print(f"overlap: {overlap_len}")

video_dataset = data.MyDataset(
    frame_root=frame_root,
    csv_in=test_csv,
    clip_len=framenum_in_,
    overlap=overlap_len,
    split=[0, 1],
    transform=frame_trans,
)

tr_data_loader = DataLoader(
    dataset=video_dataset,
    batch_size=batch_size_in,
    shuffle=False,
)

assign_labels_to_frames(
    frame_csv_path="/local/scratch/hendrik/dataset_frame_lists.csv",  # must come after dataset init
    label_csv_path="/local/scratch/hendrik/val_set.csv",
    output_csv_path="/local/scratch/hendrik/merged_csv.csv",
)
eval_csv = "/local/scratch/hendrik/merged_csv.csv"

# Dictionary to store errors by video name
errors_by_video = {}

with torch.no_grad():
    for batch_idx, (video_name, frames) in enumerate(tr_data_loader):
        # Process frames for each clip
        frames = frames.to(device)
        frames = frames.view(frames.size(0), chnum_in_, framenum_in_, height, width)
        print(f"[batch {batch_idx + 1}/{len(tr_data_loader)}]")

        if opt.ModelName == "MemAE":
            recon_res = model(frames)
            recon_frames = recon_res["output"]
            r = recon_frames - frames
            r = utils.crop_image(r, img_crop_size)
            sp_error_map = torch.sum(r**2, dim=1) ** 0.5
            sp_error_vec = sp_error_map.view(sp_error_map.size(0), -1)
            recon_error = torch.mean(sp_error_vec, dim=-1)  # Average error per clip in batch

            # Clean up video_name tuple to get the actual name
            clean_video_name = str(video_name[0]).strip("()'")

            # Initialize the error list for this video if it doesn't exist
            if clean_video_name not in errors_by_video:
                errors_by_video[clean_video_name] = []

            # Append the error for this clip to the video-specific list
            errors_by_video[clean_video_name].extend(recon_error.cpu().tolist())
        else:
            raise ValueError("Wrong model name.")

# Save the accumulated error lists for each video
for video_name, error_list in errors_by_video.items():
    np.save(os.path.join(test_results_path, f"{video_name}.npy"), error_list)
## evaluation
utils.my_eval_video(frame_root, test_results_path, eval_csv, normal=False, is_show=True)
