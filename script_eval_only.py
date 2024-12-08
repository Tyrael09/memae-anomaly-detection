import data
from data.merge_csvs_val import assign_labels_to_frames
import utils
from torchvision import transforms
import re


# Paths
frame_root = "/local/scratch/hendrik/cataract_frames_downsized/"
test_csv = "/local/scratch/hendrik/test_set.csv"
res_path = "results/6/MemAE_MemDim1000_FrameNum16_Overlap0.25_ChNum1"

# Extract FrameNum
frame_num_match = re.search(r'FrameNum(\d+)', res_path)
frame_num = int(frame_num_match.group(1)) if frame_num_match else None

# Extract ChNum
ch_num_match = re.search(r'ChNum(\d+)', res_path)
ch_num = int(ch_num_match.group(1)) if ch_num_match else 1

# Extract Overlap
overlap_match = re.search(r'Overlap(\d+\.\d+)', res_path)
overlap = float(overlap_match.group(1)) if overlap_match else None

# Calculate overlapping frames
if frame_num is not None and overlap is not None:
    overlapping_frames = int(frame_num * overlap)
else:
    overlapping_frames = None

# Frame transformations & data normalisation
if ch_num == 1:
    norm_mean = [0.5]
    norm_std = [0.5]
    frame_trans = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )
elif ch_num == 3:
    norm_mean = (0.5, 0.5, 0.5)
    norm_std = (0.5, 0.5, 0.5)
    frame_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])

# Instantiate Dataset for dataframe creation
video_dataset = data.MyDataset(
    frame_root=frame_root,
    csv_in=test_csv,
    clip_len=frame_num,
    overlap=overlapping_frames,
    split=[0, 1],
    transform=frame_trans,
)
assign_labels_to_frames(
    frame_csv_path="/local/scratch/hendrik/dataset_frame_lists.csv",  # updated by MyDataset, so wait for it to be initialised
    label_csv_path="/local/scratch/hendrik/val_set.csv",
    output_csv_path="/local/scratch/hendrik/merged_csv.csv",
)
eval_csv = "/local/scratch/hendrik/merged_csv.csv"


# Evaluation
utils.my_eval_video(res_path, eval_csv, frame_num, index=8)
