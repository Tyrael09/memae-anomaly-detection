import data
from data.merge_csvs_val import assign_labels_to_frames
import utils
from torchvision import transforms

frame_root = "/local/scratch/hendrik/cataract_frames_downsized/"
test_csv = "/local/scratch/hendrik/test_set.csv"
res_path = "results/6/MemAE_MemDim2000_FrameNum16_Overlap0.25_ChNum1"  # TODO: change
chnum_in_ = 1

# Frame transformations & data normalisation
if chnum_in_ == 1:
    norm_mean = [0.5]
    norm_std = [0.5]
    frame_trans = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )
elif chnum_in_ == 3:
    norm_mean = (0.5, 0.5, 0.5)
    norm_std = (0.5, 0.5, 0.5)
    frame_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])

video_dataset = data.MyDataset(
    frame_root=frame_root,
    csv_in=test_csv,
    clip_len=16,
    overlap=4,
    split=[0, 1],
    transform=frame_trans,
)
assign_labels_to_frames(
    frame_csv_path="/local/scratch/hendrik/dataset_frame_lists.csv",  # updated by MyDataset, so wait for it to be initialised
    label_csv_path="/local/scratch/hendrik/val_set.csv",
    output_csv_path="/local/scratch/hendrik/merged_csv.csv",
)
eval_csv = "/local/scratch/hendrik/merged_csv.csv"


## evaluation
utils.my_eval_video(frame_root, res_path, eval_csv, normal=True, is_show=True)
