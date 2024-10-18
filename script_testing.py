from __future__ import absolute_import, print_function
import os
import utils
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import data
from options.testing_options import TestOptions
import utils
from models import AutoEncoderCov3D, AutoEncoderCov3DMem

###
opt_parser = TestOptions()
opt = opt_parser.parse(is_print=True)
use_cuda = opt.UseCUDA
device = torch.device("cuda" if use_cuda else "cpu")

###
batch_size_in = opt.BatchSize #1
chnum_in_ = opt.ImgChnNum      # channel number of the input images
framenum_in_ = opt.FrameNum  # frame number of the input images in a video clip
mem_dim_in = opt.MemDim
sparse_shrink_thres = opt.ShrinkThres

img_crop_size = 0

######
model_setting = utils.get_model_setting(opt)

## data path
# frame_root = opt.Dataroot
# data_frame_dir = data_root + 'Test/'
# data_idx_dir = data_root + 'Test_idx/'

############ model path
model_root = opt.ModelRoot
if(opt.ModelFilePath): # True if string is not empty (!) or None
    model_path = opt.ModelFilePath
else:
    model_path = os.path.join(model_root, model_setting + '.pt')

### test result path
te_res_root = opt.OutRoot # ./results/1/
te_res_path = te_res_root + '/' + 'res_' + model_setting
utils.mkdir(te_res_path)

###### loading trained model
if (opt.ModelName == 'AE'):
    print("not MemAE")
    # model = AutoEncoderCov3D(chnum_in_)
elif(opt.ModelName=='MemAE'):
    model = AutoEncoderCov3DMem(chnum_in_, mem_dim_in, shrink_thres=sparse_shrink_thres)
else:
    model = []
    print('Wrong Name.')

##
model_para = torch.load(model_path)
model.load_state_dict(model_para)
model.to(device)
model.eval()

##
if(chnum_in_==1):
    norm_mean = [0.5]
    norm_std = [0.5]
elif(chnum_in_==3):
    norm_mean = (0.5, 0.5, 0.5)
    norm_std = (0.5, 0.5, 0.5)

frame_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
unorm_trans = utils.UnNormalize(mean=norm_mean, std=norm_std)

##
frame_root = '/local/scratch/hendrik/cataract_test_frames_downsized/'
my_csv = '/local/scratch/hendrik/test_set.csv'

###### data
my_dataset = data.MyDataset(frame_root=frame_root, csv_in=my_csv, transform=frame_trans)
tr_data_loader = DataLoader(my_dataset,
                            batch_size=batch_size_in,
                            shuffle=True,
                            )

## Testing loop
with torch.no_grad():
    for batch_idx, (video_name, frames) in enumerate(tr_data_loader):  
        # frames = frames.view(frames.size(0), 1, 16, 128, 128) # toy around with this to match required dimensions?
        frames = frames.to(device)  
        print(f'[batch {batch_idx + 1}/{len(tr_data_loader)}]')  
        recon_error_list = []

        if opt.ModelName == 'MemAE':
            recon_res = model(frames)
            recon_frames = recon_res['output']
            r = recon_frames - frames
            r = utils.crop_image(r, img_crop_size)
            sp_error_map = torch.sum(r ** 2, dim=1) ** 0.5
            sp_error_vec = sp_error_map.view(sp_error_map.size(0), -1)
            recon_error = torch.mean(sp_error_vec, dim=-1)
            recon_error_list += recon_error.cpu().tolist()
        else:
            print('Wrong ModelName.')
        np.save(os.path.join(te_res_path, f'{video_name}.npy'), recon_error_list) 

## evaluation
utils.my_eval_video(frame_root, te_res_path, my_csv, is_show=False) # TODO: rewrite this too??
