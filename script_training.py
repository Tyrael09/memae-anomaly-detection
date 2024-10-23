import os
import utils
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import data
from options.training_options import TrainOptions
import utils
from models import AutoEncoderCov3DMem
from models import EntropyLossEncap

# Parser options
opt_parser = TrainOptions()
opt = opt_parser.parse(is_print=True)
use_cuda = opt.UseCUDA
device = torch.device("cuda" if use_cuda else "cpu")

# Setting the random seed
utils.seed(opt.Seed)
if opt.IsDeter:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Model and training configuration
model_setting = utils.get_model_setting(opt)
print('Setting: %s' % (model_setting))

batch_size_in = opt.BatchSize
learning_rate = opt.LR
max_epoch_num = opt.EpochNum
chnum_in_ = opt.ImgChnNum # channel number of the input images
framenum_in_ = opt.FrameNum # num of frames in a video clip
mem_dim_in = opt.MemDim
entropy_loss_weight = opt.EntropyLossWeight
sparse_shrink_thres = opt.ShrinkThres

print('bs=%d, lr=%f, entrloss=%f, shr=%f, memdim=%d' % (batch_size_in, learning_rate, entropy_loss_weight, sparse_shrink_thres, mem_dim_in))

# model saving dir path
saving_root = opt.ModelRoot
saving_model_path = os.path.join(saving_root, 'model_' + model_setting + '/') # './models/model_Memae...epoch_0100_final.pt/'
utils.mkdir(saving_model_path)

# tensorboard log
if opt.IsTbLog:
    log_path = os.path.join(saving_root, 'log_'+ model_setting + '/') # tensorboard logs at models/log_MemAE_...Seed1_Non/events.out.tfevents.1728225989.pc3054
    utils.mkdir(log_path)
    tb_logger = utils.Logger(log_path)

# Frame transformations & data normalisation
if chnum_in_ == 1:
    norm_mean = [0.5]
    norm_std = [0.5]
elif chnum_in_ == 3:
    norm_mean = (0.5, 0.5, 0.5)
    norm_std = (0.5, 0.5, 0.5)

height = width = 128
frame_trans = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                  transforms.ToTensor(), # could change to only run this on cpu and rest on gpu.. might be faster
                                  transforms.Normalize([0.5], [0.5])
                                  ])

unorm_trans = utils.UnNormalize(mean=norm_mean, std=norm_std)

# Data
frame_root = '/local/scratch/hendrik/cataract_frames_downsized/'
my_csv = '/local/scratch/hendrik/train_set.csv'
overlap_ratio = 3 / 4 # TODO: make sure to change name in training_options.py accordingly!
overlap_len = framenum_in_ * overlap_ratio # overlap = (clip_length / 4) * overlap_ratio, experiment with [1,2,3]
print(f"overlap: {overlap_len}")

video_dataset = data.MyDataset(frame_root=frame_root, 
                                csv_in=my_csv, 
                                clip_len=framenum_in_, 
                                overlap=overlap_len, 
                                transform=frame_trans
                                ) 

tr_data_loader = DataLoader(video_dataset,
                            batch_size=batch_size_in,
                            shuffle=True,
                            num_workers=opt.NumWorker
                            )

# Model initialization
if opt.ModelName == 'MemAE':
    model = AutoEncoderCov3DMem(chnum_in_, mem_dim_in, shrink_thres=sparse_shrink_thres)
else:
    raise ValueError('Wrong model name.')

# Checkpointing parameters
data_loader_len = len(tr_data_loader)
textlog_interval = opt.TextLogInterval
snap_save_interval = opt.SnapInterval
save_check_interval = opt.SaveCheckInterval
tb_img_log_interval = opt.TBImgLogInterval

# Optimizer setup
tr_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Load checkpoint if exists
start_epoch = 0
global_ite_idx = 0
checkpoint_path = opt.CheckpointPath # defaults to None

if checkpoint_path is not None:
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        # model.load_state_dict(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict']) 
        tr_optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
        global_ite_idx = checkpoint['global_ite_idx']
        start_epoch = checkpoint['epoch_idx'] 
        # global_ite_idx = start_epoch * data_loader_len # calculate correct ite_idx for logging purposes
        print(f"Resuming training from epoch {start_epoch}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}. Please check the path.")
else:
    # If no checkpoint path is provided, initialize model weights
    print("No checkpoint provided. Applying initial weights.")
    model.apply(utils.weights_init)

# Moving the model to the correct device
model.to(device)

# Loss functions
tr_recon_loss_func = nn.MSELoss().to(device)
tr_entropy_loss_func = EntropyLossEncap().to(device)

# Training loop
for epoch_idx in range(start_epoch, max_epoch_num):
    for batch_idx, (_, frames) in enumerate(tr_data_loader): 
        frames = frames.to(device)        
        frames = frames.view(frames.size(0), chnum_in_, framenum_in_, height, width) # Assuming the shape of `frames` is [14, 16, 1, 128, 128] (batch, frames, channels, height, width)
        if opt.ModelName == 'MemAE':
            recon_res = model(frames)
            recon_frames = recon_res['output']
            att_w = recon_res['att']
            loss = tr_recon_loss_func(recon_frames, frames)
            recon_loss_val = loss.item()
            entropy_loss = tr_entropy_loss_func(att_w)
            entropy_loss_val = entropy_loss.item()
            loss +=  entropy_loss_weight * entropy_loss
            loss_val = loss.item()
            
            # Backpropagation
            tr_optimizer.zero_grad()
            loss.backward()
            tr_optimizer.step()
            
        ## TB log val
        if opt.IsTbLog:
            tb_info = {
                'loss': loss_val,
                'recon_loss': recon_loss_val,
                'entropy_loss': entropy_loss_val
            }
            for tag, value in tb_info.items():
                tb_logger.scalar_summary(tag, value, global_ite_idx)
            # TB log img
            if (global_ite_idx % tb_img_log_interval) == 0:
                frames_vis = utils.vframes2imgs(unorm_trans(frames.data), step=5, batch_idx=0)
                frames_vis = np.concatenate(frames_vis, axis=-1)
                frames_vis = frames_vis[None, :, :] * np.ones(3, dtype=int)[:, None, None]
                frames_recon_vis = utils.vframes2imgs(unorm_trans(recon_frames.data), step=5, batch_idx=0)
                frames_recon_vis = np.concatenate(frames_recon_vis, axis=-1)
                frames_recon_vis = frames_recon_vis[None, :, :] * np.ones(3, dtype=int)[:, None, None]
                tb_info = {
                    'x': frames_vis,
                    'x_rec': frames_recon_vis
                }
                for tag, imgs in tb_info.items():
                    tb_logger.image_summary(tag, imgs, global_ite_idx)

        # Text logging
        if (batch_idx % textlog_interval) == 0:
            print('[%s, epoch %d/%d, bt %d/%d] loss=%f, rc_losss=%f, ent_loss=%f' % (model_setting, epoch_idx, max_epoch_num, batch_idx, data_loader_len, loss_val, recon_loss_val, entropy_loss_val))
        
        # Snapshot saving
        if (global_ite_idx % snap_save_interval) == 0:
            torch.save(model.state_dict(), '%s/%s_snap.pt' % (saving_model_path, model_setting))
        global_ite_idx += 1

    # Epoch-level model saving    
    if (epoch_idx % save_check_interval) == 0:
        # torch.save(model.state_dict(), '%s/%s_epoch_%04d.pt' % (saving_model_path, model_setting, epoch_idx))
        # For future runs, additionally save state dict to enable loading from checkpoint:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': tr_optimizer.state_dict(),
            'global_ite_idx': global_ite_idx,
            'epoch': epoch_idx
        }, '%s/%s_epoch_%04d.pt' % (saving_model_path, model_setting, epoch_idx))

# Final model saving
torch.save(model.state_dict(), '%s/%s_epoch_%04d_final.pt' % (saving_model_path, model_setting, epoch_idx))

