

# train reconstruction network

import os
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from pathlib import Path
from loader_isbi_large_dataset import SSFrameDataset
from options.train_options_rec_reg import TrainOptions

from utilits_grid_data import *
from utils_rec_volume_2 import *


torch.manual_seed(4)
opt = TrainOptions().parse()
writer = SummaryWriter(os.path.join(opt.SAVE_PATH))

# if not opt.multi_gpu:
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



opt.FILENAME_VAL=opt.FILENAME_VAL+'_seqlen'+str(opt.NUM_SAMPLES)+'_'+opt.split_type+'_'+opt.train_set+'.json'
opt.FILENAME_TEST=opt.FILENAME_TEST+'_seqlen'+str(opt.NUM_SAMPLES)+'_'+opt.split_type+'_'+opt.train_set+'.json'
opt.FILENAME_TRAIN=[opt.FILENAME_TRAIN[i]+'_seqlen'+str(opt.NUM_SAMPLES)+'_'+opt.split_type+'_'+opt.train_set+'.json' for i in range(len(opt.FILENAME_TRAIN))]

# dset_val = SSFrameDataset.read_json(Path(os.getcwd()).parent.absolute().as_posix()+opt.DATA_PATH,opt.FILENAME_VAL,opt.h5_file_name)
# dset_test = SSFrameDataset.read_json(Path(os.getcwd()).parent.absolute().as_posix()+opt.DATA_PATH,opt.FILENAME_TEST,opt.h5_file_name)
# dset_train_list = [SSFrameDataset.read_json(Path(os.getcwd()).parent.absolute().as_posix()+opt.DATA_PATH,opt.FILENAME_TRAIN[i],opt.h5_file_name) for i in range(len(opt.FILENAME_TRAIN))]
# dset_train = dset_train_list[0]+dset_train_list[1]+dset_train_list[2]
# print('using %s'%opt.h5_file_name)

dset_val = SSFrameDataset.read_json(Path(os.getcwd()).parent.absolute().as_posix()+opt.DATA_PATH,opt.FILENAME_VAL,opt.h5_file_name)
dset_test = SSFrameDataset.read_json(Path(os.getcwd()).parent.absolute().as_posix()+opt.DATA_PATH,opt.FILENAME_TEST,opt.h5_file_name)
dset_train_list = [SSFrameDataset.read_json(Path(os.getcwd()).parent.absolute().as_posix()+opt.DATA_PATH,opt.FILENAME_TRAIN[i],opt.h5_file_name) for i in range(len(opt.FILENAME_TRAIN))]
dset_train = dset_train_list[0]+dset_train_list[1]+dset_train_list[2]
# dset_val = dset_val+dset_train_list[2]
print('using %s'%opt.h5_file_name)


# for generate registration use
# dset_val_reg = SSFrameDataset.read_json(opt.DATA_PATH,opt.FILENAME_VAL,opt.h5_file_name,num_samples = -1)
# dset_test_reg = SSFrameDataset.read_json(opt.DATA_PATH,opt.FILENAME_TEST,opt.h5_file_name,num_samples = -1)
# dset_train_list_reg = [SSFrameDataset.read_json(opt.DATA_PATH,opt.FILENAME_TRAIN[i],opt.h5_file_name,num_samples = -1) for i in range(len(opt.FILENAME_TRAIN))]
# dset_train_reg = dset_train_list_reg[0]+dset_train_list_reg[1]+dset_train_list_reg[2]


# dset_all = dset_train+dset_val+dset_test
               
saved_folder = opt.SAVE_PATH+'/'+ 'test_plotting'
if not os.path.exists(saved_folder):
    os.makedirs(saved_folder)


train_rec_reg_model = Train_Rec_Reg_Model(opt = opt,
                        non_improve_maxmum = 1e10, 
                        reg_loss_weight = 1000,
                        val_loss_min = 1e10,
                        val_dist_min = 1e10,
                        val_loss_min_reg = 1e10,
                        dset_train = dset_train,
                        dset_val = dset_val,
                        dset_train_reg = None,
                        dset_val_reg = None,
                        device = device,
                        writer = writer)


# train_rec_reg_model.load_best_rec_model_initial()
# train_rec_reg_model.load_best_rec_model()
# train_rec_reg_model.load_best_rec_model_half_converge()
# train_rec_reg_model.load_rec_model400()


# train_rec_reg_model.load_best_rec_model_initial()

# train_rec_reg_model.load_best_reg_model()
train_rec_reg_model.train_rec_model()
    



# # Make sure the size if the width/height is a 64 step (512, 576, 640...),
# otherwise, it would report a runtime error:
# Sizes of tensors must match except in dimension 1. Expected size 23 but got size 22 for tensor number 1 in the list.
# this method is not used as DivisiblePad will detach the variable
# instead, I initialise a divisible dimention from the intepoleted volume
# divisible_pad_16 = DivisiblePad(k=16,mode = 'minimum')


# if opt.multi_gpu:
#     model= nn.DataParallel(model)
#     VoxelMorph_net = nn.DataParallel(model)

# if opt.retain:
#     model.load_state_dict(torch.load(os.path.join(opt.SAVE_PATH,'saved_model', 'model_epoch'+str(opt.retain_epoch)),map_location=torch.device(device)))
#     # model.load_state_dict(torch.load(os.path.join(opt.SAVE_PATH,'saved_model', 'best_validation_dist_model'),map_location=torch.device(device)))
#     VoxelMorph_net.load_state_dict(torch.load(os.path.join(opt.SAVE_PATH,'saved_model', 'model_reg_epoch'+str(opt.retain_epoch)),map_location=torch.device(device)))






