

import os
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from pathlib import Path
from loader_isbi_large_dataset_test import SSFrameDataset
from options.train_options_rec_reg import TrainOptions

from utilits_grid_data import *
from utils_rec_volume_test import *



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

dset_val = SSFrameDataset.read_json(Path(os.getcwd()).parent.absolute().as_posix()+opt.DATA_PATH,opt.FILENAME_VAL,opt.h5_file_name)
dset_test = SSFrameDataset.read_json(Path(os.getcwd()).parent.absolute().as_posix()+opt.DATA_PATH,opt.FILENAME_TEST,opt.h5_file_name)
dset_train_list = [SSFrameDataset.read_json(Path(os.getcwd()).parent.absolute().as_posix()+opt.DATA_PATH,opt.FILENAME_TRAIN[i],opt.h5_file_name) for i in range(len(opt.FILENAME_TRAIN))]
dset_train = dset_train_list[0]+dset_train_list[1]#+dset_train_list[2]
dset_val = dset_val+dset_train_list[2]
print('using %s'%opt.h5_file_name)

# print('using %s'%opt.h5_file_name)

# dset_all = dset_train+dset_val+dset_test


# # for generate registration use
# dset_val_reg = SSFrameDataset.read_json(opt.DATA_PATH,opt.FILENAME_VAL,opt.h5_file_name,num_samples = -1)
# dset_test_reg = SSFrameDataset.read_json(opt.DATA_PATH,opt.FILENAME_TEST,opt.h5_file_name,num_samples = -1)
# dset_train_list_reg = [SSFrameDataset.read_json(opt.DATA_PATH,opt.FILENAME_TRAIN[i],opt.h5_file_name,num_samples = -1) for i in range(len(opt.FILENAME_TRAIN))]
# dset_train_reg = dset_train_list_reg[0]+dset_train_list_reg[1]+dset_train_list_reg[2]


               
saved_folder = opt.SAVE_PATH+'/'+ 'test_plotting'
if not os.path.exists(saved_folder):
    os.makedirs(saved_folder)




# reset train set
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
                        writer = writer,
                        option = 'common_volume')

# # train_rec_reg_model.load_best_rec_model()
# train_rec_reg_model.load_best_reg_model()
# # train_rec_reg_model.load_best_rec_model_initial()
# # train_rec_reg_model.load_best_rec_model_epoch500()
# if opt.initial == 'InitialBest':
#     train_rec_reg_model.load_best_rec_model_initial()
# elif opt.initial == 'InitialHalf':
#     train_rec_reg_model.load_best_rec_model_half_converge()

train_rec_reg_model.load_rec_model_epoch('000001000_311')
# train_rec_reg_model.load_rec_model_epoch('00000200')

train_rec_reg_model.train_rec_model()



# for iteration in range(opt.max_inter_rec_reg):
#     # train reg model
#     train_rec_reg_model.load_best_rec_model() # load the best model to generate the volume for tarining the registration network  
#     train_rec_reg_model.load_best_reg_model_initial()

#     # train_rec_reg_model.train_reg_model()
    
#     # train rec model until validation loss doesn't decrease or has the maxmum training epoch
    
#     train_rec_reg_model.train_rec_model()
    
