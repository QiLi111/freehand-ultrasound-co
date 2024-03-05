#  get the volume size of all sample (100 slices) from all avaliable scans

import os
import torch,sys
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from pathlib import Path
sys.path.append("/raid/candi/Qi/freehand-ultrasound")
sys.path.append("/raid/candi/Qi/freehand-ultrasound")
sys.path.append("/raid/candi/Qi/freehand-ultrasound/new_data_process")

from loader_isbi_large_dataset import SSFrameDataset
from options.train_options_rec_reg import TrainOptions

from utilits_grid_data import *
from utils_rec_volume_1 import *

def min_max(volume_ori,volume_opt,scan_name):
    min_x_ori = np.min(volume_ori[:,0])
    min_y_ori = np.min(volume_ori[:,1])
    min_z_ori = np.min(volume_ori[:,2])

    max_x_ori = np.max(volume_ori[:,0])
    max_y_ori = np.max(volume_ori[:,1])
    max_z_ori = np.max(volume_ori[:,2])


    min_x_opt = np.min(volume_opt[:,0])
    min_y_opt = np.min(volume_opt[:,1])
    min_z_opt = np.min(volume_opt[:,2])

    max_x_opt = np.max(volume_opt[:,0])
    max_y_opt = np.max(volume_opt[:,1])
    max_z_opt = np.max(volume_opt[:,2])

    # find the scan name of the min or max volume
    # too many indexes for min
    # min_x_ori_name = scan_name[np.where(volume_ori[:,0] == min_x_ori)[0]]
    # min_y_ori_name = scan_name[np.where(volume_ori[:,1] == min_y_ori)[0]]
    # min_z_ori_name = scan_name[np.where(volume_ori[:,2] == min_z_ori)[0]]

    max_x_ori_name = scan_name[np.where(volume_ori[:,0] == max_x_ori)[0]]
    max_y_ori_name = scan_name[np.where(volume_ori[:,1] == max_y_ori)[0]]
    max_z_ori_name = scan_name[np.where(volume_ori[:,2] == max_z_ori)[0]]

    # min_x_opt_name = scan_name[np.where(volume_opt[:,0] == min_x_opt)[0]]
    # min_y_opt_name = scan_name[np.where(volume_opt[:,1] == min_y_opt)[0]]
    # min_z_opt_name = scan_name[np.where(volume_opt[:,2] == min_z_opt)[0]]

    max_x_opt_name = scan_name[np.where(volume_opt[:,0] == max_x_opt)[0]]
    max_y_opt_name = scan_name[np.where(volume_opt[:,1] == max_y_opt)[0]]
    max_z_opt_name = scan_name[np.where(volume_opt[:,2] == max_z_opt)[0]]



    return min_x_ori,min_y_ori,min_z_ori,max_x_ori,max_y_ori,max_z_ori,min_x_opt,min_y_opt,min_z_opt,\
            max_x_opt,max_y_opt,max_z_opt,\
            max_x_ori_name,max_y_ori_name,max_z_ori_name,\
            max_x_opt_name,max_y_opt_name,max_z_opt_name

def plot_figs(volume_ori,volume_opt, folder_name,save_name):

    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.plot(range(len(volume_ori)),sorted(np.prod(volume_ori, axis=1)),'.-')
    ax2.plot(range(len(volume_opt)),sorted(np.prod(volume_opt, axis=1)),'.-')
    ax1.set_xlabel('scan_idx')
    ax1.set_ylabel('volume size')

    ax2.set_xlabel('scan_idx')
    ax2.set_ylabel('volume size')
    
    fig.savefig(os.getcwd()+'/new_data_process/'+folder_name+'/'+save_name)





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

dset_val = SSFrameDataset.read_json(Path(os.getcwd()).parent.absolute().as_posix()+opt.DATA_PATH,opt.FILENAME_VAL,opt.h5_file_name,num_samples=-1)
dset_test = SSFrameDataset.read_json(Path(os.getcwd()).parent.absolute().as_posix()+opt.DATA_PATH,opt.FILENAME_TEST,opt.h5_file_name,num_samples=-1)
dset_train_list = [SSFrameDataset.read_json(Path(os.getcwd()).parent.absolute().as_posix()+opt.DATA_PATH,opt.FILENAME_TRAIN[i],opt.h5_file_name,num_samples=-1) for i in range(len(opt.FILENAME_TRAIN))]
dset_train = dset_train_list[0]+dset_train_list[1]#+dset_train_list[2]
dset_val = dset_val+dset_train_list[2]
print('using %s'%opt.h5_file_name)

dset_all = dset_train+dset_val+dset_test


# # for generate registration use
# dset_val_reg = SSFrameDataset.read_json(opt.DATA_PATH,opt.FILENAME_VAL,opt.h5_file_name,num_samples = -1)
# dset_test_reg = SSFrameDataset.read_json(opt.DATA_PATH,opt.FILENAME_TEST,opt.h5_file_name,num_samples = -1)
# dset_train_list_reg = [SSFrameDataset.read_json(opt.DATA_PATH,opt.FILENAME_TRAIN[i],opt.h5_file_name,num_samples = -1) for i in range(len(opt.FILENAME_TRAIN))]
# dset_train_reg = dset_train_list_reg[0]+dset_train_list_reg[1]+dset_train_list_reg[2]


               
saved_folder = opt.SAVE_PATH+'/'+ 'test_plotting'
if not os.path.exists(saved_folder):
    os.makedirs(saved_folder)

csv_name = 'label_volume_size60.csv'
csv_name_anaysis = 'label_volume_size60_analys.csv'
folder_name = 'check_data_volume_60'

# with open(os.getcwd()+'/new_data_process/'+csv_name, 'a', encoding='UTF8') as f:
#     writer = csv.writer(f)
#     # writer.writerow(['the volume of each sequence'])
#     writer.writerow(['scan_name','volume_size_origin_coordinates','volume_size_optimised_coordinates'])


# # # # reset train set
# train_rec_reg_model = Train_Rec_Reg_Model(opt = opt,
#                         non_improve_maxmum = 1e10, 
#                         reg_loss_weight = 1,
#                         val_loss_min = 1e10,
#                         val_dist_min = 1e10,
#                         val_loss_min_reg = 1e10,
#                         dset_train = dset_train,
#                         dset_val = dset_val,
#                         dset_train_reg = None,
#                         dset_val_reg = None,
#                         device = device,
#                         writer = writer)


# for i_scan in range(len(dset_all)):
#     frames_all, tforms_all, tforms_inv_all = dset_all[i_scan]
#     sub_name = str([dset_all.indices_in_use[i_scan][0], dset_all.indices_in_use[i_scan][1]])
#     scan_name = dset_all.name_scan[dset_all.indices_in_use[i_scan][0], dset_all.indices_in_use[i_scan][1]].decode("utf-8")
    
#     for seq in range(0,frames_all.shape[0]-100):
#         frames = torch.from_numpy(frames_all[seq:seq+100])[None,...]
#         tforms = torch.from_numpy(tforms_all[seq:seq+100])[None,...]
#         tforms_inv = torch.from_numpy(tforms_inv_all[seq:seq+100])[None,...]

#         frames, tforms, tforms_inv = frames.to(train_rec_reg_model.device), tforms.to(train_rec_reg_model.device), tforms_inv.to(train_rec_reg_model.device)  
#         tforms_each_frame2frame0 = train_rec_reg_model.transform_label(tforms, tforms_inv)
#         labels = torch.matmul(tforms_each_frame2frame0,torch.matmul(train_rec_reg_model.tform_calib,train_rec_reg_model.image_points))[:,:,0:3,...]
        
#         gt_volume_ori, pred_volume_ori = train_rec_reg_model.scatter_pts_intepolation(labels,labels,frames,step=0)

#         # change the coordinates 
#         ori_pts = torch.matmul(tforms_each_frame2frame0,torch.matmul(train_rec_reg_model.tform_calib,train_rec_reg_model.image_points)).permute(0,1,3,2)

#         labels, labels = ConvPose(labels, ori_pts, ori_pts, 'auto_PCA',train_rec_reg_model.device)

#         gt_volume_opt, pred_volume_opt = train_rec_reg_model.scatter_pts_intepolation(labels,labels,frames,step=0)
        
        
        
#         row = [sub_name+'-'+scan_name,list(gt_volume_ori.shape)[1:],list(gt_volume_opt.shape)[1:]]
#         with open(os.getcwd()+'/new_data_process/'+csv_name, 'a', encoding='UTF8') as f:
#             writer = csv.writer(f)
#             writer.writerow(row)


# find the maxmum and minmum for all scans and liner scans

print('done')        



import pandas as pd
df = pd.read_csv(os.getcwd()+'/new_data_process'+'/'+csv_name)
volume_ori = df['volume_size_origin_coordinates']#df.column_name #you can also use df['column_name']
volume_opt = df['volume_size_optimised_coordinates']
scan_name = df['scan_name']

volume_ori_num = np.zeros((len(volume_ori),3))
volume_opt_num = np.zeros((len(volume_opt),3))

for i in range(len(volume_ori)):
    volume_ori_num[i,0] = int(volume_ori[i].split('[')[1].split(']')[0].split(',')[0])
    volume_ori_num[i,1] = int(volume_ori[i].split('[')[1].split(']')[0].split(',')[1])
    volume_ori_num[i,2] = int(volume_ori[i].split('[')[1].split(']')[0].split(',')[2])

for i in range(len(volume_opt_num)):
    volume_opt_num[i,0] = int(volume_opt[i].split('[')[1].split(']')[0].split(',')[0])
    volume_opt_num[i,1] = int(volume_opt[i].split('[')[1].split(']')[0].split(',')[1])
    volume_opt_num[i,2] = int(volume_opt[i].split('[')[1].split(']')[0].split(',')[2])

volume_ori = volume_ori_num
volume_opt = volume_opt_num

# plot
plot_figs(volume_ori,volume_opt,folder_name, 'all_data')


min_x_ori,min_y_ori,min_z_ori,max_x_ori,max_y_ori,max_z_ori,min_x_opt,min_y_opt,min_z_opt,max_x_opt,max_y_opt,max_z_opt,\
max_x_ori_name,max_y_ori_name,max_z_ori_name,\
max_x_opt_name,max_y_opt_name,max_z_opt_name\
= min_max(volume_ori,volume_opt,scan_name)



with open(os.getcwd()+'/new_data_process'+'/'+csv_name_anaysis, 'a', encoding='UTF8') as f:
    writer = csv.writer(f)
    row1 = ['original volume min x:',min_x_ori]
    row2 = ['original volume min y:',min_y_ori]
    row3 = ['original volume min z:',min_z_ori]

    row4 = ['original volume max x:',max_x_ori,max_x_ori_name]
    row5 = ['original volume max y:',max_y_ori,max_y_ori_name]
    row6 = ['original volume max z:',max_z_ori,max_z_ori_name]
    row = ['\n']
    row7 = ['optimal volume min x:',min_x_opt]
    row8 = ['optimal volume min y:',min_y_opt]
    row9 = ['optimal volume min z:',min_z_opt]

    row10 = ['optimal volume max x:',max_x_opt,max_x_opt_name]
    row11 = ['optimal volume max y:',max_y_opt,max_y_opt_name]
    row12 = ['optimal volume max z:',max_z_opt,max_z_opt_name]

    writer.writerow(row1)
    writer.writerow(row2)
    writer.writerow(row3)
    writer.writerow(row4)
    writer.writerow(row5)
    writer.writerow(row6)
    writer.writerow(row)
    writer.writerow(row7)
    writer.writerow(row8)
    writer.writerow(row9)
    writer.writerow(row10)
    writer.writerow(row11)
    writer.writerow(row12)


# pick up scans Ver_L and Par_L
Ver_L = []
Par_L = []  
for i in range(len(scan_name)):
    if 'Ver_L' in scan_name[i]:
        Ver_L.append(i)

    if 'Par_L' in scan_name[i]:
        Par_L.append(i)

volume_ori_Ver_L = volume_ori[Ver_L,...]
volume_opt_Ver_L = volume_opt[Ver_L,...]

plot_figs(volume_ori_Ver_L,volume_opt_Ver_L,folder_name, '_Ver_L')

min_x_ori_Ver_L,min_y_ori_Ver_L,min_z_ori_Ver_L,max_x_ori_Ver_L,max_y_ori_Ver_L,max_z_ori_Ver_L,min_x_opt_Ver_L,min_y_opt_Ver_L,min_z_opt_Ver_L,max_x_opt_Ver_L,max_y_opt_Ver_L,max_z_opt_Ver_L,\
max_x_ori_name_Ver_L,max_y_ori_name_Ver_L,max_z_ori_name_Ver_L,\
max_x_opt_name_Ver_L,max_y_opt_name_Ver_L,max_z_opt_name_Ver_L\
= min_max(volume_ori_Ver_L,volume_opt_Ver_L,scan_name)

with open(os.getcwd()+'/new_data_process'+'/'+csv_name_anaysis, 'a', encoding='UTF8') as f:
    writer = csv.writer(f)
    row1 = ['Ver_L original volume min x:',min_x_ori_Ver_L]
    row2 = ['Ver_L original volume min y:',min_y_ori_Ver_L]
    row3 = ['Ver_L original volume min z:',min_z_ori_Ver_L]

    row4 = ['Ver_L original volume max x:',max_x_ori_Ver_L,max_x_ori_name_Ver_L]
    row5 = ['Ver_L original volume max y:',max_y_ori_Ver_L,max_y_ori_name_Ver_L]
    row6 = ['Ver_L original volume max z:',max_z_ori_Ver_L,max_z_ori_name_Ver_L]
    row = ['\n']
    row7 = ['Ver_L optimal volume min x:',min_x_opt_Ver_L]
    row8 = ['Ver_L optimal volume min y:',min_y_opt_Ver_L]
    row9 = ['Ver_L optimal volume min z:',min_z_opt_Ver_L]

    row10 = ['Ver_L optimal volume max x:',max_x_opt_Ver_L,max_x_opt_name_Ver_L]
    row11 = ['Ver_L optimal volume max y:',max_y_opt_Ver_L,max_y_opt_name_Ver_L]
    row12 = ['Ver_L optimal volume max z:',max_z_opt_Ver_L,max_z_opt_name_Ver_L]

    writer.writerow(row)
    writer.writerow(row1)
    writer.writerow(row2)
    writer.writerow(row3)
    writer.writerow(row4)
    writer.writerow(row5)
    writer.writerow(row6)
    writer.writerow(row)
    writer.writerow(row7)
    writer.writerow(row8)
    writer.writerow(row9)
    writer.writerow(row10)
    writer.writerow(row11)
    writer.writerow(row12)




volume_ori_Par_L = volume_ori[Par_L,...]
volume_opt_Par_L = volume_opt[Par_L,...]

plot_figs(volume_ori_Par_L,volume_opt_Par_L,folder_name, '_Par_L')

min_x_ori_Par_L,min_y_ori_Par_L,min_z_ori_Par_L,max_x_ori_Par_L,max_y_ori_Par_L,max_z_ori_Par_L,min_x_opt_Par_L,min_y_opt_Par_L,min_z_opt_Par_L,max_x_opt_Par_L,max_y_opt_Par_L,max_z_opt_Par_L,\
max_x_ori_name_Par_L,max_y_ori_name_Par_L,max_z_ori_name_Par_L,\
max_x_opt_name_Par_L,max_y_opt_name_Par_L,max_z_opt_name_Par_L\
= min_max(volume_ori_Par_L,volume_opt_Par_L,scan_name)

with open(os.getcwd()+'/new_data_process'+'/'+csv_name_anaysis, 'a', encoding='UTF8') as f:
    writer = csv.writer(f)
    row1 = ['Par_L original volume min x:',min_x_ori_Par_L]
    row2 = ['Par_L original volume min y:',min_y_ori_Par_L]
    row3 = ['Par_L original volume min z:',min_z_ori_Par_L]

    row4 = ['Par_L original volume max x:',max_x_ori_Par_L,max_x_ori_name_Par_L]
    row5 = ['Par_L original volume max y:',max_y_ori_Par_L,max_y_ori_name_Par_L]
    row6 = ['Par_L original volume max z:',max_z_ori_Par_L,max_z_ori_name_Par_L]
    row = ['\n']
    row7 = ['Par_L optimal volume min x:',min_x_opt_Par_L]
    row8 = ['Par_L optimal volume min y:',min_y_opt_Par_L]
    row9 = ['Par_L optimal volume min z:',min_z_opt_Par_L]

    row10 = ['Par_L optimal volume max x:',max_x_opt_Par_L,max_x_opt_name_Par_L]
    row11 = ['Par_L optimal volume max y:',max_y_opt_Par_L,max_y_opt_name_Par_L]
    row12 = ['Par_L optimal volume max z:',max_z_opt_Par_L,max_z_opt_name_Par_L]

    writer.writerow(row)
    writer.writerow(row1)
    writer.writerow(row2)
    writer.writerow(row3)
    writer.writerow(row4)
    writer.writerow(row5)
    writer.writerow(row6)
    writer.writerow(row)
    writer.writerow(row7)
    writer.writerow(row8)
    writer.writerow(row9)
    writer.writerow(row10)
    writer.writerow(row11)
    writer.writerow(row12)


