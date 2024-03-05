import torch.nn as nn
import torch,os
import json
from scipy import stats
from torch.nn import MSELoss
from loss import PointDistance_3
from visualizer_rec_reg import Visualizer_plot_volume
from visualizer_rec_reg_non_common import Visualizer_plot_volume_non_common

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from transform import Transform_to_Params
from matplotlib import cm
import torch.nn as nn
from loader_isbi_large_dataset import SSFrameDataset
from network_isbi import build_model
from loss import PointDistance, MTL_loss
from data.calib import read_calib_matrices
from transform import LabelTransform, PredictionTransform, ImageTransform
from utils_isbi import pair_samples, reference_image_points, type_dim,compute_plane_normal,angle_between_planes
from options.train_options_rec_reg import TrainOptions
# from utils_isbi import add_scalars,save_best_network_rec_reg
from utils_4_test_ipcai import sample_dists4plot,str2list

from utilits_grid_data import *
from utils_rec_reg import *

# from monai.networks.nets.voxelmorph import VoxelMorphUNet, VoxelMorph
# from monai.transforms import DivisiblePad
# from monai.losses import BendingEnergyLoss



opt = TrainOptions().parse()
opt_test = opt

writer = SummaryWriter(os.path.join(opt.SAVE_PATH))
os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_ids
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_pairs = data_pairs_adjacent(opt.NUM_SAMPLES)
data_pairs=torch.tensor(data_pairs)

opt.FILENAME_VAL=opt.FILENAME_VAL+'_seqlen'+str(opt.NUM_SAMPLES)+'_'+opt.split_type+'_'+opt.train_set+'.json'
opt.FILENAME_TEST=opt.FILENAME_TEST+'_seqlen'+str(opt.NUM_SAMPLES)+'_'+opt.split_type+'_'+opt.train_set+'.json'
opt.FILENAME_TRAIN=[opt.FILENAME_TRAIN[i]+'_seqlen'+str(opt.NUM_SAMPLES)+'_'+opt.split_type+'_'+opt.train_set+'.json' for i in range(len(opt.FILENAME_TRAIN))]

dset_val = SSFrameDataset.read_json(Path(os.getcwd()).parent.absolute().as_posix()+opt.DATA_PATH,opt.FILENAME_VAL,opt.h5_file_name,num_samples = -1)
dset_test = SSFrameDataset.read_json(Path(os.getcwd()).parent.absolute().as_posix()+opt.DATA_PATH,opt.FILENAME_TEST,opt.h5_file_name,num_samples = -1)
dset_train_list = [SSFrameDataset.read_json(Path(os.getcwd()).parent.absolute().as_posix()+opt.DATA_PATH,opt.FILENAME_TRAIN[i],opt.h5_file_name,num_samples = -1) for i in range(len(opt.FILENAME_TRAIN))]

dset_train = dset_train_list[0]+dset_train_list[1]#+dset_train_list[2]
dset_val = dset_val+dset_train_list[2]
print('using %s'%opt.h5_file_name)

# opt.SAVE_PATH = '/raid/candi/Qi/tracked-ultrasound/seq_len100__efficientnet_b1__lr0.0001__scan_len108__output_parameter__Loss_MSE_points__forth__optimised_coord'





# used when using bash shell to pass parameters
if opt_test.use_bash_shell:
    # convert string that passed by bash shell to list
    opt_test.PAIR_INDEX = str2list(opt_test.PAIR_INDEX)



viridis = cm.get_cmap('viridis', len(opt_test.PAIR_INDEX))

# models_name = ['model_epoch00000100','model_reg_epoch00000100']

options = ['generate_reg_volume_data','reconstruction_vlume']
# folders = ['seq_len100__efficientnet_b1__lr0.0001_0.0001__scan_len108__output_parameter__Loss_rec_reg__forth__optimised_coord__fixed_interval__InitialHalf__BNoff__bs_1__pro_coord__in_ch_reg_1__Move_baseline_311',
# folders = ['seq_len100__efficientnet_b1__lr0.0001_0.0001__scan_len108__output_parameter__Loss_rec_reg__forth__optimised_coord__fixed_interval__InitialHalf__BNoff__bs_4__pro_coord__inc_reg_1__Move_baseline_311__multiGPU']

test_folders = 'final_models_non_common'
csv_name = 'non_common_metrics.csv'
fd_name_save = 'non_common_2'

folders = [f for f in os.listdir(os.getcwd()+'/'+test_folders) if f.startswith('seq_len')  and not os.path.isfile(os.path.join(os.getcwd()+'/'+test_folders, f))]
folders = sorted(folders)

# folders = ['seq_len100__efficientnet_b1__lr0.0001_0.0001__scan_len108__output_parameter__Loss_rec_reg__forth__optimised_coord__fixed_interval__iteratively__meta__InitialBest__BNoff__bs_1__pro_coord__inc_reg_1__Move_test']

# csv file
if not os.path.exists(os.getcwd()+'/'+test_folders+'/'+fd_name_save):
    os.makedirs(os.getcwd()+'/'+test_folders+'/'+fd_name_save)

with open(os.getcwd()+'/'+test_folders+'/'+fd_name_save+'/'+csv_name, 'a', encoding='UTF8') as f:
    writer = csv.writer(f)
    # writer.writerow(['the volume of each sequence'])
    writer.writerow(['file_name','model_name','points dist on all points using global T (mean.std) & ','points dist on all points using global T+R (prdiction + DDF)(mean.std) & '\
                     'points dist on 4 points using global T (mean.std) & ','points dist on 4 points using global T+R (prdiction + DDF)(mean.std) & '\
                      'points dist on all points using local T (mean.std) & ',\
                        'points dist on 4 points using local T (mean.std) & '])

for sub_fd in folders:
    # get paramteters from folder name
    print(sub_fd)
    sub_fd_split = sub_fd.split('__')
    fn = sub_fd.split('__')[0]
    opt.NUM_SAMPLES = int(fn[len(fn.rstrip('0123456789')):])
    opt.LEARNING_RATE_rec = None #float(sub_fd_split[2].split('_')[0][2:])
    opt.LEARNING_RATE_reg = None #float(sub_fd_split[2].split('_')[1])
    opt.Loss_type = None #sub_fd_split[5].split('_',1)[1]
    # opt.initial = sub_fd_split[9]
    # ind_bs = None
    # for i in range(len(sub_fd_split)):
    #     if 'bs' in sub_fd_split[i]:
    #         ind_bs = i
        
    # try:
    #     opt.MINIBATCH_SIZE_rec = int(sub_fd_split[ind_bs].split('_')[1])
    #     opt.MINIBATCH_SIZE_reg = opt.MINIBATCH_SIZE_rec
    # except:
    #     raise('Not batchsize information can be founded')
    # print('opt.MINIBATCH_SIZE_rec = %d'%opt.MINIBATCH_SIZE_rec)
    opt.MINIBATCH_SIZE_rec = None
    opt.MINIBATCH_SIZE_reg = None
    
    ind_inc = None
    for i in range(len(sub_fd_split)):
        if 'inc_reg' in sub_fd_split[i] or 'in_ch_reg' in sub_fd_split[i]:
            ind_inc = i
    try:
        opt.in_ch_reg = int(sub_fd_split[ind_inc].split('_')[-1])
        
    except:
            
        opt.in_ch_reg = 2
    print('opt.in_ch_reg = %d'%opt.in_ch_reg)

    opt.ddf_dirc = None
    if 'Move' in sub_fd:
        opt.ddf_dirc = 'Move'
    
    print('opt.ddf_dirc = %s'%opt.ddf_dirc)


    opt.saved_results = os.getcwd()+'/'+test_folders+'/'+sub_fd
    opt.SAVE_PATH = opt.saved_results

    models_all = [f for f in os.listdir(opt.SAVE_PATH+'/saved_model') if f.startswith('best_val_dist') and os.path.isfile(os.path.join(opt.SAVE_PATH+'/saved_model', f))]
    
    if len(models_all)==4:
        # non-meta training models
        models_name = [['best_val_dist_R_T','best_val_dist_R_R'],['best_val_dist_T_T','best_val_dist_T_R']]
    
    elif len(models_all)==8:
        # meta training models
        models_name = [['best_val_dist_R_R_T','best_val_dist_R_R_R'],\
                       ['best_val_dist_R_T_T','best_val_dist_R_T_R'],\
                        ['best_val_dist_T_R_T','best_val_dist_T_R_R'],\
                       ['best_val_dist_T_T_T','best_val_dist_T_T_R']]                 
    
    elif len(models_all)==0:
        # previous trained model
        models_pre = [f for f in os.listdir(opt.SAVE_PATH+'/saved_model') if f.startswith('best_validation') and os.path.isfile(os.path.join(opt.SAVE_PATH+'/saved_model', f))]
        if len(models_pre) ==4:
            models_name = [['best_validation_loss_model','best_validation_loss_model_reg'],\
                           ['best_validation_dist_model','best_validation_dist_model_reg']]
        elif len(models_pre) ==2:
            models_name = [['best_validation_loss_model'],['best_validation_dist_model']]



    else:
        raise('Not implenment')




    for i_m in range(len(models_name)):
        model_name = models_name[i_m]

        
        saved_folder_test = os.path.join(os.getcwd()+'/'+test_folders+'/'+fd_name_save+'/'+sub_fd, str(model_name)+'__TestSet')
        
        if not os.path.exists(saved_folder_test):
            os.makedirs(saved_folder_test)
    


        # visualizer_scan_train = Visualizer_plot_volume(opt,device, dset_train,model_name,data_pairs,options[1])
        # visualizer_scan_val = Visualizer_plot_volume(opt,device, dset_val,model_name,data_pairs,options[1])
        visualizer_scan_test = Visualizer_plot_volume_non_common(opt,device, dset_test,model_name,data_pairs,options[1])

        # test - test set
        for scan_index in range(len(dset_test)):
            # scan_index = 1
            visualizer_scan_test.generate_volume_data(
                                                        scan_index,
                                                        saved_folder_test,
                                                        based_volume = 'common_volume',# use common volume to reconstruct volume both for registartion use and for final vlulisation use
                                                        local_global = 'global', # use global transformation or local transformation
                                                        pts_num = 'all'
                                                        ) # use all points or 4 corner points
        # save value for future use
        metric1 = np.array(visualizer_scan_test.T_Global_AllPts_Dist)[None,...]
        metric2 = np.array(visualizer_scan_test.T_R_Warp_Global_AllPts_Dist)[None,...]
        metric3 = np.array(visualizer_scan_test.T_Global_FourPts_Dist)[None,...]
        metric4 = np.array(visualizer_scan_test.T_R_Warp_Global_FourPts_Dist)[None,...]
        metric5 = np.array(visualizer_scan_test.T_Local_AllPts_Dist)[None,...]
        metric6 = np.array(visualizer_scan_test.T_Local_FourPts_Dist)[None,...]
        
        metrics = np.concatenate((metric1, metric2, metric3, metric4,metric5, metric6), axis=0)

        with open(os.getcwd()+'/'+test_folders+'/'+fd_name_save+'/'+sub_fd[29:] + '__' + str(model_name)+'_non_common.npy', 'wb') as f:
            np.save(f, metrics)
        
        
        with open(os.getcwd()+'/'+test_folders+'/'+fd_name_save+'/'+csv_name, 'a', encoding='UTF8') as f:
            f.write(sub_fd)
            f.write(' &  ')
            f.write(str(model_name))
            f.write(' &  ')
            f.write('$%.2f\pm%.2f$ &  ' % (np.array(visualizer_scan_test.T_Global_AllPts_Dist).mean(), np.std(np.array(visualizer_scan_test.T_Global_AllPts_Dist))))
            f.write('$%.2f\pm%.2f$ &  ' % (np.array(visualizer_scan_test.T_R_Warp_Global_AllPts_Dist).mean(), np.std(np.array(visualizer_scan_test.T_R_Warp_Global_AllPts_Dist))))
            
            f.write('$%.2f\pm%.2f$ &  ' % (np.array(visualizer_scan_test.T_Global_FourPts_Dist).mean(), np.std(np.array(visualizer_scan_test.T_Global_FourPts_Dist))))
            f.write('$%.2f\pm%.2f$ &  ' % (np.array(visualizer_scan_test.T_R_Warp_Global_FourPts_Dist).mean(), np.std(np.array(visualizer_scan_test.T_R_Warp_Global_FourPts_Dist))))

            f.write('$%.2f\pm%.2f$ &  ' % (np.array(visualizer_scan_test.T_Local_AllPts_Dist).mean(), np.std(np.array(visualizer_scan_test.T_Local_AllPts_Dist))))
            f.write('$%.2f\pm%.2f$ &  ' % (np.array(visualizer_scan_test.T_Local_FourPts_Dist).mean(), np.std(np.array(visualizer_scan_test.T_Local_FourPts_Dist))))

            f.write('\n')
        

# get statistic significance
npys = [f for f in os.listdir(os.getcwd()+'/'+test_folders+'/'+fd_name_save) if f.endswith('.npy')]
sorted(npys)
# get all metrics for all models in a variable
metric_all = {}
for i_npy in range(len(npys)):  
    with open(os.getcwd()+'/'+test_folders+'/'+fd_name_save + '/' + npys[i_npy], 'rb') as f:
        metric = np.load(f)
        metric_all[npys[i_npy]] = metric
        
        # try:
        #     metric_all[i] = metric
        # except:
        #     metric_all = torch.zeros(len(npys),metric.shape) # [model_num, metric_num, scan_num]

# calculate p-value
baseline_idx = 0#38

with open(os.getcwd()+'/'+test_folders+'/'+fd_name_save+'/'+csv_name, 'a', encoding='UTF8') as f:
    f.write('\n')
    f.write('baseline: stats.ttest_rel   ')
    f.write('\n')
    f.write(str(list(metric_all.keys())[baseline_idx]))
    f.write('\n')
    f.write('mean:\n')
    f.write(str(np.mean(list(metric_all.values())[baseline_idx],1)))
    f.write('\n')
    f.write('std:\n')
    f.write(str(np.std(list(metric_all.values())[baseline_idx],1)))
    f.write('\n')
    f.write('\n')


for i in range(len(metric_all)):
    if i != baseline_idx:
        with open(os.getcwd()+'/'+test_folders+'/'+fd_name_save+'/'+csv_name, 'a', encoding='UTF8') as f:
            f.write(str(list(metric_all.keys())[i]))
            f.write('\n')
            f.write('mean:\n')
            f.write(str(np.mean(list(metric_all.values())[i],1)))
            f.write('\n')
            f.write('std:\n')
            f.write(str(np.std(list(metric_all.values())[i],1)))
            f.write('\n')

            for num_m in range(list(metric_all.values())[i].shape[0]): 
                f.write(str(stats.ttest_rel(list(metric_all.values())[i][num_m], list(metric_all.values())[baseline_idx][num_m])[1]))
                f.write('\n') 
        

# str(stats.ttest_rel(list(metric_all.values())[9][0], list(metric_all.values())[16][0]))

            



print('done')

# test - training set
# for scan_index in range(len(dset_train)):
#     visualizer_scan_train.generate_volume_data(scan_index,saved_folder_train)

# test - val set
# for scan_index in range(len(dset_val)):
#     visualizer_scan_val.plot_scan_and_cal_ave_dist(scan_index,saved_folder_val)





