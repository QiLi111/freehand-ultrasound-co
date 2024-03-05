
import torch
import heapq
import MDAnalysis.lib.transformations as MDA
import numpy as np
import pytorch3d.transforms
import pandas as pd
import os
import json,pickle
import random
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from loss import PointDistance
from loader import SSFrameDataset
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from metric import frame_volume_overlap
# from colorspacious import cspace_converter


# def plot_color_gradients(category, cmap_list):
#     cmaps = {}
#
#     gradient = np.linspace(0, 1, 256)
#     gradient = np.vstack((gradient, gradient))
#     Create figure and adjust figure height to number of colormaps
#     nrows = len(cmap_list)
#     figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
#     fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
#     fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
#                         left=0.2, right=0.99)
#     axs[0].set_title(f'{category} colormaps', fontsize=14)
#
#     for ax, name in zip(axs, cmap_list):
#         ax.imshow(gradient, aspect='auto', cmap=mpl.colormaps[name])
#         ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
#                 transform=ax.transAxes)
#
#     Turn off *all* ticks & spines, not just the ones with colormaps.
#     for ax in axs:
#         ax.set_axis_off()
#
#     # Save colormap list for later.
#     cmaps[category] = cmap_list

def pair_samples(num_samples, num_pred, single_interval):
    """
    :param num_samples:
    :param num_pred: number of the (last) samples, for which the transformations are predicted
        For each "pred" frame, pairs are formed with every one previous frame 
    :param single_interval: 0 - use all interval predictions
                            1,2,3,... - use only specific intervals
    """

    if single_interval == 0:
        return torch.tensor([[n0,n1] for n1 in range(num_samples-num_pred,num_samples) for n0 in range(n1)])
    else:
        return torch.tensor([[n1-single_interval,n1] for n1 in range(single_interval,num_samples,single_interval) ])


def type_dim(label_pred_type, num_points=None, num_pairs=1):
    type_dim_dict = {
        "transform": 12,
        "parameter": 6,
        "point": num_points*3
    }
    return type_dim_dict[label_pred_type] * num_pairs  # num_points=self.image_points.shape[1]), num_pairs=self.pairs.shape[0]


def reference_image_points(image_size, density=2):
    """
    :param image_size: (x, y), used for defining default grid image_points
    :param density: (x, y), point sample density in each of x and y, default n=2
    """
    if isinstance(density,int):
        density=(density,density)

    # image_points = torch.cartesian_prod(
    #     torch.linspace(-image_size[0]/2,image_size[0]/2,density[0]),
    #     torch.linspace(-image_size[1]/2,image_size[1]/2,density[1])
    #     ).t()  # transpose to 2-by-n

    image_points = torch.cartesian_prod(
        torch.linspace(0, image_size[0] , density[0]),
        torch.linspace(0, image_size[1], density[1])
    ).t()
    
    image_points = torch.cat([
        image_points, 
        torch.zeros(1,image_points.shape[1])*image_size[0]/2,
        torch.ones(1,image_points.shape[1])
        ], axis=0)
    
    return image_points



def tran624_np(parameter,seq= 'rzyx'):
    '''
    for numpy use: 6 parameter --> 4*4 transformation matrix
    :param parameter: numpy type, [6], angle_x, angle_y, angle_z, x, y, z
    :param seq: e.g.,'rxyz'
    :return: transform: 4*4 transformation matrix
    '''

    transform =MDA.euler_matrix(parameter[0], parameter[1], parameter[2], seq)

    transform[0,3]=parameter[3]
    transform[1,3]=parameter[4]
    transform[2,3]=parameter[5]
    transform[3,3]=1

    return transform

def tran426_np(transform,seq= 'rzyx'):
    '''
    :param transform: numpy type, 4*4 transformation matrix
    :param seq: e.g.,'rxyz'
    :return: parameter: [6], angle_x, angle_y, angle_z, x, y, z
    '''
    parameter = np.zeros(6)
    parameter[0:3]=MDA.euler_from_matrix(transform,seq)
    parameter[3]=transform[0,3]
    parameter[4]=transform[1,3]
    parameter[5]=transform[2,3]

    return parameter

def tran624_tensor(parameter, seq = 'ZYX'):
    '''
    # for tensor use: 6 parameter --> 4*4 transformation matrix
    # this can preserve grad in tensor which need to be backforward
    :param parameter: tensor type, [6], angle_x, angle_y, angle_z, x, y, z
    :param seq: e.g.,'XYZ'
    :return: transform: 4*4 transformation matrix
    '''
    Rotation = pytorch3d.transforms.euler_angles_to_matrix(parameter[0:3], seq)
    transform = torch.row_stack((torch.column_stack((Rotation, torch.t(parameter[3:6]))), torch.tensor([0, 0, 0, 1])))

    return transform

def tran426_tensor(transform,seq = 'ZYX'):
    '''
    # this can preserve grad in tensor which need to be backforward
    :param transform:tensor type, 4*4 transformation matrix
    :param seq: e.g.,'XYZ'
    :return: parameter: [6], angle_x, angle_y, angle_z, x, y, z
    '''
    Rotation = pytorch3d.transforms.matrix_to_euler_angles(transform[0:3, 0:3], seq)
    parameter = torch.cat((Rotation, transform[0:3, 3]))
    return parameter

def save_best_network(opt, model, epoch_label, running_loss_val, running_dist_val, val_loss_min, val_dist_min):
    '''

    :param opt: parameters of this projects
    :param model: model that need to be saved
    :param epoch_label: epoch of this model
    :param running_loss_val: validation loss of this epoch
    :param running_dist_val: validation distance of this epoch
    :param val_loss_min: min of previous validation loss
    :param val_dist_min: min of previous validation distance
    :return:
    '''

    if running_loss_val < val_loss_min:
        val_loss_min = running_loss_val
        file_name = os.path.join(opt.SAVE_PATH, 'config.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write('------------ best validation loss result - epoch %s: -------------\n' % (str(epoch_label)))

        torch.save(model.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'best_validation_loss_model' ))
        print('Best validation loss parameters saved.')
    else:
        val_loss_min = val_loss_min

    if running_dist_val < val_dist_min:
        val_dist_min = running_dist_val
        file_name = os.path.join(opt.SAVE_PATH, 'config.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write('------------ best validation dist result - epoch %s: -------------\n' % (str(epoch_label)))

        torch.save(model.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'best_validation_dist_model'))
        print('Best validation dist parameters saved.')
    else:
        val_dist_min = val_dist_min

    return val_loss_min, val_dist_min

def get_interval(opt, data_pairs):
    if opt.NUM_SAMPLES == 7 and opt.SAMPLE_RANGE == 7 and opt.NUM_PRED == 6 :
        interval_1 = sample_adjacent_pair(start=0, step=2, data_pairs=data_pairs)
        interval_2 = sample_adjacent_pair(start=1, step=3, data_pairs=data_pairs)[0::2]
        interval_3 = sample_adjacent_pair(start=3, step=4, data_pairs=data_pairs)[0::3]
        interval_6 = sample_adjacent_pair(start=15, step=7, data_pairs=data_pairs)[0::6]

        interval = {'0': interval_1, '1': interval_2, '2': interval_3, '3': interval_6}

        # if opt.single_interval == 1:
        #     interval_1 = sample_adjacent_pair(start=0, step=1, data_pairs=data_pairs)
        #     interval_2 = sample_adjacent_pair(start=0, step=2, data_pairs=data_pairs)
        #     interval_3 = sample_adjacent_pair(start=0, step=3, data_pairs=data_pairs)
        #     interval_6 = sample_adjacent_pair(start=0, step=6, data_pairs=data_pairs)
        #     interval = {'0': interval_1, '1': interval_2, '2': interval_3, '3': interval_6}
        #
        # if opt.single_interval == 2:
        #     interval_2 = sample_adjacent_pair(start=0, step=1, data_pairs=data_pairs)
        #     interval_4 = sample_adjacent_pair(start=0, step=2, data_pairs=data_pairs)
        #     interval_6 = sample_adjacent_pair(start=0, step=3, data_pairs=data_pairs)
        #     interval = {'0': interval_2, '1': interval_4, '2': interval_6}
        #
        # if opt.single_interval == 3:
        #     interval_3 = sample_adjacent_pair(start=0, step=1, data_pairs=data_pairs)
        #     interval_6 = sample_adjacent_pair(start=0, step=2, data_pairs=data_pairs)
        #     interval = {'0': interval_3, '1': interval_6}


    elif opt.NUM_SAMPLES == 36 and opt.SAMPLE_RANGE == 36 and opt.NUM_PRED == 35:

        interval_1 = sample_adjacent_pair(start=0, step=2, data_pairs=data_pairs)
        interval_5 = sample_adjacent_pair(start=10, step=6, data_pairs=data_pairs)[0::5]
        interval_7 = sample_adjacent_pair(start=21, step=8, data_pairs=data_pairs)[0::7]
        interval_35 = sample_adjacent_pair(start=595, step=36, data_pairs=data_pairs)[0::35]

        interval = {'0': interval_1, '1': interval_5, '2': interval_7, '3': interval_35}

    elif opt.NUM_SAMPLES == 97 and opt.SAMPLE_RANGE == 97 and opt.NUM_PRED == 96:

        interval_1 = sample_adjacent_pair(start=0, step=2, data_pairs=data_pairs)
        interval_2 = sample_adjacent_pair(start=1, step=3, data_pairs=data_pairs)[0::2]
        interval_3 = sample_adjacent_pair(start=3, step=4, data_pairs=data_pairs)[0::3]
        interval_4 = sample_adjacent_pair(start=6, step=5, data_pairs=data_pairs)[0::4]
        interval_6 = sample_adjacent_pair(start=15, step=7, data_pairs=data_pairs)[0::6]
        interval_8 = sample_adjacent_pair(start=28, step=9, data_pairs=data_pairs)[0::8]
        interval_12 = sample_adjacent_pair(start=66, step=13, data_pairs=data_pairs)[0::12]
        interval_16 = sample_adjacent_pair(start=120, step=17, data_pairs=data_pairs)[0::16]
        interval_24 = sample_adjacent_pair(start=276, step=25, data_pairs=data_pairs)[0::24]
        interval_32 = sample_adjacent_pair(start=496, step=33, data_pairs=data_pairs)[0::32]
        interval_48 = sample_adjacent_pair(start=1128, step=49, data_pairs=data_pairs)[0::48]
        interval_96 = sample_adjacent_pair(start=4560, step=97, data_pairs=data_pairs)[0::96]

        interval = {'0': interval_1, '1': interval_2, '2': interval_3, '3': interval_4,
                    '4': interval_6, '5': interval_8, '6': interval_12, '7': interval_16,
                    '8': interval_24, '9': interval_32, '10': interval_48, '11': interval_96
                    }
    elif opt.NUM_SAMPLES == 30 and opt.SAMPLE_RANGE == 30 and opt.NUM_PRED == 29:
        added_pairs = torch.tensor([[0, 29], [0, 15], [15, 29]])
        # find the interval index
        saved_flag = []
        for j in range(len(added_pairs)):
            for i, e in enumerate(data_pairs.cpu()):
                if (e == added_pairs[j, :]).all():
                    saved_flag.append(i)

        interval= {'0': [saved_flag[0]],'1':[saved_flag[1],saved_flag[2]]}
    elif opt.NUM_SAMPLES == 20 and opt.SAMPLE_RANGE == 20 and opt.NUM_PRED == 19:
        added_pairs = torch.tensor([[0, 19], [0, 10], [10, 19]])
        # find the interval index
        saved_flag = []
        for j in range(len(added_pairs)):
            for i, e in enumerate(data_pairs.cpu()):
                if (e == added_pairs[j, :]).all():
                    saved_flag.append(i)

        interval= {'0': [saved_flag[0]],'1':[saved_flag[1],saved_flag[2]]}


    else:
        interval = {'0': [0]}
    return interval

def sample_adjacent_pair(start, step, data_pairs):
    adjacent_pair = []
    while 1:
        adjacent_pair.append(start)
        start = start + step
        step = step + 1
        if start >= data_pairs.shape[0]:
            break
    return adjacent_pair # data_pairs[adjacent_pair]

def add_scalars(writer,epoch, loss_dists,preds_dist_all_train, label_dist_all_train,preds_dist_all_val,label_dist_all_val,data_pairs,opt,data_pairs_samples_index):
    # loss and average distance in training and val
    train_epoch_loss = loss_dists['train_epoch_loss']
    train_epoch_dist = loss_dists['train_epoch_dist']
    epoch_loss_val = loss_dists['epoch_loss_val']
    epoch_dist_val = loss_dists['epoch_dist_val']

    for i in range(len(preds_dist_all_train)):
        dist_train=(((preds_dist_all_train[str(i)]-label_dist_all_train[str(i)])**2).sum(dim=1).sqrt().mean())
        dist_val=(((preds_dist_all_val[str(i)]-label_dist_all_val[str(i)])**2).sum(dim=1).sqrt().mean())
        writer.add_scalars('accumulated_dists', {'train_%d' % i: dist_train.item()}, epoch)
        writer.add_scalars('accumulated_dists', {'val_%d' % i: dist_val.item()}, epoch)

    writer.add_scalars('loss', {'train_loss': train_epoch_loss},epoch)
    writer.add_scalars('loss', {'val_loss': epoch_loss_val},epoch)
    writer.add_scalars('dist', {'train_dist': train_epoch_dist.sum()}, epoch)
    writer.add_scalars('dist', {'val_dist': epoch_dist_val.sum()}, epoch)

    # add dists to scalars, seperatatly dist and each dist is divided by the interval
    if len(train_epoch_dist)<opt.MAXNUM_PAIRS or len(data_pairs)==len(data_pairs_samples_index):
        for i in range(len(train_epoch_dist)):
            writer.add_scalars('dists', {'train_%s' % str(data_pairs[i][0].item())+'_'+str(data_pairs[i][1].item()): train_epoch_dist[i]/(data_pairs[i][1]-data_pairs[i][0])},epoch)
            writer.add_scalars('dists', {'val_%s' % str(data_pairs[i][0].item())+'_'+str(data_pairs[i][1].item()): epoch_dist_val[i]/(data_pairs[i][1]-data_pairs[i][0])},epoch)
    else:
        # obatin the monitored data paired

        for i in data_pairs_samples_index:
            writer.add_scalars('dists', {'train_%s' % str(data_pairs[data_pairs_samples_index[i]][0].item()) + '_' + str(data_pairs[data_pairs_samples_index[i]][1].item()): train_epoch_dist[i] / (data_pairs[data_pairs_samples_index[i]][1] - data_pairs[data_pairs_samples_index[i]][0])}, epoch)
            writer.add_scalars('dists', {'val_%s' % str(data_pairs[data_pairs_samples_index[i]][0].item()) + '_' + str(data_pairs[data_pairs_samples_index[i]][1].item()): epoch_dist_val[i] / (data_pairs[data_pairs_samples_index[i]][1] - data_pairs[data_pairs_samples_index[i]][0])}, epoch)


def add_scalars_loss(writer, epoch, loss_dists):
    # loss and average distance in training and val
    train_epoch_loss = loss_dists['train_epoch_loss']
    train_epoch_dist = loss_dists['train_epoch_dist']
    epoch_loss_val = loss_dists['epoch_loss_val']
    epoch_dist_val = loss_dists['epoch_dist_val']



    writer.add_scalars('loss_average_dists', {'train_loss': train_epoch_loss.item(), 'val_loss': epoch_loss_val.item()},epoch)
    writer.add_scalars('loss_average_dists',{'train_dists': train_epoch_dist.item(), 'val_dists': epoch_dist_val.item()}, epoch)


def add_scalars_params(writer,epoch,params_gt_train,params_np_train,params_gt_val,params_np_val):
#     compute the absolute error of 6 parameters
    abs_errs_train,abs_errs_val = {},{}
    for i in range(len(params_gt_train)):
        abs_errs_train[str(i)],abs_errs_val[str(i)] = None, None

    for i in range(len(params_gt_train)):
        abs_errs_train[str(i)] = torch.abs(params_gt_train[str(i)] - params_np_train[str(i)]).mean(dim=0)
        abs_errs_val[str(i)] = torch.abs(params_gt_val[str(i)] - params_np_val[str(i)]).mean(dim=0)

        writer.add_scalars('params_abs_err_angle_1', {'train_%d' % i: abs_errs_train[str(i)][0], 'val_%d' % i: abs_errs_val[str(i)][0]},epoch)
        writer.add_scalars('params_abs_err_angle_2', {'train_%d' % i: abs_errs_train[str(i)][1], 'val_%d' % i: abs_errs_val[str(i)][1]},epoch)
        writer.add_scalars('params_abs_err_angle_3', {'train_%d' % i: abs_errs_train[str(i)][2], 'val_%d' % i: abs_errs_val[str(i)][2]},epoch)
        writer.add_scalars('params_abs_err_x', {'train_%d' % i: abs_errs_train[str(i)][3], 'val_%d' % i: abs_errs_val[str(i)][3]},epoch)
        writer.add_scalars('params_abs_err_y', {'train_%d' % i: abs_errs_train[str(i)][4], 'val_%d' % i: abs_errs_val[str(i)][4]},epoch)
        writer.add_scalars('params_abs_err_z', {'train_%d' % i: abs_errs_train[str(i)][5], 'val_%d' % i: abs_errs_val[str(i)][5]},epoch)


def write_to_txt(opt,epoch, loss_dists,preds_dist_all_train, label_dist_all_train,preds_dist_all_val,label_dist_all_val):
    # write loss, average distance, accumulated distance into txt
    # for the last step in each epoch
    # loss and average distance in training and val
    train_epoch_loss = loss_dists['train_epoch_loss']
    train_epoch_dist = loss_dists['train_epoch_dist'].mean()
    epoch_loss_val = loss_dists['epoch_loss_val']
    epoch_dist_val = loss_dists['epoch_dist_val'].mean()
    dist_train,dist_val = [],[]

    for i in range(len(preds_dist_all_train)):
        dist_train.append(((preds_dist_all_train[str(i)] - label_dist_all_train[str(i)]) ** 2).sum(dim=1).sqrt().mean())
        dist_val.append(((preds_dist_all_val[str(i)]-label_dist_all_val[str(i)])**2).sum(dim=1).sqrt().mean())

    dist_train = torch.tensor(dist_train)
    dist_val = torch.tensor(dist_val)
    file_name_train = os.path.join(opt.SAVE_PATH, 'train_results', 'train_loss.txt')
    with open(file_name_train, 'a') as opt_file_train:
        print('[Epoch %d], for one epoch, train-loss=%.3f, train-dist=%.3f' % (epoch, train_epoch_loss, train_epoch_dist),file=opt_file_train)
        print('[for one epoch, %d kinds of accumulated dists]:' % (len(dist_train)),file=opt_file_train)
        print('%.3f ' * len(dist_train) % tuple(dist_train),file=opt_file_train)

    file_name_val = os.path.join(opt.SAVE_PATH, 'val_results', 'val_loss.txt')
    with open(file_name_val, 'a') as opt_file_val:
        print('[Epoch %d], for one epoch, val-loss=%.3f, val-dist=%.3f' % (epoch, epoch_loss_val, epoch_dist_val), file=opt_file_val)
        print('[for one step, %d kinds of accumulated dists]' % (len(dist_val)),file=opt_file_val)
        print('%.3f ' * len(dist_val) % tuple(dist_val),file=opt_file_val)


def write_to_txt_2(opt,num_pairs,train_epoch_dist, epoch_dist_val):
    # write all pairs of dists, for the last step of each epoch
    file_name_val = os.path.join(opt.SAVE_PATH, 'val_results', 'val_loss.txt')
    with open(file_name_val, 'a') as opt_file_val:
        # if epoch_dist_val.shape[0] > 1:
        print('[for one step, %d pairs of dists]'  % (num_pairs),file=opt_file_val)
        print('%.3f ' * epoch_dist_val.shape[0] % (tuple(epoch_dist_val)), file=opt_file_val)
        print('\n',file=opt_file_val)
    file_name_train = os.path.join(opt.SAVE_PATH, 'train_results', 'train_loss.txt')
    with open(file_name_train, 'a') as opt_file_train:
        # if train_epoch_dist.shape[0]>1: # torch.tensor([dist]).shape[0]>1
        print('[for one step, %d pairs of dists] ' % (num_pairs),file=opt_file_train)
        print(' %.3f '*train_epoch_dist.shape[0] % (tuple(train_epoch_dist)),file=opt_file_train)
        print('\n',file=opt_file_train)


def write_scan_value2text(opt, rmse_scan_var_intervals,rmse_scan_var_intervals_gt_based,train_val):
    rmse_scan_var_intervals = np.array(rmse_scan_var_intervals)
    rmse_scan_var_intervals_gt_based = np.array(rmse_scan_var_intervals_gt_based)

    if train_val == 'train':
        file_name = os.path.join(opt.SAVE_PATH, 'train_results', 'train_scan_distance.txt')
    else:
        file_name = os.path.join(opt.SAVE_PATH, 'val_results', 'val_scan_distance.txt')

    with open(file_name, 'a') as opt_file:
        print('accumulated error' , file=opt_file)
        for i in range(rmse_scan_var_intervals.shape[1]):
            print('interval_%d' %(i),file=opt_file)
            print(rmse_scan_var_intervals[:,i],file=opt_file)

        print('/*************************************************/', file=opt_file)
        print('non-accumulated error', file=opt_file)
        for i in range(rmse_scan_var_intervals_gt_based.shape[1]):
            print('interval_%d' %(i),file=opt_file)
            print(rmse_scan_var_intervals_gt_based[:,i],file=opt_file)



def plot_each_img(viridis,opt,ax2,ax3,ax4,opt_test,rmse_img_var_intervals,rmse_img_var_intervals_gt_based, train_val,model_name,dset_val,scan_index, data_pair):




    if train_val == 'train':
        saved_folder = os.path.join(opt.SAVE_PATH, 'testing_train_results')
    elif train_val == 'val':
        saved_folder = os.path.join(opt.SAVE_PATH, 'testing_val_results')

    else:
        saved_folder = os.path.join(opt.SAVE_PATH, 'testing_test_results')

    plt.savefig(saved_folder + '/'  + dset_val.name_scan[dset_val.indices_in_use[scan_index][0], dset_val.indices_in_use[scan_index][1]].decode("utf-8") + '_img_level_' + model_name  + '.png')
    plt.close()

def plot_each_scan(viridis,opt,opt_test,rmse_scan_var_intervals, rmse_scan_var_intervals_gt_based,train_val,model_name,dset_val,scan_index, data_pair,scan_len):
    rmse_scan_var_intervals = np.array(rmse_scan_var_intervals)
    rmse_scan_var_intervals_gt_based = np.array(rmse_scan_var_intervals_gt_based)
    # sort error of all scans by the length of each scan
    sort_idx = np.argsort(scan_len)
    sorted_scan_len = sorted(scan_len)
    rmse_scan_var_intervals = rmse_scan_var_intervals[sort_idx]
    rmse_scan_var_intervals_gt_based = rmse_scan_var_intervals_gt_based[sort_idx]


    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 3)
    # ax3 = fig.add_subplot(2, 2, 3)
    # ax4 = fig.add_subplot(2, 2, 4)
    for i in range(rmse_scan_var_intervals.shape[1]):
        ax1.plot(sorted_scan_len,rmse_scan_var_intervals[:,i],color=viridis.colors[i], linestyle='dashed', marker='o',  alpha=.5,label='interval_'+str(data_pair[opt_test.PAIR_INDEX[i]][0].item())+'_'+str(data_pair[opt_test.PAIR_INDEX[i]][1].item()))

    ax1.legend()
    # ax1.set_xticks(sorted_scan_len)
    ax1.set_xlabel('scan length')
    ax1.set_ylabel('accumulated error')

    for i in range(rmse_scan_var_intervals_gt_based.shape[1]):
        ax2.plot(sorted_scan_len,rmse_scan_var_intervals_gt_based[:,i]/(data_pair[opt_test.PAIR_INDEX[i]][1].item()-data_pair[opt_test.PAIR_INDEX[i]][0].item()),color=viridis.colors[i],linestyle='dashed', marker='o',  alpha=.5,label='interval_'+str(data_pair[opt_test.PAIR_INDEX[i]][0].item())+'_'+str(data_pair[opt_test.PAIR_INDEX[i]][1].item()))

    ax2.legend()
    # ax2.set_xticks(sorted_scan_len)
    ax2.set_xlabel('scan length')
    ax2.set_ylabel('non-accumulated error')

    if train_val == 'train':
        saved_folder = os.path.join(opt.SAVE_PATH, 'testing_train_results')
    elif train_val == 'val':
        saved_folder = os.path.join(opt.SAVE_PATH, 'testing_val_results')

    else:
        saved_folder = os.path.join(opt.SAVE_PATH, 'testing_test_results')

    # plt.savefig(saved_folder + '/'  + model_name +'_scan_level' + '.png')
    # plt.close()


    # B-A plot
    fig2 = plt.figure(figsize=(20, 10))
    num_fig = len(opt_test.PAIR_INDEX)
    rows = 3
    if num_fig / rows != 0:
        cols = int(num_fig / rows) + 1
    else:
        cols = int(num_fig / rows)
    data1 = np.asarray(rmse_scan_var_intervals[:, 0])
    for i in range(1, rmse_scan_var_intervals.shape[1]):
        data2 = np.asarray(rmse_scan_var_intervals[:, i])
        diff = data2 - data1
        # mean = (data2 + data1)/2
        ax1 = fig2.add_subplot(rows, cols, i)

        ax1.plot(data1, diff, color=viridis.colors[i], linestyle='none', marker='o', alpha=.5,
                 label='interval_' + str(data_pair[opt_test.PAIR_INDEX[i]][0].item()) + '_' + str(
                     data_pair[opt_test.PAIR_INDEX[i]][1].item()))
        ax1.axhline(0, color='gray', linestyle='--')
        ax1.legend()
        # ax1.set_xlabel('1st transform error')
        ax1.set_ylabel('difference')
        ax1.set_title('accumulated B_A_plot')
    plt.savefig(saved_folder + '/' + model_name + '_scan_level_accumulated_B_A_plot' + '.png')
    plt.close()

    fig3 = plt.figure(figsize=(20, 10))
    data1 = np.asarray(rmse_scan_var_intervals_gt_based[:,0]/(data_pair[opt_test.PAIR_INDEX[0]][1].item()-data_pair[opt_test.PAIR_INDEX[0]][0].item()))
    for i in range(1,rmse_scan_var_intervals_gt_based.shape[1]):
        data2 = np.asarray(rmse_scan_var_intervals_gt_based[:,i]/(data_pair[opt_test.PAIR_INDEX[i]][1].item()-data_pair[opt_test.PAIR_INDEX[i]][0].item()))
        diff = data2 - data1
        ax4 = fig3.add_subplot(rows, cols, i)
        ax4.plot(data1, diff,color=viridis.colors[i], linestyle='none', marker='o', alpha=.5,label='interval_'+str(data_pair[opt_test.PAIR_INDEX[i]][0].item())+'_'+str(data_pair[opt_test.PAIR_INDEX[i]][1].item()))
        ax4.axhline(0, color='gray', linestyle='--')
        ax4.legend()
        # ax4.set_xlabel('1st transform error')
        ax4.set_ylabel('difference')
        ax4.set_title('non-accumulated B_A_plot')


    plt.savefig(saved_folder + '/'  + model_name +'_scan_level_non_accmulated_B_A_plot' + '.png')
    plt.close()

    return fig

def plot_each_scan_avg_pixel_dist(viridis,opt,opt_test,avg_pixel_dist_all_interval_all_scan,train_val,model_name,dset_val,scan_index, data_pair,scan_len):
    avg_pixel_dist_all_interval_all_scan = np.array(avg_pixel_dist_all_interval_all_scan)
    # sort error of all scans by the length of each scan
    sort_idx = np.argsort(scan_len)
    sorted_scan_len = sorted(scan_len)
    avg_pixel_dist_all_interval_all_scan = avg_pixel_dist_all_interval_all_scan[sort_idx]

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 1, 1)
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax3 = fig.add_subplot(2, 2, 3)
    # ax4 = fig.add_subplot(2, 2, 4)
    for i in range(avg_pixel_dist_all_interval_all_scan.shape[1]):
        ax1.plot(sorted_scan_len,avg_pixel_dist_all_interval_all_scan[:,i],color=viridis.colors[i], linestyle='dashed', marker='o',  alpha=.5,label='interval_'+str(data_pair[opt_test.PAIR_INDEX[i]][0].item())+'_'+str(data_pair[opt_test.PAIR_INDEX[i]][1].item()))

    ax1.legend()
    # ax1.set_xticks(sorted_scan_len)
    ax1.set_xlabel('scan length')
    ax1.set_ylabel('avg pixel dists')

    if train_val == 'train':
        saved_folder = os.path.join(opt.SAVE_PATH, 'testing_train_results')
    elif train_val == 'val':
        saved_folder = os.path.join(opt.SAVE_PATH, 'testing_val_results')

    else:
        saved_folder = os.path.join(opt.SAVE_PATH, 'testing_test_results')

    plt.savefig(saved_folder + '/'  + model_name +'_avg_pixel_dists_scan_level' + '.png')
    plt.close()

def plot_each_scan_overlap(viridis,opt,opt_test,overlap_all_interval_all_scan_test,train_val,model_name,dset_val,scan_index, data_pair,scan_len):
    overlap_all_interval_all_scan_test = np.array(overlap_all_interval_all_scan_test)
    # sort error of all scans by the length of each scan
    sort_idx = np.argsort(scan_len)
    sorted_scan_len = sorted(scan_len)
    overlap_all_interval_all_scan_test = overlap_all_interval_all_scan_test[sort_idx]

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 1, 1)
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax3 = fig.add_subplot(2, 2, 3)
    # ax4 = fig.add_subplot(2, 2, 4)
    for i in range(overlap_all_interval_all_scan_test.shape[1]):
        ax1.plot(sorted_scan_len,overlap_all_interval_all_scan_test[:,i],color=viridis.colors[i], linestyle='dashed', marker='o',  alpha=.5,label='interval_'+str(data_pair[opt_test.PAIR_INDEX[i]][0].item())+'_'+str(data_pair[opt_test.PAIR_INDEX[i]][1].item()))

    ax1.legend()
    # ax1.set_xticks(sorted_scan_len)
    ax1.set_xlabel('scan length')
    ax1.set_ylabel('overlap')

    if train_val == 'train':
        saved_folder = os.path.join(opt.SAVE_PATH, 'testing_train_results')
    elif train_val == 'val':
        saved_folder = os.path.join(opt.SAVE_PATH, 'testing_val_results')

    else:
        saved_folder = os.path.join(opt.SAVE_PATH, 'testing_test_results')

    plt.savefig(saved_folder + '/'  + model_name +'_overlap_scan_level' + '.png')
    plt.close()

def plot_each_epoch(viridis,opt,opt_test,epoch_id_val,rmse_epoch_var_intervals, train_val,model_name,dset_val,scan_index):
    rmse_epoch_var_intervals = np.array(rmse_epoch_var_intervals)
    ax = plt.figure()
    for i in range(rmse_epoch_var_intervals.shape[1]):
        plt.plot(epoch_id_val,rmse_epoch_var_intervals[:,i],color=viridis.colors[i], label='interval_'+str(opt_test.INTERVAL_LIST[i]))
    ax.legend()
    plt.xlabel('epoch index')
    plt.ylabel('accumulated error')
    if train_val == 'train':
        saved_folder = os.path.join(opt.SAVE_PATH, 'train_results')
    else:
        saved_folder = os.path.join(opt.SAVE_PATH, 'val_results')


    plt.savefig(saved_folder + '/'  +   model_name+'_epoch_level' +  '.png')
    plt.close()



def bland_altman_plot(ax,data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = data1
    diff      = data2 - data1                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    # plt.axhline(md,           color='gray', linestyle='--')
    # plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    # plt.axhline(md - 1.96*sd, color='gray', linestyle='--')


def sample_dists4plot(NUM_SAMPLES,CONSIATENT_LOSS, ACCUMULAT_LOSS,data_pairs):
    # first method for sample
    # data_pairs = data_pairs.cpu()
    # num_samples = opt.NUM_SAMPLES
    # saved_pairs = torch.tensor([[0,1],[int(num_samples/8)-1,int(num_samples/8)],[int(num_samples/4)-1,int(num_samples/4)],[0,int(num_samples/8)],[int(num_samples/8),int(num_samples/4)],
    #                         [int(num_samples*3/8),int(num_samples*3/8)+1],[int(num_samples/2),int(num_samples/2)+1],[int(num_samples/4),int(num_samples*3/8)],[int(num_samples*3/8),int(num_samples/2)],
    #                         [int(num_samples*3/4),int(num_samples*3/4)+1],[int(num_samples*5/8),int(num_samples*5/8)+1],[int(num_samples/2),int(num_samples*5/8)+1],[int(num_samples*5/8),int(num_samples*3/4)+1],
    #                         [num_samples-2,num_samples-1],[int(num_samples*7/8),int(num_samples*7/8)+1],[int(num_samples*3/4),int(num_samples*7/8)],[int(num_samples*7/8),num_samples-1],
    #                         [0,int(num_samples/4)],[0,int(num_samples/2)],[0,int(num_samples*3/4)],[0,num_samples-1],[int(num_samples/4),int(num_samples*3/4)],
    #                         [int(num_samples/4),num_samples-1],[int(num_samples/2),num_samples-1]])
    #
    #
    # saved_flag = []
    # for i, e in enumerate(data_pairs):
    #     for j in range(len(saved_pairs)):
    #         if (e == saved_pairs[j, :]).all():
    #             saved_flag.append(i)
    #

    # # second method for sample
    # data_pairs = data_pairs.cpu().numpy()
    # # sort transformations for sample
    # # data_pairs[np.lexsort(data_pairs[:, ::-1].T)]
    # maximum_pair = opt.MAXNUM_PAIRS
    # sample_rate = max(1,int(data_pairs.shape[0]/maximum_pair))
    # first_col = np.array(list(set(data_pairs[:,0])))
    # used_pairs = []
    #
    # for i in range(len(first_col)):
    #     if i == len(first_col)-1:
    #         temp = np.squeeze(data_pairs[np.where(data_pairs[:,0]==first_col[i]),:],axis=0)
    #     else:
    #         temp = (np.squeeze(data_pairs[np.where(data_pairs[:,0]==first_col[i]),:]))
    #     if i ==0:
    #         used_pairs.append(temp[0::sample_rate])
    #         used_pairs = np.squeeze(np.array(used_pairs))
    #     else:
    #         used_pairs = np.append(used_pairs,temp[0::sample_rate],axis=0)
    #
    # used_pairs = np.array(used_pairs)
    # used_pairs.reshape(-1, used_pairs.shape[-1])

    # # # third method for sample
    # # uniformly select transformations
    # data_pairs = data_pairs.cpu().numpy()
    # maximum_pair = opt.MAXNUM_PAIRS
    # sample_rate = max(1,int(data_pairs.shape[0]/maximum_pair))
    # used_pairs = data_pairs[0::sample_rate]
    # # add those typical transformations
    # num_samples = opt.NUM_SAMPLES
    # used_pairs = np.append(used_pairs, np.array([[0, int(num_samples / 4)],[0, int(num_samples / 2)],[0, int(num_samples*3 / 4)],[0, num_samples -1],
    #                         [int(num_samples / 4), int(num_samples / 2)],[int(num_samples / 4), int(num_samples*3 / 4)],[int(num_samples / 4), num_samples-1],
    #                         [int(num_samples / 2), int(num_samples*3 / 4)],[int(num_samples / 2), num_samples-1 ],[int(num_samples*3 / 4), num_samples-1]]),axis=0)
    # used_pairs = np.unique(used_pairs, axis=0)
    # used_pairs_index = []
    # for i, e in enumerate(data_pairs):
    #     for j in range(len(used_pairs)):
    #         if (e == used_pairs[j, :]).all():
    #             used_pairs_index.append(i)

    # # # # forth method for sample - choose according to interval first, and then past frames
    # selected_intervals = [1,5,10,15,20,25,30,40,50,60,90]
    # selected_pf = [0,1,5,10,15,20,25,30,40,50,60,90]
    # selected_ff = [0, 1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 90]
    # num_intervals = [10,10,10,10,10,10,10,5,5,5,5]
    # num_pf = [10,10, 10, 10, 10, 10, 10, 10, 5, 5, 5, 5]
    # num_ff = [10, 10, 10, 10, 10, 10, 10, 10, 5, 5, 5, 5]
    # past_frames_in_data_pairs = data_pairs[:,0]
    # future_frames_in_data_pairs = NUM_SAMPLES - 1-data_pairs[:, 1]
    # intervals_in_data_pairs = data_pairs[:,1]-data_pairs[:,0] # interval of each transformation
    # for i in range(len(selected_intervals)):
    #     temp = intervals_in_data_pairs==selected_intervals[i]
    #     idx_intervals = torch.squeeze(temp.nonzero())  # all index in data_pairs of the selected interval
    #     idx_pf = [torch.squeeze((past_frames_in_data_pairs[idx_intervals] == selected_pf[i]).nonzero()).numpy().tolist() for i in range(len(selected_pf))]  # find selected interval that has various past frames
    #     idx_pf = [x for x in idx_pf if x != []]
    #     indices = idx_intervals[idx_pf]
    #     if len(idx_pf) < num_intervals[i]:
    #         additional = idx_intervals.numpy().tolist()
    #         setA = set(additional)
    #         setB = set(indices.numpy().tolist())
    #         additional = list(setA.difference(setB)) # delete all ready indexed index
    #
    #         random.shuffle(additional)
    #         additional_idx = additional[:(num_intervals[i] - len(idx_pf))] # random select the first samples
    #         indices = torch.cat((indices, torch.Tensor(additional_idx)))
    #
    #     if i==0:
    #         final_indices = indices
    #     else:
    #         final_indices = torch.cat((final_indices,indices))
    #
    # final_indices = list(map(int, final_indices))
    # used_pairs = data_pairs[final_indices]
    # used_pairs_index = final_indices

    # # # fifth method for sample - choose according to pf anf ff first, and then intervals
    past_frames_in_data_pairs = data_pairs[:, 0]
    future_frames_in_data_pairs = NUM_SAMPLES - 1 - data_pairs[:, 1]
    intervals_in_data_pairs = data_pairs[:, 1] - data_pairs[:, 0]  # interval of each transformation
    second_max_interval = heapq.nlargest(2, torch.unique(intervals_in_data_pairs))[1].numpy()
    second_max_pf = heapq.nlargest(2, torch.unique(past_frames_in_data_pairs))[1].numpy()
    second_max_ff = heapq.nlargest(2, torch.unique(future_frames_in_data_pairs))[1].numpy()
    second_min_interval = sorted(torch.unique(intervals_in_data_pairs))[1].numpy()
    second_min_pf = sorted(torch.unique(past_frames_in_data_pairs))[1].numpy()
    second_min_ff = sorted(torch.unique(future_frames_in_data_pairs))[1].numpy()

    selected_intervals = [min(intervals_in_data_pairs),second_min_interval,max(intervals_in_data_pairs),second_max_interval, 5, 10, 15, 20, 25, 30, 40, 50, 60, 90]
    selected_pf = [min(past_frames_in_data_pairs),second_min_pf,max(past_frames_in_data_pairs),second_max_pf,1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 90]
    selected_ff = [min(future_frames_in_data_pairs),second_min_ff,max(future_frames_in_data_pairs),second_max_ff, 1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 90]
    num_intervals = [ 10,10,10,10, 10, 10, 10, 10, 10, 10, 5, 5, 5, 5]
    num_pf = [10,10,10,10, 10, 10, 10, 10, 10, 10, 10, 5, 5, 5, 5]
    num_ff = [10,10,10,10, 10, 10, 10, 10, 10, 10, 10, 5, 5, 5, 5]
    selected_intervals = list(map(int, selected_intervals))
    selected_pf = list(map(int, selected_pf))
    selected_ff = list(map(int, selected_ff))


    # data pairs for past frames
    for i in range(len(selected_pf)):
        temp = past_frames_in_data_pairs == selected_pf[i]
        idx_pf = torch.squeeze(temp.nonzero())  # all index in data_pairs of the selected interval
        idx_interval = [torch.squeeze((intervals_in_data_pairs[idx_pf] == selected_intervals[j]).nonzero()).numpy().tolist() for j in range(len(selected_intervals))]  # find selected interval that has various past frames
        idx_interval = [x for x in idx_interval if x != []]
        if not idx_interval: # idx_interval is empty
            indices = idx_pf
        else:
            indices = idx_pf[idx_interval]

        try:
            len(indices)
        except:
            indices = [indices.item()]


        if len(indices) < num_pf[i]:
            additional = idx_pf.numpy().tolist()
            try:
                setA = set(additional)
            except: # idx_pf just have one number
                setA = set([additional])
            try:
                setB = set(indices.numpy().tolist())
            except:
                setB = set(indices)
            additional = list(setA.difference(setB))  # delete all ready indexed index

            # random.Random(5).shuffle(additional)
            # if num_pf[i] - len(idx_interval)==0:
            if len(additional)<(num_pf[i] - len(idx_interval)):
                additional_idx = additional
            else:
                additional_idx = additional[0:-1:int(len(additional)/(num_pf[i] - len(idx_interval)))]  # random select the first samples
            # except:
            #     additional_idx=[]
            try:
                indices = torch.cat((indices, torch.Tensor(additional_idx)))
            except:
                indices = torch.cat((torch.tensor(indices), torch.Tensor(additional_idx)))


        elif len(indices) > num_pf[i]:
            indices = indices[0:-1:int(len(indices)/num_pf[i])]


        if i == 0:
            final_indices = indices
        else:
            final_indices = torch.cat((final_indices, indices))



    # data pairs for future frames
    for i in range(len(selected_ff)):
        temp = future_frames_in_data_pairs == selected_ff[i]
        idx_ff = torch.squeeze(temp.nonzero())  # all index in data_pairs of the selected interval
        idx_interval = [torch.squeeze((intervals_in_data_pairs[idx_ff] == selected_intervals[j]).nonzero()).numpy().tolist() for j in range(len(selected_intervals))]  # find selected interval that has various past frames
        idx_interval = [x for x in idx_interval if x != []]
        if not idx_interval: # idx_interval is empty
            indices_ff = idx_ff
        else:
            indices_ff = idx_ff[idx_interval]

        try:
            len(indices_ff)
        except:
            indices_ff = [indices_ff.item()]

        if len(indices_ff) < num_ff[i]:
            additional = idx_ff.numpy().tolist()
            try:
                setA = set(additional)
            except:  # idx_pf just have one number
                setA = set([additional])
            try:
                setB = set(indices.numpy().tolist())
            except:
                setB = set(indices)

            additional = list(setA.difference(setB))  # delete all ready indexed index

            # random.Random(5).shuffle(additional)
            if len(additional) < (num_ff[i] - len(idx_interval)):
                additional_idx=additional
            else:
                additional_idx = additional[0:-1:int(len(additional) /(num_ff[i] - len(idx_interval)))]  # random select the first samples
            try:
                indices_ff = torch.cat((indices_ff, torch.Tensor(additional_idx)))
            except:
                indices_ff = torch.cat((torch.tensor(indices_ff), torch.Tensor(additional_idx)))


        elif len(indices_ff) > num_ff[i]:
            indices_ff = indices_ff[0: -1:int(len(indices_ff) / num_ff[i])]

        if i == 0:
            final_indices_ff = indices_ff
        else:
            final_indices_ff = torch.cat((final_indices_ff, indices_ff))

    final_indices_pf_ff = torch.cat((final_indices, final_indices_ff))
    final_indices_pf_ff = torch.unique(final_indices_pf_ff, dim=0)

    final_indices_pf_ff = list(map(int, final_indices_pf_ff))
    used_pairs = data_pairs[final_indices_pf_ff]
    used_pairs_index = final_indices_pf_ff

    # add some transformations that we want
    # if CONSIATENT_LOSS or ACCUMULAT_LOSS: # add some indices for computing consistence or accmulate loss
    #     added_pairs = torch.tensor([[0,29],[0,15],[15,29]])
    #     used_pairs = torch.cat((used_pairs, added_pairs))
    #     used_pairs = torch.unique(used_pairs, dim=0)
    #
    #     saved_flag = []
    #     for i, e in enumerate(data_pairs):
    #         for j in range(len(used_pairs)):
    #             if (e == used_pairs[j, :]).all():
    #                 saved_flag.append(i)
    #
    #     used_pairs_index = saved_flag

    return used_pairs,used_pairs_index

def plot_statistic_scan(viridis,opt,opt_test,rmse_statistic_var_intervals, rmse_statistic_var_intervals_gt_based,train_val,model_name, data_pair,fig2,scan_len):
    # rmse_statistic_var_intervals = np.array(rmse_statistic_var_intervals)
    # rmse_statistic_var_intervals_gt_based = np.array(rmse_statistic_var_intervals_gt_based)
    rmse_statistic_var_intervals = pd.DataFrame(rmse_statistic_var_intervals)
    rmse_statistic_var_intervals_gt_based = pd.DataFrame(rmse_statistic_var_intervals_gt_based)
    # obatin the monitored data paired
    # plot_flag = sample_dists4plot(opt, data_pair.cpu())
    # print(data_pair[plot_flag])
    # for scan_idx in range(len(rmse_statistic_var_intervals)):
    sort_idx = np.argsort(scan_len)
    sorted_scan_len = sorted(scan_len)
    num_fig = len(opt_test.PAIR_INDEX)
    rows = 3
    if num_fig % rows != 0:
        cols = int(num_fig / rows) + 1
    else:
        cols = int(num_fig / rows)
    fig1 = plt.figure(figsize=(20, 10))

    ax_fig2 = fig2.add_subplot(1, 3, 2)
    fig_box_plot = plt.figure(figsize=(20, 10))
    fig_box_plot_gt = plt.figure(figsize=(20, 10))
    fig_box_plot_wo_outliers = plt.figure(figsize=(20, 10))
    fig_box_plot_gt_wo_outliers = plt.figure(figsize=(20, 10))
    fig_box_plot_all = plt.figure(figsize=(20, 10))

    # B-A plot
    # fig_B_A = plt.figure(figsize=(15, 10))
    # fig_B_A_gt_based = plt.figure(figsize=(15, 10))

    # plot_var_intervals, mean, std = [], [], []
    ax1 = fig1.add_subplot(2, 2, 1)
    ax2 = fig1.add_subplot(2, 2, 2)
    ax1_median = fig1.add_subplot(2, 2, 3)
    ax2_median = fig1.add_subplot(2, 2, 4)
    ax_box_plot_all = fig_box_plot_all.add_subplot(1, 2, 1)
    ax_box_plot_all_wo_outlier = fig_box_plot_all.add_subplot(1, 2, 1)

    for pair_idx in range(len(opt_test.PAIR_INDEX)):
        each_pair_all_scan = rmse_statistic_var_intervals.loc[:][pair_idx]
        each_pair_all_scan = pd.DataFrame(item for item in each_pair_all_scan)
        each_pair_all_scan_gt_based = rmse_statistic_var_intervals_gt_based.loc[:][pair_idx]
        each_pair_all_scan_gt_based = pd.DataFrame(item for item in each_pair_all_scan_gt_based)
        each_pair_all_scan_gt_based = each_pair_all_scan_gt_based/(data_pair[opt_test.PAIR_INDEX[pair_idx]][1]-data_pair[opt_test.PAIR_INDEX[pair_idx]][0]).numpy()
        interval = int(data_pair[opt_test.PAIR_INDEX[pair_idx]][1]-data_pair[opt_test.PAIR_INDEX[pair_idx]][0])
        # plot ave dists of each scan
        # scan_len = each_pair_all_scan.count(axis=1).to_numpy()
        # sort_idx = np.argsort(scan_len)
        # sorted_scan_len = sorted(scan_len)
        mean_all_scan_data = []
        for i in range(each_pair_all_scan.shape[0]):
            each_scan_data = each_pair_all_scan.loc[i].to_numpy()
            each_scan_data = each_scan_data[~np.isnan(each_scan_data)]  # delete nan
            each_scan_data = np.diff(each_scan_data) / interval
            mean_all_scan_data.append(each_scan_data.mean())

        mean_all_scan_data = np.array(mean_all_scan_data)
        sorted_mean_all_scan_data = mean_all_scan_data[sort_idx]
        ax_fig2.plot(sorted_scan_len,sorted_mean_all_scan_data, color=viridis.colors[pair_idx], linestyle='dashed', marker='o',  alpha=.5,label='interval_'+str(data_pair[opt_test.PAIR_INDEX[pair_idx]][0].item())+'_'+str(data_pair[opt_test.PAIR_INDEX[pair_idx]][1].item()))

        # plot_var_intervals.append(each_pair_all_scan)
        # mean.append(pd.DataFrame(each_pair_all_scan).mean(axis=0).to_numpy())
        # std.append(pd.DataFrame(each_pair_all_scan).std(axis=0).to_numpy())
        mean = pd.DataFrame(each_pair_all_scan).mean(axis=0).to_numpy()
        std = pd.DataFrame(each_pair_all_scan).std(axis=0,ddof=0).to_numpy()

        mean_gt_based = pd.DataFrame(each_pair_all_scan_gt_based).mean(axis=0).to_numpy()
        std_gt_based = pd.DataFrame(each_pair_all_scan_gt_based).std(axis=0,ddof=0).to_numpy()

        # plot mean and std with a shaded region
        x = range(data_pair[opt_test.PAIR_INDEX[pair_idx]][1], data_pair[opt_test.PAIR_INDEX[pair_idx]][1]+len(mean)*(data_pair[opt_test.PAIR_INDEX[pair_idx]][1]-data_pair[opt_test.PAIR_INDEX[pair_idx]][0]), (data_pair[opt_test.PAIR_INDEX[pair_idx]][1]-data_pair[opt_test.PAIR_INDEX[pair_idx]][0]))

        ax1.fill_between(x,mean - std, mean + std,color=viridis.colors[pair_idx],alpha = 0.3)
        ax1.plot(x,mean,color=viridis.colors[pair_idx], label='interval_'+str(data_pair[opt_test.PAIR_INDEX[pair_idx]][0].item())+'_'+str(data_pair[opt_test.PAIR_INDEX[pair_idx]][1].item()))
        ax1.axhline(0, color='gray', linestyle='--')

        ax2.fill_between(x, mean_gt_based - std_gt_based, mean_gt_based + std_gt_based, color=viridis.colors[pair_idx], alpha=0.3)
        ax2.plot(x,mean_gt_based, color=viridis.colors[pair_idx],label='interval_' + str(data_pair[opt_test.PAIR_INDEX[pair_idx]][0].item()) + '_' + str(data_pair[opt_test.PAIR_INDEX[pair_idx]][1].item()))
        ax2.axhline(0, color='gray', linestyle='--')

        median = pd.DataFrame(each_pair_all_scan).median(axis=0).to_numpy()
        median_gt_based = pd.DataFrame(each_pair_all_scan_gt_based).median(axis=0).to_numpy()

        # plot median and std with a shaded region

        ax1_median.fill_between(x, median - std, median + std, color=viridis.colors[pair_idx], alpha=0.3)
        ax1_median.plot(x, median, color=viridis.colors[pair_idx],label='interval_' + str(data_pair[opt_test.PAIR_INDEX[pair_idx]][0].item()) + '_' + str(data_pair[opt_test.PAIR_INDEX[pair_idx]][1].item()))
        ax1_median.axhline(0, color='gray', linestyle='--')

        ax2_median.fill_between(x, median_gt_based - std_gt_based, median_gt_based + std_gt_based, color=viridis.colors[pair_idx],alpha=0.3)
        ax2_median.plot(x, median_gt_based, color=viridis.colors[pair_idx],label='interval_' + str(data_pair[opt_test.PAIR_INDEX[pair_idx]][0].item()) + '_' + str(data_pair[opt_test.PAIR_INDEX[pair_idx]][1].item()))
        ax2_median.axhline(0, color='gray', linestyle='--')

        # box plot
        ax = fig_box_plot.add_subplot(rows, cols, pair_idx + 1)
        # each_pair_all_scan_np = each_pair_all_scan.to_numpy()
        each_pair_all_scan.boxplot(ax=ax,positions=x,medianprops={"linewidth": 4})
        # ax.set_xlabel('img index')
        ax.set_title('interval_'+str(data_pair[opt_test.PAIR_INDEX[pair_idx]][0].item())+'_'+str(data_pair[opt_test.PAIR_INDEX[pair_idx]][1].item()))

        ax_gt = fig_box_plot_gt.add_subplot(rows, cols, pair_idx + 1)
        each_pair_all_scan_gt_based.boxplot(ax=ax_gt,positions=x,medianprops={"linewidth": 4}) #,positions=list(range(10,59))
        # ax_gt.set_xlabel('img index')
        ax_gt.set_title('non-accumulated interval_' + str(data_pair[opt_test.PAIR_INDEX[pair_idx]][0].item()) + '_' + str(data_pair[opt_test.PAIR_INDEX[pair_idx]][1].item()))

        # box plot without outliers
        ax_wo_outliers = fig_box_plot_wo_outliers.add_subplot(rows, cols, pair_idx + 1)
        # each_pair_all_scan_np = each_pair_all_scan.to_numpy()
        each_pair_all_scan.boxplot(ax=ax_wo_outliers, positions=x,medianprops={"linewidth": 4},showfliers = False)
        # ax.set_xlabel('img index')
        ax_wo_outliers.set_title('interval_' + str(data_pair[opt_test.PAIR_INDEX[pair_idx]][0].item()) + '_' + str(
            data_pair[opt_test.PAIR_INDEX[pair_idx]][1].item()))

        ax_gt_wo_outliers = fig_box_plot_gt_wo_outliers.add_subplot(rows, cols, pair_idx + 1)
        each_pair_all_scan_gt_based.boxplot(ax=ax_gt_wo_outliers, positions=x,medianprops={"linewidth": 4},showfliers = False)  # ,positions=list(range(10,59))
        # ax_gt.set_xlabel('img index')
        ax_gt_wo_outliers.set_title('non-accumulated interval_' + str(data_pair[opt_test.PAIR_INDEX[pair_idx]][0].item()) + '_' + str(
            data_pair[opt_test.PAIR_INDEX[pair_idx]][1].item()))

        # box-plot all

        # each_pair_all_scan.boxplot(ax=ax_box_plot_all,positions=x,medianprops={"linewidth": 4})
        # sns.boxplot(x=x, y=each_pair_all_scan,data=[x,each_pair_all_scan], palette=viridis.colors[pair_idx],alpha = 0.3,ax=ax_box_plot_all,labels = str(data_pair[opt_test.PAIR_INDEX[pair_idx]][0].item()) + '_' + str(
        #     data_pair[opt_test.PAIR_INDEX[pair_idx]][1].item()))

        # # B-A plot
        # if pair_idx == 0:
        #     data1 = mean
        #     data1_gt_based = mean_gt_based
        # else:
        #     data2 = mean
        #     data2_gt_based = mean_gt_based
        #     diff = data2 - data1
        #     diff_gt_based = data2_gt_based-data1_gt_based
        #     ax_BA_plot = fig_B_A.add_subplot(rows, cols, pair_idx)
        #     ax_BA_plot_gt_based = fig_B_A_gt_based.add_subplot(rows, cols, pair_idx)
        #
        #     ax_BA_plot_gt_based.plot(data1_gt_based, diff_gt_based, color=viridis.colors[pair_idx], linestyle='none', marker='o', alpha=.5,
        #              label='interval_' + str(data_pair[opt_test.PAIR_INDEX[pair_idx]][0].item()) + '_' + str(
        #                  data_pair[opt_test.PAIR_INDEX[pair_idx]][1].item()))
        #     ax_BA_plot_gt_based.axhline(0, color='gray', linestyle='--')
        #     ax_BA_plot_gt_based.legend()
        #     # ax1.set_xlabel('1st transform error')
        #     ax_BA_plot_gt_based.set_ylabel('difference')
        #     ax_BA_plot_gt_based.set_title('gt_based B_A_plot')
        #
        #     ax_BA_plot.plot(data1, diff, color=viridis.colors[pair_idx], linestyle='none', marker='o', alpha=.5,
        #                     label='interval_' + str(data_pair[opt_test.PAIR_INDEX[pair_idx]][0].item()) + '_' + str(
        #                         data_pair[opt_test.PAIR_INDEX[pair_idx]][1].item()))
        #     ax_BA_plot.axhline(0, color='gray', linestyle='--')
        #     ax_BA_plot.legend()
        #     # ax1.set_xlabel('1st transform error')
        #     ax_BA_plot.set_ylabel('difference')
        #     ax_BA_plot.set_title('accumulated B_A_plot')
        #


    ax1.legend()
    # ax1.set_xlabel('img index')
    ax1.set_ylabel('accumulated error (mean)')
    ax2.legend()
    ax2.set_xlabel('img index')
    ax2.set_ylabel('non-accumulated error (mean)')


    ax_fig2.legend()
    # ax_fig2.set_xticks(sorted_scan_len)
    ax_fig2.set_xlabel('scan length')
    ax_fig2.set_ylabel('ave dist error')

    ax1_median.legend()
    # ax1.set_xlabel('img index')
    ax1_median.set_ylabel('accumulated error (median)')
    ax2_median.legend()
    ax2_median.set_xlabel('img index')
    ax2_median.set_ylabel('non-accumulated error (median)')
    # plt.show()
    if train_val == 'train':
        saved_folder = os.path.join(opt.SAVE_PATH, 'testing_train_results')
    elif train_val == 'val':
        saved_folder = os.path.join(opt.SAVE_PATH, 'testing_val_results')

    else:
        saved_folder = os.path.join(opt.SAVE_PATH, 'testing_test_results')

    fig1.savefig(saved_folder + '/'  +   model_name+'_ave_over_all_scan' +  '.png')
    # fig1.close()
    fig2.savefig(saved_folder + '/' + model_name + '_scan_level' + '.png')
    # plt.close()

    fig_box_plot.savefig(saved_folder + '/'  +   model_name+'_ave_over_all_scan_box_plot' +  '.png')
    fig_box_plot_gt.savefig(saved_folder + '/'  +   model_name+'_ave_over_all_scan_box_plot_non_accumulated' +  '.png')
    fig_box_plot_wo_outliers.savefig(saved_folder + '/' + model_name + '_ave_over_all_scan_box_plot_wo_outliers' + '.png')
    fig_box_plot_gt_wo_outliers.savefig(saved_folder + '/' + model_name + '_ave_over_all_scan_box_plot_non_accumulated_wo_outliers' + '.png')

    fig_box_plot_all.savefig(saved_folder + '/' + model_name + '_ave_over_all_scan_box_plot_all(mean)' + '.png')
    fig2.savefig(saved_folder + '/' + model_name + '_ave_dists_over_all_scan' + '.png')

    # fig_B_A.savefig(saved_folder + '/' + model_name + '_ave_over_all_scan_B-A_plot_' + '.png')
    # fig_B_A_gt_based.savefig(saved_folder + '/' + model_name + '_ave_over_all_scan_B-A_gt_based' + '.png')

def str2list(string):
    string = ''.join(string)
    string = string[1:-1]
    token = string.split(',')
    list = [int(token_i) for token_i in token]
    return list

def load_json(opt,json_fn):
    if os.path.isfile(opt.SAVE_PATH+'/'+json_fn+".json"):
        with open(opt.SAVE_PATH + '/' + json_fn+".json", "r", encoding='utf-8') as f:
            rmse_intervals_each_scan= json.load(f)
    else:
        rmse_intervals_each_scan= {}

    return rmse_intervals_each_scan


def load_json_MICCAI(folder,json_fn):
    if os.path.isfile(folder+'/'+json_fn+".json"):
        with open(folder + '/' + json_fn+".json", "r", encoding='utf-8') as f:
            rmse_intervals_each_scan= json.load(f)
    else:
        rmse_intervals_each_scan= {}

    return rmse_intervals_each_scan


def get_common_data(data,frame_index_each_scan_test, common_frames,overlap):

    '''
    get the cpmmon frames that existing in each interval and sequence length
    data: metirc for each transformation pair, for each sunject, for all avaliable frames in a scan
    frame_index_each_scan_test: the index of each value in a scan
    common_frames: all the shared frames for all avaliable transformations and input sequence length
    '''



    data_df = pd.DataFrame(data)
    intervals = data_df[data_df.keys()[0]].keys().to_list()
    frame_index_df=pd.DataFrame(frame_index_each_scan_test)
    all_scan_all_err={}
    # get the common frames in each scan
    sub_names = list(data.keys())
    for sub_idx in range(len(data)):
        sub_name = sub_names[sub_idx]
        if sub_name not in all_scan_all_err.keys():
            all_scan_all_err[sub_name] = {}
        each_scan_all_pairs=data[sub_name]
        each_scan_all_pairs_fi = frame_index_each_scan_test[sub_name]
        each_scan_common_frames = common_frames[sub_name]
        for pair_idx in list(each_scan_all_pairs.keys()):

            each_scan_each_pair_all_start_ref=each_scan_all_pairs[pair_idx]
            each_scan_each_pair_all_start_ref_fi=each_scan_all_pairs_fi[pair_idx]

            for start_ref_idx in list(each_scan_each_pair_all_start_ref_fi.keys()):
                mask = np.isin(each_scan_each_pair_all_start_ref_fi[start_ref_idx],each_scan_common_frames)

                if not overlap:
                    temp=(np.array(each_scan_each_pair_all_start_ref[start_ref_idx])[mask]).tolist()
                    if pair_idx not in all_scan_all_err[sub_name].keys():

                        all_scan_all_err[sub_name][pair_idx] = temp
                    else:
                        temp_2 = np.array(all_scan_all_err[sub_name][pair_idx])
                        all_scan_all_err[sub_name][pair_idx] =np.append(temp_2,temp).tolist()
                else:
                    current_frm_idx=np.array(each_scan_each_pair_all_start_ref_fi[start_ref_idx])[mask]
                    temp=(np.array(each_scan_each_pair_all_start_ref[start_ref_idx])[:,:,mask]).tolist()
                    if pair_idx not in all_scan_all_err[sub_name].keys():
                        all_frm_idx=current_frm_idx
                        all_scan_all_err[sub_name][pair_idx] = temp
                    else:
                        all_frm_idx=np.append(all_frm_idx,current_frm_idx)
                        sorted_idx=np.argsort(all_frm_idx)

                        temp_2 = np.array(all_scan_all_err[sub_name][pair_idx])
                        temp_2=np.append(temp_2, temp, 2)
                        temp_2=temp_2[:, :, sorted_idx]

                        all_scan_all_err[sub_name][pair_idx] =temp_2.tolist()

            if not overlap:

                if  len(all_scan_all_err[sub_name][pair_idx] )!= len(each_scan_common_frames):
                    raise ("Inconsistent common_frames.")
            else:
                if  (np.array(all_scan_all_err[sub_name][pair_idx]) ).shape[2]!= len(each_scan_common_frames):
                    raise ("Inconsistent common_frames.")


    return all_scan_all_err
    # plot the average error of each scan & transformations

def merge_data(data_all,data):
    keys=list(data.keys())
    for sub_name in list(keys):
        # all_pair_each_scan=data[sub_name]
        # all_pair_each_scan_all = data_all[sub_name]
        for data_pair in list(data[sub_name].keys()):
            if data_pair not in data_all[sub_name].keys():
                data_all[sub_name][data_pair] = data[sub_name][data_pair]
            else:
                data_all[sub_name][data_pair] =np.append(data_all[sub_name][data_pair],data[sub_name][data_pair]).tolist()
    return data_all

def pf_ff(data,saved_str,input_num,final_drift,sta_in_each_interval_each_scan, sta_in_each_interval_all_scan ,
        sta_in_each_pf_each_scan, sta_in_each_pf_all_scan ,
        sta_in_each_ff_each_scan, sta_in_each_ff_all_scan ,
        sta_in_each_pf_add_ff_each_scan, sta_in_each_pf_add_ff_all_scan ,
        sta_in_each_pf_div_ff_each_scan, sta_in_each_pf_div_ff_all_scan ):
    # compute the error & past frames, and error & past frames
    data_df = pd.DataFrame(data)
    intervals = data_df[data_df.keys()[0]].keys().to_list()
    mean_each_task_each_scan = {}

    for pair_idx in range(data_df.shape[0]):
        interval_name = intervals[pair_idx]
        interval_all_scan = data_df.loc[interval_name]
        interval_all_scan = pd.DataFrame(item for item in interval_all_scan)
        string = interval_name[1:-1]
        token = string.split(',')
        interval_name_list = [int(token_i) for token_i in token]
        # std = pd.DataFrame(interval_all_scan).std(axis=1, ddof=0).to_numpy()

        if interval_name not in mean_each_task_each_scan.keys():
            if final_drift:
                mean = interval_all_scan.ffill(axis=1).iloc[:, -1].to_numpy()

            else:
                mean = pd.DataFrame(interval_all_scan).mean(axis=1).to_numpy()

            mean = mean[~np.isnan(mean)]



        # interval_value = interval_name_list[1] - interval_name_list[0]
        # num_pf = interval_name_list[0]
        # num_ff = input_num - 1 - interval_name_list[1]
        # num_pf_add_ff = interval_name_list[0] + input_num - 1 - interval_name_list[1]
        # if interval_name_list[0] == 0:
        #     num_ff_div_pf = (input_num - 1 - interval_name_list[1])
        # else:
        #     num_ff_div_pf = (input_num - 1 - interval_name_list[1]) / interval_name_list[0]

        if interval_name_list[1]-interval_name_list[0] in sta_in_each_interval_each_scan:
            sta_in_each_interval_each_scan[interval_name_list[1]-interval_name_list[0]] = np.append(sta_in_each_interval_each_scan[interval_name_list[1]-interval_name_list[0]],mean)
            sta_in_each_interval_all_scan[interval_name_list[1]-interval_name_list[0]] = np.append(sta_in_each_interval_all_scan[interval_name_list[1]-interval_name_list[0]],mean.mean())
        else:
            sta_in_each_interval_each_scan[interval_name_list[1] - interval_name_list[0]]=mean
            sta_in_each_interval_all_scan[interval_name_list[1] - interval_name_list[0]]=mean.mean()

        # dist error (mean of all images error in a scan) v.s. past frames
        if interval_name_list[0] in sta_in_each_pf_each_scan:
            sta_in_each_pf_each_scan[interval_name_list[0]] = np.append(sta_in_each_pf_each_scan[interval_name_list[0]], mean)
            sta_in_each_pf_all_scan[interval_name_list[0]] = np.append(sta_in_each_pf_all_scan[interval_name_list[0]], mean.mean())
        else:
            sta_in_each_pf_each_scan[interval_name_list[0]] = mean
            sta_in_each_pf_all_scan[interval_name_list[0]] = mean.mean()

        # dist error (mean of all images error in a scan) v.s. future frames
        if input_num - 1 - interval_name_list[1] in sta_in_each_ff_each_scan:
            sta_in_each_ff_each_scan[input_num - 1 - interval_name_list[1]] = np.append(sta_in_each_ff_each_scan[input_num - 1 - interval_name_list[1]], mean)
            sta_in_each_ff_all_scan[input_num - 1 - interval_name_list[1]] = np.append(sta_in_each_ff_all_scan[input_num - 1 - interval_name_list[1]], mean.mean())
        else:
            sta_in_each_ff_each_scan[input_num - 1 - interval_name_list[1]] = mean
            sta_in_each_ff_all_scan[input_num - 1 - interval_name_list[1]] = mean.mean()

        # dist error (mean of all images error in a scan) v.s. past frames + future frames
        if interval_name_list[0]+input_num - 1 - interval_name_list[1] in sta_in_each_pf_add_ff_each_scan:
            sta_in_each_pf_add_ff_each_scan[interval_name_list[0]+input_num - 1 - interval_name_list[1] ] = np.append(sta_in_each_pf_add_ff_each_scan[interval_name_list[0]+input_num - 1 - interval_name_list[1] ], mean)
            sta_in_each_pf_add_ff_all_scan[interval_name_list[0]+input_num - 1 - interval_name_list[1] ] = np.append(sta_in_each_pf_add_ff_all_scan[interval_name_list[0]+input_num - 1 - interval_name_list[1] ], mean.mean())
        else:
            sta_in_each_pf_add_ff_each_scan[interval_name_list[0]+input_num - 1 - interval_name_list[1] ] = mean
            sta_in_each_pf_add_ff_all_scan[interval_name_list[0]+input_num - 1 - interval_name_list[1] ] = mean.mean()

        # dist error (mean of all images error in a scan) v.s. future frames / past frames
        if interval_name_list[0] == 0:

            if (input_num - 1 - interval_name_list[1]) in sta_in_each_pf_div_ff_each_scan:
                sta_in_each_pf_div_ff_each_scan[(input_num - 1 - interval_name_list[1])] = np.append(sta_in_each_pf_div_ff_each_scan[(input_num - 1 - interval_name_list[1])], mean)
                sta_in_each_pf_div_ff_all_scan[(input_num - 1 - interval_name_list[1])] = np.append(sta_in_each_pf_div_ff_all_scan[(input_num - 1 - interval_name_list[1])],mean.mean())
            else:
                sta_in_each_pf_div_ff_each_scan[(input_num - 1 - interval_name_list[1])] = mean
                sta_in_each_pf_div_ff_all_scan[(input_num - 1 - interval_name_list[1])] = mean.mean()

        else:
            if (input_num - 1 - interval_name_list[1])/interval_name_list[0] in sta_in_each_pf_div_ff_each_scan:
                sta_in_each_pf_div_ff_each_scan[(input_num - 1 - interval_name_list[1])/interval_name_list[0]] = np.append(sta_in_each_pf_div_ff_each_scan[(input_num - 1 - interval_name_list[1])/interval_name_list[0]], mean)
                sta_in_each_pf_div_ff_all_scan[(input_num - 1 - interval_name_list[1])/interval_name_list[0]] = np.append(sta_in_each_pf_div_ff_all_scan[(input_num - 1 - interval_name_list[1])/interval_name_list[0]],mean.mean())
            else:
                sta_in_each_pf_div_ff_each_scan[(input_num - 1 - interval_name_list[1])/interval_name_list[0]] = mean
                sta_in_each_pf_div_ff_all_scan[(input_num - 1 - interval_name_list[1])/interval_name_list[0]] = mean.mean()

    return sta_in_each_interval_each_scan, sta_in_each_interval_all_scan ,\
        sta_in_each_pf_each_scan, sta_in_each_pf_all_scan ,\
        sta_in_each_ff_each_scan, sta_in_each_ff_all_scan ,\
        sta_in_each_pf_add_ff_each_scan, sta_in_each_pf_add_ff_all_scan ,\
        sta_in_each_pf_div_ff_each_scan, sta_in_each_pf_div_ff_all_scan

def plot_mean_var_all(smooth_rate,saved_name,saved_folder_test,train_set_type,\
        avg_pixel_dists_interval_each_scan, avg_pixel_dists_interval_all_scan ,\
        avg_pixel_dists_pf_each_scan, avg_pixel_dists_pf_all_scan ,\
        avg_pixel_dists_ff_each_scan, avg_pixel_dists_ff_all_scan ,\
        avg_pixel_dists_pf_add_ff_each_scan, avg_pixel_dists_pf_add_ff_all_scan ,\
        avg_pixel_dists_pf_div_ff_each_scan, avg_pixel_dists_pf_div_ff_all_scan, \
                      baseline_avg_pixel_dists_interval_each_scan, baseline_avg_pixel_dists_interval_all_scan, \
                      baseline_avg_pixel_dists_pf_each_scan, baseline_avg_pixel_dists_pf_all_scan, \
                      baseline_avg_pixel_dists_ff_each_scan, baseline_avg_pixel_dists_ff_all_scan, \
                      baseline_avg_pixel_dists_pf_add_ff_each_scan, baseline_avg_pixel_dists_pf_add_ff_all_scan, \
                      baseline_avg_pixel_dists_pf_div_ff_each_scan, baseline_avg_pixel_dists_pf_div_ff_all_scan
                      ):


    # viridis = cm.get_cmap('hsv', 5)
    viridis=mpl.colormaps['Set1']

    fig = plt.figure(figsize=(20, 10))
    ax1,ax2 = fig.add_subplot(1, 2, 1),fig.add_subplot(1, 2, 2)
    fig1 = plt.figure(figsize=(20, 10))
    ax3, ax4 = fig1.add_subplot(1, 2, 1), fig1.add_subplot(1, 2, 2)

    # plot baseline
    plot_mean_var(smooth_rate,True,ax3, baseline_avg_pixel_dists_pf_div_ff_each_scan, '', saved_name, viridis.colors[7], 'baseline')
    # plot_mean_var(True,ax3, baseline_avg_pixel_dists_pf_div_ff_each_scan, '', saved_name, viridis.colors[8], 'pf_add_ff')
    # plot_mean_var(True,ax3, baseline_avg_pixel_dists_interval_each_scan, '', saved_name, viridis.colors[6], 'interval')
    plot_mean_var(smooth_rate,True,ax1, baseline_avg_pixel_dists_pf_each_scan, '', saved_name, viridis.colors[7], 'baseline')
    # plot_mean_var(True,ax1, baseline_avg_pixel_dists_ff_each_scan, '', saved_name, viridis.colors[8], 'ff')

    plot_mean_var(smooth_rate,False,ax3, avg_pixel_dists_pf_div_ff_each_scan, '', saved_name,viridis.colors[2],'ff_div_pf')
    plot_mean_var(smooth_rate,False,ax3, avg_pixel_dists_pf_add_ff_each_scan, '', saved_name,viridis.colors[3],'pf_add_ff')
    plot_mean_var(smooth_rate,False,ax3, avg_pixel_dists_interval_each_scan, '', saved_name,viridis.colors[1],'interval')
    plot_mean_var(smooth_rate,False,ax1, avg_pixel_dists_pf_each_scan, '', saved_name,viridis.colors[3],'pf')
    plot_mean_var(smooth_rate,False,ax1, avg_pixel_dists_ff_each_scan, '', saved_name,viridis.colors[1],'ff')
    ax1.set_title('each scan')
    ax3.set_title('each scan')

    plot_mean_var(smooth_rate,True,ax4, baseline_avg_pixel_dists_pf_div_ff_all_scan, '', saved_name, viridis.colors[7], 'baseline')
    # plot_mean_var(True,ax4, baseline_avg_pixel_dists_pf_add_ff_all_scan, '', saved_name, viridis.colors[8], 'pf_add_ff')
    # plot_mean_var(True,ax4, baseline_avg_pixel_dists_interval_all_scan, '', saved_name, viridis.colors[6], 'interval')
    plot_mean_var(smooth_rate,True,ax2, baseline_avg_pixel_dists_pf_all_scan, '', saved_name, viridis.colors[7], 'baseline')
    # plot_mean_var(True,ax2, baseline_avg_pixel_dists_ff_all_scan, '', saved_name, viridis.colors[8], 'ff')

    plot_mean_var(smooth_rate,False,ax4, avg_pixel_dists_pf_div_ff_all_scan, '', saved_name, viridis.colors[2], 'ff_div_pf')
    plot_mean_var(smooth_rate,False,ax4, avg_pixel_dists_pf_add_ff_all_scan, '', saved_name, viridis.colors[3], 'pf_add_ff')
    plot_mean_var(smooth_rate,False,ax4, avg_pixel_dists_interval_all_scan, '', saved_name, viridis.colors[1], 'interval')
    plot_mean_var(smooth_rate,False,ax2, avg_pixel_dists_pf_all_scan, '', saved_name, viridis.colors[3], 'pf')
    plot_mean_var(smooth_rate,False,ax2, avg_pixel_dists_ff_all_scan, '', saved_name, viridis.colors[1], 'ff')
    ax2.set_title('all scan')
    ax4.set_title('all scan')
    if not train_set_type:
        fig.savefig(saved_folder_test + '/' + saved_name + '_pf_ff' + '_smooth_rate_' + str(
            smooth_rate)  + '.png')
        fig.savefig(saved_folder_test + '/' + saved_name + '_pf_ff' + '_smooth_rate_' + str(
            smooth_rate)  + '.pdf')
        fig.savefig(saved_folder_test + '/' + saved_name + '_pf_ff' + '_smooth_rate_' + str(
            smooth_rate)  + '.eps')

        fig1.savefig(saved_folder_test + '/' + saved_name + '_interval_pf_add_div_ff' + '_smooth_rate_' + str(
            smooth_rate) + '.png')
        fig1.savefig(saved_folder_test + '/' + saved_name + '_interval_pf_add_div_ff' + '_smooth_rate_' + str(
            smooth_rate)  + '.pdf')
        fig1.savefig(saved_folder_test + '/' + saved_name + '_interval_pf_add_div_ff' + '_smooth_rate_' + str(
            smooth_rate)  + '.eps')


    else:
        fig.savefig(saved_folder_test + '/'+ saved_name+ '_pf_ff'+'_smooth_rate_'+str(smooth_rate)+'_'+train_set_type+'.png')
        fig.savefig(saved_folder_test + '/'+ saved_name+ '_pf_ff'+'_smooth_rate_'+str(smooth_rate)+'_'+train_set_type+'.pdf')
        fig.savefig(saved_folder_test + '/'+ saved_name+ '_pf_ff'+'_smooth_rate_'+str(smooth_rate)+'_'+train_set_type+'.eps')

        fig1.savefig(saved_folder_test + '/'+ saved_name+ '_interval_pf_add_div_ff'+'_smooth_rate_'+str(smooth_rate)+'_'+train_set_type+'.png')
        fig1.savefig(saved_folder_test + '/'+ saved_name+ '_interval_pf_add_div_ff'+'_smooth_rate_'+str(smooth_rate)+'_'+train_set_type+'.pdf')
        fig1.savefig(saved_folder_test + '/'+ saved_name+ '_interval_pf_add_div_ff'+'_smooth_rate_'+str(smooth_rate)+'_'+train_set_type+'.eps')




def plot_mean_var(smooth_rate,baseline_plot,ax, each_scan, x_label, y_label,colors,legend):
    # plot dists v.s. intervals - plot each scan
    labels, data = each_scan.keys(), each_scan.values()
    # sort data increasing
    label_np = np.array(list(labels))
    index = np.argsort(label_np)
    data_list = list(data)
    labels_list = list(labels)
    data_list = [x for _, x in sorted(zip(labels_list, data_list))]

    labels_sorted = sorted(labels)

    # plot mean and var
    mean = [np.array(data_list[i]).mean() for i in range(len(data_list))]
    std = [np.array(data_list[i]).std() for i in range(len(data_list))]
    mean=np.array(mean)
    std=np.array(std)
    if baseline_plot:
        labels_sorted=range(0,100)
        mean=np.ones(len(labels_sorted))*mean
        std = np.ones(len(labels_sorted)) * std

    # smoth mean and std
    if not baseline_plot and smooth_rate:

        # mean[0:len(mean)-smooth_rate+1]=\
        temp1=([(mean[i:i+smooth_rate]).mean() for i in range(len(mean)-smooth_rate+1)])
        temp2=([(mean[i]) for i in range(len(mean)-(smooth_rate-1),len(mean))])
        mean=np.append(np.array(temp1),np.array(temp2))

        temp1 = ([math.sqrt((std[i:i + smooth_rate]**2).mean()) for i in range(len(std) - smooth_rate + 1)])
        temp2 = ([(std[i]) for i in range(len(std) - (smooth_rate - 1), len(std))])
        std = np.append(np.array(temp1), np.array(temp2))

    ax.fill_between(labels_sorted, mean - std, mean + std, color=colors, alpha=0.3)
    if baseline_plot:
        ax.plot(labels_sorted, mean,'--',linewidth=2, color=colors,label=legend)
    else:
        ax.plot(labels_sorted, mean,linewidth=2, color=colors,label=legend)

    ax.legend()


    # box plot
    # if mean_median == 'mean':
    #     ax.boxplot(data_list, showmeans=True, meanprops={"linewidth": 4}, medianprops={"linewidth": 4})
    # elif mean_median == 'median':
    #     ax.boxplot(data_list, medianprops={"linewidth": 4})
    #
    # ax.set_xticks(range(1, len(labels) + 1), labels_sorted)


    # # plot original data
    # for i in range(len(list(labels))):
    #     y = list(data)[index[i]]
    #     try:
    #         # Add some random "jitter" to the x-axis
    #         x = np.random.normal(i + 1, 0.04, size=y.shape[0])
    #     except:
    #         y = [y]
    #         x = np.random.normal(i + 1, 0.04, size=len(y))
    #
    #     ax.plot(x, y, 'r.', alpha=0.2, markersize=12)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def compute_DSC(y_actual_overlap,y_predicted_overlap):
    DSC={}
    for sub_name in list(y_actual_overlap.keys()):
        if sub_name not in DSC.keys():
            DSC[sub_name] = {}
        actual_all_pair = y_actual_overlap[sub_name]
        pred_all_pair = y_predicted_overlap[sub_name]
        all_pair_name=list(y_actual_overlap[sub_name].keys())
        for pair_name in all_pair_name:
            if pair_name not in DSC[sub_name].keys():
                DSC[sub_name][pair_name] = {}

            y_actual=actual_all_pair[pair_name]
            y_pred = pred_all_pair[pair_name]
            if np.array(y_actual).shape[2]==0:
                dice=np.nan
            else:
                # select 10 frames to compute dice, but the last frame maybe not included
                # if math.floor(np.array(y_actual).shape[2] / 10) == 0:
                #     slice_step = 1
                # else:
                #     slice_step = math.floor(np.array(y_actual).shape[2] / 10)
                # dice = frame_volume_overlap(np.array(y_actual)[:, :, ::slice_step],
                #                             np.array(y_pred)[:, :, ::slice_step])

                # select the first and last frames, and one middle frame
                index = [0, math.floor(np.array(y_actual).shape[2] / 2), -1]


                dice = frame_volume_overlap(np.array(y_actual)[:, :, index],
                                           np.array(y_pred)[:, :, index])
            DSC[sub_name][pair_name]=dice

    return DSC

def box_plot_ipcai(data,saved_folder_test,saved_str):
    data_df = pd.DataFrame(data)
    intervals = data_df[data_df.keys()[0]].keys().to_list()
    mean_each_task_each_scan={}
    mean_each_task_all_scan = {}
    std_each_task_all_scan={}
    for pair_idx in range(data_df.shape[0]):
        interval_name = intervals[pair_idx]
        interval_all_scan = data_df.loc[interval_name]
        interval_all_scan = pd.DataFrame(item for item in interval_all_scan)
        string = interval_name[1:-1]
        token = string.split(',')
        interval_name_list = [int(token_i) for token_i in token]
        mean = pd.DataFrame(interval_all_scan).mean(axis=1).to_numpy()


        # std = pd.DataFrame(interval_all_scan).std(axis=1, ddof=0).to_numpy()

        if interval_name not in mean_each_task_each_scan.keys():
            if 'final_drift' in saved_str:
                mean=interval_all_scan.ffill(axis=1).iloc[:, -1]
                mean_each_task_each_scan[interval_name] =mean
            # elif 'overlap_dice' in saved_str:

            else:
                # mean = mean[~np.isnan(mean)]
                mean_each_task_each_scan[interval_name] = mean

            if np.isnan(mean).any():  # delete nan
                mean = mean[~np.isnan(mean)]
            mean_each_task_all_scan[interval_name] = mean.mean()
            std_each_task_all_scan[interval_name] = mean.std()

        # if interval_name not in std_each_task_each_scan.keys():
        #     std_each_task_each_scan[interval_name] = std
    # box plotting
    mean_each_task_each_scan = pd.DataFrame(mean_each_task_each_scan)
    # std_each_task_each_scan = pd.DataFrame(std_each_task_each_scan)

    # find mininum mean
    if 'overlap' not in saved_str: # find the minimum
        mean_each=pd.DataFrame(mean_each_task_each_scan).mean(axis=0).to_numpy()
        mean_min = min(mean_each)
        idx_min= mean_each.tolist().index(mean_min)
        mean_std = pd.DataFrame(mean_each_task_each_scan).std(axis=0).to_numpy()[idx_min]

        # mean_all=np.array(list(mean_each_task_all_scan.values())).mean()

        # std_all=np.array(list(mean_each_task_all_scan.values())).std()

        mean_all=np.nanmean(mean_each_task_each_scan.to_numpy())
        std_all=np.nanstd(mean_each_task_each_scan.to_numpy())

        best_perf = list(mean_each_task_each_scan.keys())[idx_min]



        # best_perf = mean_each_task_each_scan[list(mean_each_task_each_scan.keys())[idx_min]]
        # with open(opt.SAVE_PATH + '/' + str('seq_len') + str(opt.NUM_SAMPLES)+ str('_protocol_cls')+str(opt.class_protocol)+ str('_module')+str(opt.branch_id) + '_y_predicted_overlap_each_scan_test_MICCAI.json',
        #       'w', encoding='utf-8') as fp:
        #     json.dump(y_predicted_overlap_each_scan_test, fp, ensure_ascii=False, indent=4)


    else: #find the maxmum
        mean_each = pd.DataFrame(mean_each_task_each_scan).mean(axis=0).to_numpy()
        mean_min = max(mean_each)
        idx_min = mean_each.tolist().index(mean_min)
        mean_std = pd.DataFrame(mean_each_task_each_scan).std(axis=0).to_numpy()[idx_min]

        # mean_all = np.array(list(mean_each_task_all_scan.values())).mean()

        # std_all = np.array(list(mean_each_task_all_scan.values())).std()

        mean_all=np.nanmean(mean_each_task_each_scan.to_numpy())
        std_all=np.nanstd(mean_each_task_each_scan.to_numpy())

        best_perf = list(mean_each_task_each_scan.keys())[idx_min]

    # fig = plt.figure(figsize=(20, 10))
    # ax=fig.add_subplot(1, 1, 1)
    # mean_each_task_each_scan.boxplot(ax=ax, medianprops={"linewidth": 4}, showmeans=True)
    # ax.set_xticklabels(list(mean_each_task_each_scan.keys()), rotation=90)
    # # save
    #
    # fig2 = plt.figure(figsize=(20, 10))
    # ax2 = fig2.add_subplot(1, 1, 1)
    #
    # ax2.boxplot(np.array(list(mean_each_task_all_scan.values())), medianprops={"linewidth": 4}, showmeans=True)
    # # save
    # fig.savefig(saved_folder_test + '/' + saved_str + '_individual.png')
    # fig2.savefig(saved_folder_test + '/' + saved_str + '_overall.png')

    fig3 = plt.figure(figsize=(30, 10))
    ax3 = fig3.add_subplot(1, 1, 1)
    mean_each_task_each_scan.boxplot(ax=ax3,
                                     meanprops={"marker": "s", 'markersize': 2, 'markerfacecolor': "#EDC948",
                                                'markeredgecolor': "#EDC948"},
                                     medianprops={'linestyle': '-', "linewidth": 1, 'color': "k"},
                                     showfliers=False,
                                     patch_artist=True,
                                     boxprops=dict(facecolor='#4E79A7'),
                                     showmeans=True)
    ax3.set_xticklabels(list(mean_each_task_each_scan.keys()), rotation=90)
    ax3.grid(False)
    ax3.set_xlabel('Transformation tasks',fontsize=15)
    ax3.set_ylabel('Accumulated tracking error (mm)',fontsize=15)


    # save

    # fig4 = plt.figure(figsize=(20, 10))
    # ax4 = fig4.add_subplot(1, 1, 1)
    #
    # ax4.boxplot(np.array(list(mean_each_task_all_scan.values())), medianprops={"linewidth": 4}, showmeans=True,showfliers=False)
    # save
    if not os.path.exists(saved_folder_test):
        os.makedirs(saved_folder_test)

    fig3.savefig(saved_folder_test + '/' + saved_str + '_individual_wo.png')
    fig3.savefig(saved_folder_test + '/' + saved_str + '_individual_wo.pdf')
    fig3.savefig(saved_folder_test + '/' + saved_str + '_individual_wo.eps')

    # fig4.savefig(saved_folder_test + '/' + saved_str + '_overall_wo.png')


    return mean_min, mean_std, mean_all, std_all,mean_each_task_each_scan,best_perf


def get_metric_data(mean_each_task_each_scan,best_perf,overlap_flag,test_scan_idx):
# get the evaluation metric for one transformation
    mean_each=pd.DataFrame(mean_each_task_each_scan).mean(axis=0).to_numpy()
    idx_min=list(mean_each_task_each_scan.keys()).index(best_perf)
    mean_min = mean_each[idx_min]
    
    mean_std = pd.DataFrame(mean_each_task_each_scan).std(axis=0).to_numpy()[idx_min]

    # find the maxnum error and recomputing the mean and std after deleting it
    all_scan = mean_each_task_each_scan[best_perf]
    all_scan_np = all_scan.to_numpy()
    all_scan = all_scan.to_numpy()
    all_scan=all_scan[~np.isnan(all_scan)]
    # find the maximun error and re-computed the mean and std of the err
    all_scan=all_scan.tolist()
    
    if overlap_flag:
        max_idx=np.where(all_scan_np==np.nanmin(all_scan_np))
        all_scan.remove(min(all_scan))
    else:
        max_idx=np.where(all_scan_np==np.nanmax(all_scan_np))
        all_scan.remove(max(all_scan))

    mean_deleting=np.mean(np.array(all_scan))
    std_deleting=np.std(np.array(all_scan))

    idx=test_scan_idx['indices_in_use'][max_idx[0].tolist()[0]]
    


    return mean_min, mean_std,mean_deleting,std_deleting,idx

       
def get_hist(ax,ax_delete_max,mean_each_task_each_scan,best_perf,x_label,y_label,saved_folder):
    
    
    all_scan = mean_each_task_each_scan[best_perf]
    

    
    ax.hist(all_scan,20)

    ax.set_xlabel(x_label, fontsize=20) # , family='Times New Roman'
    ax.set_ylabel(y_label, fontsize=20, math_fontfamily='cm') # ,fontdict=csfont)#
    # ax.legend(fontsize=35)
    ax.tick_params(axis='both', labelsize=20)
    
    all_scan = all_scan.to_numpy()
    all_scan=all_scan[~np.isnan(all_scan)]
    # find the maximun error and re-computed the mean and std of the err
    all_scan=all_scan.tolist()
    

    max_idx=all_scan.index(max(all_scan))



    all_scan.remove(max(all_scan))
    ax_delete_max.hist(all_scan,20)

    ax_delete_max.set_xlabel(x_label, fontsize=20) # , family='Times New Roman'
    ax_delete_max.set_ylabel(y_label, fontsize=20, math_fontfamily='cm') # ,fontdict=csfont)#
    # ax.legend(fontsize=35)
    ax_delete_max.tick_params(axis='both', labelsize=20)



def get_error_each_scan(ax,mean_each_task_each_scan,best_perf,test_scan_idx,x_label,y_label):
# get the error of each scan in test set
    all_scan = mean_each_task_each_scan[best_perf]
    idx_scan=test_scan_idx['indices_in_use']
    all_scan = all_scan.to_numpy()
    non_nan_idx=np.argwhere(~np.isnan(all_scan))

    ax.bar(np.array(range(len(all_scan[non_nan_idx]))), np.squeeze(all_scan[non_nan_idx]))

    x_xtick=[str(idx_scan[int(i)]) for i in non_nan_idx]
    ax.set_xticks(np.array(range(len(all_scan[non_nan_idx]))))
    ax.set_xticklabels(x_xtick, rotation=90)

    



    ax.set_xlabel(x_label, fontsize=20) # , family='Times New Roman'
    ax.set_ylabel(y_label, fontsize=20, math_fontfamily='cm') # ,fontdict=csfont)#
    # ax.legend(fontsize=35)
    ax.tick_params(axis='both', labelsize=20)


 

    

    









def plot_box_points(ax,sta_in_each_interval_each_scan,x_label,y_label,mean_median):
    # plot dists v.s. intervals - plot each scan
    labels, data = sta_in_each_interval_each_scan.keys(), sta_in_each_interval_each_scan.values()
    # sort data increasing
    label_np = np.array(list(labels))
    index = np.argsort(label_np)
    data_list = list(data)
    labels_list = list(labels)
    data_list = [x for _, x in sorted(zip(labels_list, data_list))]

    labels_sorted = sorted(labels)

    if mean_median == 'mean':
        ax.boxplot(data_list, showmeans=True, meanprops={"linewidth": 4}, medianprops={"linewidth": 4})
    elif mean_median == 'median':
        ax.boxplot(data_list, medianprops={"linewidth": 4})

    ax.set_xticks(range(1, len(labels) +1), labels_sorted)
    # plot original data
    for i in range(len(list(labels))):
        y = list(data)[index[i]]
        try:
            # Add some random "jitter" to the x-axis
            x = np.random.normal(i+1, 0.04, size=y.shape[0])
        except:
            y=[y]
            x = np.random.normal(i+1, 0.04, size=len(y))

        ax.plot(x, y, 'r.', alpha=0.2,markersize=12)


    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def plot_2d_statistic(ax,sta_in_2d,x_label,y_label,title):
    keys = list(sta_in_2d.keys())
    x =[float(key.split('_')[0]) for key in keys]
    y = [float(key.split('_')[1]) for key in keys]
    value = [np.array([sta_in_2d[key]]) if isinstance(sta_in_2d[key], np.floating) else sta_in_2d[key] for key in keys]


    if pd.DataFrame(value).shape[1]>1:#each scan has a value; use DataFrame to avoid various length in different keys

        # color denete mean, radius note std
        mean_value = [value[i].mean() for i in range(len(value))]
        std_value = [value[i].std() for i in range(len(value))]
        mean_value = np.array(mean_value)
        if mean_value.mean()<1:
            mean_value = 100*mean_value # let _v ststistic multipy 100 to nake sure they are distinguishable when ploting, as the value of _v is less than 1
        mean_value = np.round(mean_value).astype(int)
        viridis = cm.get_cmap('viridis', max(mean_value)-min(mean_value)+1)
        im = ax.scatter(x, y,color=viridis.colors[mean_value-min(mean_value)], alpha=0.5, s=50*(np.round(std_value)+1))

    else:
        value = np.squeeze(np.array(value))
        if value.mean() < 1:
            value = 100 * value
        # value = list(map(int,np.round(value) ))
        value = np.round(value).astype(int)
        viridis = cm.get_cmap('viridis', max(value)-min(value)+1)

        im = ax.scatter(x, y,color=viridis.colors[value-min(value)], alpha=0.5, s=100)
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.grid(linestyle='--', linewidth=1,alpha=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    # add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    return cax, im



def plot_mean_std(interval_name, mean, std, viridis,pair_idx,ax):
    # plot mean and std with a shaded region

    x = range(interval_name[1], interval_name[1] + len(mean) * (interval_name[1] - interval_name[0]),interval_name[1] - interval_name[0])

    ax.fill_between(x, mean - std, mean + std, color=viridis.colors[pair_idx], alpha=0.3)
    ax.plot(x, mean, color=viridis.colors[pair_idx],label='interval_' + str(interval_name[0]) + '_' + str(interval_name[1]))
    ax.axhline(0, color='gray', linestyle='--')


def get_scan_len_hist(dataset_all):
    # get the length histogram of all the scans
    plt.hist(dataset_all.num_frames.flatten())
    plt.hist(dataset_all.num_frames)
    temp = dataset_all.num_frames.flatten()
    len_80 = len(temp[temp>80])
    plt.hist(temp[temp>80])

def get_img_normalization_mean_std():
    dataset_all = SSFrameDataset(
        min_scan_len=0,
        filename_h5=os.path.join(os.path.expanduser("~"), "workspace", 'frames_res{}'.format(4)+".h5"),
        num_samples=-1,
        sample_range=None
    )
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    psum,psum_sq=0,0
    for scan_index in range(len(dataset_all)):
        frames, tforms, tforms_inv = dataset_all[scan_index]
        frames = frames.astype('float64')
        if scan_index == 0:
            all_frames = frames
        else:
            all_frames = np.append(all_frames,frames, axis=0)
        channels_sum += (frames).mean()
        channels_squared_sum += (frames ** 2).mean()
        psum += frames.sum()
        psum_sq += (frames ** 2).sum()


    all_frames_flatten = all_frames.flatten()
    mean = all_frames_flatten.mean()
    std = all_frames_flatten.std()

    num_frames = dataset_all.num_frames.sum()
    mean_2 = channels_sum/len(dataset_all)
    std_2 = (channels_squared_sum/len(dataset_all) - mean_2 ** 2) ** 0.5

    count = num_frames * 120 * 160
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = np.sqrt(total_var)

    return all_frames


# def extract_frame_features(frames,device):
#     # encode frames
#     frames = torch.unsqueeze(frames, 2)
#     pretrained_model = Pretrained_model(1).to(device)
#     frame_frets = torch.empty(frames.shape[0], frames.shape[1], 1000)
#     for i in range(frames.shape[0]):
#         frame_frets[i, ...] = pretrained_model(frames[i, :, :, :])
#     # test = frames.view(-1, frames.shape[2], frames.shape[3], frames.shape[4])
#     # test_f = pretrained_model(test)
#     # test_f==frame_frets
#
#     return frame_frets

def save_dict_to_pickle(datapath,data_list,metric_name,train_set_type,model_type):

    string_type=['interval_each_scan','interval_all_scan',
                 'pf_each_scan','pf_all_scan',
                 'ff_each_scan','ff_all_scan',
                 'pf_add_ff_each_scan','pf_add_ff_all_scan',
                 'pf_div_ff_each_scan','pf_div_ff_all_scan']
    for i in range(len(data_list)):
        data=data_list[i]
        if model_type == 'efficientnet_b1':
            if not train_set_type:
                saved_name= metric_name+'_'+string_type[i]

            else:
                saved_name= metric_name+'_'+string_type[i]+'_'+str(train_set_type)
        elif model_type == 'LSTM_E':
            saved_name = metric_name + '_' + string_type[i] + '_' + str('LSTM_E')

        with open(datapath+'/'+saved_name+'.pkl', 'wb') as f:
            pickle.dump(data, f)

def load_dict_from_pickle(datapath):

    with open(datapath+'.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    return loaded_dict

def plot_all_results(datapath,metric_name,watched_variable,all_each_scan,type_all,smooth_rate):
    # metric_name: 'frame_acc'
    # watched_variable: 'pf'
    # all_each_scan:'all_scan'
    # type_all: [None,'linear','c_s','remain_ind_in_use_25','remain_ind_in_use_50','remain_ind_in_use_75']
    #
    suffixs=[None,'linear','c_s','remain_ind_in_use_25','remain_ind_in_use_50','remain_ind_in_use_75','half_use','75_use']
    type_all=['All train set','Straight','C-shape and s-shape','Subject reduction 75%','Subject reduction 50%','Subject reduction 25%','Half length','75% length']

    COLORS = mpl.colormaps['Dark2']
    viridis=['#A52A2A','tab:green','#E3CF57','#D2691E','#9932CC','violet','tab:cyan','#3D59AB']
    variables = ['pf', 'ff', 'interval', 'pf_add_ff', 'pf_div_ff']
    x_labels = ['Number of past frames', 'Number of future frames','Interval value', 'Number of past add future frames', 'Number of future divide past frames']
    x_label = x_labels[variables.index(watched_variable)]

    metric_names=['avg_pixel_dists','frame_acc', 'frame_acc_unite','final_drift','overlap']
    y_labels = ['Accumulated tracking error (mm)', 'Non_divide frame prediction accuracy (mm)','Frame prediction accuracy (mm)', 'Final drift (mm)', 'Volume reconstruction overlap (mm)']
    y_label=y_labels[metric_names.index(metric_name)]
    y_limits=[[0,50],[10,50],[0,1],[0,1]]




    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(1, 1, 1)

    # plot baseline
    load_datapath = datapath + '/' + 'baseline_' + metric_name + '_' + watched_variable + '_' + all_each_scan
    data = load_dict_from_pickle(load_datapath)

    plot_mean_var(smooth_rate, True, ax, data,  x_label, y_label, 'gray', 'Baseline')

    for i in range(len(type_all)):

        suffix=suffixs[i]
        # suffix=suffixs[3]
        if not suffix:
            load_datapath=datapath+'/'+metric_name+'_'+watched_variable+'_'+all_each_scan
            data = load_dict_from_pickle(load_datapath)
            plot_mean_var(smooth_rate, False, ax, data,  x_label, y_label, viridis[i], type_all[i])


        else:
            load_datapath=datapath+'/'+metric_name+'_'+watched_variable+'_'+all_each_scan+'_'+suffix
            data = load_dict_from_pickle(load_datapath)
            # cnn_ff79_remain_in_use_20 = data[79]
            # with open('cnn_ff79_remain_in_use_20.json', 'w', encoding='utf-8') as fp:
            #     json.dump(cnn_ff79_remain_in_use_20.tolist(), fp, ensure_ascii=False, indent=4)

            # cnn_ff74_remain_in_use_20 = data[74]
            # with open('cnn_ff74_remain_in_use_20.json', 'w', encoding='utf-8') as fp:
            #     json.dump(cnn_ff74_remain_in_use_20.tolist(), fp, ensure_ascii=False, indent=4)

            plot_mean_var(smooth_rate, False, ax, data, 'the number of ' + x_label, y_label, viridis[i], type_all[i])

    # if y_label=='accumulated tracking error' and x_label=='the number of past frames':
    #     y_limit=[0,50]
    #     ax.set_ylim(y_limit)
    if y_label=='Accumulated tracking error (mm)' and x_label=='Number of future frames':
        y_limit = [None, 120]
        ax.set_ylim(y_limit)
    elif y_label=='Frame prediction accuracy (mm)' and x_label=='Number of past frames':
        y_limit = [None,2]
        ax.set_ylim(y_limit)
    # elif y_label=='frame prediction accuracy' and x_label=='the number of future frames':
    #     y_limit = [0,1]
    #     ax.set_ylim(y_limit)

    ax.set_xlabel(x_label, fontsize=50)
    ax.set_ylabel(y_label, fontsize=50)
    ax.legend(fontsize=30)
    ax.tick_params(axis='both', labelsize=50)
    fig.savefig(datapath + '/' + y_label + '_'+x_label + '_smooth_rate_' + str(smooth_rate)  + '.png')
    fig.savefig(datapath + '/' + y_label + '_'+x_label + '_smooth_rate_' + str(smooth_rate)  + '.pdf')
    fig.savefig(datapath + '/' + y_label + '_'+x_label + '_smooth_rate_' + str(smooth_rate)  + '.eps')



def plot_cnn_lstm_results(datapath,metric_name,watched_variable,all_each_scan,type_all,smooth_rate):
    # metric_name: 'frame_acc'
    # watched_variable: 'pf'
    # all_each_scan:'all_scan'
    # type_all: [None,'linear','c_s','remain_ind_in_use_25','remain_ind_in_use_50','remain_ind_in_use_75']
    #
    suffixs=[None,'LSTM_E']
    type_all=['pf', 'ff']
    legend=['Past frame dependency', 'Future frame dependency']

    viridis = ['tab:blue','tab:purple']
    variables = ['efficientnet_b1', 'LSTM_E']
    x_label = 'Number of frames'
    suffix = suffixs[variables.index(watched_variable)]

    metric_names=['avg_pixel_dists','frame_acc', 'frame_acc_unite','final_drift','overlap']
    y_labels = ['Accumulated tracking error (mm)', 'Non_divide frame prediction accuracy (mm)','Frame prediction accuracy (mm)', 'Final drift (mm)', 'Volume reconstruction overlap (mm)']
    y_label=y_labels[metric_names.index(metric_name)]
    # y_limits=[[0,50],[10,50],[0,1],[0,1]]


    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)

    # plot baseline
    load_datapath = datapath + '/' + 'baseline_' + metric_name + '_' + 'pf' + '_' + all_each_scan
    data = load_dict_from_pickle(load_datapath)
    # cnn_baseline=data[0]
    # with open('cnn_baseline.json','w', encoding='utf-8') as fp:
    #     json.dump(cnn_baseline.tolist(), fp, ensure_ascii=False, indent=4)

    plot_mean_var(smooth_rate, True, ax, data,  x_label, y_label, 'gray', 'Baseline')

    for i in range(len(type_all)):
        # suffix=suffixs[i]
        if not suffix:
            load_datapath=datapath+'/'+metric_name+'_'+type_all[i]+'_'+all_each_scan
            data = load_dict_from_pickle(load_datapath)
            plot_mean_var(smooth_rate, False, ax, data,  x_label, y_label, viridis[i], legend[i])
            #     save data to compute p-value
            # cnn_pf_74=data[74]
            # with open('cnn_pf_74.json','w', encoding='utf-8') as fp:
            #     json.dump(cnn_pf_74.tolist(), fp, ensure_ascii=False, indent=4)
            # cnn_pf_20 = data[20]
            # with open('cnn_pf_20.json', 'w', encoding='utf-8') as fp:
            #     json.dump(cnn_pf_20.tolist(), fp, ensure_ascii=False, indent=4)

        else:
            load_datapath=datapath+'/'+metric_name+'_'+type_all[i]+'_'+all_each_scan+'_'+suffix
            data = load_dict_from_pickle(load_datapath)
            # lstm_pf_79=data[79]
            # with open('lstm_pf_79.json','w', encoding='utf-8') as fp:
            #     json.dump(lstm_pf_79.tolist(), fp, ensure_ascii=False, indent=4)
            # lstm_pf_89 = data[89]
            # with open('lstm_pf_89.json', 'w', encoding='utf-8') as fp:
            #     json.dump(lstm_pf_89.tolist(), fp, ensure_ascii=False, indent=4)

            #
            # lstm_pf_20 = data[20]
            # with open('lstm_pf_20.json', 'w', encoding='utf-8') as fp:
            #     json.dump(lstm_pf_20.tolist(), fp, ensure_ascii=False, indent=4)

            plot_mean_var(smooth_rate, False, ax, data,  x_label, y_label, viridis[i], legend[i])

    # if y_label=='accumulated tracking error' and x_label=='the number of past frames':
    #     y_limit=[0,50]
    #     ax.set_ylim(y_limit)
    # if y_label=='accumulated tracking error (mm)' and x_label=='the number of future frames':
    #     y_limit = [None, 120]
    #     ax.set_ylim(y_limit)
    # elif y_label=='frame prediction accuracy (mm)' and x_label=='the number of past frames':
    #     y_limit = [None,2]
    #     ax.set_ylim(y_limit)
    # elif y_label=='frame prediction accuracy' and x_label=='the number of future frames':
    #     y_limit = [0,1]
    #     ax.set_ylim(y_limit)

    ax.set_xlabel(x_label, fontsize=35)
    ax.set_ylabel(y_label, fontsize=35)
    ax.legend(fontsize=20)
    ax.tick_params(axis='both', labelsize=35)
    fig.savefig(datapath + '/' + y_label + '_'+x_label + '_smooth_rate_' + str(smooth_rate)+'_'+watched_variable  + '.png')
    fig.savefig(datapath + '/' + y_label + '_'+x_label + '_smooth_rate_' + str(smooth_rate)+'_'+watched_variable   + '.pdf')
    fig.savefig(datapath + '/' + y_label + '_'+x_label + '_smooth_rate_' + str(smooth_rate)+'_'+watched_variable   + '.eps')





