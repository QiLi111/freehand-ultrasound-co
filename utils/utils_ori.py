
import torch
import heapq
import MDAnalysis.lib.transformations as MDA
import numpy as np
import pytorch3d.transforms
import pandas as pd
import os
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from utils.loss import PointDistance
from utils.loader import SSFrameDataset


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

def pair_samples_DCL(num_samples, num_pred, single_interval):
    """
    :param num_samples:
    :param num_pred: number of the (last) samples, for which the transformations are predicted
        For each "pred" frame, pairs are formed with every one previous frame
    :param single_interval: 0 - use all interval predictions
                            1,2,3,... - use only specific intervals
    """

    return torch.tensor([[n0, n0+1] for n0 in range(0,num_samples-1) ])



def type_dim(label_pred_type, num_points=None, num_pairs=1):
    type_dim_dict = {
        "transform": 12,
        "parameter": 6,
        "point": num_points*3,
        "quaternion": 7
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
        if opt.multi_gpu:
            torch.save(model.module.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'best_validation_loss_model' ))

        else:
            torch.save(model.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'best_validation_loss_model' ))
        print('Best validation loss parameters saved.')
    else:
        val_loss_min = val_loss_min

    if running_dist_val < val_dist_min:
        val_dist_min = running_dist_val
        file_name = os.path.join(opt.SAVE_PATH, 'config.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write('------------ best validation dist result - epoch %s: -------------\n' % (str(epoch_label)))
        
        if opt.multi_gpu:
            torch.save(model.module.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'best_validation_loss_model' ))
        else:
            torch.save(model.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'best_validation_dist_model'))
        print('Best validation dist parameters saved.')
    else:
        val_dist_min = val_dist_min

    return val_loss_min, val_dist_min

def save_best_network_rec(opt, model, epoch_label, running_loss_val, running_dist_val, val_loss_min, val_dist_min,count_non_improved_loss):
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
            opt_file.write('------------ rec - best validation loss result - epoch %s: -------------\n' % (str(epoch_label)))
        if opt.multi_gpu:
            torch.save(model.module.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'best_validation_loss_model' ))
        else:
            torch.save(model.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'best_validation_loss_model' ))

        print('Best rec validation loss parameters saved.')
        count_non_improved_loss = 0
    else:
        val_loss_min = val_loss_min
        count_non_improved_loss += 1

    if running_dist_val < val_dist_min:
        val_dist_min = running_dist_val
        file_name = os.path.join(opt.SAVE_PATH, 'config.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write('------------ rec - best validation dist result - epoch %s: -------------\n' % (str(epoch_label)))
        if opt.multi_gpu:
            torch.save(model.module.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'best_validation_dist_model'))

        else:
            torch.save(model.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'best_validation_dist_model'))

        print('Best rec validation dist parameters saved.')
    else:
        val_dist_min = val_dist_min

    return val_loss_min, val_dist_min, count_non_improved_loss


def save_best_network_reg(opt,VoxelMorph_net, epoch_label, running_loss_val, val_loss_min,count_non_improved_loss):
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
            opt_file.write('------------reg - best validation loss result - epoch %s: -------------\n' % (str(epoch_label)))
        if opt.multi_gpu:
            torch.save(VoxelMorph_net.module.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'best_validation_loss_model_reg' ))
        else:
            torch.save(VoxelMorph_net.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'best_validation_loss_model_reg' ))

        print('Best reg validation loss parameters saved.')
        count_non_improved_loss = 0
    else:
        val_loss_min = val_loss_min
        count_non_improved_loss += 1

    

    return val_loss_min, count_non_improved_loss



def sample_adjacent_pair(start, step, data_pairs):
    adjacent_pair = []
    while 1:
        adjacent_pair.append(start)
        start = start + step
        step = step + 1
        if start >= data_pairs.shape[0]:
            break
    return adjacent_pair # data_pairs[adjacent_pair]

def add_scalars(writer,epoch, loss_dists):
    # loss and average distance in training and val
    train_epoch_loss = loss_dists['train_epoch_loss']
    train_epoch_dist = loss_dists['train_epoch_dist']
    epoch_loss_val = loss_dists['epoch_loss_val']
    epoch_dist_val = loss_dists['epoch_dist_val']
    train_epoch_loss_reg = loss_dists['train_epoch_loss_reg']
    epoch_loss_val_reg = loss_dists['epoch_loss_val_reg']
    train_epoch_loss_rec = loss_dists['train_epoch_loss_rec']
    epoch_loss_val_rec = loss_dists['epoch_loss_val_rec']


    
    writer.add_scalars('loss_rec_reg', {'train_loss': train_epoch_loss},epoch)
    writer.add_scalars('loss_rec_reg', {'val_loss': epoch_loss_val},epoch)
    writer.add_scalars('dist', {'train_dist': train_epoch_dist}, epoch)
    writer.add_scalars('dist', {'val_dist': epoch_dist_val}, epoch)

    writer.add_scalars('loss_reg', {'train_loss': train_epoch_loss_reg},epoch)
    writer.add_scalars('loss_reg', {'val_loss': epoch_loss_val_reg},epoch)

    writer.add_scalars('loss_rec', {'train_loss': train_epoch_loss_rec},epoch)
    writer.add_scalars('loss_rec', {'val_loss': epoch_loss_val_rec},epoch)

def add_scalars_rec_volume(writer,epoch, loss_dists):
    # loss and average distance in training and val
    train_epoch_loss = loss_dists['train_epoch_loss_all']
    train_epoch_dist = loss_dists['train_epoch_dist']
    epoch_loss_val = loss_dists['epoch_loss_val_all']
    epoch_dist_val = loss_dists['epoch_dist_val']
    train_epoch_loss_reg = loss_dists['train_epoch_loss_reg']
    epoch_loss_val_reg = loss_dists['epoch_loss_val_reg']
    train_epoch_loss_rec = loss_dists['train_epoch_loss_rec']
    epoch_loss_val_rec = loss_dists['epoch_loss_val_rec']


    
    writer.add_scalars('loss_rec_all', {'train_loss': train_epoch_loss},epoch)
    writer.add_scalars('loss_rec_all', {'val_loss': epoch_loss_val},epoch)
    writer.add_scalars('dist', {'train_dist': train_epoch_dist}, epoch)
    writer.add_scalars('dist', {'val_dist': epoch_dist_val}, epoch)

    writer.add_scalars('loss_rec_volume', {'train_loss': train_epoch_loss_reg},epoch)
    writer.add_scalars('loss_rec_volume', {'val_loss': epoch_loss_val_reg},epoch)

    writer.add_scalars('loss_rec', {'train_loss': train_epoch_loss_rec},epoch)
    writer.add_scalars('loss_rec', {'val_loss': epoch_loss_val_rec},epoch)

    

def add_scalars_reg(writer,epoch, loss_dists,model_name):
    # loss in training and val
    train_epoch_loss = loss_dists['train_epoch_loss_reg_only']
    epoch_loss_val = loss_dists['epoch_loss_val_reg_only']

    writer.add_scalars('loss_reg_only', {'train_loss_'+model_name: train_epoch_loss},epoch)
    writer.add_scalars('loss_reg_only', {'val_loss_'+model_name: epoch_loss_val},epoch)

def add_scalars_reg_T(writer,epoch, loss_dists,model_name):
    # loss in training and val
    train_epoch_loss = loss_dists['train_dist_reg_T']
    epoch_loss_val = loss_dists['val_dist_reg_T']

    writer.add_scalars('T_dist_in_R', {'train_loss_'+model_name: train_epoch_loss},epoch)
    writer.add_scalars('T_dist_in_R', {'val_loss_'+model_name: epoch_loss_val},epoch)

def add_scalars_wrap_dist(writer,epoch, loss_dists,model_name):
    # loss in training and val
    train_epoch_loss = loss_dists['train_wrap_dist']
    epoch_loss_val = loss_dists['val_wrap_dist']

    writer.add_scalars('wrap_dist_'+model_name, {'train_wrap_dist_'+model_name: train_epoch_loss},epoch)
    writer.add_scalars('wrap_dist_'+model_name, {'val_wrap_dist_'+model_name: epoch_loss_val},epoch)



def add_scalars_loss(writer, epoch, loss_dists):
    # loss and average distance in training and val
    train_epoch_loss = loss_dists['train_epoch_loss']
    train_epoch_dist = loss_dists['train_epoch_dist']
    epoch_loss_val = loss_dists['epoch_loss_val']
    epoch_dist_val = loss_dists['epoch_dist_val']



    writer.add_scalars('loss_average_dists', {'train_loss': train_epoch_loss.item(), 'val_loss': epoch_loss_val.item()},epoch)
    writer.add_scalars('loss_average_dists',{'train_dists': train_epoch_dist.item(), 'val_dists': epoch_dist_val.item()}, epoch)


def add_scalars_params_1(writer,epoch,params_gt_train,params_np_train,params_gt_val,params_np_val):
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


def add_scalars_params(writer,epoch,error_6DOF_train,error_6DOF_val):
#     compute the absolute error of 6 parameters
    # abs_errs_train,abs_errs_val = {},{}
    # for i in range(len(params_gt_train)):
    #     abs_errs_train[str(i)],abs_errs_val[str(i)] = None, None

    # for i in range(len(error_6DOF_val)):
        # abs_errs_train[str(i)] = torch.abs(params_gt_train[str(i)] - params_np_train[str(i)])
        # abs_errs_val[str(i)] = torch.abs(params_gt_val[str(i)] - params_np_val[str(i)])

    writer.add_scalars('params_abs_err_angle_1', {'train': error_6DOF_train[0], 'val': error_6DOF_val[0]},epoch)
    writer.add_scalars('params_abs_err_angle_2', {'train': error_6DOF_train[1], 'val': error_6DOF_val[1]},epoch)
    writer.add_scalars('params_abs_err_angle_3', {'train': error_6DOF_train[2], 'val': error_6DOF_val[2]},epoch)
    writer.add_scalars('params_abs_err_x', {'train': error_6DOF_train[3], 'val': error_6DOF_val[3]},epoch)
    writer.add_scalars('params_abs_err_y', {'train': error_6DOF_train[4], 'val': error_6DOF_val[4]},epoch)
    writer.add_scalars('params_abs_err_z', {'train': error_6DOF_train[5], 'val': error_6DOF_val[5]},epoch)

def add_scalars_params_1(writer,epoch,params_gt_train,params_np_train,params_gt_val,params_np_val):
#     compute the absolute error of 6 parameters
    abs_errs_train,abs_errs_val = {},{}
    for i in range(len(params_gt_train)):
        abs_errs_train[str(i)],abs_errs_val[str(i)] = None, None

    for i in range(len(params_gt_train)):
        abs_errs_train[str(i)] = torch.abs(params_gt_train[str(i)] - params_np_train[str(i)])
        abs_errs_val[str(i)] = torch.abs(params_gt_val[str(i)] - params_np_val[str(i)])

        writer.add_scalars('params_abs_err_angle_1', {'train_%d' % i: abs_errs_train[str(i)][0], 'val_%d' % i: abs_errs_val[str(i)][0]},epoch)
        writer.add_scalars('params_abs_err_angle_2', {'train_%d' % i: abs_errs_train[str(i)][1], 'val_%d' % i: abs_errs_val[str(i)][1]},epoch)
        writer.add_scalars('params_abs_err_angle_3', {'train_%d' % i: abs_errs_train[str(i)][2], 'val_%d' % i: abs_errs_val[str(i)][2]},epoch)
        writer.add_scalars('params_abs_err_x', {'train_%d' % i: abs_errs_train[str(i)][3], 'val_%d' % i: abs_errs_val[str(i)][3]},epoch)
        writer.add_scalars('params_abs_err_y', {'train_%d' % i: abs_errs_train[str(i)][4], 'val_%d' % i: abs_errs_val[str(i)][4]},epoch)
        writer.add_scalars('params_abs_err_z', {'train_%d' % i: abs_errs_train[str(i)][5], 'val_%d' % i: abs_errs_val[str(i)][5]},epoch)



def write_to_txt(opt,epoch, loss_dists):
    # write loss, average distance, accumulated distance into txt
    # for the last step in each epoch
    # loss and average distance in training and val
    train_epoch_loss = loss_dists['train_epoch_loss']
    train_epoch_dist = loss_dists['train_epoch_dist'].mean()
    epoch_loss_val = loss_dists['epoch_loss_val']
    epoch_dist_val = loss_dists['epoch_dist_val'].mean()
    dist_train,dist_val = [],[]

    # for i in range(len(preds_dist_all_train)):
    #     dist_train.append(((preds_dist_all_train[str(i)] - label_dist_all_train[str(i)]) ** 2).sum(dim=1).sqrt().mean())
    #     dist_val.append(((preds_dist_all_val[str(i)]-label_dist_all_val[str(i)])**2).sum(dim=1).sqrt().mean())

    # dist_train = torch.tensor(dist_train)
    # dist_val = torch.tensor(dist_val)
    file_name_train = os.path.join(opt.SAVE_PATH, 'train_results', 'train_loss.txt')
    with open(file_name_train, 'a') as opt_file_train:
        print('[Epoch %d], for one epoch, train-loss=%.3f, train-dist=%.3f' % (epoch, train_epoch_loss, train_epoch_dist),file=opt_file_train)
        # print('[for one epoch, %d kinds of accumulated dists]:' % (len(dist_train)),file=opt_file_train)
        # print('%.3f ' * len(dist_train) % tuple(dist_train),file=opt_file_train)

    file_name_val = os.path.join(opt.SAVE_PATH, 'val_results', 'val_loss.txt')
    with open(file_name_val, 'a') as opt_file_val:
        print('[Epoch %d], for one epoch, val-loss=%.3f, val-dist=%.3f' % (epoch, epoch_loss_val, epoch_dist_val), file=opt_file_val)
        # print('[for one step, %d kinds of accumulated dists]' % (len(dist_val)),file=opt_file_val)
        # print('%.3f ' * len(dist_val) % tuple(dist_val),file=opt_file_val)


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

    for i in range(len(rmse_img_var_intervals)):
        # x = range(opt_test.INTERVAL_LIST[i], (len(rmse_img_var_intervals[i])+1)*opt_test.INTERVAL_LIST[i], opt_test.INTERVAL_LIST[i])
        x = range(data_pair[opt_test.PAIR_INDEX[i]][1], data_pair[opt_test.PAIR_INDEX[i]][1]+(len(rmse_img_var_intervals[i]))*(data_pair[opt_test.PAIR_INDEX[i]][1]-data_pair[opt_test.PAIR_INDEX[i]][0]), (data_pair[opt_test.PAIR_INDEX[i]][1]-data_pair[opt_test.PAIR_INDEX[i]][0]))

        ax2.plot(x,rmse_img_var_intervals[i],color=viridis.colors[i], linestyle='dashed', marker='o', markersize=3,label='interval_'+str(data_pair[opt_test.PAIR_INDEX[i]][0].item())+'_'+str(data_pair[opt_test.PAIR_INDEX[i]][1].item()))
    ax2.legend()
    ax2.set_xlabel('img index')
    ax2.set_ylabel('accumulated error')
    # plt.title('accumulated')

    for i in range(len(rmse_img_var_intervals_gt_based)):
        # x = range(opt_test.INTERVAL_LIST[i], (len(rmse_img_var_intervals[i])+1)*opt_test.INTERVAL_LIST[i], opt_test.INTERVAL_LIST[i])
        x = range(data_pair[opt_test.PAIR_INDEX[i]][1], data_pair[opt_test.PAIR_INDEX[i]][1]+(len(rmse_img_var_intervals_gt_based[i]))*(data_pair[opt_test.PAIR_INDEX[i]][1]-data_pair[opt_test.PAIR_INDEX[i]][0]), (data_pair[opt_test.PAIR_INDEX[i]][1]-data_pair[opt_test.PAIR_INDEX[i]][0]))

        ax3.plot(x,rmse_img_var_intervals_gt_based[i]/(data_pair[opt_test.PAIR_INDEX[i]][1].item()-data_pair[opt_test.PAIR_INDEX[i]][0].item()),color=viridis.colors[i],linestyle='dashed', marker='o', markersize=3, label='interval_'+str(data_pair[opt_test.PAIR_INDEX[i]][0].item())+'_'+str(data_pair[opt_test.PAIR_INDEX[i]][1].item()))
    ax3.legend()
    ax3.set_xlabel('img index')
    ax3.set_ylabel('each image error')
    plt.title('non-accumulated')

    for i in range(len(rmse_img_var_intervals_gt_based)):
        # x = range(opt_test.INTERVAL_LIST[i], (len(rmse_img_var_intervals[i])+1)*opt_test.INTERVAL_LIST[i], opt_test.INTERVAL_LIST[i])
        x = range(data_pair[opt_test.PAIR_INDEX[i]][1], data_pair[opt_test.PAIR_INDEX[i]][1]+(len(rmse_img_var_intervals_gt_based[i]))*(data_pair[opt_test.PAIR_INDEX[i]][1]-data_pair[opt_test.PAIR_INDEX[i]][0]), (data_pair[opt_test.PAIR_INDEX[i]][1]-data_pair[opt_test.PAIR_INDEX[i]][0]))
        acc_error = []
        for j in range(len(rmse_img_var_intervals_gt_based[i])):
            if j == 0:
                acc_error.append(rmse_img_var_intervals_gt_based[i][j])
            else:
                acc_error.append(rmse_img_var_intervals_gt_based[i][j]+acc_error[-1])

        ax4.plot(x,np.array(acc_error),color=viridis.colors[i],linestyle='dashed', marker='o', markersize=3, label='interval_'+str(data_pair[opt_test.PAIR_INDEX[i]][0].item())+'_'+str(data_pair[opt_test.PAIR_INDEX[i]][1].item()))
    ax4.legend()
    ax4.set_xlabel('img index')
    ax4.set_ylabel('accumulated error')
    plt.title('non-accumulated')



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

def plot_all_intervals(opt,json_fn,saved_folder,model_name,input_num,dataset_all):
    # plot the mean and std of each interval over all scans
    with open(opt.SAVE_PATH+'/'+json_fn+'.json', "r", encoding='utf-8') as f:
        rmse_intervals_each_scan = json.load(f)

    if 'avg_pixel_dists' in json_fn:
        avg_pixel_dists=True
    else:
        avg_pixel_dists = False

    rmse_intervals_each_scan_df = pd.DataFrame(rmse_intervals_each_scan)
    intervals = rmse_intervals_each_scan_df[rmse_intervals_each_scan_df.keys()[0]].keys().to_list()

    # get scan name
    scan_name_all = [dataset_all.name_scan[i, j].decode("utf-8") for i in range(dataset_all.name_scan.shape[0]) for j in range(dataset_all.name_scan.shape[1])]
    scan_len_all = [dataset_all.num_frames[i, j] for i in range(dataset_all.name_scan.shape[0]) for j in range(dataset_all.name_scan.shape[1])]

    scan_name = rmse_intervals_each_scan_df.keys().tolist()
    scan_len=[scan_len_all[scan_name_all.index(scan_name[i])]   for i in range(len(scan_name))]

    viridis = cm.get_cmap('viridis', rmse_intervals_each_scan_df.shape[0])
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    fig1 = plt.figure(figsize=(20, 10))
    ax1 = fig1.add_subplot(1, 2, 1)
    ax2 = fig1.add_subplot(1, 2, 2)
    fig2 = plt.figure(figsize=(20, 10))
    ax3 = fig2.add_subplot(2, 2, 1)
    ax4 = fig2.add_subplot(2, 2, 2)
    ax3_1 = fig2.add_subplot(2, 2, 3)
    ax4_1 = fig2.add_subplot(2, 2, 4)


    fig3 = plt.figure(figsize=(20, 10))
    ax5 = fig3.add_subplot(2, 2, 1)
    ax6 = fig3.add_subplot(2, 2, 2)
    ax5_1 = fig3.add_subplot(2, 2, 3)
    ax6_1 = fig3.add_subplot(2, 2, 4)

    fig4 = plt.figure(figsize=(20, 10))
    ax7 = fig4.add_subplot(2, 2, 1)
    ax8 = fig4.add_subplot(2, 2, 2)
    ax7_1 = fig4.add_subplot(2, 2, 3)
    ax8_1 = fig4.add_subplot(2, 2, 4)

    fig5 = plt.figure(figsize=(20, 10))
    ax9 = fig5.add_subplot(2, 2, 1)
    ax10 = fig5.add_subplot(2, 2, 2)
    ax9_1 = fig5.add_subplot(2, 2, 3)
    ax10_1 = fig5.add_subplot(2, 2, 4)

    fig6 = plt.figure(figsize=(20, 10))
    ax11 = fig6.add_subplot(2, 2, 1)
    ax12 = fig6.add_subplot(2, 2, 2)
    ax11_1 = fig6.add_subplot(2, 2, 3)
    ax12_1 = fig6.add_subplot(2, 2, 4)

    fig7 = plt.figure(figsize=(20, 10))
    ax13 = fig7.add_subplot(2, 2, 1)
    ax14 = fig7.add_subplot(2, 2, 2)
    ax13_1 = fig7.add_subplot(2, 2, 3)
    ax14_1 = fig7.add_subplot(2, 2, 4)

    fig8 = plt.figure(figsize=(20, 10))
    ax15 = fig8.add_subplot(2, 2, 1)
    ax16 = fig8.add_subplot(2, 2, 2)
    ax15_1 = fig8.add_subplot(2, 2, 3)
    ax16_1 = fig8.add_subplot(2, 2, 4)

    fig9 = plt.figure(figsize=(20, 10))
    ax17 = fig9.add_subplot(2, 2, 1)
    ax18 = fig9.add_subplot(2, 2, 2)
    ax17_1 = fig9.add_subplot(2, 2, 3)
    ax18_1 = fig9.add_subplot(2, 2, 4)

    fig10 = plt.figure(figsize=(20, 10))
    ax19 = fig10.add_subplot(2, 2, 1)
    ax20 = fig10.add_subplot(2, 2, 2)
    ax19_1 = fig10.add_subplot(2, 2, 3)
    ax20_1 = fig10.add_subplot(2, 2, 4)

    fig11 = plt.figure(figsize=(20, 10))
    ax23 = fig11.add_subplot(2, 2, 1)
    ax24 = fig11.add_subplot(2, 2, 2)
    ax23_1 = fig11.add_subplot(2, 2, 3)
    ax24_1 = fig11.add_subplot(2, 2, 4)

    fig12 = plt.figure(figsize=(20, 10))
    ax28_1, ax29_1 = fig12.add_subplot(1, 2, 1),fig12.add_subplot(1, 2, 2)

    fig13 = plt.figure(figsize=(20, 10))
    ax28, ax29 = fig13.add_subplot(1, 2, 1), fig13.add_subplot(1, 2, 2)

    fig14 = plt.figure(figsize=(20, 10))
    ax32, ax33 = fig14.add_subplot(1, 2, 1), fig14.add_subplot(1, 2, 2)
    fig15 = plt.figure(figsize=(20, 10))
    ax36, ax37 = fig15.add_subplot(1, 2, 1), fig15.add_subplot(1, 2, 2)

    fig16 = plt.figure(figsize=(20, 10))
    ax40, ax41 = fig16.add_subplot(1, 2, 1), fig16.add_subplot(1, 2, 2)

    fig17 = plt.figure(figsize=(20, 10))
    ax44, ax45 = fig17.add_subplot(1, 2, 1), fig17.add_subplot(1, 2, 2)

    fig18 = plt.figure(figsize=(20, 10))
    ax48, ax49 = fig18.add_subplot(1, 2, 1), fig18.add_subplot(1, 2, 2)

    fig19 = plt.figure(figsize=(20, 10))
    ax52, ax53 = fig19.add_subplot(1, 2, 1), fig19.add_subplot(1, 2, 2)


    fig_box_plot = plt.figure(figsize=(20, 10))
    fig_box_plot_wo_outliers= plt.figure(figsize=(20, 10))
    num_fig = rmse_intervals_each_scan_df.shape[0]
    rows = 3
    if num_fig % rows != 0:
        cols = int(num_fig / rows) + 1
    else:
        cols = int(num_fig / rows)

    #
    sta_in_each_interval_each_scan,sta_in_each_interval_all_scan = {},{}
    sta_in_each_interval_each_scan_v, sta_in_each_interval_all_scan_v = {},{}
    sta_in_each_pf_each_scan, sta_in_each_pf_all_scan = {}, {} # past frames
    sta_in_each_ff_each_scan, sta_in_each_ff_all_scan = {}, {} # future frames
    sta_in_each_pf_add_ff_each_scan, sta_in_each_pf_add_ff_all_scan = {}, {} # past frames add future frames
    sta_in_each_pf_div_ff_each_scan, sta_in_each_pf_div_ff_all_scan = {}, {} # past frames divide future frames

    sta_in_each_pf_each_scan_v, sta_in_each_pf_all_scan_v = {}, {}
    sta_in_each_ff_each_scan_v, sta_in_each_ff_all_scan_v = {}, {}
    sta_in_each_pf_add_ff_each_scan_v, sta_in_each_pf_add_ff_all_scan_v = {}, {}
    sta_in_each_pf_div_ff_each_scan_v, sta_in_each_pf_div_ff_all_scan_v = {}, {}

    sta_in_interval_pf_each_scan,sta_in_interval_pf_all_scan = {},{}
    sta_in_interval_pf_each_scan_v, sta_in_interval_pf_all_scan_v = {}, {}
    sta_in_interval_ff_each_scan,sta_in_interval_ff_all_scan = {},{}
    sta_in_interval_ff_each_scan_v,sta_in_interval_ff_all_scan_v = {},{}

    sta_in_interval_pf_add_ff_each_scan, sta_in_interval_pf_add_ff_all_scan = {}, {}
    sta_in_interval_pf_add_ff_each_scan_v, sta_in_interval_pf_add_ff_all_scan_v = {}, {}
    sta_in_interval_pf_div_ff_each_scan, sta_in_interval_pf_div_ff_all_scan = {}, {}
    sta_in_interval_pf_div_ff_each_scan_v, sta_in_interval_pf_div_ff_all_scan_v = {}, {}



    for pair_idx in range(rmse_intervals_each_scan_df.shape[0]):
        interval_name = intervals[pair_idx]
        interval_all_scan = rmse_intervals_each_scan_df.loc[interval_name]
        interval_all_scan = pd.DataFrame(item for item in interval_all_scan)
        string = interval_name[1:-1]
        token = string.split(',')
        interval_name_list = [int(token_i) for token_i in token]



        if 'avg_pixel_dists' not in json_fn:
            mean = pd.DataFrame(interval_all_scan).mean(axis=0).to_numpy()
            std = pd.DataFrame(interval_all_scan).std(axis=0,ddof=0).to_numpy()

            # plot mean and std
            plot_mean_std(interval_name_list, mean, std , viridis,pair_idx,ax)

            # plot average v of the dist error in each scan,
            # the x axis is the index of scan, which is sorted increasing of the scan length
            sort_idx = np.argsort(scan_len)
            sorted_scan_len=sorted(scan_len)
            mean_all_scan_data,mean_all_scan,delete_scan = [],[],[]
            for i in range(interval_all_scan.shape[0]):
                each_scan_data = interval_all_scan.loc[i].to_numpy()
                each_scan_data = each_scan_data[~np.isnan(each_scan_data)] # delete nan
                # this is the mean of error of all images in a scan
                mean_all_scan.append(each_scan_data.mean())
                # average v of the dist error in each scan
                if len(each_scan_data)>1: # for scans whose length is too short, and the statistic is only one value, the v cannot be computed using only one value
                    each_scan_data = np.diff(each_scan_data)/(interval_name_list[1]-interval_name_list[0])
                    mean_all_scan_data.append(each_scan_data.mean())
                else:
                    delete_scan.append(i) # delete this scan when ploting
                    # print(i)
                    # print(delete_scan)
                    # print(interval_name_list)
                    # print(each_scan_data)

            mean_all_scan=np.array(mean_all_scan)
            sorted_mean_all_scan = mean_all_scan[sort_idx]
            ax2.plot(sorted_scan_len,sorted_mean_all_scan, color=viridis.colors[pair_idx], linestyle='dashed', marker='o',  alpha=.5,label='interval_' + str(interval_name_list[0]) + '_' + str(interval_name_list[1]))

            # delete_scan = np.array(delete_scan)
            scan_len_del = np.delete(scan_len,delete_scan)
            sort_idx = np.argsort(scan_len_del)
            sorted_scan_len = sorted(scan_len_del)
            mean_all_scan_data = np.array(mean_all_scan_data)
            sorted_mean_all_scan_data = mean_all_scan_data[sort_idx]
            ax1.plot(sorted_scan_len,sorted_mean_all_scan_data, color=viridis.colors[pair_idx], linestyle='dashed', marker='o',  alpha=.5,label='interval_' + str(interval_name_list[0]) + '_' + str(interval_name_list[1]))
        else:
            # plot average v of the dist error in each scan,
            # the x axis is the index of scan, which is sorted increasing of the scan length
            sort_idx = np.argsort(scan_len)
            sorted_scan_len = sorted(scan_len)
            mean_all_scan = np.squeeze(np.array(interval_all_scan))
            sorted_mean_all_scan = mean_all_scan[sort_idx]
            ax2.plot(sorted_scan_len, sorted_mean_all_scan, color=viridis.colors[pair_idx], linestyle='dashed',marker='o', alpha=.5,label='interval_' + str(interval_name_list[0]) + '_' + str(interval_name_list[1]))

        # dist error (mean of all images error in a scan) v.s. intervals
        interval_value = interval_name_list[1]-interval_name_list[0]
        num_pf = interval_name_list[0]
        num_ff = input_num - 1 - interval_name_list[1]
        num_pf_add_ff = interval_name_list[0]+input_num - 1 - interval_name_list[1]
        if interval_name_list[0] == 0:
            num_ff_div_pf = (input_num - 1 - interval_name_list[1])
        else:
            num_ff_div_pf = (input_num - 1 - interval_name_list[1]) / interval_name_list[0]

        interval_pf = str(interval_value) + '_' + str(num_pf)
        inetrval_ff = str(interval_value) + '_' + str(num_ff)
        inetrval_pf_add_ff = str(interval_value) + '_' + str(num_pf_add_ff)
        inetrval_ff_div_pf = str(interval_value) + '_' + str(num_ff_div_pf)

        if interval_name_list[1]-interval_name_list[0] in sta_in_each_interval_each_scan:
            sta_in_each_interval_each_scan[interval_name_list[1]-interval_name_list[0]] = np.append(sta_in_each_interval_each_scan[interval_name_list[1]-interval_name_list[0]],mean_all_scan)
            sta_in_each_interval_all_scan[interval_name_list[1]-interval_name_list[0]] = np.append(sta_in_each_interval_all_scan[interval_name_list[1]-interval_name_list[0]],mean_all_scan.mean())
        else:
            sta_in_each_interval_each_scan[interval_name_list[1] - interval_name_list[0]]=mean_all_scan
            sta_in_each_interval_all_scan[interval_name_list[1] - interval_name_list[0]]=mean_all_scan.mean()

        # dist error (mean of all images error in a scan) v.s. past frames
        if interval_name_list[0] in sta_in_each_pf_each_scan:
            sta_in_each_pf_each_scan[interval_name_list[0]] = np.append(sta_in_each_pf_each_scan[interval_name_list[0]], mean_all_scan)
            sta_in_each_pf_all_scan[interval_name_list[0]] = np.append(sta_in_each_pf_all_scan[interval_name_list[0]], mean_all_scan.mean())
        else:
            sta_in_each_pf_each_scan[interval_name_list[0]] = mean_all_scan
            sta_in_each_pf_all_scan[interval_name_list[0]] = mean_all_scan.mean()

        # dist error (mean of all images error in a scan) v.s. future frames
        if input_num - 1 - interval_name_list[1] in sta_in_each_ff_each_scan:
            sta_in_each_ff_each_scan[input_num - 1 - interval_name_list[1]] = np.append(sta_in_each_ff_each_scan[input_num - 1 - interval_name_list[1]], mean_all_scan)
            sta_in_each_ff_all_scan[input_num - 1 - interval_name_list[1]] = np.append(sta_in_each_ff_all_scan[input_num - 1 - interval_name_list[1]], mean_all_scan.mean())
        else:
            sta_in_each_ff_each_scan[input_num - 1 - interval_name_list[1]] = mean_all_scan
            sta_in_each_ff_all_scan[input_num - 1 - interval_name_list[1]] = mean_all_scan.mean()

        # dist error (mean of all images error in a scan) v.s. past frames + future frames
        if interval_name_list[0]+input_num - 1 - interval_name_list[1] in sta_in_each_pf_add_ff_each_scan:
            sta_in_each_pf_add_ff_each_scan[interval_name_list[0]+input_num - 1 - interval_name_list[1] ] = np.append(sta_in_each_pf_add_ff_each_scan[interval_name_list[0]+input_num - 1 - interval_name_list[1] ], mean_all_scan)
            sta_in_each_pf_add_ff_all_scan[interval_name_list[0]+input_num - 1 - interval_name_list[1] ] = np.append(sta_in_each_pf_add_ff_all_scan[interval_name_list[0]+input_num - 1 - interval_name_list[1] ], mean_all_scan.mean())
        else:
            sta_in_each_pf_add_ff_each_scan[interval_name_list[0]+input_num - 1 - interval_name_list[1] ] = mean_all_scan
            sta_in_each_pf_add_ff_all_scan[interval_name_list[0]+input_num - 1 - interval_name_list[1] ] = mean_all_scan.mean()

        # dist error (mean of all images error in a scan) v.s. future frames / past frames
        if interval_name_list[0] == 0:

            if (input_num - 1 - interval_name_list[1]) in sta_in_each_pf_div_ff_each_scan:
                sta_in_each_pf_div_ff_each_scan[(input_num - 1 - interval_name_list[1])] = np.append(sta_in_each_pf_div_ff_each_scan[(input_num - 1 - interval_name_list[1])], mean_all_scan)
                sta_in_each_pf_div_ff_all_scan[(input_num - 1 - interval_name_list[1])] = np.append(sta_in_each_pf_div_ff_all_scan[(input_num - 1 - interval_name_list[1])],mean_all_scan.mean())
            else:
                sta_in_each_pf_div_ff_each_scan[(input_num - 1 - interval_name_list[1])] = mean_all_scan
                sta_in_each_pf_div_ff_all_scan[(input_num - 1 - interval_name_list[1])] = mean_all_scan.mean()

        else:
            if (input_num - 1 - interval_name_list[1])/interval_name_list[0] in sta_in_each_pf_div_ff_each_scan:
                sta_in_each_pf_div_ff_each_scan[(input_num - 1 - interval_name_list[1])/interval_name_list[0]] = np.append(sta_in_each_pf_div_ff_each_scan[(input_num - 1 - interval_name_list[1])/interval_name_list[0]], mean_all_scan)
                sta_in_each_pf_div_ff_all_scan[(input_num - 1 - interval_name_list[1])/interval_name_list[0]] = np.append(sta_in_each_pf_div_ff_all_scan[(input_num - 1 - interval_name_list[1])/interval_name_list[0]],mean_all_scan.mean())
            else:
                sta_in_each_pf_div_ff_each_scan[(input_num - 1 - interval_name_list[1])/interval_name_list[0]] = mean_all_scan
                sta_in_each_pf_div_ff_all_scan[(input_num - 1 - interval_name_list[1])/interval_name_list[0]] = mean_all_scan.mean()

        # dist error (mean of v error in a scan) v.s. intervals
        if 'avg_pixel_dists' not in json_fn:
            if interval_name_list[1]-interval_name_list[0] in sta_in_each_interval_each_scan_v:
                sta_in_each_interval_each_scan_v[interval_name_list[1]-interval_name_list[0]] = np.append(sta_in_each_interval_each_scan_v[interval_name_list[1]-interval_name_list[0]],mean_all_scan_data)
                sta_in_each_interval_all_scan_v[interval_name_list[1]-interval_name_list[0]] = np.append(sta_in_each_interval_all_scan_v[interval_name_list[1]-interval_name_list[0]],mean_all_scan_data.mean())
            else:
                sta_in_each_interval_each_scan_v[interval_name_list[1] - interval_name_list[0]]=mean_all_scan_data
                sta_in_each_interval_all_scan_v[interval_name_list[1] - interval_name_list[0]]=mean_all_scan_data.mean()

            # dist error (mean of v error in a scan) v.s. past frames
            if interval_name_list[0] in sta_in_each_pf_each_scan_v:
                sta_in_each_pf_each_scan_v[interval_name_list[0]] = np.append(sta_in_each_pf_each_scan_v[interval_name_list[0]], mean_all_scan_data)
                sta_in_each_pf_all_scan_v[interval_name_list[0]] = np.append(sta_in_each_pf_all_scan_v[interval_name_list[0]],mean_all_scan_data.mean())
            else:
                sta_in_each_pf_each_scan_v[interval_name_list[0]] = mean_all_scan_data
                sta_in_each_pf_all_scan_v[interval_name_list[0]] = mean_all_scan_data.mean()

            # dist error (mean of all images error in a scan) v.s. future frames
            if input_num - 1 - interval_name_list[1] in sta_in_each_ff_each_scan_v:
                sta_in_each_ff_each_scan_v[input_num - 1 - interval_name_list[1]] = np.append(sta_in_each_ff_each_scan_v[input_num - 1 - interval_name_list[1]], mean_all_scan_data)
                sta_in_each_ff_all_scan_v[input_num - 1 - interval_name_list[1]] = np.append(sta_in_each_ff_all_scan_v[input_num - 1 - interval_name_list[1]], mean_all_scan_data.mean())
            else:
                sta_in_each_ff_each_scan_v[input_num - 1 - interval_name_list[1]] = mean_all_scan_data
                sta_in_each_ff_all_scan_v[input_num - 1 - interval_name_list[1]] = mean_all_scan_data.mean()

            # dist error (mean of all images error in a scan) v.s. past frames + future frames
            if interval_name_list[0]+input_num - 1 - interval_name_list[1] in sta_in_each_pf_add_ff_each_scan_v:
                sta_in_each_pf_add_ff_each_scan_v[interval_name_list[0]+input_num - 1 - interval_name_list[1]] = np.append(sta_in_each_pf_add_ff_each_scan_v[interval_name_list[0]+input_num - 1 - interval_name_list[1]], mean_all_scan_data)
                sta_in_each_pf_add_ff_all_scan_v[interval_name_list[0]+input_num - 1 - interval_name_list[1]] = np.append(sta_in_each_pf_add_ff_all_scan_v[interval_name_list[0]+input_num - 1 - interval_name_list[1]],mean_all_scan_data.mean())
            else:
                sta_in_each_pf_add_ff_each_scan_v[interval_name_list[0]+input_num - 1 - interval_name_list[1]] = mean_all_scan_data
                sta_in_each_pf_add_ff_all_scan_v[interval_name_list[0]+input_num - 1 - interval_name_list[1]] = mean_all_scan_data.mean()

            # dist error (mean of all images error in a scan) v.s. past frames / future frames



            if interval_name_list[0] == 0:
                # if the number of past frame = 0, then let the number of past frame = 1
                if (input_num - 1 - interval_name_list[1])  in sta_in_each_pf_div_ff_each_scan_v:
                    sta_in_each_pf_div_ff_each_scan_v[(input_num - 1 - interval_name_list[1])] = np.append(sta_in_each_pf_div_ff_each_scan_v[(input_num - 1 - interval_name_list[1])],mean_all_scan_data)
                    sta_in_each_pf_div_ff_all_scan_v[(input_num - 1 - interval_name_list[1])] = np.append(sta_in_each_pf_div_ff_all_scan_v[(input_num - 1 - interval_name_list[1])],mean_all_scan_data.mean())
                else:
                    sta_in_each_pf_div_ff_each_scan_v[(input_num - 1 - interval_name_list[1])] = mean_all_scan_data
                    sta_in_each_pf_div_ff_all_scan_v[(input_num - 1 - interval_name_list[1])] = mean_all_scan_data.mean()

            else:
                if (input_num - 1 - interval_name_list[1])/interval_name_list[0]  in sta_in_each_pf_div_ff_each_scan_v:
                    sta_in_each_pf_div_ff_each_scan_v[(input_num - 1 - interval_name_list[1])/interval_name_list[0]] = np.append(sta_in_each_pf_div_ff_each_scan_v[(input_num - 1 - interval_name_list[1])/interval_name_list[0]],mean_all_scan_data)
                    sta_in_each_pf_div_ff_all_scan_v[(input_num - 1 - interval_name_list[1])/interval_name_list[0]] = np.append(sta_in_each_pf_div_ff_all_scan_v[(input_num - 1 - interval_name_list[1])/interval_name_list[0]],mean_all_scan_data.mean())
                else:
                    sta_in_each_pf_div_ff_each_scan_v[(input_num - 1 - interval_name_list[1])/interval_name_list[0]] = mean_all_scan_data
                    sta_in_each_pf_div_ff_all_scan_v[(input_num - 1 - interval_name_list[1])/interval_name_list[0]] = mean_all_scan_data.mean()

        if interval_pf in sta_in_interval_pf_each_scan:
            sta_in_interval_pf_each_scan[interval_pf] = np.append(sta_in_interval_pf_each_scan[interval_pf],
                                                                  mean_all_scan)
            sta_in_interval_pf_all_scan[interval_pf] = np.append(sta_in_interval_pf_all_scan[interval_pf],
                                                                 mean_all_scan.mean())
        else:
            sta_in_interval_pf_each_scan[interval_pf] = mean_all_scan
            sta_in_interval_pf_all_scan[interval_pf] = mean_all_scan.mean()

        if inetrval_ff in sta_in_interval_ff_each_scan:
            sta_in_interval_ff_each_scan[inetrval_ff] = np.append(sta_in_interval_ff_each_scan[inetrval_ff],
                                                                  mean_all_scan)
            sta_in_interval_ff_all_scan[inetrval_ff] = np.append(sta_in_interval_ff_all_scan[inetrval_ff],
                                                                 mean_all_scan.mean())
        else:
            sta_in_interval_ff_each_scan[inetrval_ff] = mean_all_scan
            sta_in_interval_ff_all_scan[inetrval_ff] = mean_all_scan.mean()

        if inetrval_pf_add_ff in sta_in_interval_pf_add_ff_each_scan:
            sta_in_interval_pf_add_ff_each_scan[inetrval_pf_add_ff] = np.append(
                sta_in_interval_pf_add_ff_each_scan[inetrval_pf_add_ff], mean_all_scan)
            sta_in_interval_pf_add_ff_all_scan[inetrval_pf_add_ff] = np.append(
                sta_in_interval_pf_add_ff_all_scan[inetrval_pf_add_ff], mean_all_scan.mean())
        else:
            sta_in_interval_pf_add_ff_each_scan[inetrval_pf_add_ff] = mean_all_scan
            sta_in_interval_pf_add_ff_all_scan[inetrval_pf_add_ff] = mean_all_scan.mean()

        if inetrval_ff_div_pf in sta_in_interval_pf_div_ff_each_scan:
            sta_in_interval_pf_div_ff_each_scan[inetrval_ff_div_pf] = np.append(
                sta_in_interval_pf_div_ff_each_scan[inetrval_ff_div_pf], mean_all_scan)
            sta_in_interval_pf_div_ff_all_scan[inetrval_ff_div_pf] = np.append(
                sta_in_interval_pf_div_ff_all_scan[inetrval_ff_div_pf], mean_all_scan.mean())
        else:
            sta_in_interval_pf_div_ff_each_scan[inetrval_ff_div_pf] = mean_all_scan
            sta_in_interval_pf_div_ff_all_scan[inetrval_ff_div_pf] = mean_all_scan.mean()
        if 'avg_pixel_dists' not in json_fn:
            if interval_pf in sta_in_interval_pf_each_scan_v:
                sta_in_interval_pf_each_scan_v[interval_pf] = np.append(sta_in_interval_pf_each_scan_v[interval_pf],
                                                                        mean_all_scan_data)
                sta_in_interval_pf_all_scan_v[interval_pf] = np.append(sta_in_interval_pf_all_scan_v[interval_pf],
                                                                       mean_all_scan_data.mean())
            else:
                sta_in_interval_pf_each_scan_v[interval_pf] = mean_all_scan_data
                sta_in_interval_pf_all_scan_v[interval_pf] = mean_all_scan_data.mean()

            if inetrval_ff in sta_in_interval_ff_each_scan_v:
                sta_in_interval_ff_each_scan_v[inetrval_ff] = np.append(sta_in_interval_ff_each_scan_v[inetrval_ff],
                                                                        mean_all_scan_data)
                sta_in_interval_ff_all_scan_v[inetrval_ff] = np.append(sta_in_interval_ff_all_scan_v[inetrval_ff],
                                                                       mean_all_scan_data.mean())
            else:
                sta_in_interval_ff_each_scan_v[inetrval_ff] = mean_all_scan_data
                sta_in_interval_ff_all_scan_v[inetrval_ff] = mean_all_scan_data.mean()

            if inetrval_pf_add_ff in sta_in_interval_pf_add_ff_each_scan_v:
                sta_in_interval_pf_add_ff_each_scan_v[inetrval_pf_add_ff] = np.append(
                    sta_in_interval_pf_add_ff_each_scan_v[inetrval_pf_add_ff], mean_all_scan_data)
                sta_in_interval_pf_add_ff_all_scan_v[inetrval_pf_add_ff] = np.append(
                    sta_in_interval_pf_add_ff_all_scan_v[inetrval_pf_add_ff], mean_all_scan_data.mean())
            else:
                sta_in_interval_pf_add_ff_each_scan_v[inetrval_pf_add_ff] = mean_all_scan_data
                sta_in_interval_pf_add_ff_all_scan_v[inetrval_pf_add_ff] = mean_all_scan_data.mean()

            if inetrval_ff_div_pf in sta_in_interval_pf_div_ff_each_scan_v:
                sta_in_interval_pf_div_ff_each_scan_v[inetrval_ff_div_pf] = np.append(
                    sta_in_interval_pf_div_ff_each_scan_v[inetrval_ff_div_pf], mean_all_scan_data)
                sta_in_interval_pf_div_ff_all_scan_v[inetrval_ff_div_pf] = np.append(
                    sta_in_interval_pf_div_ff_all_scan_v[inetrval_ff_div_pf], mean_all_scan_data.mean())
            else:
                sta_in_interval_pf_div_ff_each_scan_v[inetrval_ff_div_pf] = mean_all_scan_data
                sta_in_interval_pf_div_ff_all_scan_v[inetrval_ff_div_pf] = mean_all_scan_data.mean()

        # box plot - plot all box plot of all intervals in a fifure
        # x = range(interval_name_list[1], interval_name_list[1] + len(mean) * (interval_name_list[1] - interval_name_list[0]),interval_name_list[1] - interval_name_list[0])
        ax_box_plot = fig_box_plot.add_subplot(rows, cols, pair_idx + 1)
        interval_all_scan.boxplot(ax=ax_box_plot,  medianprops={"linewidth": 4})
        # ax.set_xlabel('img index')
        ax_box_plot.set_title('interval_' + str(interval_name_list[0]) + '_' + str(interval_name_list[1]))

        # box plot without outliers
        ax_wo_outliers = fig_box_plot_wo_outliers.add_subplot(rows, cols, pair_idx + 1)
        interval_all_scan.boxplot(ax=ax_wo_outliers, medianprops={"linewidth": 4}, showfliers=False)
        # ax.set_xlabel('img index')
        ax_wo_outliers.set_title('interval_' + str(interval_name_list[0]) + '_' + str(interval_name_list[1]))

    # plot mean dists error v.s. intervals - plot each scan
    plot_box_points(ax3, sta_in_each_interval_each_scan,'interval','dists error in each scan (median)','median')
    # plot mean dists error v.s. intervals - plot mean of all scans
    plot_box_points(ax4, sta_in_each_interval_all_scan,'interval','mean dists error of all scan (median)','median')
    plot_box_points(ax3_1, sta_in_each_interval_each_scan,'interval','dists error in each scan (mean)','mean')
    plot_box_points(ax4_1, sta_in_each_interval_all_scan,'interval','mean dists error of all scan (mean)','mean')

    # plot mean v error v.s. intervals - plot each scan
    if not avg_pixel_dists:
        plot_box_points(ax5, sta_in_each_interval_each_scan_v,'interval', 'v error in each scan (median)','median')
        # plot mean v error v.s. intervals - plot mean of all scans
        plot_box_points(ax6, sta_in_each_interval_all_scan_v,'interval','mean v error in all scans (median)','median')
        plot_box_points(ax5_1, sta_in_each_interval_each_scan_v,'interval', 'v error in each scan (mean)','mean')
        plot_box_points(ax6_1, sta_in_each_interval_all_scan_v,'interval','mean v error in all scans (mean)','mean')

    # plot mean dists error v.s. past frames - plot each scan
    plot_box_points(ax7, sta_in_each_pf_each_scan, 'past frames', 'dists error in each scan (median)','median')
    # plot mean dists error v.s. intervals - plot mean of all scans
    plot_box_points(ax8, sta_in_each_pf_all_scan, 'past frames', 'mean dists error of all scan (median)','median')
    plot_box_points(ax7_1, sta_in_each_pf_each_scan, 'past frames', 'dists error in each scan (mean)','mean')
    plot_box_points(ax8_1, sta_in_each_pf_all_scan, 'past frames', 'mean dists error of all scan (mean)','mean')

    # plot mean v error v.s. past frames - plot each scan
    if not avg_pixel_dists:
        plot_box_points(ax9, sta_in_each_pf_each_scan_v, 'past frames', 'v error in each scan (median)','median')
        # plot mean v error v.s. intervals - plot mean of all scans
        plot_box_points(ax10, sta_in_each_pf_all_scan_v, 'past frames', 'mean v error in all scans (median)','median')
        plot_box_points(ax9_1, sta_in_each_pf_each_scan_v, 'past frames', 'v error in each scan (mean)','mean')
        plot_box_points(ax10_1, sta_in_each_pf_all_scan_v, 'past frames', 'mean v error in all scans (mean)','mean')

    # plot mean dists error v.s. future frames - plot each scan
    plot_box_points(ax11, sta_in_each_ff_each_scan, 'future frames', 'dists error in each scan (median)', 'median')
    # plot mean dists error v.s. future frames - plot mean of all scans
    plot_box_points(ax12, sta_in_each_ff_all_scan, 'future frames', 'mean dists error of all scan (median)', 'median')
    plot_box_points(ax11_1, sta_in_each_ff_each_scan, 'future frames', 'dists error in each scan (mean)', 'mean')
    plot_box_points(ax12_1, sta_in_each_ff_all_scan, 'future frames', 'mean dists error of all scan (mean)', 'mean')

    # plot mean v error v.s. past frames  - plot each scan
    if not avg_pixel_dists:
        plot_box_points(ax13, sta_in_each_ff_each_scan_v, 'future frames', 'v error in each scan (median)', 'median')
        # plot mean v error v.s. past frames + future frames - plot mean of all scans
        plot_box_points(ax14, sta_in_each_ff_all_scan_v, 'future frames', 'mean v error in all scans (median)', 'median')
        plot_box_points(ax13_1, sta_in_each_ff_each_scan_v, 'future frames', 'v error in each scan (mean)', 'mean')
        plot_box_points(ax14_1, sta_in_each_ff_all_scan_v, 'future frames', 'mean v error in all scans (mean)', 'mean')

    # plot mean dists error v.s. past frames+future frames - plot each scan
    plot_box_points(ax15, sta_in_each_pf_add_ff_each_scan, 'past frames+future frames', 'dists error in each scan (median)', 'median')
    # plot mean dists error v.s. intervals - plot mean of all scans
    plot_box_points(ax16, sta_in_each_pf_add_ff_all_scan, 'past frames+future frames', 'mean dists error of all scan (median)', 'median')
    plot_box_points(ax15_1, sta_in_each_pf_add_ff_each_scan, 'past frames+future frames', 'dists error in each scan (mean)', 'mean')
    plot_box_points(ax16_1, sta_in_each_pf_add_ff_all_scan, 'past frames+future frames', 'mean dists error of all scan (mean)', 'mean')

    # plot mean v error v.s. past frames - plot each scan
    if not avg_pixel_dists:
        plot_box_points(ax17, sta_in_each_pf_add_ff_each_scan_v, 'past frames+future frames', 'v error in each scan (median)', 'median')
        # plot mean v error v.s. intervals - plot mean of all scans
        plot_box_points(ax18, sta_in_each_pf_add_ff_all_scan_v, 'past frames+future frames', 'mean v error in all scans (median)', 'median')
        plot_box_points(ax17_1, sta_in_each_pf_add_ff_each_scan_v, 'past frames+future frames', 'v error in each scan (mean)', 'mean')
        plot_box_points(ax18_1, sta_in_each_pf_add_ff_all_scan_v, 'past frames+future frames', 'mean v error in all scans (mean)', 'mean')

    # plot mean dists error v.s. past frames/future frames - plot each scan
    plot_box_points(ax19, sta_in_each_pf_div_ff_each_scan, 'future frames/past frames', 'dists error in each scan (median)', 'median')
    # plot mean dists error v.s. intervals - plot mean of all scans
    plot_box_points(ax20, sta_in_each_pf_div_ff_all_scan, 'future frames/past frames', 'mean dists error of all scan (median)', 'median')
    plot_box_points(ax19_1, sta_in_each_pf_div_ff_each_scan, 'future frames/past frames', 'dists error in each scan (mean)', 'mean')
    plot_box_points(ax20_1, sta_in_each_pf_div_ff_all_scan, 'future frames/past frames', 'mean dists error of all scan (mean)', 'mean')

    # plot mean v error v.s. past frames/future frames - plot each scan
    if not avg_pixel_dists:
        plot_box_points(ax23, sta_in_each_pf_div_ff_each_scan_v, 'future frames/past frames', 'v error in each scan (median)', 'median')
        # plot mean v error v.s. intervals - plot mean of all scans
        plot_box_points(ax24, sta_in_each_pf_div_ff_all_scan_v, 'future frames/past frames', 'mean v error in all scans (median)', 'median')
        plot_box_points(ax23_1, sta_in_each_pf_div_ff_each_scan_v, 'future frames/past frames', 'v error in each scan (mean)', 'mean')
        plot_box_points(ax24_1, sta_in_each_pf_div_ff_all_scan_v, 'future frames/past frames', 'mean v error in all scans (mean)', 'mean')

    cax1,im_1 = plot_2d_statistic(ax28_1, sta_in_interval_pf_each_scan, 'interval', 'past frames','each scan')
    fig12.colorbar(im_1, cax=cax1, orientation='vertical')
    cax_2,im_2 = plot_2d_statistic(ax29_1, sta_in_interval_pf_all_scan, 'interval', 'past frames','mean of all scans')
    fig12.colorbar(im_2, cax=cax_2, orientation='vertical')

    cax_1,im_1 = plot_2d_statistic(ax28, sta_in_interval_ff_each_scan, 'interval', 'future frames','each scan')
    cax_2,im_2 = plot_2d_statistic(ax29, sta_in_interval_ff_all_scan, 'interval', 'future frames','mean of all scans')
    fig13.colorbar(im_1, cax=cax_1, orientation='vertical')
    fig13.colorbar(im_2, cax=cax_2, orientation='vertical')

    cax_1,im_1 = plot_2d_statistic(ax32, sta_in_interval_pf_add_ff_each_scan, 'interval', 'past+future frames', 'each scan')
    cax_2,im_2 =plot_2d_statistic(ax33, sta_in_interval_pf_add_ff_all_scan, 'interval', 'past+future frames', 'mean of all scans')
    fig14.colorbar(im_1, cax=cax_1, orientation='vertical')
    fig14.colorbar(im_2, cax=cax_2, orientation='vertical')

    cax_1,im_1 = plot_2d_statistic(ax36, sta_in_interval_pf_div_ff_each_scan, 'interval', 'future/past frames', 'each scan')
    cax_2,im_2 =plot_2d_statistic(ax37, sta_in_interval_pf_div_ff_all_scan, 'interval', 'future/past frames', 'mean of all scans')
    fig15.colorbar(im_1, cax=cax_1, orientation='vertical')
    fig15.colorbar(im_2, cax=cax_2, orientation='vertical')

    if not avg_pixel_dists:
        cax_1,im_1 = plot_2d_statistic(ax40, sta_in_interval_pf_each_scan_v, 'interval', 'past frames', 'v_each scan')
        cax_2,im_2 =plot_2d_statistic(ax41, sta_in_interval_pf_all_scan_v, 'interval', 'past frames', 'v_mean of all scans')
        fig16.colorbar(im_1, cax=cax_1, orientation='vertical')
        fig16.colorbar(im_2, cax=cax_2, orientation='vertical')

        cax_1,im_1 = plot_2d_statistic(ax44, sta_in_interval_ff_each_scan_v, 'interval', 'future frames', 'v_each scan')
        cax_2,im_2 =plot_2d_statistic(ax45, sta_in_interval_ff_all_scan_v, 'interval', 'future frames', 'v_mean of all scans')
        fig17.colorbar(im_1, cax=cax_1, orientation='vertical')
        fig17.colorbar(im_2, cax=cax_2, orientation='vertical')

        cax_1,im_1 = plot_2d_statistic(ax48, sta_in_interval_pf_add_ff_each_scan_v, 'interval', 'past+future frames', 'v_each scan')
        cax_2,im_2 =plot_2d_statistic(ax49, sta_in_interval_pf_add_ff_all_scan_v, 'interval', 'past+future frames', 'vmean of all scans')
        fig18.colorbar(im_1, cax=cax_1, orientation='vertical')
        fig18.colorbar(im_2, cax=cax_2, orientation='vertical')

        cax_1,im_1 =plot_2d_statistic(ax52, sta_in_interval_pf_div_ff_each_scan_v, 'interval', 'future/past frames', 'v_each scan')
        cax_2,im_2 =plot_2d_statistic(ax53, sta_in_interval_pf_div_ff_all_scan_v, 'interval', 'future/past frames', 'v_mean of all scans')
        fig19.colorbar(im_1, cax=cax_1, orientation='vertical')
        fig19.colorbar(im_2, cax=cax_2, orientation='vertical')

    ax.legend()
    ax.set_xlabel('img index')
    ax.set_ylabel('accumulated error (mean/std)')

    ax1.legend()
    ax1.set_xlabel('scan length')
    ax1.set_ylabel('ave v (change of error / interval)')

    ax2.legend()
    ax2.set_xlabel('scan length')
    ax2.set_ylabel('ave dists error of each scan')

    if avg_pixel_dists:
        saved_str = '_avg_pixel_dists'
    else:
        saved_str = '_acc_err'

    fig.savefig(saved_folder + '/'  +   model_name+saved_str+'_ave_all_intervals_over_all_scan' +  '.png')
    fig1.savefig(saved_folder + '/'  +   model_name+saved_str+'_ave_dists_all_intervals' +  '.png')
    fig2.savefig(saved_folder + '/'  +   model_name+saved_str+'_dists_error_vs_intervals' +  '.png')
    fig3.savefig(saved_folder + '/'  +   model_name+saved_str+'_v_vs_intervals' +  '.png')
    fig4.savefig(saved_folder + '/'  +   model_name+saved_str+'_dists_error_vs_pf' +  '.png')
    fig5.savefig(saved_folder + '/'  +   model_name+saved_str+'_v_vs_pf' +  '.png')
    fig6.savefig(saved_folder + '/'  +   model_name+saved_str+'_dists_error_vs_ff' +  '.png')
    fig7.savefig(saved_folder + '/'  +   model_name+saved_str+'_v_vs_ff' +  '.png')

    fig8.savefig(saved_folder + '/'  +   model_name+saved_str+'_dists_error_vs_pf_add_ff' +  '.png')
    fig9.savefig(saved_folder + '/'  +   model_name+saved_str+'_v_vs_pf_add_ff' +  '.png')
    fig10.savefig(saved_folder + '/'  +   model_name+saved_str+'_dists_error_vs_ff_div_pf' +  '.png')
    fig11.savefig(saved_folder + '/'  +   model_name+saved_str+'_v_vs_ff_div_pf' +  '.png')
    fig12.savefig(saved_folder + '/' + model_name +saved_str+ '_err_interval_vs_pf' + '.png')
    fig13.savefig(saved_folder + '/' + model_name + saved_str+'_err_interval_vs_ff' + '.png')
    fig14.savefig(saved_folder + '/' + model_name + saved_str+'_err_interval_vs_pf_add_ff' + '.png')
    fig15.savefig(saved_folder + '/' + model_name + saved_str+'_err_interval_vs_ff_div_pf' + '.png')

    fig16.savefig(saved_folder + '/' + model_name + saved_str+'_v_err_interval_vs_pf' + '.png')
    fig17.savefig(saved_folder + '/' + model_name + saved_str+'_v_err_interval_vs_ff' + '.png')
    fig18.savefig(saved_folder + '/' + model_name + saved_str+'_v_err_interval_vs_pf_add_ff' + '.png')
    fig19.savefig(saved_folder + '/' + model_name + saved_str+'_v_err_interval_vs_ff_div_pf' + '.png')

    fig_box_plot.savefig(saved_folder + '/' + model_name + saved_str+'_ave_all_intervals_over_all_scan_box_plot' + '.png')
    fig_box_plot_wo_outliers.savefig(saved_folder + '/' + model_name +saved_str+ '_ave_all_intervals_over_all_scan_box_plot_wo_outliers' + '.png')
    plt.close(fig)
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    plt.close(fig5)
    plt.close(fig6)
    plt.close(fig7)
    plt.close(fig8)
    plt.close(fig9)
    plt.close(fig10)
    plt.close(fig11)
    plt.close(fig12)
    plt.close(fig13)
    plt.close(fig14)
    plt.close(fig15)
    plt.close(fig16)
    plt.close(fig17)
    plt.close(fig18)
    plt.close(fig19)


    plt.close(fig_box_plot)
    plt.close(fig_box_plot_wo_outliers)


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


def extract_frame_features(frames,device):
    # encode frames
    frames = torch.unsqueeze(frames, 2)
    pretrained_model = Pretrained_model(1).to(device)
    frame_frets = torch.empty(frames.shape[0], frames.shape[1], 1000)
    for i in range(frames.shape[0]):
        frame_frets[i, ...] = pretrained_model(frames[i, :, :, :])
    # test = frames.view(-1, frames.shape[2], frames.shape[3], frames.shape[4])
    # test_f = pretrained_model(test)
    # test_f==frame_frets

    return frame_frets





def get_split_data(opt,SSFrameDataset):

    dataset_all = SSFrameDataset(
        min_scan_len = opt.MIN_SCAN_LEN,
        filename_h5=opt.FILENAME_FRAMES,
        num_samples=opt.NUM_SAMPLES,
        sample_range=opt.SAMPLE_RANGE
        )
    ## setup for cross-validation
    dset_folds = dataset_all.partition_by_ratio(
        ratios = [1]*5, 
        randomise=True, 
        subject_level=False
        )
    for (idx, ds) in enumerate(dset_folds):
        ds.write_json(os.path.join(opt.DATA_PATH,"fold_{:02d}_seqlen{:d}_{:s}.json".format(idx,opt.NUM_SAMPLES,opt.train_set)))  # see test.py for file reading

    # seperate the dataset into train, validation, and test
    # the validation set is use to tune the hyper-parameter, and the test set is used to evaluate the model performance
    dset_train = dset_folds[0]+dset_folds[1]+dset_folds[2]
    dset_val = dset_folds[3]
    dset_test = dset_folds[4]

    return dset_train,dset_val,dset_test


def get_data_pairs(opt):
    
    data_pairs = pair_samples(opt.NUM_SAMPLES, opt.NUM_PRED, opt.single_interval).to(device)
    if opt.NUM_PRED > 1:
        data_pairs_samples,data_pairs_samples_index = sample_dists4plot(opt.NUM_SAMPLES,opt.CONSIATENT_LOSS, opt.ACCUMULAT_LOSS,data_pairs.cpu())
        data_pairs_samples = data_pairs_samples.to(device)
    else:
        data_pairs_samples = data_pairs
        data_pairs_samples_index = list(range(data_pairs.shape[0]))
    # save data_pairs_samples for use
    with open(opt.DATA_PATH +'/'+ 'data_pairs_'+str(opt.NUM_SAMPLES)+'.json', 'w', encoding='utf-8') as fp:
        json.dump(data_pairs.cpu().numpy().tolist(), fp, ensure_ascii=False, indent=4)
    with open(opt.DATA_PATH +'/'+ 'data_pairs_samples_'+str(opt.NUM_SAMPLES)+'.json', 'w', encoding='utf-8') as fp:
        json.dump(data_pairs_samples.cpu().numpy().tolist(), fp, ensure_ascii=False, indent=4)
    with open(opt.DATA_PATH +'/'+ 'data_pairs_samples_index_'+str(opt.NUM_SAMPLES)+'.json', 'w', encoding='utf-8') as fp:
        json.dump(data_pairs_samples_index, fp, ensure_ascii=False, indent=4)

    return data_pairs,data_pairs_samples,data_pairs_samples_index


from torch import linalg as LA
def compute_plane_normal(pts):
    # Create vectors from the points
    vector1 = pts[:,:,:,1]-pts[:,:,:,0]
    vector2 = pts[:,:,:,2]-pts[:,:,:,0]
    
    # Compute the cross product of vector1 and vector2
    cross_product = torch.linalg.cross(vector1, vector2)
    
    # Normalize the cross product to get the plane's normal vector
    matrix_norm = LA.norm(cross_product, dim= 2)
    normal_vector = cross_product / matrix_norm.unsqueeze(2).repeat(1, 1, 3)
    
    return normal_vector



def angle_between_planes(normal_vector1, normal_vector2):
    # compute the cos value between two norm vertorc of two planes
   
    # Calculate the dot product of the two normal vectors
    normal_vector1 = normal_vector1.to(torch.float)
    normal_vector2 = normal_vector2.to(torch.float)

    dot_product = torch.sum(normal_vector1 * normal_vector2, dim=(2))

    # dot_product = torch.dot(normal_vector1, normal_vector2)
    
    # Calculate the magnitudes of the two normal vectors
   
    magnitude1 = LA.norm(normal_vector1, dim= 2)
    magnitude2 = LA.norm(normal_vector2, dim= 2)
    
    # Calculate the cos value using the dot product and magnitudes
    cos_value = dot_product / (magnitude1 * magnitude2)
    # np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    
    return cos_value





