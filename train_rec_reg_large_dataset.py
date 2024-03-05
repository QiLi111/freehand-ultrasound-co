
import os
from torch.autograd import Variable
import torch.nn as nn
import torch
import json
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
from transform import Transform_to_Params
import torch.nn as nn
from loader_isbi_large_dataset import SSFrameDataset
from network_isbi import build_model
from loss import PointDistance, MTL_loss
from data.calib import read_calib_matrices
from transform import LabelTransform, PredictionTransform, ImageTransform
from utils_isbi import pair_samples, reference_image_points, type_dim,compute_plane_normal,angle_between_planes
from options.train_options_rec_reg import TrainOptions
from utils_isbi import add_scalars,save_best_network_rec_reg

from utilits_grid_data import *
from utils_rec_reg import *

from monai.networks.nets.voxelmorph import VoxelMorphUNet, VoxelMorph
from monai.transforms import DivisiblePad
from monai.losses import BendingEnergyLoss

opt = TrainOptions().parse()
writer = SummaryWriter(os.path.join(opt.SAVE_PATH))

# if not opt.multi_gpu:
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




# # get data pairs
# data_pairs,data_pairs_samples,data_pairs_samples_index = get_data_pairs(opt)
data_pairs = data_pairs_adjacent(opt.NUM_SAMPLES)

# with open(opt.DATA_PATH + '/' +"data_pairs_"+str(opt.NUM_SAMPLES)+".json", "r", encoding='utf-8') as f:
#     data_pairs= json.load(f)
# with open(opt.DATA_PATH + '/' +"data_pairs_samples_index_"+str(opt.NUM_SAMPLES)+".json", "r", encoding='utf-8') as f:
#     data_pairs_samples_index= json.load(f)
# with open(opt.DATA_PATH + '/' +"data_pairs_samples_"+str(opt.NUM_SAMPLES)+".json", "r", encoding='utf-8') as f:
#     data_pairs_samples= json.load(f)
# data_pairs=torch.tensor(data_pairs).to(device)
# data_pairs_samples_index=torch.tensor(data_pairs_samples_index).to(device)
# data_pairs_samples=torch.tensor(data_pairs_samples).to(device)


# from utils import get_img_normalization_mean_std
# all_frames = get_img_normalization_mean_std()


# if opt.NUM_PRED > 1 and opt.single_interval == 0:
#     interval = get_interval(opt, data_pairs)

# elif opt.NUM_PRED==1 or opt.single_interval != 0:
#     interval = {'0':[0]}

# in this multi-task problem, if the input number of frames of too larger, use a sample of data pairs,
# which can decrease the tasks that the network must learn
# if opt.sample:
#     if data_pairs.shape[0]>50:
#         data_pairs = data_pairs_samples
#     # interval = {'0': [0]}

# if opt.CONSIATENT_LOSS or opt.ACCUMULAT_LOSS:
#     interval = get_interval(opt, data_pairs)


# get split data - train, val, test
# dset_train,dset_val,dset_test = get_split_data(opt,SSFrameDataset)




opt.FILENAME_VAL=opt.FILENAME_VAL+'_seqlen'+str(opt.NUM_SAMPLES)+'_'+opt.train_set+'.json'
opt.FILENAME_TEST=opt.FILENAME_TEST+'_seqlen'+str(opt.NUM_SAMPLES)+'_'+opt.train_set+'.json'
opt.FILENAME_TRAIN=[opt.FILENAME_TRAIN[i]+'_seqlen'+str(opt.NUM_SAMPLES)+'_'+opt.train_set+'.json' for i in range(len(opt.FILENAME_TRAIN))]

dset_val = SSFrameDataset.read_json(opt.DATA_PATH,opt.FILENAME_VAL,opt.h5_file_name)
dset_test = SSFrameDataset.read_json(opt.DATA_PATH,opt.FILENAME_TEST,opt.h5_file_name)
dset_train_list = [SSFrameDataset.read_json(opt.DATA_PATH,opt.FILENAME_TRAIN[i],opt.h5_file_name) for i in range(len(opt.FILENAME_TRAIN))]
dset_train = dset_train_list[0]+dset_train_list[1]+dset_train_list[2]
               
saved_folder = opt.SAVE_PATH+'/'+ 'test_plotting'
if not os.path.exists(saved_folder):
    os.makedirs(saved_folder)

train_loader = torch.utils.data.DataLoader(
    dset_train,
    batch_size=opt.MINIBATCH_SIZE,
    shuffle=True,
    num_workers=8
    )

val_loader = torch.utils.data.DataLoader(
    dset_val,
    batch_size=1, 
    shuffle=False,
    num_workers=8
    )


## loss
tform_calib_scale,tform_calib_R_T, tform_calib = read_calib_matrices(filename_calib=opt.FILENAME_CALIB, resample_factor=opt.RESAMPLE_FACTOR, device=device)


image_points = reference_image_points(dset_train.frame_size,dset_train.frame_size).to(device)
pred_dim = compute_dimention(opt.PRED_TYPE, image_points.shape[1],opt.NUM_SAMPLES,'pred')
label_dim = compute_dimention(opt.LABEL_TYPE, image_points.shape[1],opt.NUM_SAMPLES,'label')



transform_label = LabelTransform(
    opt.LABEL_TYPE,
    pairs=data_pairs,  #
    image_points=image_points,
    in_image_coords=True,
    tform_image_to_tool=tform_calib,
    tform_image_mm_to_tool=tform_calib_R_T
    )
# transform_label_to_params = LabelTransform(
#     label_type = "parameter",
#     pairs=data_pairs,
#     image_points=image_points,
#     in_image_coords=True,
#     tform_image_to_tool=tform_calib,
#     tform_image_mm_to_tool=tform_calib_R_T
#     )
transform_prediction = PredictionTransform(
    opt.PRED_TYPE,
    "transform",
    num_pairs=data_pairs.shape[0]-1,
    image_points=image_points,
    in_image_coords=True,
    tform_image_to_tool=tform_calib,
    tform_image_mm_to_tool=tform_calib_R_T
    )

# transform_prediction_to_params = PredictionTransform(
#     pred_type = opt.PRED_TYPE,
#     label_type = "parameter",
#     num_pairs=data_pairs.shape[0],
#     image_points=image_points,
#     in_image_coords=True,
#     tform_image_to_tool=tform_calib,
#     tform_image_mm_to_tool=tform_calib_R_T
#     )


# transform_image = ImageTransform(mean=30.873100930319428, std=31.349069347795712)

# loss
criterion = torch.nn.MSELoss()
metrics = PointDistance(paired=False)
img_loss = MSELoss()
regularization = BendingEnergyLoss()


# init_data_pairs4lstm = init_data_pairs4LSTM(opt,device,tform_calib,tform_calib_R_T)
## network
model = build_model(
    opt,
    in_frames = opt.NUM_SAMPLES,
    pred_dim = pred_dim,
    label_dim = label_dim,
    image_points = image_points,
    tform_calib = tform_calib,
    tform_calib_R_T = tform_calib_R_T
    ).to(device)


# First, a backbone network is constructed. In this case, we use a VoxelMorphUNet as the backbone network.
backbone = VoxelMorphUNet(
    spatial_dims=3,
    in_channels=2,
    unet_out_channels=32,#16,#32,
    channels=(16, 32, 32, 32, 32, 32), #(8, 16, 16, 16, 16, 16),#(16, 32, 32, 32, 32, 32),  # this indicates the down block at the top takes 16 channels as
                                        # input, the corresponding up block at the top produces 32
                                        # channels as output, the second down block takes 32 channels as
                                        # input, and the corresponding up block at the same level
                                        # produces 32 channels as output, etc.
    final_conv_channels=(16,16),#(8, 8)#(16,16)
).to(device)

# Then, a full VoxelMorph network is constructed using the specified backbone network.
VoxelMorph_net = VoxelMorph(
    backbone=backbone,
    integration_steps=7,
    half_res=False
).to(device)

# # Make sure the size if the width/height is a 64 step (512, 576, 640...),
# otherwise, it would report a runtime error:
# Sizes of tensors must match except in dimension 1. Expected size 23 but got size 22 for tensor number 1 in the list.
# this method is not used as DivisiblePad will detach the variable
# instead, I initialise a divisible dimention from the intepoleted volume
# divisible_pad_16 = DivisiblePad(k=16,mode = 'minimum')


if opt.multi_gpu:
    model= nn.DataParallel(model)
    VoxelMorph_net = nn.DataParallel(model)

if opt.retain:
    model.load_state_dict(torch.load(os.path.join(opt.SAVE_PATH,'saved_model', 'model_epoch'+str(opt.retain_epoch)),map_location=torch.device(device)))
    # model.load_state_dict(torch.load(os.path.join(opt.SAVE_PATH,'saved_model', 'best_validation_dist_model'),map_location=torch.device(device)))
    VoxelMorph_net.load_state_dict(torch.load(os.path.join(opt.SAVE_PATH,'saved_model', 'model_reg_epoch'+str(opt.retain_epoch)),map_location=torch.device(device)))



## train
val_loss_min = 1e10
val_dist_min = 1e10

if opt.Loss_type == "MSE_points":
    optimiser = torch.optim.Adam(model.parameters(), lr=opt.LEARNING_RATE)
elif opt.Loss_type == "reg":
    optimiser = torch.optim.Adam(VoxelMorph_net.parameters(), lr=opt.LEARNING_RATE)
elif opt.Loss_type == "rec_reg":
    optimiser = torch.optim.Adam(list(model.parameters())+list(VoxelMorph_net.parameters()), lr=opt.LEARNING_RATE)


for epoch in range(int(opt.retain_epoch), int(opt.retain_epoch)+opt.NUM_EPOCHS):
    train_epoch_loss = 0
    train_epoch_dist = 0
    train_epoch_loss_reg = 0
    for step, (frames, tforms, tforms_inv) in enumerate(train_loader):
        frames, tforms, tforms_inv = frames.to(device), tforms.to(device), tforms_inv.to(device)
        # frames = transform_image(frames)

        # cannot use the ground truth coordinates based on the camera coordinates system, 
        # which will depend on the posotion of camera
        # the transformation between each frame and frame 0
        tforms_each_frame2frame0 = transform_label(tforms, tforms_inv)
        # obtain the coordinates of each frame, set frame 0 as the reference frame
        
        # result1 = torch.linalg.multi_dot([*labels[0,...]])  # X[0] @ X[1] ... @ X[N-1]
        # the coordinates of each pixel points
        labels = torch.matmul(tforms_each_frame2frame0,torch.matmul(tform_calib,image_points))[:,:,0:3,...]
       
        # change labels to a convenient coordinates system
        
        # # check if the labels is correct
        # # ground truth, use camera as the reference
        # labels1 = torch.matmul(tforms,torch.matmul(tform_calib,image_points))
        # scatter_plot_3D(labels[0,...].cpu().numpy(),saved_folder,save_name = 'frame0.png')
        # scatter_plot_3D(labels1[0,...].cpu().numpy(),saved_folder,save_name = 'camera.png')


        frames = frames/255 # normalise image into range (0,1)

        optimiser.zero_grad()
        
        
        if opt.model_name == 'LSTM_E':
            outputs = torch.squeeze(model(frames),dim=1)
        else:
            outputs = model(frames)

        # 6 parameter to 4*4 transformation
        pred_transfs = transform_prediction(outputs)
        # make the predicted transformations are based on frame 0
        # predict only opt.NUM_FRAES-1 transformatons,and let the first frame equals to identify matrix
        predframe0 = torch.eye(4,4)[None,...].repeat(pred_transfs.shape[0],1, 1,1).to(device)
        pred_transfs = torch.cat((predframe0,pred_transfs),1)

        # transformtion to points
        pred_pts = torch.matmul(pred_transfs,torch.matmul(tform_calib,image_points))[:,:,0:3,...]

        if opt.Conv_Coords == 'optimised_coord':
            convR_batched = calculateConvPose_batched(labels,option = 'first_last_frames_centroid',device=device)    
            labels = torch.matmul(convR_batched,torch.matmul(tforms_each_frame2frame0,torch.matmul(tform_calib,image_points)))[:,:,0:3,...]
            
            pred_pts = torch.matmul(convR_batched,torch.matmul(pred_transfs,torch.matmul(tform_calib,image_points)))[:,:,0:3,...]
        


        
        if opt.Loss_type == "MSE_points":
            loss = criterion(pred_pts, labels)
        elif opt.Loss_type == "Plane_norm":
            loss1 = criterion(pred_pts, labels)
            normal_gt = compute_plane_normal(labels)
            normal_np = compute_plane_normal(pred_pts)
            cos_value = angle_between_planes(normal_gt,normal_np)
            loss = loss1-sum(sum(cos_value))
        elif opt.Loss_type == "reg" or opt.Loss_type == "rec_reg":
            # scatter points to grid points
            gt_volume, gt_volume_position = interpolation_3D_pytorch_batched(scatter_pts = labels,
                                                    frames = frames,
                                                    time_log=None,
                                                    saved_folder_test = saved_folder,
                                                    scan_name='gt_step'+str(step),
                                                    device = device,
                                                    option = opt.intepoletion_method,
                                                    volume_size = opt.intepoletion_volume,
                                                    volume_position = None
                                                    )
            
            # use the volume of ground truth to intepolete the prediction,
            # this will enforce the prediction become colser to the ground truth.
            # to do this, delete thoes points outside the groundtruth volume 
            # and then use the parameter (size and step) of the ground truth volume to intepolete
            # 
            pred_volume,pred_volume_position = interpolation_3D_pytorch_batched(scatter_pts = pred_pts,
                                                    frames = frames,
                                                    time_log=None,
                                                    saved_folder_test = saved_folder,
                                                    scan_name='pred_step'+str(step),
                                                    device = device,
                                                    option = opt.intepoletion_method,
                                                    volume_size = opt.intepoletion_volume,
                                                    volume_position = gt_volume_position
                                                    )
            warped, ddf = VoxelMorph_net(moving = torch.unsqueeze(pred_volume, 1), 
                          fixed = torch.unsqueeze(gt_volume, 1))
            if opt.Loss_type == "reg":
                # test if only use registartion can backward
                loss = img_loss(torch.squeeze(warped),gt_volume) + regularization(ddf)
            elif opt.Loss_type == "rec_reg":
                loss1 = criterion(pred_pts, labels)
                loss2 = img_loss(torch.squeeze(warped),gt_volume) + regularization(ddf)
                loss = loss1+100*loss2


        dist = metrics(pred_pts, labels).detach()


        
    
        # 3D registration use 
        # not used
        #  divisible_pad will treat the first dimension as channel, such that it would not pad the first dimension
        # pred_volume = torch.unsqueeze(pred_volume, 0)
        # gt_volume = torch.unsqueeze(gt_volume, 0)
        # pred_volume_pad = divisible_pad_16(pred_volume)
        # gt_volume_pad = divisible_pad_16(gt_volume)

        # gt_volume_before = interpolation_3D_pytorch(scatter_pts = labels[0,...],
        #                                            frames = frames[0,...],
        #                                            time_log=None,
        #                                            saved_folder_test = saved_folder,
        #                                            scan_name='gt_step'+str(step),
        #                                            device = device,
        #                                            option = opt.intepoletion_method,
        #                                            )
        
        # save2mha(gt_volume_before.cpu().numpy(),sx = 1,sy=1,sz=1,
        #     save_folder=saved_folder+'/'+'gt_before.mha'
        #     )
        # save2mha(gt_volume.cpu().numpy(),sx = 1,sy=1,sz=1,
        #     save_folder=saved_folder+'/'+'gt.mha'
        #     )
        # save2mha(pred_volume.detach().cpu().numpy(),sx = 1,sy=1,sz=1,
        #     save_folder=saved_folder+'/'+'pred.mha'
        #     )
       
        
        # save2mha(torch.squeeze(warped).detach().cpu().numpy(),sx = 1,sy=1,sz=1,
        #     save_folder=saved_folder+'/'+'warped.mha'
        #     )
                                                   
    
        


        

        # preds_dist_all_train, prev_tform_all_train = predict_accumulated_pts_dists(outputs, interval)
        # label_dist_all_train, label_prev_tform_all_train = label_accumulated_pts_dists(tforms, tforms_inv, interval)

        # if opt.single_interval == 0:
        #     loss = consistent_accumulated_loss(loss, opt,preds_dist_all_train, label_dist_all_train)
        # elif opt.single_interval != 0:
        #     if opt.single_interval_ACCUMULAT_LOSS:
        #         preds_dist_all_train_single_interval = predict_accumulated_pts_dists_single_interval(outputs, interval)
        #         label_dist_all_train_single_interval = label_accumulated_pts_dists_single_interval(tforms, tforms_inv,interval)
        #         loss = loss + torch.mean(torch.stack([criterion_MSE(preds_dist_all_train_single_interval[i],label_dist_all_train_single_interval[i]) for i in range(len(label_dist_all_train_single_interval))]))
        #     else:
        #         loss = loss
        train_epoch_loss += loss.item()
        train_epoch_dist += dist

        if opt.Loss_type == "rec_reg":
            train_epoch_loss_reg += loss2.item()


        loss.backward()
        optimiser.step()

        # pred_volume.detach(),warped.detach(),ddf.detach()
        

    train_epoch_loss /= (step + 1)
    train_epoch_dist /= (step + 1)
    train_epoch_loss_reg /= (step + 1)

    if epoch in range(0, opt.NUM_EPOCHS, opt.FREQ_INFO):
        print('[Epoch %d, Step %05d] train-loss=%.3f, train-dist=%.3f' % (epoch, step, loss, dist))


    # validation    
    if epoch in range(0, opt.NUM_EPOCHS, opt.val_fre):

        model.train(False)
        VoxelMorph_net.train(False)

        epoch_loss_val = 0
        epoch_dist_val = 0
        epoch_loss_val_reg = 0
        for step, (fr_val, tf_val, tf_val_inv) in enumerate(val_loader):

            fr_val, tf_val, tf_val_inv = fr_val.to(device), tf_val.to(device), tf_val_inv.to(device)
            tforms_each_frame2frame0_val = transform_label(tf_val, tf_val_inv)
            labels_val = torch.matmul(tforms_each_frame2frame0_val,torch.matmul(tform_calib,image_points))[:,:,0:3,...]
            
            
            fr_val = fr_val/255
        
            if opt.model_name == 'LSTM_E':
                out_val = torch.squeeze(model(fr_val),dim=1)
            else:
                out_val = model(fr_val)

            pr_transfs_val = transform_prediction(out_val)
            predframe0_val = torch.eye(4,4)[None,...].repeat(pr_transfs_val.shape[0],1, 1,1).to(device)
            pr_transfs_val = torch.cat((predframe0_val,pr_transfs_val),1)


            pred_pts_val = torch.matmul(pr_transfs_val,torch.matmul(tform_calib,image_points))[:,:,0:3,...]
        
            if opt.Conv_Coords == 'optimised_coord':
                # change labels to a convenient coordinates system
                convR_batched_val = calculateConvPose_batched(labels_val,option = 'first_last_frames_centroid',device=device)    
                labels_val = torch.matmul(convR_batched_val,torch.matmul(tforms_each_frame2frame0_val,torch.matmul(tform_calib,image_points)))[:,:,0:3,...]
                
                pred_pts_val = torch.matmul(convR_batched_val,torch.matmul(pr_transfs_val,torch.matmul(tform_calib,image_points)))[:,:,0:3,...]
        



            if opt.Loss_type == "MSE_points":
                loss_val = criterion(pred_pts_val, labels_val)
            elif opt.Loss_type == "Plane_norm":
                loss1_val = criterion(pred_pts_val, labels_val)
                normal_gt_val = compute_plane_normal(labels_val)
                normal_np_val = compute_plane_normal(pred_pts_val)
                cos_value_val = angle_between_planes(normal_gt_val,normal_np_val)
                loss_val = loss1_val-sum(sum(cos_value_val))
            elif opt.Loss_type == "reg" or opt.Loss_type == "rec_reg":
                # scatter points to grid points
                gt_volume_val, gt_volume_position_val = interpolation_3D_pytorch_batched(scatter_pts = labels_val,
                                                    frames = torch.unsqueeze(fr_val,0),
                                                    time_log=None,
                                                    saved_folder_test = saved_folder,
                                                    scan_name='gt_step'+str(step)+'_val',
                                                    device = device,
                                                    option = opt.intepoletion_method,
                                                    volume_position = None
                                                    )
                pred_volume_val,pred_volume_position_val = interpolation_3D_pytorch_batched(scatter_pts = pred_pts_val,
                                                    frames = torch.unsqueeze(fr_val,0),
                                                    time_log=None,
                                                    saved_folder_test = saved_folder,
                                                    scan_name='pred_step'+str(step)+'_val',
                                                    device = device,
                                                    option = opt.intepoletion_method,
                                                    volume_position = gt_volume_position_val
                                                    )
                
                warped_val, ddf_val = VoxelMorph_net(moving = torch.unsqueeze(pred_volume_val, 1), 
                            fixed = torch.unsqueeze(gt_volume_val, 1))

                if opt.Loss_type == "reg":
                    # test if only use registartion can backward
                    loss_val = img_loss(torch.squeeze(warped_val,0),gt_volume_val) + regularization(ddf_val)
                elif opt.Loss_type == "rec_reg":
                    loss1_val = criterion(pred_pts_val, labels_val)
                    loss2_val = img_loss(torch.squeeze(warped_val,0),gt_volume_val) + regularization(ddf_val)
                    loss_val = loss1_val+100*loss2_val

            
            dist_val = metrics(pred_pts_val, labels_val).detach()
           
            
            epoch_loss_val += loss_val.item()
            epoch_dist_val += dist_val
            if opt.Loss_type == "rec_reg":
                epoch_loss_val_reg += loss2_val.item()
            
            # pred_volume_val.detach(),warped_val.detach(),ddf_val.detach()


        epoch_loss_val /= (step+1)
        epoch_dist_val /= (step+1)
        epoch_loss_val_reg /= (step+1)

        if epoch in range(0, opt.NUM_EPOCHS, opt.FREQ_INFO):
            print('[Epoch %d] val-loss=%.3f, val-dist=%.3f' % (epoch, epoch_loss_val, epoch_dist_val))

        if epoch in range(0, opt.NUM_EPOCHS, opt.FREQ_SAVE):
            torch.save(model.state_dict(), os.path.join(opt.SAVE_PATH,'saved_model','model_epoch%08d' % epoch))
            torch.save(VoxelMorph_net.state_dict(), os.path.join(opt.SAVE_PATH,'saved_model','model_reg_epoch%08d' % epoch))

            print('Model parameters saved.')
            list_dir = os.listdir(os.path.join(opt.SAVE_PATH, 'saved_model'))
            saved_models = [i for i in list_dir if i.startswith('model_epoch')]
            if len(saved_models)>4:
                print(saved_models)
                os.remove(os.path.join(opt.SAVE_PATH,'saved_model',sorted(saved_models)[0]))

            saved_models_reg = [i for i in list_dir if i.startswith('model_reg_epoch')]
            if len(saved_models_reg)>4:
                print(saved_models_reg)
                os.remove(os.path.join(opt.SAVE_PATH,'saved_model',sorted(saved_models_reg)[0]))

        # save best validation model
        val_loss_min, val_dist_min = save_best_network_rec_reg(opt, model, VoxelMorph_net, epoch, epoch_loss_val, epoch_dist_val, val_loss_min, val_dist_min)
        # add to tensorboard
        loss_dists = {'train_epoch_loss': train_epoch_loss, 
                      'train_epoch_dist': train_epoch_dist,
                      'train_epoch_loss_reg':train_epoch_loss_reg,
                      'epoch_loss_val':epoch_loss_val,
                      'epoch_dist_val':epoch_dist_val,
                      'epoch_loss_val_reg':epoch_loss_val_reg}
        add_scalars(writer, epoch, loss_dists)

        # add_scalars_params(writer, epoch,error_6DOF_train,error_6DOF_val)
        # write_to_txt(opt, epoch, loss_dists)
        # write_to_txt_2(opt, data_pairs.shape[0], dist, metrics(pr_val, la_val))


        model.train(True)
        VoxelMorph_net.train(True)
