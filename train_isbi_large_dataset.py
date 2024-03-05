
import os
from torch.autograd import Variable
import torch.nn as nn
import torch
import json
from torch.utils.tensorboard import SummaryWriter
from transform import Transform_to_Params
# from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
# from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import torch.nn as nn
from loader_isbi_large_dataset import SSFrameDataset
from network_isbi import build_model
from loss import PointDistance, MTL_loss
from data.calib import read_calib_matrices
from transform import LabelTransform, PredictionTransform, ImageTransform
from utils_isbi import pair_samples, reference_image_points, type_dim,compute_plane_normal,angle_between_planes
from options.train_options_isbi import TrainOptions
from utils_isbi import save_best_network
from loss import PredictionPointDistanceAccumulation
from loss import LabelPointDistanceAccumulation
from utils_isbi import add_scalars
from utils_isbi import add_scalars_params
from utils_isbi import write_to_txt
from utils_isbi import write_to_txt_2
from utils_isbi import add_scalars_loss
from utils_isbi import get_interval
from utils_isbi import sample_dists4plot
from utils_isbi import get_data_pairs,get_split_data
from loss import consistent_accumulated_loss
from loss import WeightedLoss
from pcgrad import PCGrad
from rigid_transform_3D import rigid_transform_3D
from utilits_grid_data import *


# combine regristration and reconstruction
# input US sequence for example 100*120*160
# output coordinates for the each pixel in the US sequnece, and then intepolete
# 


opt = TrainOptions().parse()
writer = SummaryWriter(os.path.join(opt.SAVE_PATH))

# if not opt.multi_gpu:
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get data pairs
# data_pairs,data_pairs_samples,data_pairs_samples_index = get_data_pairs(opt)

with open(opt.DATA_PATH + '/' +"data_pairs_"+str(opt.NUM_SAMPLES)+".json", "r", encoding='utf-8') as f:
    data_pairs= json.load(f)
with open(opt.DATA_PATH + '/' +"data_pairs_samples_index_"+str(opt.NUM_SAMPLES)+".json", "r", encoding='utf-8') as f:
    data_pairs_samples_index= json.load(f)
with open(opt.DATA_PATH + '/' +"data_pairs_samples_"+str(opt.NUM_SAMPLES)+".json", "r", encoding='utf-8') as f:
    data_pairs_samples= json.load(f)
data_pairs=torch.tensor(data_pairs).to(device)
data_pairs_samples_index=torch.tensor(data_pairs_samples_index).to(device)
data_pairs_samples=torch.tensor(data_pairs_samples).to(device)


# from utils import get_img_normalization_mean_std
# all_frames = get_img_normalization_mean_std()


# if opt.NUM_PRED > 1 and opt.single_interval == 0:
#     interval = get_interval(opt, data_pairs)

# elif opt.NUM_PRED==1 or opt.single_interval != 0:
#     interval = {'0':[0]}

# in this multi-task problem, if the input number of frames of too larger, use a sample of data pairs,
# which can decrease the tasks that the network must learn
if opt.sample:
    if data_pairs.shape[0]>50:
        data_pairs = data_pairs_samples
    # interval = {'0': [0]}

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


if (opt.PRED_TYPE=="point") or (opt.LABEL_TYPE=="point"):
    image_points = reference_image_points(dset_train.frame_size,2).to(device)
    pred_dim = type_dim(opt.PRED_TYPE, image_points.shape[1], data_pairs.shape[0])
    label_dim = type_dim(opt.LABEL_TYPE, image_points.shape[1], data_pairs.shape[0])
else:
    image_points = None
    pred_dim = type_dim(opt.PRED_TYPE)
    label_dim = type_dim(opt.LABEL_TYPE)

transform_label = LabelTransform(
    opt.LABEL_TYPE,
    pairs=data_pairs,
    image_points=image_points,
    in_image_coords=True,
    tform_image_to_tool=tform_calib,
    tform_image_mm_to_tool=tform_calib_R_T
    )
transform_label_to_params = LabelTransform(
    label_type = "parameter",
    pairs=data_pairs,
    image_points=image_points,
    in_image_coords=True,
    tform_image_to_tool=tform_calib,
    tform_image_mm_to_tool=tform_calib_R_T
    )
transform_prediction = PredictionTransform(
    opt.PRED_TYPE,
    opt.LABEL_TYPE,
    num_pairs=data_pairs.shape[0],
    image_points=image_points,
    in_image_coords=True,
    tform_image_to_tool=tform_calib,
    tform_image_mm_to_tool=tform_calib_R_T
    )

transform_prediction_to_params = PredictionTransform(
    pred_type = opt.PRED_TYPE,
    label_type = "parameter",
    num_pairs=data_pairs.shape[0],
    image_points=image_points,
    in_image_coords=True,
    tform_image_to_tool=tform_calib,
    tform_image_mm_to_tool=tform_calib_R_T
    )


transform_image = ImageTransform(mean=30.873100930319428, std=31.349069347795712)

# loss
criterion = torch.nn.MSELoss()
metrics = PointDistance()



# if opt.LABEL_TYPE == "MSE_points" and opt.weight_option == 'assigned_weight':
#     # criterion = torch.nn.MSELoss() # compute the squre error of each axis
#     criterion_MSE = torch.nn.MSELoss()
#     criterion = WeightedLoss(data_pairs,opt.weight_option)
#     metrics = PointDistance() # compute the root error of each axis
# elif opt.LABEL_TYPE == "point" and opt.weight_option == 'trained_weight':
#     # criterion =
#     criterion_MSE = torch.nn.MSELoss()
#     metrics = PointDistance()
# elif opt.LABEL_TYPE == "point" and opt.weight_option == 'none':
#     criterion_MSE = torch.nn.MSELoss()
#     criterion = torch.nn.MSELoss()
#     metrics = PointDistance()
#     criterion_MTL = MTL_loss()
# elif opt.LABEL_TYPE == "parameter":
#     criterion_MSE = torch.nn.MSELoss()
#     criterion = torch.nn.L1Loss()
#     metrics = torch.nn.L1Loss()





# predict_accumulated_pts_dists = PredictionPointDistanceAccumulation(
#         single_interval = opt.single_interval,
#         interval = True,
#         single_weight = 'none',
#         pred_type = opt.PRED_TYPE,
#         num_pairs = data_pairs.shape[0],
#         frame_points = image_points,
#         tform_calib = tform_calib,
#         tform_calib_R_T = tform_calib_R_T
# ) # compute accumulated distance for prediction

# predict_accumulated_pts_dists_single_interval = PredictionPointDistanceAccumulation(
#         single_interval = opt.single_interval,
#         interval = False,
#         single_weight = opt.single_weight_option,
#         pred_type = opt.PRED_TYPE,
#         num_pairs = data_pairs.shape[0],
#         frame_points = image_points,
#         tform_calib = tform_calib,
#         tform_calib_R_T = tform_calib_R_T
# )

# label_accumulated_pts_dists = LabelPointDistanceAccumulation(
#         single_interval = opt.single_interval,
#         interval = True,
#         single_weight = 'none',
#         data_pairs = data_pairs,
#         frame_points = image_points,
#         tform_calib = tform_calib,
#         tform_calib_R_T = tform_calib_R_T
# )
# label_accumulated_pts_dists_single_interval = LabelPointDistanceAccumulation(
#         single_interval = opt.single_interval,
#         interval = False,
#         single_weight = opt.single_weight_option,
#         data_pairs = data_pairs,
#         frame_points = image_points,
#         tform_calib = tform_calib,
#         tform_calib_R_T = tform_calib_R_T
# )


# compute abs error between 6 parameters of label and prediction
# transform_label_to_params = LabelTransform(
#     label_type = "parameter",
#     pairs=data_pairs,
#     image_points=image_points,
#     tform_image_to_tool=tform_calib,
#     tform_image_mm_to_tool = tform_calib_R_T
#     )
# transform_prediction_to_params = PredictionTransform(
#     pred_type = opt.PRED_TYPE,
#     label_type="parameter",
#     num_pairs=data_pairs.shape[0],
#     image_points=image_points,
#     tform_image_to_tool=tform_calib,
#     tform_image_mm_to_tool = tform_calib_R_T
#     )

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

if opt.multi_gpu:
    model= nn.DataParallel(model)

if opt.retain:
    model.load_state_dict(torch.load(os.path.join(opt.SAVE_PATH,'saved_model', 'model_epoch'+str(opt.retain_epoch)),map_location=torch.device(device)))
    # model.load_state_dict(torch.load(os.path.join(opt.SAVE_PATH,'saved_model', 'best_validation_dist_model'),map_location=torch.device(device)))


## train
val_loss_min = 1e10
val_dist_min = 1e10
optimiser = torch.optim.Adam(model.parameters(), lr=opt.LEARNING_RATE)


for epoch in range(int(opt.retain_epoch), int(opt.retain_epoch)+opt.NUM_EPOCHS):
    train_epoch_loss = 0
    train_epoch_dist = 0
    for step, (frames, tforms, tforms_inv) in enumerate(train_loader):
        frames, tforms, tforms_inv = frames.to(device), tforms.to(device), tforms_inv.to(device)
        frames = transform_image(frames)
        labels = transform_label(tforms, tforms_inv)

        optimiser.zero_grad()
        if opt.model_name == 'LSTM_E':
            outputs = torch.squeeze(model(frames),dim=1)
        else:
            outputs = model(frames)
        preds = transform_prediction(outputs)


        if opt.Loss_type == "MSE_points":
            loss = criterion(preds, labels)
        elif opt.Loss_type == "Plane_norm":
            loss1 = criterion(preds, labels)
            normal_gt = compute_plane_normal(labels)
            normal_np = compute_plane_normal(preds)
            cos_value = angle_between_planes(normal_gt,normal_np)
            loss = loss1-sum(sum(cos_value))

        dist = metrics(preds, labels).detach()

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

        # compute 6 parameters of each pair of output, this is not be used as accumulated error should be used
        
        
        # params_gt = transform_label_to_params(tforms, tforms_inv)
        # params_np =transform_prediction_to_params(outputs)
        # params_np_train = Transform_to_Params(prev_tform_all_train)
        # params_gt_train = Transform_to_Params(label_prev_tform_all_train)

        # compute the average error of 6DOF, over all possiable transformations
        #  

        # error_6DOF_train = torch.mean(torch.abs(params_gt-params_np),dim=[0,1])
        
        # # compute the best performance of 6DOF (this only for each batch), over all possiable transformations
        # best_value,best_pair = torch.mean(torch.abs(params_gt-params_np),dim=[0,2])
        # error_6DOF_train = torch.mean(torch.abs(params_gt-params_np),dim=[0])[best_pair,...]




        loss.backward()
    

        optimiser.step()

    train_epoch_loss /= (step + 1)
    train_epoch_dist /= (step + 1)

    if epoch in range(0, opt.NUM_EPOCHS, opt.FREQ_INFO):
        print('[Epoch %d, Step %05d] train-loss=%.3f, train-dist=%.3f' % (epoch, step, loss, dist.mean()))
        if dist.shape[0]>1: # torch.tensor([dist]).shape[0]>1
            print('%.2f '*dist.shape[0] % tuple(dist))


    # validation    
    if epoch in range(0, opt.NUM_EPOCHS, opt.val_fre):

        model.train(False)

        epoch_loss_val = 0
        epoch_dist_val = 0
        for step, (fr_val, tf_val, tf_val_inv) in enumerate(val_loader):

            fr_val, tf_val, tf_val_inv = fr_val.to(device), tf_val.to(device), tf_val_inv.to(device)
            la_val = transform_label(tf_val, tf_val_inv)
            fr_val = transform_image(fr_val)

            if opt.model_name == 'LSTM_E':
                out_val = torch.squeeze(model(fr_val),dim=1)
            else:
                out_val = model(fr_val)

            pr_val = transform_prediction(out_val)
            
            if opt.Loss_type == "MSE_points":
                loss_val = criterion(pr_val, la_val)
            elif opt.Loss_type == "Plane_norm":
                loss_val_1 = criterion(pr_val, la_val)
                normal_gt = compute_plane_normal(la_val)
                normal_np = compute_plane_normal(pr_val)
                cos_value = angle_between_planes(normal_gt,normal_np)
                loss_val = loss_val_1-sum(sum(cos_value))

            
            dist_val = metrics(pr_val, la_val).detach()
            # preds_dist_all_val, prev_tform_all_val = predict_accumulated_pts_dists(out_val, interval)
            # label_dist_all_val, label_prev_tform_all_val = label_accumulated_pts_dists(tf_val, tf_val_inv, interval)

            # if opt.single_interval == 0:
            #     loss_val = consistent_accumulated_loss(loss_val, opt, preds_dist_all_val, label_dist_all_val)
            # elif opt.single_interval != 0:
            #     if opt.single_interval_ACCUMULAT_LOSS:
            #         preds_dist_all_val_single_interval = predict_accumulated_pts_dists_single_interval(out_val, interval)
            #         label_dist_all_val_single_interval = label_accumulated_pts_dists_single_interval(tf_val, tf_val_inv,interval)
            #         loss_val = loss_val + torch.mean(torch.stack([criterion_MSE(preds_dist_all_val_single_interval[i], label_dist_all_val_single_interval[i]) for i in range(len(preds_dist_all_val_single_interval))]))
            #     else:
            #         loss_val = loss_val

            # params_np_val = Transform_to_Params(prev_tform_all_val)
            # params_gt_val = Transform_to_Params(label_prev_tform_all_val)



            # params_gt_all = transform_label_to_params(tf_val, tf_val_inv)
            # params_np_all =transform_prediction_to_params(out_val)
            
            # error_6DOF_val = torch.mean(torch.abs(params_gt_all-params_np_all),dim=[0,1])
            # # compute the best performance of 6DOF (this only for each batch), over all possiable transformations
            # best_value,best_pair = torch.mean(torch.abs(params_gt_all-params_np_all),dim=[0,2])
            # error_6DOF_val = torch.mean(torch.abs(params_gt_all-params_np_all),dim=[0])[best_pair,...]

            
            epoch_loss_val += loss_val.item()
            epoch_dist_val += dist_val


        epoch_loss_val /= (step+1)
        epoch_dist_val /= (step+1)

        if epoch in range(0, opt.NUM_EPOCHS, opt.FREQ_INFO):
            print('[Epoch %d] val-loss=%.3f, val-dist=%.3f' % (epoch, epoch_loss_val, epoch_dist_val.mean()))
            if epoch_dist_val.shape[0]>1:
                print('%.2f '*epoch_dist_val.shape[0] % tuple(epoch_dist_val))

        if epoch in range(0, opt.NUM_EPOCHS, opt.FREQ_SAVE):
            torch.save(model.state_dict(), os.path.join(opt.SAVE_PATH,'saved_model','model_epoch%08d' % epoch))
            print('Model parameters saved.')
            list_dir = os.listdir(os.path.join(opt.SAVE_PATH, 'saved_model'))
            saved_models = [i for i in list_dir if i.startswith('model_epoch')]
            if len(saved_models)>4:
                print(saved_models)
                os.remove(os.path.join(opt.SAVE_PATH,'saved_model',sorted(saved_models)[0]))

        # save best validation model
        val_loss_min, val_dist_min = save_best_network(opt, model, epoch, epoch_loss_val, epoch_dist_val.mean(), val_loss_min, val_dist_min)
        # add to tensorboard
        loss_dists = {'train_epoch_loss': train_epoch_loss, 'train_epoch_dist': train_epoch_dist,'epoch_loss_val':epoch_loss_val,'epoch_dist_val':epoch_dist_val}

        add_scalars(writer, epoch, loss_dists,data_pairs,opt,data_pairs_samples_index)
        # add_scalars_params(writer, epoch,error_6DOF_train,error_6DOF_val)
        write_to_txt(opt, epoch, loss_dists)
        write_to_txt_2(opt, data_pairs.shape[0], dist, metrics(pr_val, la_val))


        model.train(True)
