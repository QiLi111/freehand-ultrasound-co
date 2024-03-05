

import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py,csv
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
from utils_isbi import add_scalars,add_scalars_reg,save_best_network_rec,save_best_network_reg
from visualizer_reg_data import Visualizer
from loader_reg_large_dataset import SSFrameDataset_reg
from utilits_grid_data import *


# from monai.networks.nets.voxelmorph import VoxelMorphUNet, VoxelMorph
from monai.networks.nets import VoxelMorph
from monai.networks.blocks import Warp
from monai.transforms import DivisiblePad
from monai.losses import BendingEnergyLoss

def compute_dimention(label_pred_type,num_points_each_frame=None,num_frames=None,type_option=None):
    if type_option == 'pred':
        num_frames = num_frames-1

    type_dim_dict = {
        "transform": 12*num_frames,
        "parameter": 6*num_frames,
        "point": 3*4*num_frames,  # predict four corner points, and then intepolete the other points in a frame
        "quaternion": 7*num_frames
    }
    return type_dim_dict[label_pred_type]   # num_points=self.image_points.shape[1]), num_pairs=self.pairs.shape[0]



def data_pairs_adjacent(num_frames):
# obtain the data_pairs to compute the tarnsfomration between adjacent frames

    # return torch.tensor([[n0,n0+1] for n0 in range(num_frames-1)])# [0,1],[1,2],...[n-1,n]

    return torch.tensor([[0,n0] for n0 in range(num_frames)])

def scatter_plot_3D(data,save_folder,save_name):
    # plot 3D scatter points

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:,0,:], data[:,1,:], data[:,2,:], marker='o')
    plt.show()
    # plt.savefig(save_folder+'/'+save_name)


# def ():

# obtain all the coordinates within all the boxes’ area
# def convert_transform_to_pts(tarnsform,tform_calib,image_points,num_frames):
#     # obtain the coordinates of each frame from the transformations


def union_volome(gt_volume_position,pred_volume,pred_volume_position):
    # crop the volume2 based on volume 1
    # get the boundary of ground truth volume
    #  not completed
    gt_X = gt_volume_position[0]
    gt_Y = gt_volume_position[1]
    gt_Z = gt_volume_position[2]
    min_x = torch.min(gt_X)
    max_x = torch.max(gt_X)
    min_y = torch.min(gt_Y)
    max_y = torch.max(gt_Y)
    min_z = torch.min(gt_Z)
    max_z = torch.max(gt_Z)

    #  the length of each dimention is larger than the volume, because of the torch.ceil operation, we need to
    #  add one additional length for each dimention to allow the torch.ceil opration
    pred_X = torch.zeros((pred_volume.shape[0],pred_volume.shape[1],pred_volume.shape[2])) 

    pred_X = pred_volume_position[0] + pred_X[:-1,:-1,:-1]
    pred_Y = pred_volume_position[1]
    pred_Z = pred_volume_position[2]


    inside_min_x = torch.where(pred_X > min_x, 1.0, 0.0)
    inside_max_x = torch.where(pred_X < max_x, 1.0, 0.0)
    inside_min_y = torch.where(pred_Y > min_y, 1.0, 0.0)
    inside_max_y = torch.where(pred_Y < max_y, 1.0, 0.0)
    inside_min_z = torch.where(pred_Z > min_z, 1.0, 0.0)
    inside_max_z = torch.where(pred_Z < max_z, 1.0, 0.0)

    return pred_volume*inside_min_x*inside_max_x*inside_min_y*inside_max_y*inside_min_z*inside_max_z

def calculateConvPose_batched(pts_batched,option,device):
    for i_batch in range(pts_batched.shape[0]):
        
        ConvR = calculateConvPose(pts_batched[i_batch,...],option,device)
        ConvR = ConvR.repeat(pts_batched[i_batch,...].shape[0], 1,1)[None,...]
        if i_batch == 0:
            ConvR_batched = ConvR
        else:
            ConvR_batched = torch.cat((ConvR_batched,ConvR),0)
    return ConvR_batched
            


def calculateConvPose(pts,option,device):
    """Calculate roto-translation matrix from global reference frame to *convenient* reference frame.
    Voxel-array dimensions are calculated in this new refence frame. This rotation is important whenever the US scans sihouette is remarkably
    oblique to some axis of the global reference frame. In this case, the voxel-array dimensions (calculated by the smallest parallelepipedon 
    wrapping all the realigned scans), calculated in the global refrence frame, would not be optimal, i.e. larger than necessary.
    
    .. image:: diag_scan_direction.png
        :scale: 30 %          
        
    Parameters
    ----------
    convR : mixed
        Roto-translation matrix.
        If str, it specifies the method for automatically calculate the matrix.
        If 'auto_PCA', PCA is performed on all US image corners. The x, y and z of the new convenient reference frame are represented by the eigenvectors out of the PCA.
        If 'first_last_frames_centroid', the convenent reference frame is expressed as:
        
        - x from first image centroid to last image centroid
        - z orthogonal to x and the axis and the vector joining the top-left corner to the top-right corner of the first image
        - y orthogonal to z and x
        
        If np.ndarray, it must be manually specified as a 4 x 4 affine matrix.
        
    """
    # pts = torch.reshape(pts,(pts.shape[0],-1,3))
    # pts = torch.permute(pts, (2, 0, 1))
    
    # Calculating best pose automatically, if necessary
    # ivx = np.array(self.voxFrames)
    if option == 'auto_PCA':
        # Perform PCA on image corners
        # print ('Performing PCA on images corners...')
        U, s = pca(pts)
        # Build convenience affine matrix
        convR = np.vstack((np.hstack((U,np.zeros((3,1)))),[0,0,0,1])).T
        # print ('PCA perfomed')
    elif option == 'first_last_frames_centroid':
        # Search connection from first image centroid to last image centroid (X)
        # print ('Performing convenient reference frame calculation based on first and last image centroids...')
        C0 = torch.mean(pts[0,:,:], 1)  # 3
        C1 = torch.mean(pts[-1,:,:], 1)  # 3
        X = C1 - C0
        # Define Y and Z axis
        Ytemp = pts[0,:,0] - pts[0,:,1]   # from top-left corner to top-right corner
        
        Z = torch.cross(X, Ytemp)
        Y = torch.cross(Z, X)
        # Normalize axis length
        X = X / torch.linalg.norm(X)
        Y = Y / torch.linalg.norm(Y)
        Z = Z / torch.linalg.norm(Z)
        # Create rotation matrix
        # M = np.array([X, Y, Z]).T
        M = torch.transpose(torch.stack((X,Y,Z),0),0,1)
        # Build convenience affine matrix
        # convR = np.vstack((np.hstack((M,np.zeros((3,1)))),[0,0,0,1])).T
        convR = torch.transpose(torch.vstack((torch.hstack((M,torch.zeros((3,1)).to(device))),torch.tensor([0,0,0,1]).to(device))),0,1)
        # print ('Convenient reference frame calculated')

        return convR


def pca(D):
    """Run Principal Component Analysis on data matrix. It performs SVD
    decomposition on data covariance matrix.
    
    Parameters
    ----------
    D : np.ndarray
        Nv x No matrix, where Nv is the number of variables 
        and No the number of observations.
    
    Returns
    -------
    list
        U, s as out of SVD (``see np.linalg.svd``)

    """
    cov = np.cov(D)
    U, s, V = np.linalg.svd(cov)
    return U, s


def select_rec_reg_model(current_epoch,max_epoch,count_non_improved_loss,non_improve_maxmum,rec_model_training,reg_model_training):
    # count the number of non-improved epoches, and check if the model has been trained by the max_epoch
    # then convert trained model, from reg to rec, or vise 


    if count_non_improved_loss <= non_improve_maxmum and current_epoch <= max_epoch:
        # non improved epoch less than non_improve_maxmum, current_epoch less than max_epoch 
        rec_model_training = rec_model_training
        reg_model_training = reg_model_training
    elif count_non_improved_loss > non_improve_maxmum or current_epoch > max_epoch:
        rec_model_training = not rec_model_training
        reg_model_training = not reg_model_training


    return rec_model_training,reg_model_training


def select_optimiser(rec_model_training,reg_model_training,optimiser_rec,optimiser_reg):
    if rec_model_training and not reg_model_training:
        optimiser = optimiser_rec
    elif not rec_model_training and reg_model_training:
        optimiser = optimiser_reg
    elif rec_model_training and reg_model_training:
        raise("Optimise rec and reg at the same time is not supported")
    
    return optimiser


def append_to_list_max_40(original_list, new_element):
    original_list.append(new_element)
    if len(original_list) > 40:
        original_list.pop(0)  # Remove the oldest element when the list exceeds 40 elements

class Train_Rec_Reg_Model():

    def __init__(
        self, 
        opt,
        non_improve_maxmum, 
        reg_loss_weight,
        val_loss_min,
        val_dist_min,
        val_loss_min_reg,
        dset_train,
        dset_val,
        dset_train_reg,
        dset_val_reg,
        device,
        writer
        
        ):

        self.non_improve_maxmum = non_improve_maxmum
        self.val_loss_min = val_loss_min
        self.val_dist_min = val_dist_min
        self.val_loss_min_reg = val_loss_min_reg
        self.device = device
        self.writer = writer
        self.opt = opt
        self.dset_train = dset_train
        self.dset_val = dset_val
        self.dset_train_reg = dset_train_reg
        self.dset_val_reg = dset_val_reg
        
        
        self.datasets = {'train':self.dset_train_reg, 'val':self.dset_val_reg}

        self.data_pairs = data_pairs_adjacent(opt.NUM_SAMPLES)

        
        



        self.train_loader_rec = torch.utils.data.DataLoader(
            self.dset_train,
            batch_size=self.opt.MINIBATCH_SIZE,
            shuffle=True,
            num_workers=8
            )

        self.val_loader_rec = torch.utils.data.DataLoader(
            self.dset_val,
            batch_size=1, 
            shuffle=False,
            num_workers=8
            )


        ## loss
        self.tform_calib_scale, self.tform_calib_R_T,  self.tform_calib = read_calib_matrices(filename_calib=self.opt.FILENAME_CALIB, resample_factor=self.opt.RESAMPLE_FACTOR, device=self.device)


        self.image_points = reference_image_points((self.dset_train[0][0].shape[1],self.dset_train[0][0].shape[2]),(self.dset_train[0][0].shape[1],self.dset_train[0][0].shape[2])).to(self.device)
        self.pred_dim = compute_dimention(self.opt.PRED_TYPE, self.image_points.shape[1],self.opt.NUM_SAMPLES,'pred')
        self.label_dim = compute_dimention(self.opt.LABEL_TYPE, self.image_points.shape[1],self.opt.NUM_SAMPLES,'label')



        self.transform_label = LabelTransform(
            self.opt.LABEL_TYPE,
            pairs= self.data_pairs,  #
            image_points= self.image_points,
            in_image_coords=True,
            tform_image_to_tool= self.tform_calib,
            tform_image_mm_to_tool= self.tform_calib_R_T
            )
        # transform_label_to_params = LabelTransform(
        #     label_type = "parameter",
        #     pairs=data_pairs,
        #     image_points=image_points,
        #     in_image_coords=True,
        #     tform_image_to_tool=tform_calib,
        #     tform_image_mm_to_tool=tform_calib_R_T
        #     )
        self.transform_prediction = PredictionTransform(
             self.opt.PRED_TYPE,
            "transform",
            num_pairs= self.data_pairs.shape[0]-1,
            image_points= self.image_points,
            in_image_coords=True,
            tform_image_to_tool= self.tform_calib,
            tform_image_mm_to_tool= self.tform_calib_R_T
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
        self.criterion = torch.nn.MSELoss()
        self.metrics = PointDistance(paired=False)
        self.img_loss = MSELoss()
        self.regularization = BendingEnergyLoss()


        # init_data_pairs4lstm = init_data_pairs4LSTM(opt,device,tform_calib,tform_calib_R_T)
        ## network
        self.model = build_model(
             self.opt,
            in_frames =  self.opt.NUM_SAMPLES,
            pred_dim =  self.pred_dim,
            label_dim =  self.label_dim,
            image_points =  self.image_points,
            tform_calib =  self.tform_calib,
            tform_calib_R_T =  self.tform_calib_R_T
            ).to( self.device)


        # First, a backbone network is constructed. In this case, we use a VoxelMorphUNet as the backbone network.
        # self.backbone = VoxelMorphUNet(
        #     spatial_dims=3,
        #     in_channels=2,
        #     unet_out_channels=32,#16,#32,
        #     channels=(16, 32, 32, 32, 32, 32), #(8, 16, 16, 16, 16, 16),#(16, 32, 32, 32, 32, 32),  # this indicates the down block at the top takes 16 channels as
        #                                         # input, the corresponding up block at the top produces 32
        #                                         # channels as output, the second down block takes 32 channels as
        #                                         # input, and the corresponding up block at the same level
        #                                         # produces 32 channels as output, etc.
        #     final_conv_channels=(16,16),#(8, 8)#(16,16)
        #     ).to( self.device)

        # # Then, a full VoxelMorph network is constructed using the specified backbone network.
        # self.VoxelMorph_net = VoxelMorph(
        #     backbone= self.backbone,
        #     integration_steps=7,
        #     half_res=False
        #     ).to( self.device)

        self.VoxelMorph_net = VoxelMorph().to(self.device)
        # self.warp_layer = Warp().to(self.device) 

        
        
        self.PATH_SAVE = os.path.join(os.path.expanduser("~"), "/public_data/forearm_US_large_dataset")

        self.fh5_frames_train_path = os.path.join(self.PATH_SAVE,'data4reg_seqlen'+str(self.opt.NUM_SAMPLES)+'_'+self.opt.Conv_Coords+'_train.h5')
        self.fh5_frames_val_path = os.path.join(self.PATH_SAVE,'data4reg_seqlen'+str(self.opt.NUM_SAMPLES)+'_'+self.opt.Conv_Coords+'_val.h5')
        # self.fh5_frames_train = h5py.File(self.fh5_frames_train_path,'a')
        # self.fh5_frames_val = h5py.File(self.fh5_frames_val_path,'a')

        self.current_epoch = 0
        self.reg_loss_weight = reg_loss_weight

        # set optimiser
        self.optimiser_rec = torch.optim.Adam(self.model.parameters(), lr=self.opt.LEARNING_RATE)
        self.optimiser_reg = torch.optim.Adam(self.VoxelMorph_net.parameters(), lr=self.opt.LEARNING_RATE)


    def generate_reg_train_val_data(self):

        
        
        
        # delete the exist file
        os.remove(self.fh5_frames_train_path)
        os.remove(self.fh5_frames_val_path)



        # visualizer_scan_train = Visualizer(self.opt,self.device, self.dset_train,self.model,self.data_pairs,fh5_frames_train)
        # visualizer_scan_val = Visualizer(self.opt,self.device, self.dset_val,self.model,self.data_pairs,fh5_frames_val)

        #  - training set
        self.generate_reg_data(self.PATH_SAVE,'train',self.fh5_frames_train)
        #  - val set
        self.generate_reg_data(self.PATH_SAVE,'val',self.fh5_frames_val)



    def train_rec_model(self):
        # train reconstruction network
        count_non_improved_loss = 0

        self.model.train(True)
        self.VoxelMorph_net.train(False)
        
        

        # torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.current_epoch, self.current_epoch+self.opt.max_rec_epoch_each_interation):
            if count_non_improved_loss > self.non_improve_maxmum:
                break

            train_epoch_loss = 0
            train_epoch_dist = 0
            train_epoch_loss_reg = 0
            train_epoch_loss_rec = 0
            for step, (frames, tforms, tforms_inv) in enumerate(self.train_loader_rec):
                frames, tforms, tforms_inv = frames.to(self.device), tforms.to(self.device), tforms_inv.to(self.device)

                # cannot use the ground truth coordinates based on the camera coordinates system, 
                # which will depend on the posotion of camera
                # the transformation between each frame and frame 0
                tforms_each_frame2frame0 = self.transform_label(tforms, tforms_inv)
                # obtain the coordinates of each frame, set frame 0 as the reference frame
                
                # result1 = torch.linalg.multi_dot([*labels[0,...]])  # X[0] @ X[1] ... @ X[N-1]
                # the coordinates of each pixel points
                labels = torch.matmul(tforms_each_frame2frame0,torch.matmul(self.tform_calib,self.image_points))[:,:,0:3,...]
            
                # change labels to a convenient coordinates system
                
                # # check if the labels is correct
                # # ground truth, use camera as the reference
                # labels1 = torch.matmul(tforms,torch.matmul(tform_calib,image_points))
                # scatter_plot_3D(labels[0,...].cpu().numpy(),saved_folder,save_name = 'frame0.png')
                # scatter_plot_3D(labels1[0,...].cpu().numpy(),saved_folder,save_name = 'camera.png')


                frames = frames/255 # normalise image into range (0,1)

                self.optimiser_rec.zero_grad()
                
                
                if self.opt.model_name == 'LSTM_E':
                    outputs = torch.squeeze(self.model(frames),dim=1)
                else:
                    outputs = self.model(frames)

                # 6 parameter to 4*4 transformation
                pred_transfs = self.transform_prediction(outputs)
                # make the predicted transformations are based on frame 0
                # predict only opt.NUM_FRAES-1 transformatons,and let the first frame equals to identify matrix
                predframe0 = torch.eye(4,4)[None,...].repeat(pred_transfs.shape[0],1, 1,1).to(self.device)
                pred_transfs = torch.cat((predframe0,pred_transfs),1)

                # transformtion to points
                pred_pts = torch.matmul(pred_transfs,torch.matmul(self.tform_calib,self.image_points))[:,:,0:3,...]

                if self.opt.Conv_Coords == 'optimised_coord':
                    convR_batched = calculateConvPose_batched(labels,option = 'first_last_frames_centroid',device=self.device)    
                    labels = torch.matmul(convR_batched,torch.matmul(tforms_each_frame2frame0,torch.matmul(self.tform_calib,self.image_points)))[:,:,0:3,...]
                    
                    pred_pts = torch.matmul(convR_batched,torch.matmul(pred_transfs,torch.matmul(self.tform_calib,self.image_points)))[:,:,0:3,...]
                


                
                if self.opt.Loss_type == "MSE_points":
                    loss = self.criterion(pred_pts, labels)
                elif self.opt.Loss_type == "Plane_norm":
                    loss1 = self.criterion(pred_pts, labels)
                    normal_gt = compute_plane_normal(labels)
                    normal_np = compute_plane_normal(pred_pts)
                    cos_value = angle_between_planes(normal_gt,normal_np)
                    loss = loss1-sum(sum(cos_value))
                elif self.opt.Loss_type == "reg" or self.opt.Loss_type == "rec_reg":
                    # scatter points to grid points
                    gt_volume,pred_volume, warped, ddf = self.scatter_pts_registration(labels,pred_pts,frames,step)
                    
                    if self.opt.Loss_type == "reg":
                        # test if only use registartion can backward
                        loss = self.img_loss(torch.squeeze(warped,1),gt_volume) + self.regularization(ddf)
                    elif self.opt.Loss_type == "rec_reg":
                        loss1 = self.criterion(pred_pts, labels)
                        loss2 = self.img_loss(torch.squeeze(warped,1),gt_volume) + self.regularization(ddf)
                        loss = loss1+self.reg_loss_weight*loss2
                    elif self.opt.Loss_type == "rec_volume":
                        gt_volume, pred_volume = self.scatter_pts_intepolation(self,labels,pred_pts,frames,step)
                        loss1 = self.criterion(pred_pts, labels)
                        loss2 = self.criterion(gt_volume, pred_volume)
                        loss = loss1 + loss2


                dist = self.metrics(pred_pts, labels).detach()

                save2mha(gt_volume[0,...].cpu().numpy(),sx = 1,sy=1,sz=1,
                    save_folder='gt_volume_only_initial_0.mha'
                    )
                save2mha(pred_volume[0,...].detach().cpu().numpy(),sx = 1,sy=1,sz=1,
                    save_folder='pred_volume_only_initial_0.mha'
                    )
                
                save2mha(warped[0,0,...].detach().cpu().numpy(),sx = 1,sy=1,sz=1,
                    save_folder='wraped_volume_only_initial_0.mha'
                    )

                
            
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
                train_epoch_loss_rec = train_epoch_loss_rec + loss1.item()
                train_epoch_loss_reg += loss2.item()


                loss.backward()
                self.optimiser_rec.step()


                # pred_volume.detach(),warped.detach(),ddf.detach()
                

            train_epoch_loss /= (step + 1)
            train_epoch_dist /= (step + 1)
            train_epoch_loss_reg /= (step + 1)
            train_epoch_loss_rec /= (step + 1)

            if epoch in range(0, self.opt.NUM_EPOCHS, self.opt.FREQ_INFO):
                print('[Rec - Epoch %d] train-loss-rec=%.3f, train-dist=%.3f' % (epoch, train_epoch_loss_rec, train_epoch_dist))


            # validation    
            if epoch in range(0, self.opt.NUM_EPOCHS, self.opt.val_fre):

                self.model.train(False)
                self.VoxelMorph_net.train(False)

                epoch_loss_val = 0
                epoch_dist_val = 0
                epoch_loss_val_reg = 0
                epoch_loss_val_rec = 0
                for step, (fr_val, tf_val, tf_val_inv) in enumerate(self.val_loader_rec):

                    fr_val, tf_val, tf_val_inv = fr_val.to(self.device), tf_val.to(self.device), tf_val_inv.to(self.device)
                    tforms_each_frame2frame0_val = self.transform_label(tf_val, tf_val_inv)
                    labels_val = torch.matmul(tforms_each_frame2frame0_val,torch.matmul(self.tform_calib,self.image_points))[:,:,0:3,...]
                    
                    
                    fr_val = fr_val/255
                
                    if self.opt.model_name == 'LSTM_E':
                        out_val = torch.squeeze(self.model(fr_val),dim=1)
                    else:
                        out_val = self.model(fr_val)

                    pr_transfs_val = self.transform_prediction(out_val)
                    predframe0_val = torch.eye(4,4)[None,...].repeat(pr_transfs_val.shape[0],1, 1,1).to(self.device)
                    pr_transfs_val = torch.cat((predframe0_val,pr_transfs_val),1)


                    pred_pts_val = torch.matmul(pr_transfs_val,torch.matmul(self.tform_calib,self.image_points))[:,:,0:3,...]
                
                    if self.opt.Conv_Coords == 'optimised_coord':
                        # change labels to a convenient coordinates system
                        convR_batched_val = calculateConvPose_batched(labels_val,option = 'first_last_frames_centroid',device=self.device)    
                        labels_val = torch.matmul(convR_batched_val,torch.matmul(tforms_each_frame2frame0_val,torch.matmul(self.tform_calib,self.image_points)))[:,:,0:3,...]
                        
                        pred_pts_val = torch.matmul(convR_batched_val,torch.matmul(pr_transfs_val,torch.matmul(self.tform_calib,self.image_points)))[:,:,0:3,...]
                



                    if self.opt.Loss_type == "MSE_points":
                        loss_val = self.criterion(pred_pts_val, labels_val)
                    elif self.opt.Loss_type == "Plane_norm":
                        loss1_val = self.criterion(pred_pts_val, labels_val)
                        normal_gt_val = compute_plane_normal(labels_val)
                        normal_np_val = compute_plane_normal(pred_pts_val)
                        cos_value_val = angle_between_planes(normal_gt_val,normal_np_val)
                        loss_val = loss1_val-sum(sum(cos_value_val))
                    elif self.opt.Loss_type == "reg" or self.opt.Loss_type == "rec_reg":
                        # scatter points to grid points
                        gt_volume_val,pred_volume_val, warped_val, ddf_val = self.scatter_pts_registration(labels_val,pred_pts_val,fr_val,step)

                        if self.opt.Loss_type == "reg":
                            # test if only use registartion can backward
                            loss_val = self.img_loss(torch.squeeze(warped_val,1),gt_volume_val) + self.regularization(ddf_val)
                        elif self.opt.Loss_type == "rec_reg":
                            loss1_val = self.criterion(pred_pts_val, labels_val)
                            loss2_val = self.img_loss(torch.squeeze(warped_val,1),gt_volume_val) + self.regularization(ddf_val)
                            loss_val = loss1_val+self.reg_loss_weight*loss2_val

                    
                    dist_val = self.metrics(pred_pts_val, labels_val).detach()
                
                    
                    epoch_loss_val += loss_val.item()
                    epoch_dist_val += dist_val
                    epoch_loss_val_reg += loss2_val.item()
                    epoch_loss_val_rec = epoch_loss_val_rec + loss1_val.item()
                    
                    # pred_volume_val.detach(),warped_val.detach(),ddf_val.detach()


                epoch_loss_val /= (step+1)
                epoch_dist_val /= (step+1)
                epoch_loss_val_reg /= (step+1)
                epoch_loss_val_rec /= (step+1)

                # save model
                self.save_rec_model(epoch)
                # save best validation model - based on the valisation loss on transformation prediction, get rid of the registartion part
                self.val_loss_min, self.val_dist_min,count_non_improved_loss = save_best_network_rec(self.opt, self.model, epoch, epoch_loss_val_rec, epoch_dist_val, self.val_loss_min, self.val_dist_min,count_non_improved_loss)
                
                if epoch in range(0, self.opt.NUM_EPOCHS, self.opt.FREQ_INFO):
                    print('[Rec - Epoch %d] val-loss-rec=%.3f, val-dist=%.3f' % (epoch, epoch_loss_val_rec, epoch_dist_val))
                    print('[Rec - Epoch %d] count_non_improved_loss=%d' % (epoch,count_non_improved_loss))
                # add to tensorboard
                loss_dists = {'train_epoch_loss': train_epoch_loss, 
                            'train_epoch_dist': train_epoch_dist,
                            'train_epoch_loss_reg':train_epoch_loss_reg,
                            'train_epoch_loss_rec':train_epoch_loss_rec,

                            'epoch_loss_val':epoch_loss_val,
                            'epoch_dist_val':epoch_dist_val,
                            'epoch_loss_val_reg':epoch_loss_val_reg,
                            'epoch_loss_val_rec':epoch_loss_val_rec}
                add_scalars(self.writer, epoch, loss_dists)


                

                self.model.train(True)
                self.VoxelMorph_net.train(False)
        
        self.current_epoch = epoch+1

        
        # after train the rec model, generate training data for registration use
        # self.generate_reg_train_val_data()


    def scatter_pts_registration(self,labels,pred_pts,frames,step):
        # intepelote scatter points and thenregistartion

        gt_volume, pred_volume = self.scatter_pts_intepolation(labels,pred_pts,frames,step)

        warped, ddf = self.VoxelMorph_net(moving = torch.unsqueeze(pred_volume, 1), 
                    fixed = torch.unsqueeze(gt_volume, 1))

        # ddf = self.VoxelMorph_net(torch.cat((torch.unsqueeze(pred_volume,1), torch.unsqueeze(gt_volume,1)), dim=1)).float()

        # warped = self.warp_layer(pred_volume, ddf)

        
        return gt_volume,pred_volume, warped, ddf
    
    def scatter_pts_intepolation(self,labels,pred_pts,frames,step):
        # intepelote scatter points

        gt_volume, gt_volume_position = interpolation_3D_pytorch_batched(scatter_pts = labels,
                                                            frames = frames,
                                                            time_log=None,
                                                            saved_folder_test = None,
                                                            scan_name='gt_step'+str(step),
                                                            device = self.device,
                                                            option = self.opt.intepoletion_method,
                                                            volume_size = self.opt.intepoletion_volume,
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
                                                saved_folder_test = None,
                                                scan_name='pred_step'+str(step),
                                                device = self.device,
                                                option = self.opt.intepoletion_method,
                                                volume_size = self.opt.intepoletion_volume,
                                                volume_position = gt_volume_position
                                                )
        
        return gt_volume, pred_volume

        
    def train_reg_model(self):
        # train reconstruction network
        count_non_improved_loss_reg = 0

        self.model.train(False)
        self.VoxelMorph_net.train(True)
        

        # data loader
        # fh5_frames_train = h5py.File(self.fh5_frames_train_path,'a')
        # fh5_frames_val = h5py.File(self.fh5_frames_val_path,'a')

        # dataset_reg_train = SSFrameDataset_reg(
        #     min_scan_len = self.opt.MIN_SCAN_LEN,
        #     h5_file_name=self.fh5_frames_train_path,
        #     num_samples=self.opt.NUM_SAMPLES,
        #     sample_range=self.opt.SAMPLE_RANGE
        #     )
        
        # dataset_reg_val = SSFrameDataset_reg(
        #     min_scan_len = self.opt.MIN_SCAN_LEN,
        #     h5_file_name=self.fh5_frames_val_path,
        #     num_samples=self.opt.NUM_SAMPLES,
        #     sample_range=self.opt.SAMPLE_RANGE
        #     )
        
        # train_loader_reg = torch.utils.data.DataLoader(
        #     dataset_reg_train,
        #     batch_size=self.opt.MINIBATCH_SIZE,
        #     shuffle=True,
        #     num_workers=8
        #     )

        # val_loader_reg = torch.utils.data.DataLoader(
        #     dataset_reg_val,
        #     batch_size=1, 
        #     shuffle=False,
        #     num_workers=8
        #     )
        



        for epoch in range(self.current_epoch, self.current_epoch+self.opt.max_rec_epoch_each_interation):
            if count_non_improved_loss_reg > self.non_improve_maxmum:
                self.current_epoch = epoch
                break

            train_epoch_loss = 0
            train_epoch_dist = 0
            train_epoch_loss_reg = 0
            for step, (frames, tforms, tforms_inv) in enumerate(self.train_loader_rec):
                frames, tforms, tforms_inv = frames.to(self.device), tforms.to(self.device), tforms_inv.to(self.device)

                tforms_each_frame2frame0 = self.transform_label(tforms, tforms_inv)
                labels = torch.matmul(tforms_each_frame2frame0,torch.matmul(self.tform_calib,self.image_points))[:,:,0:3,...]
            
                frames = frames/255 # normalise image into range (0,1)
                self.optimiser_reg.zero_grad()                
                
                if self.opt.model_name == 'LSTM_E':
                    outputs = torch.squeeze(self.model(frames),dim=1)
                else:
                    outputs = self.model(frames)

                # 6 parameter to 4*4 transformation
                pred_transfs = self.transform_prediction(outputs)
                predframe0 = torch.eye(4,4)[None,...].repeat(pred_transfs.shape[0],1, 1,1).to(self.device)
                pred_transfs = torch.cat((predframe0,pred_transfs),1)

                pred_pts = torch.matmul(pred_transfs,torch.matmul(self.tform_calib,self.image_points))[:,:,0:3,...]

                if self.opt.Conv_Coords == 'optimised_coord':
                    convR_batched = calculateConvPose_batched(labels,option = 'first_last_frames_centroid',device=self.device)    
                    labels = torch.matmul(convR_batched,torch.matmul(tforms_each_frame2frame0,torch.matmul(self.tform_calib,self.image_points)))[:,:,0:3,...]
                    
                    pred_pts = torch.matmul(convR_batched,torch.matmul(pred_transfs,torch.matmul(self.tform_calib,self.image_points)))[:,:,0:3,...]
                
                gt_volume,pred_volume, warped, ddf = self.scatter_pts_registration(labels,pred_pts,frames,step)

                
                loss = self.img_loss(torch.squeeze(warped,1),gt_volume) + self.regularization(ddf)

                loss.backward()
                self.optimiser_reg.step()

                train_epoch_loss += loss.item()
            
            train_epoch_loss /= (step + 1)
            if epoch in range(0, self.opt.NUM_EPOCHS, self.opt.FREQ_INFO):
                print('[Reg - Epoch %d] train-loss=%f' % (epoch, train_epoch_loss))

            # validation    
            if epoch in range(0, self.opt.NUM_EPOCHS, self.opt.val_fre):

                self.model.train(False)
                self.VoxelMorph_net.train(False)
                epoch_loss_val = 0

                for step, (fr_val, tf_val, tf_val_inv) in enumerate(self.val_loader_rec):

                    fr_val, tf_val, tf_val_inv = fr_val.to(self.device), tf_val.to(self.device), tf_val_inv.to(self.device)
                    tforms_each_frame2frame0_val = self.transform_label(tf_val, tf_val_inv)
                    labels_val = torch.matmul(tforms_each_frame2frame0_val,torch.matmul(self.tform_calib,self.image_points))[:,:,0:3,...]
                    
                    
                    fr_val = fr_val/255
                
                    if self.opt.model_name == 'LSTM_E':
                        out_val = torch.squeeze(self.model(fr_val),dim=1)
                    else:
                        out_val = self.model(fr_val)

                    pr_transfs_val = self.transform_prediction(out_val)
                    predframe0_val = torch.eye(4,4)[None,...].repeat(pr_transfs_val.shape[0],1, 1,1).to(self.device)
                    pr_transfs_val = torch.cat((predframe0_val,pr_transfs_val),1)


                    pred_pts_val = torch.matmul(pr_transfs_val,torch.matmul(self.tform_calib,self.image_points))[:,:,0:3,...]
                
                    if self.opt.Conv_Coords == 'optimised_coord':
                        # change labels to a convenient coordinates system
                        convR_batched_val = calculateConvPose_batched(labels_val,option = 'first_last_frames_centroid',device=self.device)    
                        labels_val = torch.matmul(convR_batched_val,torch.matmul(tforms_each_frame2frame0_val,torch.matmul(self.tform_calib,self.image_points)))[:,:,0:3,...]
                        
                        pred_pts_val = torch.matmul(convR_batched_val,torch.matmul(pr_transfs_val,torch.matmul(self.tform_calib,self.image_points)))[:,:,0:3,...]
                
                    gt_volume_val,pred_volume_val, warped_val, ddf_val = self.scatter_pts_registration(labels_val,pred_pts_val,fr_val,step)

                    
                    loss_val = self.img_loss(torch.squeeze(warped_val,1),gt_volume_val) + self.regularization(ddf_val)
                    
                    epoch_loss_val += loss_val.item()

                epoch_loss_val /= (step + 1)
                if epoch in range(0, self.opt.NUM_EPOCHS, self.opt.FREQ_INFO):
                    print('[Reg - Epoch %d] val-loss=%f' % (epoch, epoch_loss_val))

                # save model
                self.save_reg_model(epoch)

                # save best validation model
                self.val_loss_min_reg,count_non_improved_loss_reg = save_best_network_reg(self.opt, self.VoxelMorph_net, epoch, epoch_loss_val, self.val_loss_min_reg,count_non_improved_loss_reg)
                print('[Reg - Epoch %d] count_non_improved_loss_reg=%d' % (epoch,count_non_improved_loss_reg))

                # add to tensorboard
                loss_reg = {'train_epoch_loss': train_epoch_loss, 
                            
                            'epoch_loss_val':epoch_loss_val,
                            }
                add_scalars_reg(self.writer, epoch, loss_reg,'reg')


                # add_scalars_params(writer, epoch,error_6DOF_train,error_6DOF_val)
                # write_to_txt(opt, epoch, loss_dists)
                # write_to_txt_2(opt, data_pairs.shape[0], dist, metrics(pr_val, la_val))



                self.model.train(False)
                self.VoxelMorph_net.train(True)
        





        self.current_epoch = epoch+1


    def save_rec_model(self,epoch):
        if epoch in range(0, self.opt.NUM_EPOCHS, self.opt.FREQ_SAVE):
            torch.save(self.model.state_dict(), os.path.join(self.opt.SAVE_PATH,'saved_model','model_epoch%08d' % epoch))

            print('Model parameters saved.')
            list_dir = os.listdir(os.path.join(self.opt.SAVE_PATH, 'saved_model'))
            saved_models = [i for i in list_dir if i.startswith('model_epoch')]
            if len(saved_models)>4:
                print(saved_models)
                os.remove(os.path.join(self.opt.SAVE_PATH,'saved_model',sorted(saved_models)[0]))

            
    def save_reg_model(self,epoch):
        if epoch in range(0, self.opt.NUM_EPOCHS, self.opt.FREQ_SAVE):
            torch.save(self.VoxelMorph_net.state_dict(), os.path.join(self.opt.SAVE_PATH,'saved_model','model_reg_epoch%08d' % epoch))

            print('Model parameters saved.')
            list_dir = os.listdir(os.path.join(self.opt.SAVE_PATH, 'saved_model'))
            
            saved_models_reg = [i for i in list_dir if i.startswith('model_reg_epoch')]
            if len(saved_models_reg)>4:
                print(saved_models_reg)
                os.remove(os.path.join(self.opt.SAVE_PATH,'saved_model',sorted(saved_models_reg)[0]))


    def generate_reg_data(self, saved_folder,train_val_test,fh5_frames):
        
        data_used = self.datasets[train_val_test]
        for scan_index in range(len(data_used)):
            frames, tforms, tforms_inv = data_used[scan_index]
            #  the first dimention is batchsize
            frames, tforms, tforms_inv = (torch.tensor(t)[None,...].to(self.device) for t in [frames, tforms, tforms_inv])
            frames = frames/255
            saved_folder = saved_folder
            saved_name = data_used.name_scan[data_used.indices_in_use[scan_index][0], data_used.indices_in_use[scan_index][1]].decode("utf-8")


            idx = 0
            
            while True:
                if (idx + self.opt.NUM_SAMPLES) > frames.shape[1]:
                    break

                frames_sub = frames[:,idx:idx + self.opt.NUM_SAMPLES, ...]
                tforms_sub = tforms[:,idx:idx + self.opt.NUM_SAMPLES, ...]
                tforms_inv_sub = tforms_inv[:,idx:idx + self.opt.NUM_SAMPLES, ...]
                # frames_sub = frames_sub/255

                # obtain the transformation from current frame to frame 0
                tforms_each_frame2frame0_gt_sub = self.transform_label(tforms_sub, tforms_inv_sub)
                
                outputs = self.model(frames_sub)
                # 6 parameter to 4*4 transformation
                pred_transfs = self.transform_prediction(outputs)

                # make the predicted transformations are based on frame 0
                predframe0 = torch.eye(4,4)[None,...].repeat(pred_transfs.shape[0],1, 1,1).to(self.device)
                pred_transfs = torch.cat((predframe0,pred_transfs),1)

                
                # obtain the coordinates of each frame, using frame 0 as the reference frame
                labels_gt_sub = torch.matmul(tforms_each_frame2frame0_gt_sub,torch.matmul(self.tform_calib,self.image_points))[:,:,0:3,...]
                

                # transformtion to points
                pred_pts_sub = torch.matmul(pred_transfs,torch.matmul(self.tform_calib,self.image_points))[:,:,0:3,...]
                

                
                # change labels to a convenient coordinates system
                if self.opt.Conv_Coords == 'optimised_coord':
                    convR_batched = calculateConvPose_batched(labels_gt_sub,option = 'first_last_frames_centroid',device=self.device)    
                    labels_gt_sub = torch.cat((labels_gt_sub,torch.ones([1,labels_gt_sub.shape[-1]]).repeat(labels_gt_sub.shape[0],labels_gt_sub.shape[1],1,1).to(self.device)),2)
                    labels_gt_sub = torch.matmul(convR_batched,labels_gt_sub)[:,:,0:3,...]

                    pred_pts_sub = torch.cat((pred_pts_sub,torch.ones([1,pred_pts_sub.shape[-1]]).repeat(pred_pts_sub.shape[0],pred_pts_sub.shape[1],1,1).to(self.device)),2)
                    pred_pts_sub = torch.matmul(convR_batched,pred_pts_sub)[:,:,0:3,...]
            
                
                gt_volume, gt_volume_position = interpolation_3D_pytorch_batched(scatter_pts = labels_gt_sub,
                                                    frames = frames[0,idx:idx + self.opt.NUM_SAMPLES,...],
                                                    time_log=None,
                                                    saved_folder_test = saved_folder,
                                                    scan_name=saved_name+'_gt',
                                                    device = self.device,
                                                    option = self.opt.intepoletion_method,
                                                    volume_position = None,
                                                    volume_size = self.opt.intepoletion_volume,

                                                    )
                
                pred_volume,pred_volume_position = interpolation_3D_pytorch_batched(scatter_pts = pred_pts_sub,
                                                        frames = frames[0,idx:idx + self.opt.NUM_SAMPLES,...],
                                                        time_log=None,
                                                        saved_folder_test = saved_folder,
                                                        scan_name=saved_name+'_pred',
                                                        device = self.device,
                                                        option = self.opt.intepoletion_method,
                                                        volume_position = gt_volume_position,
                                                        volume_size = self.opt.intepoletion_volume,
                                                        )
                gt_saved = torch.squeeze(gt_volume,0).cpu().numpy()
                pred_saved = torch.squeeze(pred_volume,0).detach().cpu().numpy()
                
                fh5_frames.create_dataset('/sub%03d_scan%02d_seq%04d_gt' % (data_used.indices_in_use[scan_index][0],data_used.indices_in_use[scan_index][1],idx), gt_saved.shape, dtype=gt_saved.dtype, data=gt_saved)
                fh5_frames.create_dataset('/sub%03d_scan%02d_seq%04d_pred' % (data_used.indices_in_use[scan_index][0],data_used.indices_in_use[scan_index][1],idx), pred_saved.shape, dtype=pred_saved.dtype, data=pred_saved)

                

                idx += 1

                # save2mha(gt_volume[0,...].cpu().numpy(),sx = 1,sy=1,sz=1,
                #     save_folder=saved_folder+'/'+'gt-test.mha'
                #     )

                # save2mha(pred_volume[0,...].detach().cpu().numpy(),sx = 1,sy=1,sz=1,
                #     save_folder=saved_folder+'/'+'pred-test.mha'
                #     )

                # print('done')
            
        
        
        fh5_frames.create_dataset('num_frames', data_used.num_frames.shape, dtype=data_used.num_frames.dtype, data=data_used.num_frames)
        # self.fh5_frames.create_dataset('sub_folders', self.num_frames.shape[0], data=self.num_frames.shape[0])
        # self.fh5_frames.create_dataset('frame_size', 2, data=frames_croped.shape[1:3])
        fh5_frames.create_dataset('name_scan', tuple((len(data_used.name_scan),len(data_used.name_scan[0]))), data=data_used.name_scan)

        fh5_frames.flush()
        fh5_frames.close()

    def load_best_rec_model(self):
        # load the best transformation model for training again or for generating volume data for registartion model use
        try:
            self.model.load_state_dict(torch.load(os.path.join(self.opt.SAVE_PATH,'saved_model', 'best_validation_loss_model'),map_location=torch.device(self.device)))
        except:
            print('No best rec model saved at the moment...')

    def load_best_reg_model(self):
        # load the best registation model registartion
        try:
            self.VoxelMorph_net.load_state_dict(torch.load(os.path.join(self.opt.SAVE_PATH,'saved_model', 'best_validation_loss_model_reg'),map_location=torch.device(self.device)))
        except:
            print('No best reg model saved at the moment...')

    def load_best_rec_model_initial(self):
        # load the best registation model registartion
        try:
            self.model.load_state_dict(torch.load(os.path.join(self.opt.SAVE_PATH,'saved_model', 'best_validation_loss_model_initial'),map_location=torch.device(self.device)))
        except:
            raise('No best model saved at the moment...')
        
    def load_best_reg_model_initial(self):
        # load the best registation model registartion
        try:
            self.VoxelMorph_net.load_state_dict(torch.load(os.path.join(self.opt.SAVE_PATH,'saved_model', 'best_validation_loss_model_reg_initial'),map_location=torch.device(self.device)))
        except:
            raise('No best model saved at the moment...')