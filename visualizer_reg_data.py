import os
from torch.autograd import Variable
import json
from matplotlib import pyplot as plt
import torch
import numpy as np
from torchvision.models import efficientnet_b1
from torch.utils.tensorboard import SummaryWriter
import pickle
import torch.nn as nn

from loader import SSFrameDataset
from network_isbi import build_model
from data.calib import read_calib_matrices
from transform import LabelTransform, TransformAccumulation, ImageTransform, PredictionTransform
from utils import pair_samples, reference_image_points, type_dim, sample_dists4plot
from options.train_options import TrainOptions
#from options.test_options import TestOptions
from loss import PointDistance_2

from utilits_grid_data import *



from monai.networks.nets.voxelmorph import VoxelMorphUNet, VoxelMorph
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


class Visualizer():  # plot scan

    def __init__(self, opt,device, dset,model_name,data_pairs,fh5_frames,batchsize = 1):
        self.opt = opt
        self.opt_test = self.opt
        self.device = device
        self.dset = dset
        
        self.num_frames = self.dset.num_frames
        self.name_scan = self.dset.name_scan

        self.fh5_frames = fh5_frames

        
        self.opt.MINIBATCH_SIZE = batchsize

        self.FILENAME_WEIGHTS = model_name
        self.model_name = model_name

        self.data_pairs = data_pairs
        self.tform_calib_scale, self.tform_calib_R_T, self.tform_calib = read_calib_matrices(
            filename_calib=self.opt.FILENAME_CALIB,
            resample_factor=self.opt.RESAMPLE_FACTOR,
            device=self.device#'cpu'#self.device''
        )
        # using prediction: transform frame_points from current image to starting (reference) image 0
        # four corner points
        self.four_points = reference_image_points(self.dset.frame_size, 2)#.to(self.device)
        

        # using GT: transform pixel_points from current image to starting (reference) image 0
        # all points in a frame
        self.all_points = reference_image_points(self.dset.frame_size, self.dset.frame_size).to(self.device)

        # if self.opt_test.plot_line:
        #     self.pixel_points = self.frame_points

        self.transform_label = LabelTransform(
            label_type=self.opt.LABEL_TYPE,
            pairs=self.data_pairs,  #
            image_points=self.all_points ,
            in_image_coords=True,
            tform_image_to_tool=self.tform_calib,
            tform_image_mm_to_tool=self.tform_calib_R_T
            )
        

       
       

        self.transform_prediction = PredictionTransform(
            self.opt.PRED_TYPE,
            "transform",
            num_pairs=self.data_pairs.shape[0]-1,
            image_points=self.all_points,
            in_image_coords=True,
            tform_image_to_tool=self.tform_calib,
            tform_image_mm_to_tool=self.tform_calib_R_T
            )
        
        self.pred_dim = compute_dimention(self.opt.PRED_TYPE, self.all_points.shape[1],self.opt.NUM_SAMPLES,type_option='pred')
        self.label_dim = compute_dimention(self.opt.LABEL_TYPE, self.all_points.shape[1],self.opt.NUM_SAMPLES)

        self.metrics = PointDistance_2()

        ## load the model
        self.model = build_model(
            self.opt,
            in_frames = self.opt.NUM_SAMPLES,
            pred_dim = self.pred_dim,
            label_dim = self.label_dim,
            image_points = self.all_points,
            tform_calib = self.tform_calib,
            tform_calib_R_T = self.tform_calib_R_T
            ).to(self.device)


        # First, a backbone network is constructed. In this case, we use a VoxelMorphUNet as the backbone network.
        self.backbone = VoxelMorphUNet(
            spatial_dims=3,
            in_channels=2,
            unet_out_channels=32,
            channels=(16, 32, 32, 32, 32, 32),  # this indicates the down block at the top takes 16 channels as
                                                # input, the corresponding up block at the top produces 32
                                                # channels as output, the second down block takes 32 channels as
                                                # input, and the corresponding up block at the same level
                                                # produces 32 channels as output, etc.
            final_conv_channels=(16, 16)
        ).to(self.device)

        # Then, a full VoxelMorph network is constructed using the specified backbone network.
        self.VoxelMorph_net = VoxelMorph(
            backbone=self.backbone,
            integration_steps=7,
            half_res=False
        ).to(self.device)


        
            
        self.model.load_state_dict(torch.load(os.path.join(self.opt.SAVE_PATH, self.opt_test.MODEL_FN,self.model_name[0]), map_location=torch.device(self.device)))
        self.VoxelMorph_net.load_state_dict(torch.load(os.path.join(self.opt.SAVE_PATH,self.opt_test.MODEL_FN, self.model_name[1]),map_location=torch.device(self.device)))

        
        self.model.train(False)
        self.VoxelMorph_net.train(False)
       

    def generate_reg_data(self, scan_index,saved_folder):
        
        frames, tforms, tforms_inv = self.dset[scan_index]
        #  the first dimention is batchsize
        frames, tforms, tforms_inv = (torch.tensor(t)[None,...].to(self.device) for t in [frames, tforms, tforms_inv])
        frames = frames/255
        saved_folder = saved_folder
        saved_name = self.name_scan[self.dset.indices_in_use[scan_index][0], self.dset.indices_in_use[scan_index][1]].decode("utf-8")

        # # compute all the points in ground truth volume at once
        # # reference frame is frame 0

        # data_pairs_all = data_pairs_adjacent(frames.shape[1])
        # data_pairs_all=torch.tensor(data_pairs_all)

        # transform_label_all = LabelTransform(
        #     label_type=self.opt.LABEL_TYPE,
        #     pairs=data_pairs_all,  #
        #     image_points=self.all_points ,
        #     in_image_coords=True,
        #     tform_image_to_tool=self.tform_calib,
        #     tform_image_mm_to_tool=self.tform_calib_R_T
        #     )
        # tforms_each_frame2frame0_gt_all = transform_label_all(tforms, tforms_inv)
        # labels_gt_all = torch.matmul(tforms_each_frame2frame0_gt_all,torch.matmul(self.tform_calib,self.all_points))[:,:,0:3,...]

        # # intepolete
        # gt_volume_all, gt_volume_position_all = interpolation_3D_pytorch_train(scatter_pts = labels_gt_all[0,...],
        #                                            frames = frames[0,...],
        #                                            time_log=None,
        #                                            saved_folder_test = saved_folder,
        #                                            scan_name=saved_name+'_gt_all',
        #                                            device = self.device,
        #                                            option = self.opt.intepoletion_method,
        #                                            volume_position = None
        #                                            )

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

            # if idx !=0:
            #     # if not the first sub-sequence, should be transformed into frame 0
            #     tforms_each_frame2frame0_gt_sub = torch.matmul(tform_last_frame[None,...],tforms_each_frame2frame0_gt_sub)
            #     pred_transfs = torch.matmul(tform_last_frame_pred[None,...],pred_transfs) 
            
            # tform_last_frame = tforms_each_frame2frame0_gt_sub[:,-1,...]
            # tform_last_frame_pred = pred_transfs[:,-1,...]

            # obtain the coordinates of each frame, using frame 0 as the reference frame
            labels_gt_sub = torch.matmul(tforms_each_frame2frame0_gt_sub,torch.matmul(self.tform_calib,self.all_points))[:,:,0:3,...]
            

            # transformtion to points
            pred_pts_sub = torch.matmul(pred_transfs,torch.matmul(self.tform_calib,self.all_points))[:,:,0:3,...]
            

            # if idx ==0:
            #     labels_gt = labels_gt_sub
            #     pred_pts = pred_pts_sub
            # else:
            #     labels_gt = torch.cat((labels_gt,labels_gt_sub[:,1:,...]),1)
            #     pred_pts = torch.cat((pred_pts,pred_pts_sub[:,1:,...]),1)

            # change labels to a convenient coordinates system
            if self.opt.Conv_Coords == 'optimised_coord':
                convR_batched = calculateConvPose_batched(labels_gt_sub,option = 'first_last_frames_centroid',device=self.device)    
                labels_gt_sub = torch.cat((labels_gt_sub,torch.ones([1,labels_gt_sub.shape[-1]]).repeat(labels_gt_sub.shape[0],labels_gt_sub.shape[1],1,1).to(self.device)),2)
                labels_gt_sub = torch.matmul(convR_batched,labels_gt_sub)[:,:,0:3,...]

                pred_pts_sub = torch.cat((pred_pts_sub,torch.ones([1,pred_pts_sub.shape[-1]]).repeat(pred_pts_sub.shape[0],pred_pts_sub.shape[1],1,1).to(self.device)),2)
                pred_pts_sub = torch.matmul(convR_batched,pred_pts_sub)[:,:,0:3,...]
        
             
            


            


                

       

            # intepolete
            # # compute the common volume for ground truth and prediction to intepolete
            # min_x,max_x = torch.min(torch.min(labels_gt_sub[0,:,0,:])),torch.max(torch.max(labels_gt_sub[0,:,0,:]))
            # min_y,max_y = torch.min(torch.min(labels_gt_sub[0,:,1,:])),torch.max(torch.max(labels_gt_sub[0,:,1,:]))
            # min_z,max_z = torch.min(torch.min(labels_gt_sub[0,:,2,:])),torch.max(torch.max(labels_gt_sub[0,:,2,:]))


            # x = torch.linspace((min_x.item()), (max_x.item()), int(((max_x.item())-(min_x.item()))))
            # y = torch.linspace((min_y.item()), (max_y.item()), int(((max_y.item())-(min_y.item()))))
            # z = torch.linspace((min_z.item()), (max_z.item()), int(((max_z.item())-(min_z.item()))))
            # X, Y, Z = torch.meshgrid(x, y, z,indexing='ij')
            # X, Y, Z =X.to(self.device), Y.to(self.device), Z.to(self.device) 
            
            
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
            
            self.fh5_frames.create_dataset('/sub%03d_scan%02d_seq%04d_gt' % (self.dset.indices_in_use[scan_index][0],self.dset.indices_in_use[scan_index][1],idx), gt_saved.shape, dtype=gt_saved.dtype, data=gt_saved)
            self.fh5_frames.create_dataset('/sub%03d_scan%02d_seq%04d_pred' % (self.dset.indices_in_use[scan_index][0],self.dset.indices_in_use[scan_index][1],idx), pred_saved.shape, dtype=pred_saved.dtype, data=pred_saved)

            

            idx += 1

            # save2mha(gt_volume[0,...].cpu().numpy(),sx = 1,sy=1,sz=1,
            #     save_folder=saved_folder+'/'+'gt-test.mha'
            #     )

            # save2mha(pred_volume[0,...].detach().cpu().numpy(),sx = 1,sy=1,sz=1,
            #     save_folder=saved_folder+'/'+'pred-test.mha'
            #     )

            # print('done')
        
        
        
        self.fh5_frames.create_dataset('num_frames', self.num_frames.shape, dtype=self.num_frames.dtype, data=self.num_frames)
        # self.fh5_frames.create_dataset('sub_folders', self.num_frames.shape[0], data=self.num_frames.shape[0])
        # self.fh5_frames.create_dataset('frame_size', 2, data=frames_croped.shape[1:3])
        self.fh5_frames.create_dataset('name_scan', tuple((len(self.name_scan),len(self.name_scan[0]))), data=self.name_scan)

        self.fh5_frames.flush()
        self.fh5_frames.close()

        # return self.fh5_frames
        
        

       
        #     # plot the reference
        #     px, py, pz = [torch.mm(self.tform_calib_scale, torch.mm(torch.from_numpy(np.array([[self.opt.RESAMPLE_FACTOR, 0, 0, 0], [0, self.opt.RESAMPLE_FACTOR, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],np.float32)), self.pixel_points))[ii,] for ii in range(3)]
        #     pix_intensities = (frames[idx_p0, ..., None].float() / 255).cpu().expand(-1, -1, 3).numpy()
        #     fx, fy, fz = [torch.mm(self.tform_calib_scale, torch.mm(torch.from_numpy(np.array([[self.opt.RESAMPLE_FACTOR, 0, 0, 0], [0, self.opt.RESAMPLE_FACTOR, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],np.float32)), self.pixel_points))[ii,].reshape(2, 2).cpu() for ii in range(3)]

        #     fig, ax = plt.figure().add_subplot(projection='3d')
        #     ax.plot_surface(px, py, pz, facecolors=pix_intensities, linewidth=0, edgecolors=None, antialiased=True)
        #     ax.plot_surface(fx, fy, fz, edgecolor='y', linewidth=2, alpha=0.2, antialiased=True)
        #     prev_tform = torch.eye(4)

        #     rms_img = []  # store the distance of each image
        #     rms_img_4_plot = []
        #     img_no = 0  # index of image in this scan

        #     while 1:
        #         img_no = img_no + interval_pred

        #         # prediction -> points in image coords
        #         frames_val = frames[idx_f0:idx_f0 + self.opt_test.NUM_SAMPLES, ...]
        #         frames_val = self.transform_image(frames_val)
        #         if self.opt.model_name != 'LSTM':
        #             outputs_val = self.model(frames_val.unsqueeze(0))
        #         else:
        #             encoder_h_1 = (Variable(torch.zeros(frames_val.unsqueeze(0).shape[0], 256, 30, 40)),
        #                            Variable(torch.zeros(frames_val.unsqueeze(0).shape[0], 256, 30, 40)))
        #             encoder_h_2 = (Variable(torch.zeros(frames_val.unsqueeze(0).shape[0], 512, 15, 20)),
        #                            Variable(torch.zeros(frames_val.unsqueeze(0).shape[0], 512, 15, 20)))
        #             encoder_h_3 = (Variable(torch.zeros(frames_val.unsqueeze(0).shape[0], 512, 8, 10)),
        #                            Variable(torch.zeros(frames_val.unsqueeze(0).shape[0], 512, 8, 10)))

        #             outputs_val, encoder_h_1, encoder_h_2, encoder_h_3 = self.model(
        #                 frames_val.unsqueeze(0), encoder_h_1, encoder_h_2, encoder_h_3)

        #         outputs_val = self.transform_prediction(outputs_val).detach().cpu()  # transform prediction to 4*4 transformation type
        #         current_tform = outputs_val.reshape(self.data_pairs.shape[0], 4, 4)[PAIR_INDEX, :, :].cpu()
        #         preds_val, prev_tform = self.accumulate_prediction(prev_tform, current_tform)
        #         y_predicted = preds_val

        #         fx, fy, fz = [preds_val[ii,].reshape(2, 2).cpu().detach().numpy() for ii in range(3)]

        #         ax.plot_surface(fx, fy, fz, edgecolor='y', linewidth=0.5, alpha=0.1, antialiased=True)
        #         # label -> points in image coords, wrt. the "unchanged" reference starting frame, idx_p0
        #         tforms_val, tforms_inv_val = (t[[idx_p0, idx_p1], ...] for t in [tforms, tforms_inv])
        #         label = self.transform_label(tforms_val.unsqueeze(0), tforms_inv_val.unsqueeze(0))
        #         px, py, pz = [label[:, :, ii, :].reshape(self.dset_val.frame_size[0], self.dset_val.frame_size[1]).cpu() for ii in range(3)]
        #         pix_intensities = (frames[idx_p1, ..., None].float() / 255).cpu().expand(-1, -1, 3).numpy()
        #         ax.plot_surface(px, py, pz, linewidth=0, facecolors=pix_intensities, edgecolors=None,antialiased=True)  #
        #         y_actual = torch.squeeze(label)
        #         # if img_no % self.opt_test.MAX_INTERVAL == 0 and img_no <= ((num_frames - self.opt_test.MAX_INTERVAL) / self.opt_test.MAX_INTERVAL) * self.opt_test.MAX_INTERVAL:
        #         rms_each_img = self.metrics(y_actual, y_predicted)
        #         rms_img.append(rms_each_img.detach().cpu().numpy())
        #         rms_each_img_4_plot = self.metrics(y_actual, y_predicted)
        #         rms_img_4_plot.append(rms_each_img_4_plot.detach().cpu().numpy())

        #         # update for the next prediction
        #         idx_f0 += interval_pred
        #         idx_p1 += interval_pred
        #         if (idx_f0 + self.opt_test.NUM_SAMPLES) > frames.shape[0]:
        #             break

        #     plt.show()
        # else:

        #     # plot the frame 0
        #     # px, py, pz = [torch.mm(self.tform_calib_scale, torch.mm(torch.from_numpy(np.array([[self.opt.RESAMPLE_FACTOR, 0, 0, 0], [0, self.opt.RESAMPLE_FACTOR, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],np.float32)), self.pixel_points))[ii,] for ii in range(3)]
        #     # pix_intensities = (frames[idx_p0, ..., None].float() / 255).cpu().expand(-1, -1, 3).numpy()
        #     fx, fy, fz = [torch.mm(self.tform_calib_scale.cpu(), torch.mm(torch.from_numpy(np.array([[self.opt.RESAMPLE_FACTOR, 0, 0, 0], [0, self.opt.RESAMPLE_FACTOR, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],np.float32)), self.pixel_points.cpu()))[ii,].reshape(2, 2) for ii in range(3)]

        #     # fig, ax = plt.figure().add_subplots(projection='3d')
        #     # ax = plt.axes(projection='3d')
        #     # ax.plot_surface(px, py, pz, facecolors=pix_intensities, linewidth=0, edgecolors=None, antialiased=True)
        #     ax.plot_surface(fx, fy, fz, edgecolor='r', linewidth=1, alpha=0.2, antialiased=True)

        #     # # plot the first number of frames to make sure different intervals have the same start frames
        #     # tforms_val, tforms_inv_val = (t[[0, idx_p0], ...] for t in [tforms, tforms_inv])
        #     # label = self.transform_label(tforms_val.unsqueeze(0), tforms_inv_val.unsqueeze(0))
        #     # px, py, pz = [label[:, :, ii, :] for ii in range(3)]
        #     # ax.scatter(px, py, pz, c='r', alpha=0.2, s=2)  #

        #     # fx, fy, fz = [torch.mm(self.tform_calib_scale, torch.mm(torch.from_numpy(np.array(
        #     #     [[self.opt.RESAMPLE_FACTOR, 0, 0, 0], [0, self.opt.RESAMPLE_FACTOR, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        #     #     np.float32)), self.pixel_points))[ii,].reshape(2, 2).cpu() for ii in range(3)]

        #     # if start frame is not from 0, the prev_tform label can be computed as
        #     tforms_val, tforms_inv_val = (t[[0 , idx_f0 + torch.squeeze(self.data_pairs[PAIR_INDEX])[0]], ...] for t in [tforms, tforms_inv])
        #     _label = self.transform_label_2_transformation(tforms_val.unsqueeze(0), tforms_inv_val.unsqueeze(0))
        #     # first_img = self.transform_label(tforms_val.unsqueeze(0), tforms_inv_val.unsqueeze(0))
        #     prev_tform = torch.squeeze(_label)
        #     prev_tform_all_pixel = torch.squeeze(_label)

        #     rms_img_gt_based = []  # store the distance of each image
        #     rms_img_4_plot_gt_based = []
        #     rms_img = []  # store the distance of each image
        #     rms_img_4_plot,y_actual_all_pixel_in_a_scan,y_predicted_all_pixel_in_a_scan = [],[],[]
        #     y_actual_overlap,y_predicted_overlap=[],[]
        #     img_no = 0  # index of image in this scan
        #     label_x, label_y, label_z, pre_x, pre_y, pre_z = [], [] ,[], [], [], []
        #     while 1:
        #         img_no = img_no + interval_pred

        #         # prediction -> points in image coords
        #         frames_val = frames[idx_f0:idx_f0 + self.opt_test.NUM_SAMPLES, ...]

        #         # label -> points in image coords, wrt. the "unchanged" reference starting frame, idx_p0
        #         tforms_val, tforms_inv_val = (t[[idx_p0, idx_p1], ...] for t in [tforms, tforms_inv])
        #         label = self.transform_label(tforms_val.unsqueeze(0).cpu(), tforms_inv_val.unsqueeze(0).cpu())
        #         # coordinates of all pixels
        #         label_all_pixels = self.transform_label_all_pixel(tforms_val.unsqueeze(0).cpu(), tforms_inv_val.unsqueeze(0).cpu())
        #         px, py, pz = [label[:, :, ii, :] for ii in range(3)]
        #         label_x.append(torch.squeeze(px).cpu().detach().numpy())
        #         label_y.append(torch.squeeze(py).cpu().detach().numpy())
        #         label_z.append(torch.squeeze(pz).cpu().detach().numpy())

        #         # label -> points in image coords, wrt. the "unchanged" reference starting frame, idx_p0
        #         # this label is used for the ground truth based error computation
        #         tforms_val_gt, tforms_inv_val_gt = (t[[idx_p0, idx_p1-interval_pred], ...] for t in [tforms, tforms_inv])
        #         label_matrix_gt = self.transform_label_2_transformation(tforms_val_gt.unsqueeze(0).cpu(), tforms_inv_val_gt.unsqueeze(0).cpu())
        #         frames_val = self.transform_image(frames_val)
        #         if self.opt.model_name == 'LSTM_E':
        #             outputs_val = torch.squeeze(self.model(frames_val.unsqueeze(0)), dim=1)
        #         else:
        #             outputs_val = self.model(frames_val.unsqueeze(0))



        #         outputs_val = self.transform_prediction(outputs_val).detach().cpu()  # transform prediction to 4*4 transformation type
        #         current_tform = outputs_val.reshape(self.data_pairs.shape[0], 4, 4)[PAIR_INDEX, :, :].cpu()
        #         preds_val, prev_tform = self.accumulate_prediction(prev_tform.cpu(), current_tform.cpu())
        #         preds_val_gt_based, prev_tform_gt_based = self.accumulate_prediction(torch.squeeze(label_matrix_gt), current_tform)
        #         # for all pixel
        #         current_tform_all_pixel = outputs_val.reshape(self.data_pairs.shape[0], 4, 4)[PAIR_INDEX, :, :].cpu()
        #         preds_val_all_pixel, prev_tform_all_pixel = self.accumulate_prediction_all_pixel(prev_tform_all_pixel.cpu(), current_tform_all_pixel.cpu())

        #         y_predicted = preds_val
        #         y_predicted_all_pixel = preds_val_all_pixel
        #         y_predicted_gt_based = preds_val_gt_based
        #         y_actual = torch.squeeze(label)
        #         y_actual_all_pixel = torch.squeeze(label_all_pixels)
        #         # if img_no % self.opt_test.MAX_INTERVAL == 0 and img_no <= ((num_frames - self.opt_test.MAX_INTERVAL) / self.opt_test.MAX_INTERVAL) * self.opt_test.MAX_INTERVAL:
        #         rms_each_img = self.metrics(y_actual, y_predicted)
        #         # rms_all_pixels_in_a_frame = self.metrics(y_actual_all_pixel, y_predicted_all_pixel)

        #         y_actual_overlap.append(y_actual.numpy().tolist())
        #         y_predicted_overlap.append(y_predicted.numpy().tolist())

        #         rms_img.append(rms_each_img.detach().cpu().numpy())
        #         rms_each_img_4_plot = self.metrics(y_actual, y_predicted)
        #         rms_img_4_plot.append(rms_each_img_4_plot.detach().cpu().numpy())

        #         rms_each_img_gt_based = self.metrics(y_actual, y_predicted_gt_based)
        #         rms_img_gt_based.append(rms_each_img_gt_based.detach().cpu().numpy())
        #         rms_each_img_4_plot_gt_based = self.metrics(y_actual, y_predicted_gt_based)
        #         rms_img_4_plot_gt_based.append(rms_each_img_4_plot_gt_based.detach().cpu().numpy())
        #         y_actual_all_pixel_in_a_scan.append(y_actual_all_pixel.detach().cpu().numpy())
        #         y_predicted_all_pixel_in_a_scan.append(y_predicted_all_pixel.detach().cpu().numpy())
        #         fx, fy, fz = [torch.squeeze(preds_val)[ii,].cpu().detach().numpy() for ii in range(3)]
        #         pre_x.append(fx)
        #         pre_y.append(fy)
        #         pre_z.append(fz)

        #         # ax.scatter(fx, fy, fz, color=viridis.colors[idx_color], alpha=0.2, s=2) #

        #         # pix_intensities = (frames[idx_p1, ..., None].float() / 255).cpu().expand(-1, -1, 3).numpy()
        #         # ax.scatter(px, py, pz, c='r', alpha=0.2, s=2)  #
        #         # update for the next prediction
        #         idx_f0 += interval_pred
        #         idx_p1 += interval_pred
        #         if (idx_f0 + self.opt_test.NUM_SAMPLES) > frames.shape[0]:
        #             break

        #     ax.scatter(np.array(pre_x), np.array(pre_y), np.array(pre_z), color=viridis.colors[idx_color], alpha=0.2, s=2) #
        #     ax.scatter(np.array(label_x), np.array(label_y), np.array(label_z), c='r', alpha=0.2, s=2)  #

        #     # plot the last image
        #     fx, fy, fz = fx.reshape(2, 2), fy.reshape(2, 2), fz.reshape(2, 2)
        #     ax.plot_surface(fx, fy, fz, edgecolor=viridis.colors[idx_color], linewidth=1, alpha=0.2, antialiased=True)#
        #     px, py, pz = px.reshape(2, 2), py.reshape(2, 2), pz.reshape(2, 2)
        #     ax.plot_surface(px, py, pz, edgecolor='r', linewidth=1, alpha=0.2, antialiased=True)
        #     # ax.legend()
        #     # plt.show()
        #     # if tran_val == 'train':
        #     #     saved_folder = os.path.join(self.opt.SAVE_PATH, 'train_results')
        #     # else:
        #     #     saved_folder = os.path.join(self.opt.SAVE_PATH, 'val_results')

        #     # plt.savefig(saved_folder+'/'+self.model_name+'_'+self.dset_val.name_scan[self.dset_val.indices_in_use[scan_index][0],self.dset_val.indices_in_use[scan_index][1]].decode("utf-8")+'_interval_'+PAIR_INDEX.__str__()+ '.png')
        #     # plt.close()

        #     # prediction and groundtruth coordinates
        #     # with open(saved_folder + '/' + 'y_actual_overlap_'+saved_name+'.json', 'w', encoding='utf-8') as fp:
        #     #     json.dump(y_actual_overlap, fp, ensure_ascii=False, indent=4)
        #     # with open(saved_folder + '/' + 'y_predicted_overlap_'+saved_name+'.json', 'w', encoding='utf-8') as fp:
        #     #     json.dump(y_predicted_overlap, fp, ensure_ascii=False, indent=4)

        #     y_predicted_all_pixel_in_a_scan = np.array(y_predicted_all_pixel_in_a_scan)
        #     y_actual_all_pixel_in_a_scan = np.array(y_actual_all_pixel_in_a_scan)
        # return rms_img, rms_img_4_plot, rms_img_gt_based, rms_img_4_plot_gt_based,y_actual_all_pixel_in_a_scan,y_predicted_all_pixel_in_a_scan,y_actual_overlap,y_predicted_overlap





