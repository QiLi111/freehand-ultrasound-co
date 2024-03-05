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

from loader_MICCAI import SSFrameDataset
from network import build_model
from data.calib import read_calib_matrices
from transform import LabelTransform, TransformAccumulation, ImageTransform, PredictionTransform
from utils import pair_samples, reference_image_points, type_dim, sample_dists4plot,cls_f1
from options.train_options import TrainOptions
#from options.test_options import TestOptions
from loss import PointDistance_2

class Visualizer():  # plot scan

    def __init__(self, opt, opt_test,device, dset_val, dset_train, dset_test,model_name,data_pairs):
        self.opt = opt
        self.opt_test = opt_test
        self.device = device
        self.dset_test = dset_test
        self.dset_val = dset_val
        self.dset_train = dset_train
        self.FILENAME_WEIGHTS = model_name
        self.model_name = model_name

        self.data_pairs = data_pairs
        self.tform_calib_scale, self.tform_calib_R_T, self.tform_calib = read_calib_matrices(
            filename_calib=self.opt.FILENAME_CALIB,
            resample_factor=self.opt.RESAMPLE_FACTOR,
            device='cpu'#self.device''
        )
        # using prediction: transform frame_points from current image to starting (reference) image 0
        self.frame_points = reference_image_points(self.dset_val.frame_size, 2)#.to(self.device)
        self.accumulate_prediction = TransformAccumulation(
            image_points=self.frame_points,
            tform_image_to_tool=self.tform_calib,
            tform_image_mm_to_tool=self.tform_calib_R_T,
            train_val=False
        )

        # using GT: transform pixel_points from current image to starting (reference) image 0
        self.pixel_points = reference_image_points(self.dset_val.frame_size, self.dset_val.frame_size)#.to(self.device)
        self.all_pixel_points = reference_image_points(self.dset_val.frame_size, self.dset_val.frame_size)#.to(self.device)

        if self.opt_test.plot_line:
            self.pixel_points = self.frame_points

        self.transform_label = LabelTransform(
            label_type=self.opt_test.LABEL_TYPE,  # for plotting
            pairs=torch.tensor([0, 1])[None,],
            image_points=self.pixel_points,
            in_image_coords=True,  # for plotting
            tform_image_to_tool=self.tform_calib,
            tform_image_mm_to_tool=self.tform_calib_R_T
        )
        self.transform_label_all_pixel = LabelTransform(
            label_type=self.opt_test.LABEL_TYPE,  # for plotting
            pairs=torch.tensor([0, 1])[None,],
            image_points=self.all_pixel_points,
            in_image_coords=True,  # for plotting
            tform_image_to_tool=self.tform_calib,
            tform_image_mm_to_tool=self.tform_calib_R_T
        )

        self.transform_label_2_transformation = LabelTransform(
            label_type="transform",  # for plotting
            pairs=torch.tensor([0, 1])[None,],
            image_points=self.pixel_points,
            in_image_coords=True,  # for plotting
            tform_image_to_tool=self.tform_calib,
            tform_image_mm_to_tool=self.tform_calib_R_T
        )
        self.transform_prediction = PredictionTransform(
            pred_type=self.opt_test.PRED_TYPE,
            label_type="transform",
            num_pairs=self.data_pairs.shape[0],
            image_points=self.pixel_points,
            tform_image_to_tool=self.tform_calib,
            tform_image_mm_to_tool=self.tform_calib_R_T
        )
        self.accumulate_prediction_all_pixel = TransformAccumulation(
            image_points=self.all_pixel_points,
            tform_image_to_tool=self.tform_calib,
            tform_image_mm_to_tool=self.tform_calib_R_T,
            train_val=False
        )

        self.pred_dim = type_dim(self.opt_test.PRED_TYPE, self.frame_points.shape[1], self.data_pairs.shape[0])
        self.label_dim = type_dim(self.opt_test.LABEL_TYPE, self.frame_points.shape[1], self.data_pairs.shape[0])

        self.transform_image = ImageTransform(mean=30.873100930319428, std=31.349069347795712)
        self.metrics = PointDistance_2()

        ## load the model





        self.model,self.linear_list_anatomy, self.linear_list_protocol,self.feature_map = build_model(
            opt,
            in_frames = self.opt_test.NUM_SAMPLES,
            pred_dim = self.pred_dim,
            label_dim = self.label_dim,
            image_points = self.pixel_points,
            tform_calib = self.tform_calib,
            tform_calib_R_T = self.tform_calib_R_T,
            input_size=(120,160)
            )


        self.model = self.model.to(self.device)
        if self.linear_list_anatomy:
            for idx_linear in range(len(self.linear_list_anatomy)):
                self.linear_list_anatomy[idx_linear] = self.linear_list_anatomy[idx_linear].to(self.device)
                self.linear_list_protocol[idx_linear] = self.linear_list_protocol[idx_linear].to(self.device)

        
        # self.model= nn.DataParallel(self.model)
        # state_dict1 =self.model.state_dict()
        # print(state_dict1)

        
            
        self.model.load_state_dict(torch.load(os.path.join(self.opt.SAVE_PATH, self.opt_test.MODEL_FN,model_name[0]), map_location=torch.device(self.device)))
        if self.linear_list_anatomy:
            for idx_linear in range(len(self.linear_list_anatomy)):
                self.linear_list_anatomy[idx_linear].load_state_dict(torch.load(os.path.join(self.opt.SAVE_PATH, self.opt_test.MODEL_FN,model_name[1]), map_location=torch.device(self.device))[idx_linear])
                self.linear_list_protocol[idx_linear].load_state_dict(torch.load(os.path.join(self.opt.SAVE_PATH, self.opt_test.MODEL_FN,model_name[2]), map_location=torch.device(self.device))[idx_linear])


        self.model.train(False)
        for idx_linear in range(len(self.linear_list_anatomy)):
            self.linear_list_anatomy[idx_linear] = self.linear_list_anatomy[idx_linear].train(False)
            self.linear_list_protocol[idx_linear] = self.linear_list_protocol[idx_linear].train(False)

        # self.feature_map = torch.load(os.path.join(opt.SAVE_PATH, opt_test.MODEL_FN,model_name[4]), map_location=torch.device(device)) 
        



        ###register forward hook to fetch feature maps of each block
        self.registered_feature_map = {}
        def get_feature_map(name):
          def hook(model, input, output):
            self.registered_feature_map[name] = output
          return hook

        hook_handles = {}
        for index, module in enumerate(self.model.features):
            hook_handles[index] = module.register_forward_hook(get_feature_map(f'module{index}'))




        
        # print('hello')
        
        # print(self.model)
        # self.model.load_state_dict(new_state_dict)

        # state_dict =self.model_ori.state_dict()
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()

        # for k, v in state_dict.items():
        #     if 'module' not in k:
        #         k = 'module.'+k
        #     # else:
        #     #     k = k.replace('features.module.', 'module.features.')
        #     new_state_dict[k]=v
        
        # self.model.state_dict = new_state_dict


        # print(self.model.state_dict.items())
        
        # if self.opt.multi_gpu:
    	#     self.model= nn.DataParallel(self.model)
        
        # self.model= nn.DataParallel(self.model)
        # print(self.model)
        # print(self.model.state_dict.items())
        
       

    def plot_scan_and_cal_ave_dist(self, opt_test,tran_val,scan_index, PAIR_INDEX,viridis,idx_color,device,START_FRAME_INDEX,all_parameter_cls):
        if tran_val == 'train':
            frames, tforms, tforms_inv = self.dset_train[scan_index]
            # num_frames = self.dset_train.num_frames[self.dset_train.indices_in_use[scan_index]]
            saved_folder = os.path.join(self.opt.SAVE_PATH, 'testing_train_results')
            # saved_name = self.dset_train.name_scan[self.dset_train.indices_in_use[scan_index][0],self.dset_train.indices_in_use[scan_index][1]].decode("utf-8")

        elif tran_val == 'val':
            frames, tforms, tforms_inv = self.dset_val[scan_index]
            # num_frames = self.dset_val.num_frames[self.dset_val.indices_in_use[scan_index]]
            saved_folder = os.path.join(self.opt.SAVE_PATH, 'testing_val_results')
            # saved_name =self.dset_val.name_scan[self.dset_val.indices_in_use[scan_index][0],self.dset_val.indices_in_use[scan_index][1]].decode("utf-8")

        else:
            frames, tforms, tforms_inv,scan_flag_inv,cls_pro_val, cls_atm_val = self.dset_test[scan_index]
            
            
            
            saved_folder = os.path.join(self.opt.SAVE_PATH, 'testing_test_results')
            
            
            # saved_name = self.dset_test.name_scan[self.dset_test.indices_in_use[scan_index][0], self.dset_test.indices_in_use[scan_index][1]].decode("utf-8")

        # with open(self.opt.SAVE_PATH + '/'  + '_lstm_groundtruth_img.json', 'w', encoding='utf-8') as fp:
        #     json.dump(frames.tolist(), fp, ensure_ascii=False, indent=4)
        # with open(self.opt.SAVE_PATH + '/'  + '_lstm_groundtruth_transf.json', 'w', encoding='utf-8') as fp:
        #     json.dump(tforms.tolist(), fp, ensure_ascii=False, indent=4)

        frames, tforms, tforms_inv = (torch.tensor(t).to(self.device) for t in [frames, tforms, tforms_inv])
        idx_f0 = START_FRAME_INDEX  #   # this is the reference starting frame for network prediction

        idx_p0 = idx_f0 # + torch.squeeze(self.data_pairs[PAIR_INDEX])[0]  # this is the reference frame for transformaing others to
        idx_p1 = idx_f0 + torch.squeeze(self.data_pairs[PAIR_INDEX])[1]
        interval_pred = torch.squeeze(self.data_pairs[PAIR_INDEX])[1] - torch.squeeze(self.data_pairs[PAIR_INDEX])[0]

# 
        # feature_map = {}
        # def get_feature_map(name):
        #     def hook(model, input, output):
        #         feature_map[name] = output
        #     return hook

        # hook_handles = {}
        # for index, module in enumerate(self.model.features):
        #     hook_handles[index] = module.register_forward_hook(get_feature_map(f'module{index}'))

        




        # plot
        if self.opt_test.plot_line == False:
            # plot the reference
            plt.show()
        else:

            # plot the frame 0
            px, py, pz = [torch.mm(self.tform_calib_scale, torch.mm(torch.from_numpy(np.array([[self.opt.RESAMPLE_FACTOR, 0, 0, 0], [0, self.opt.RESAMPLE_FACTOR, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],np.float32)), self.pixel_points))[ii,].reshape(2, 2) for ii in range(3)]
            pix_intensities = (frames[idx_p0, ..., None].float() / 255).cpu().expand(-1, -1, 3).numpy()
            fx, fy, fz = [torch.mm(self.tform_calib_scale.cpu(), torch.mm(torch.from_numpy(np.array([[self.opt.RESAMPLE_FACTOR, 0, 0, 0], [0, self.opt.RESAMPLE_FACTOR, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],np.float32)), self.pixel_points.cpu()))[ii,].reshape(2, 2) for ii in range(3)]

            ax = plt.figure().add_subplot(projection='3d')
            # ax = plt.axes(projection='3d')
            ax.plot_surface(px, py, pz, facecolors=pix_intensities, linewidth=0, antialiased=True)
            ax.plot_surface(fx, fy, fz, edgecolor='r', linewidth=1, alpha=0.2, antialiased=True)

            # plot the first number of frames to make sure different intervals have the same start frames
            tforms_val, tforms_inv_val = (t[[0, idx_p0], ...] for t in [tforms, tforms_inv])
            label = self.transform_label(tforms_val.unsqueeze(0).cpu(), tforms_inv_val.unsqueeze(0).cpu())
            px, py, pz = [label[:, :, ii, :] for ii in range(3)]
            ax.scatter(px, py, pz, c='r', alpha=0.2, s=2)  #

            fx, fy, fz = [torch.mm(self.tform_calib_scale, torch.mm(torch.from_numpy(np.array(
                [[self.opt.RESAMPLE_FACTOR, 0, 0, 0], [0, self.opt.RESAMPLE_FACTOR, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                np.float32)), self.pixel_points))[ii,].reshape(2, 2).cpu() for ii in range(3)]

            # if start frame is not from 0, the prev_tform label can be computed as
            tforms_val, tforms_inv_val = (t[[0 , idx_f0 + torch.squeeze(self.data_pairs[PAIR_INDEX])[0]], ...] for t in [tforms, tforms_inv])
            _label = self.transform_label_2_transformation(tforms_val.unsqueeze(0), tforms_inv_val.unsqueeze(0))
            # first_img = self.transform_label(tforms_val.unsqueeze(0), tforms_inv_val.unsqueeze(0))
            prev_tform = torch.squeeze(_label)
            prev_tform_all_pixel = torch.squeeze(_label)

            rms_img_gt_based = []  # store the distance of each image
            rms_img_4_plot_gt_based = []
            rms_img = []  # store the distance of each image
            rms_img_4_plot,y_actual_all_pixel_in_a_scan,y_predicted_all_pixel_in_a_scan = [],[],[]
            y_actual_overlap,y_predicted_overlap=[],[]
            img_no = 0  # index of image in this scan
            label_x, label_y, label_z, pre_x, pre_y, pre_z = [], [] ,[], [], [], []
            frame_index=[]
            acc_protocol_each_scan,acc_anatomy_each_scan={},{}

            while 1:
                if (idx_f0 + self.opt_test.NUM_SAMPLES) > frames.shape[0]:
                    break
                img_no = img_no + interval_pred

                # prediction -> points in image coords
                frames_val = frames[idx_f0:idx_f0 + self.opt_test.NUM_SAMPLES, ...]

                # label -> points in image coords, wrt. the "unchanged" reference starting frame, idx_p0
                tforms_val, tforms_inv_val = (t[[idx_p0, idx_p1], ...] for t in [tforms, tforms_inv])
                label = self.transform_label(tforms_val.unsqueeze(0).cpu(), tforms_inv_val.unsqueeze(0).cpu())
                # coordinates of all pixels
                label_all_pixels = self.transform_label_all_pixel(tforms_val.unsqueeze(0).cpu(), tforms_inv_val.unsqueeze(0).cpu())
                px, py, pz = [label[:, :, ii, :] for ii in range(3)]
                label_x.append(torch.squeeze(px).cpu().detach().numpy())
                label_y.append(torch.squeeze(py).cpu().detach().numpy())
                label_z.append(torch.squeeze(pz).cpu().detach().numpy())

                # label -> points in image coords, wrt. the "unchanged" reference starting frame, idx_p0
                # this label is used for the ground truth based error computation
                tforms_val_gt, tforms_inv_val_gt = (t[[idx_p0, idx_p1-interval_pred], ...] for t in [tforms, tforms_inv])
                label_matrix_gt = self.transform_label_2_transformation(tforms_val_gt.unsqueeze(0).cpu(), tforms_inv_val_gt.unsqueeze(0).cpu())
                frames_val = self.transform_image(frames_val)
                if self.opt.model_name == 'LSTM_E':
                    outputs_val = torch.squeeze(self.model(frames_val.unsqueeze(0)), dim=1)
                else:
                    outputs_val = self.model(frames_val.unsqueeze(0))


                


                outputs_val = self.transform_prediction(outputs_val).detach().cpu()  # transform prediction to 4*4 transformation type
                current_tform = outputs_val.reshape(self.data_pairs.shape[0], 4, 4)[PAIR_INDEX, :, :].cpu()
                preds_val, prev_tform = self.accumulate_prediction(prev_tform.cpu(), current_tform.cpu())
                preds_val_gt_based, prev_tform_gt_based = self.accumulate_prediction(torch.squeeze(label_matrix_gt), current_tform)
                # for all pixel
                current_tform_all_pixel = outputs_val.reshape(self.data_pairs.shape[0], 4, 4)[PAIR_INDEX, :, :].cpu()
                preds_val_all_pixel, prev_tform_all_pixel = self.accumulate_prediction_all_pixel(prev_tform_all_pixel.cpu(), current_tform_all_pixel.cpu())

                y_predicted = preds_val
                y_predicted_all_pixel = preds_val_all_pixel
                y_predicted_gt_based = preds_val_gt_based
                y_actual = torch.squeeze(label)
                y_actual_all_pixel = torch.squeeze(label_all_pixels)
                # if img_no % self.opt_test.MAX_INTERVAL == 0 and img_no <= ((num_frames - self.opt_test.MAX_INTERVAL) / self.opt_test.MAX_INTERVAL) * self.opt_test.MAX_INTERVAL:
                rms_each_img = self.metrics(y_actual, y_predicted)
                # rms_all_pixels_in_a_frame = self.metrics(y_actual_all_pixel, y_predicted_all_pixel)

                y_actual_overlap.append(y_actual.numpy().tolist())
                y_predicted_overlap.append(y_predicted.numpy().tolist())

                rms_img.append(rms_each_img.detach().cpu().numpy())
                rms_each_img_4_plot = self.metrics(y_actual, y_predicted)
                rms_img_4_plot.append(rms_each_img_4_plot.detach().cpu().numpy())

                rms_each_img_gt_based = self.metrics(y_actual, y_predicted_gt_based)
                rms_img_gt_based.append(rms_each_img_gt_based.detach().cpu().numpy())
                rms_each_img_4_plot_gt_based = self.metrics(y_actual, y_predicted_gt_based)
                rms_img_4_plot_gt_based.append(rms_each_img_4_plot_gt_based.detach().cpu().numpy())
                y_actual_all_pixel_in_a_scan.append(y_actual_all_pixel.detach().cpu().numpy())
                y_predicted_all_pixel_in_a_scan.append(y_predicted_all_pixel.detach().cpu().numpy())
                fx, fy, fz = [torch.squeeze(preds_val)[ii,].cpu().detach().numpy() for ii in range(3)]
                pre_x.append(fx)
                pre_y.append(fy)
                pre_z.append(fz)

                # ax.scatter(fx, fy, fz, color=viridis.colors[idx_color], alpha=0.2, s=2) #

                # pix_intensities = (frames[idx_p1, ..., None].float() / 255).cpu().expand(-1, -1, 3).numpy()
                # ax.scatter(px, py, pz, c='r', alpha=0.2, s=2)  #
                # update for the next prediction
                # frame index of computed accumulated error
                frame_index.append(idx_p1.numpy().tolist())

                idx_f0 += interval_pred
                idx_p1 += interval_pred
                # if (idx_f0 + self.opt_test.NUM_SAMPLES) > frames.shape[0]:
                #     break


                # compute evalation performance for classification task
                # acc_protocol_each_scan,acc_anatomy_each_scan = cls_f1(-1,self.registered_feature_map,self.linear_list_protocol,acc_protocol_each_scan,cls_pro_val,self.linear_list_anatomy,acc_anatomy_each_scan,cls_atm_val,all_parameter_cls)





            ax.scatter(np.array(pre_x), np.array(pre_y), np.array(pre_z), color=viridis.colors[idx_color], alpha=0.2, s=2) #
            ax.scatter(np.array(label_x), np.array(label_y), np.array(label_z), c='r', alpha=0.2, s=2)  #

            # plot the last image
            fx, fy, fz = fx.reshape(2, 2), fy.reshape(2, 2), fz.reshape(2, 2)
            ax.plot_surface(fx, fy, fz, edgecolor=viridis.colors[idx_color], linewidth=1, alpha=0.2, antialiased=True)#
            px, py, pz = px.reshape(2, 2), py.reshape(2, 2), pz.reshape(2, 2)
            ax.plot_surface(px, py, pz, edgecolor='r', linewidth=1, alpha=0.2, antialiased=True)
            ax.legend()


            # plt.show()
            
            plt.savefig(saved_folder+'/'+self.model_name[0]+'_'+str(self.dset_test.indices_in_use[scan_index][0])+'_'+self.dset_test.name_scan[self.dset_test.indices_in_use[scan_index][0],self.dset_test.indices_in_use[scan_index][1]].decode("utf-8")+'_interval_'+PAIR_INDEX.__str__()+ '.png')
            plt.close()

            # prediction and groundtruth coordinates
            # with open(saved_folder + '/' + 'y_actual_overlap_'+saved_name+'.json', 'w', encoding='utf-8') as fp:
            #     json.dump(y_actual_overlap, fp, ensure_ascii=False, indent=4)
            # with open(saved_folder + '/' + 'y_predicted_overlap_'+saved_name+'.json', 'w', encoding='utf-8') as fp:
            #     json.dump(y_predicted_overlap, fp, ensure_ascii=False, indent=4)

            y_predicted_all_pixel_in_a_scan = np.array(y_predicted_all_pixel_in_a_scan)
            y_actual_all_pixel_in_a_scan = np.array(y_actual_all_pixel_in_a_scan)
        # print(rms_img)
        # print(rms_img_4_plot)
        # print(rms_img_gt_based)
        # print(rms_img_4_plot_gt_based)
        # print(y_actual_all_pixel_in_a_scan)
        # print(y_predicted_all_pixel_in_a_scan)
        # print(y_actual_overlap)
        # print(y_predicted_overlap)
        # print(frame_index)

        return rms_img, rms_img_4_plot, rms_img_gt_based, rms_img_4_plot_gt_based,y_actual_all_pixel_in_a_scan,y_predicted_all_pixel_in_a_scan,y_actual_overlap,y_predicted_overlap,frame_index







