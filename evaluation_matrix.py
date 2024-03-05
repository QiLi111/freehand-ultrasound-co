import os
from torch.autograd import Variable

from matplotlib import pyplot as plt
import torch
import numpy as np
from torchvision.models import efficientnet_b1
from torch.utils.tensorboard import SummaryWriter
import pickle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from loss import PointDistance_2

from loader import SSFrameDataset
from network import build_model
from data.calib import read_calib_matrices
from transform import LabelTransform, TransformAccumulation, ImageTransform, PredictionTransform
from utils import pair_samples, reference_image_points, type_dim
from options.train_options import TrainOptions
#from options.test_options import TestOptions

class Evaluation():  # plot scan

    def __init__(self, opt, opt_test,device, dset_val, dset_train, model_name):
        self.opt = opt
        self.opt_test = opt_test
        self.device = device
        self.dset_val = dset_val
        self.dset_train = dset_train
        self.FILENAME_WEIGHTS = model_name
        self.model_name = model_name

        self.data_pairs = pair_samples(self.opt_test.NUM_SAMPLES, self.opt_test.NUM_PRED, self.opt.single_interval)

        self.tform_calib_scale, self.tform_calib_R_T, self.tform_calib = read_calib_matrices(
            filename_calib=self.opt.FILENAME_CALIB,
            resample_factor=self.opt.RESAMPLE_FACTOR,
            device=self.device
        )
        # using prediction: transform frame_points from current image to starting (reference) image 0
        self.pixel_points = reference_image_points(self.dset_val.frame_size, 2).to(self.device)
        self.accumulate_prediction = TransformAccumulation(
            image_points=self.pixel_points,
            tform_image_to_tool=self.tform_calib,
            tform_image_mm_to_tool=self.tform_calib_R_T,
            train_val=False
        )
        # using GT: transform pixel_points from current image to starting (reference) image 0

        self.transform_label = LabelTransform(
            label_type=self.opt_test.LABEL_TYPE,  # for plotting
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
        self.pred_dim = type_dim(self.opt_test.PRED_TYPE, self.pixel_points.shape[1], self.data_pairs.shape[0])
        self.label_dim = type_dim(self.opt_test.LABEL_TYPE, self.pixel_points.shape[1], self.data_pairs.shape[0])

        self.transform_image = ImageTransform(mean=32, std=32)
        self.metrics = PointDistance_2()
        ## load the model
        self.model = build_model(
            self.opt,
            in_frames=self.opt.NUM_SAMPLES,
            pred_dim=self.pred_dim,
            label_dim=self.label_dim,
            image_points=self.pixel_points,
            tform_calib=self.tform_calib,
            tform_calib_R_T=self.tform_calib_R_T
        ).to(device)
        self.model.load_state_dict(torch.load(os.path.join(self.opt.SAVE_PATH, self.opt_test.MODEL_FN,model_name), map_location=torch.device(self.device)))
        self.model.train(False)
        # if self.opt.model_name == 'LSTM':
        #     self.encoder_h_1 = (Variable(torch.zeros(opt.MINIBATCH_SIZE, 256, 8, 8)), Variable(torch.zeros(opt.MINIBATCH_SIZE, 256, 8, 8)))
        #     self.encoder_h_2 = (Variable(torch.zeros(opt.MINIBATCH_SIZE, 512, 4, 4)), Variable(torch.zeros(opt.MINIBATCH_SIZE, 512, 4, 4)))
        #     self.encoder_h_3 = (Variable(torch.zeros(opt.MINIBATCH_SIZE, 512, 2, 2)), Variable(torch.zeros(opt.MINIBATCH_SIZE, 512, 2, 2)))


    def ave_dist(self, tran_val, scan_index, PAIR_INDEX ):
        # compute the average distance of all pixels between ground truth and network prediction
        # note: for different intervals, the number of frames used are the same
        if tran_val == 'train':
            frames, tforms, tforms_inv = self.dset_train[scan_index]
            num_frames = self.dset_train.num_frames[self.dset_train.indices_in_use[scan_index]]
        else:
            frames, tforms, tforms_inv = self.dset_val[scan_index]
            num_frames = self.dset_val.num_frames[self.dset_val.indices_in_use[scan_index]]

        frames, tforms, tforms_inv = (torch.tensor(t).to(self.device) for t in [frames, tforms, tforms_inv])
        idx_f0 = self.opt_test.START_FRAME_INDEX  # this is the reference starting frame for network prediction
        idx_p0 = idx_f0 + torch.squeeze(self.data_pairs[PAIR_INDEX])[0]  # this is the reference frame for transformaing others to
        idx_p1 = idx_f0 + torch.squeeze(self.data_pairs[PAIR_INDEX])[1]
        interval_pred = torch.squeeze(self.data_pairs[PAIR_INDEX])[1] - torch.squeeze(self.data_pairs[PAIR_INDEX])[0]


        prev_tform = torch.eye(4)
        rms_img = [] # store the distance of each image
        rms_img_4_plot = []
        img_no = 0 # index of image in this scan
        while 1:
            img_no = img_no + interval_pred
            # prediction -> points in image coords
            frames_val = frames[idx_f0:idx_f0 + self.opt_test.NUM_SAMPLES, ...]
            frames_val = self.transform_image(frames_val)
            if self.opt.model_name != 'LSTM':
                outputs_val = self.model(frames_val.unsqueeze(0))
            else:
                ## init lstm state
                encoder_h_1 = (Variable(torch.zeros(frames_val.unsqueeze(0).shape[0], 256, 30, 40)),
                               Variable(torch.zeros(frames_val.unsqueeze(0).shape[0], 256, 30, 40)))
                encoder_h_2 = (Variable(torch.zeros(frames_val.unsqueeze(0).shape[0], 512, 15, 20)),
                               Variable(torch.zeros(frames_val.unsqueeze(0).shape[0], 512, 15, 20)))
                encoder_h_3 = (Variable(torch.zeros(frames_val.unsqueeze(0).shape[0], 512, 8, 10)),
                               Variable(torch.zeros(frames_val.unsqueeze(0).shape[0], 512, 8, 10)))

                outputs_val, encoder_h_1, encoder_h_2, encoder_h_3 = self.model(
                    frames_val.unsqueeze(0), encoder_h_1, encoder_h_2, encoder_h_3)

            outputs_val = self.transform_prediction(outputs_val)  # transform prediction to 4*4 transformation type
            current_tform = outputs_val.reshape(self.data_pairs.shape[0], 4, 4)[PAIR_INDEX, :, :].cpu()
            preds_val, prev_tform = self.accumulate_prediction(prev_tform, current_tform)
            y_predicted = preds_val
            # ax.scatter(fx, fy, fz, c='b', alpha=0.2, s=2)
            # label -> points in image coords, wrt. the "unchanged" reference starting frame, idx_p0
            tforms_val, tforms_inv_val = (t[[idx_p0, idx_p1], ...] for t in [tforms, tforms_inv])
            label = self.transform_label(tforms_val.unsqueeze(0), tforms_inv_val.unsqueeze(0))
            y_actual = torch.squeeze(label)
            if img_no % self.opt_test.MAX_INTERVAL == 0 and img_no <= ((num_frames-self.opt_test.MAX_INTERVAL)/self.opt_test.MAX_INTERVAL)*self.opt_test.MAX_INTERVAL:
                rms_each_img = self.metrics(y_actual, y_predicted)
                rms_img.append(rms_each_img.detach().cpu().numpy())
            rms_each_img_4_plot = self.metrics(y_actual, y_predicted)
            rms_img_4_plot.append(rms_each_img_4_plot.detach().cpu().numpy())

            # pix_intensities = (frames[idx_p1, ..., None].float() / 255).cpu().expand(-1, -1, 3).numpy()
            # ax.scatter(px, py, pz, c='r', alpha=0.2, s=2)  #
            # update for the next prediction
            idx_f0 += interval_pred
            idx_p1 += interval_pred
            if (idx_f0 + self.opt_test.NUM_SAMPLES) > frames.shape[0]:
                break

            # plot accumulated error for each scan
            # if tran_val == 'train':
            #     saved_folder = os.path.join(self.opt.SAVE_PATH, 'train_results')
            # else:
            #     saved_folder = os.path.join(self.opt.SAVE_PATH, 'val_results')
            #
            # plt.plot(rms_img)
            # plt.show()
            # plt.savefig(saved_folder + '/' + self.model_name.split('/')[1] + '_' + self.dset_val.name_scan[
            #     self.dset_val.indices_in_use[scan_index][0], self.dset_val.indices_in_use[scan_index][1]].decode(
            #     "utf-8") + '_interval_' + PAIR_INDEX.__str__() + '.png')
            # plt.close()
        return rms_img,rms_img_4_plot






