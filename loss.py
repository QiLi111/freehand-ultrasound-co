
import torch
from transform import TransformAccumulation
from transform import LabelTransform
import numpy as np
from itertools import combinations
# TBD


class PointDistance:
    def __init__(self,paired=True):
        self.paired = paired
    
    def __call__(self,preds,labels):
        if self.paired:
            return ((preds-labels)**2).sum(dim=2).sqrt().mean(dim=(0,2))
        else:
            return ((preds-labels)**2).sum(dim=2).sqrt().mean()


class MTL_loss:
    def __init__(self, paired=True):
        self.paired = paired

    def __call__(self, preds, labels):
        if self.paired:
            return ((preds - labels) ** 2).mean(dim=(0,2, 3))
        else:
            return ((preds - labels) ** 2).mean()


class PointDistance_2:
    def __init__(self, paired=True):
        self.paired = paired

    def __call__(self, preds, labels):
        return ((preds - labels) ** 2).sum(dim=0).sqrt().mean()

class PointDistance_1:
    def __init__(self, paired=True):
        self.paired = paired

    def __call__(self, preds, labels):
        return ((preds - labels) ** 2).sum(dim=1).sqrt().mean()

class PointDistance_3:
    def __init__(self, paired=True):
        self.paired = paired

    def __call__(self, preds, labels):
        if len(preds):
            return ((preds - labels) ** 2).sum(dim=1).sqrt().mean(axis=1)
        else:
            return []
class WeightedLoss:
    def __init__(self, data_pairs,weight_option):
        self.data_pairs = data_pairs
        self.weight_option = weight_option

    def __call__(self, preds, labels):
        if self.weight_option == 'none':
            criterion = torch.nn.MSELoss()
            loss = (preds, labels)
        elif self.weight_option == 'trained_weight':
            loss = 1
        elif self.weight_option == 'assigned_weight':
            _weight = torch.ones_like(labels)
            weight1 = [_weight[:,i,:,:]/((self.data_pairs[i,1]-self.data_pairs[i,0])) for i in range(self.data_pairs.shape[0])]
            weight = torch.stack(weight1, dim=1)
            loss = torch.mean(torch.mul((preds - labels)** 2,weight))

        return loss


def consistent_accumulated_loss(loss,opt,preds_dist, label_dist):

    criterion = torch.nn.MSELoss()
    # if opt.NUM_SAMPLES == 2:
    #     raise ("no consistent or accumulated loss can be calculated as the input number of images is 2.")

    pair_idx = np.array(list(combinations(list(range(len(preds_dist))),2)))

    if opt.CONSIATENT_LOSS == True and opt.ACCUMULAT_LOSS == True:
        loss = loss + torch.mean(torch.stack([criterion(preds_dist[str(pair_idx[i][0])],preds_dist[str(pair_idx[i][1])]) for i in range(pair_idx.shape[0])]))
        loss = loss + torch.mean(torch.stack([criterion(preds_dist[str(pair_idx[i][0])],label_dist[str(pair_idx[i][1])]) for i in range(pair_idx.shape[0])])) + torch.mean(torch.stack([criterion(preds_dist[str(pair_idx[i][1])],label_dist[str(pair_idx[i][0])]) for i in range(pair_idx.shape[0])]))
        loss = loss + torch.mean(torch.stack([criterion(preds_dist[str(i)],label_dist[str(i)]) for i in range(len(preds_dist))]))

    if opt.CONSIATENT_LOSS == True and opt.ACCUMULAT_LOSS == False:
        loss = loss + torch.mean(torch.stack([criterion(preds_dist[str(pair_idx[i][0])],preds_dist[str(pair_idx[i][1])]) for i in range(pair_idx.shape[0])]))

    if opt.CONSIATENT_LOSS == False and opt.ACCUMULAT_LOSS == True:
        loss = loss + torch.mean(torch.stack([criterion(preds_dist[str(pair_idx[i][0])],label_dist[str(pair_idx[i][1])]) for i in range(pair_idx.shape[0])])) + torch.mean(torch.stack([criterion(preds_dist[str(pair_idx[i][1])],label_dist[str(pair_idx[i][0])]) for i in range(pair_idx.shape[0])]))
        loss = loss + torch.mean(torch.stack([criterion(preds_dist[str(i)],label_dist[str(i)]) for i in range(len(preds_dist))]))
        # loss = loss



    return loss







class PredictionPointDistanceAccumulation:
    # compute accumulated transformation matrix and transformed points coordinates

    def __init__(self,single_interval, interval, single_weight, pred_type,num_pairs,frame_points,tform_calib,tform_calib_R_T):
        self.single_interval = single_interval
        self.interval = interval
        self.weighted = single_weight
        self.pred_type = pred_type
        self.frame_points = frame_points
        self.tform_calib = tform_calib
        self.tform_calib_R_T = tform_calib_R_T
        self.num_pairs = num_pairs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



        self.accumulate_prediction = TransformAccumulation(
            image_points=self.frame_points,
            tform_image_to_tool=self.tform_calib,
            tform_image_mm_to_tool = self.tform_calib_R_T,
            train_val=True
        )

        if self.pred_type == "point":
            raise ('Not implemented.')
        elif self.pred_type == "parameter":
            if interval == True:
                self.call_function = self.parameter_to_point_dist
            else:
                self.call_function = self.parameter_to_point_dist_2

        elif self.pred_type == "transform":
            self.call_function = self.transform_to_point_dist

    def __call__(self, outputs, interval):
        preds = outputs.reshape((outputs.shape[0], self.num_pairs, -1))
        return self.call_function(preds, interval)

    def transform_to_point_dist(self, _tforms,interval):
        last_rows = torch.cat([
            torch.zeros_like(_tforms[..., 0])[..., None, None].expand(-1, -1, 1, 3),
            torch.ones_like(_tforms[..., 0])[..., None, None]
        ], axis=3)
        _tforms = torch.cat((
            _tforms.reshape(-1, self.num_pairs, 3, 4),
            last_rows
        ), axis=2)
        preds_all,prev_tform_all = {},{}
        for i in range(len(interval)):
            preds_all[str(i)],prev_tform_all[str(i)] = None, None

        for num_inter in range (len(interval)):
            tforms_intval = _tforms[:,interval[str(num_inter)],...]
            prev_tform = torch.eye(4)[None,...].expand(tforms_intval.shape[0], -1, -1)
            for i in range (tforms_intval.shape[1]):
                preds, prev_tform = self.accumulate_prediction(prev_tform.to(self.device), tforms_intval[:,i,...])
            preds_all[str(num_inter)], prev_tform_all[str(num_inter)] = preds, prev_tform
        return preds_all, prev_tform_all

    def transform_to_point_dist_2(self, _tforms,interval):
        last_rows = torch.cat([
            torch.zeros_like(_tforms[..., 0])[..., None, None].expand(-1, -1, 1, 3),
            torch.ones_like(_tforms[..., 0])[..., None, None]
        ], axis=3)
        _tforms = torch.cat((
            _tforms.reshape(-1, self.num_pairs, 3, 4),
            last_rows
        ), axis=2)

        preds_all, prev_tform_all = [], []
        # prev_tform = torch.eye(4)[None, ...].expand(_tforms.shape[0], -1, -1)
        for i in range(_tforms.shape[1]):
            prev_tform = _tforms[:, i, ...]
            for j in range(i + 1, _tforms.shape[1]):
                preds, prev_tform = self.accumulate_prediction(prev_tform.to(self.device), _tforms[:, j, ...])
                if self.weighted == 'assigned_weight':
                    preds = preds/((j-i+1)*self.single_interval)
                preds_all.append(preds)
        return preds_all

    def parameter_to_point_dist(self, params, interval):
        _tforms = self.param_to_transform(params)

        preds_all, prev_tform_all = {}, {}
        for i in range(len(interval)):
            preds_all[str(i)], prev_tform_all[str(i)] = None, None

        for num_inter in range(len(interval)):
            tforms_intval = _tforms[:, interval[str(num_inter)], ...]
            prev_tform = torch.eye(4)[None, ...].expand(tforms_intval.shape[0], -1, -1)
            for i in range(tforms_intval.shape[1]):
                preds, prev_tform = self.accumulate_prediction(prev_tform.to(self.device), tforms_intval[:, i, ...])
            preds_all[str(num_inter)], prev_tform_all[str(num_inter)] = preds, prev_tform
        return preds_all, prev_tform_all

    def parameter_to_point_dist_2(self, params,interval):
        _tforms = self.param_to_transform(params)

        preds_all, prev_tform_all = [], []
        # prev_tform = torch.eye(4)[None, ...].expand(_tforms.shape[0], -1, -1)
        for i in range(_tforms.shape[1]):
            prev_tform = _tforms[:, i, ...]
            for j in range(i+1,_tforms.shape[1]):
                preds, prev_tform = self.accumulate_prediction(prev_tform.to(self.device), _tforms[:, j, ...])
                if self.weighted == 'assigned_weight':
                    preds = preds/((j-i+1)*self.single_interval)
                preds_all.append(preds)
        return preds_all


    @staticmethod
    def param_to_transform(params):
        # params: (batch,ch,6), "ch": num_pairs, "6":rx,ry,rz,tx,ty,tz
        # this function is equal to utils-->tran426_tensor/tran426_np, seq = 'ZYX'
        cos_x = torch.cos(params[..., 2])
        sin_x = torch.sin(params[..., 2])
        cos_y = torch.cos(params[..., 1])
        sin_y = torch.sin(params[..., 1])
        cos_z = torch.cos(params[..., 0])
        sin_z = torch.sin(params[..., 0])
        return torch.cat((
            torch.stack(
                [cos_y * cos_z, sin_x * sin_y * cos_z - cos_x * sin_z, cos_x * sin_y * cos_z + sin_x * sin_z,
                 params[..., 3]], axis=2)[:, :, None, :],
            torch.stack(
                [cos_y * sin_z, sin_x * sin_y * sin_z + cos_x * cos_z, cos_x * sin_y * sin_z - sin_x * cos_z,
                 params[..., 4]], axis=2)[:, :, None, :],
            torch.stack([-sin_y, sin_x * cos_y, cos_x * cos_y, params[..., 5]], axis=2)[:, :, None, :],
            torch.cat((torch.zeros_like(params[..., 0:3])[..., None, :],
                       torch.ones_like(params[..., 0])[..., None, None]), axis=3)
        ), axis=2)

class LabelPointDistanceAccumulation:
    # compute accumulated transformation matrix and transformed points coordinates
    def __init__(self,single_interval, interval,single_weight,data_pairs,frame_points,tform_calib,tform_calib_R_T):
        self.single_interval = single_interval
        self.interval = interval # if True, use interval list to compute accmulated distance
        self.weighted = single_weight
        self.data_pairs = data_pairs
        self.frame_points = frame_points
        self.tform_calib = tform_calib
        self.tform_calib_R_T = tform_calib_R_T
        self.data_pairs = data_pairs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.accumulate_prediction = TransformAccumulation(
            image_points=self.frame_points,
            tform_image_to_tool=self.tform_calib,
            tform_image_mm_to_tool=self.tform_calib_R_T,
            train_val=True
        )

        self.transform_label = LabelTransform(
            label_type = "transform",
            pairs=self.data_pairs,
            image_points=self.frame_points,
            tform_image_to_tool=self.tform_calib
        )
        if self.interval == True:
            self.call_function = self.transform_to_point_dist
        else:
            self.call_function = self.transform_to_point_dist_2



    def __call__(self, tforms,tforms_inv, interval):
        _tforms = self.transform_label(tforms, tforms_inv)
        return self.call_function(_tforms, interval)

    def transform_to_point_dist(self, _tforms,interval):

        preds_all,prev_tform_all = {},{}
        for i in range(len(interval)):
            preds_all[str(i)],prev_tform_all[str(i)] = None, None

        for num_inter in range (len(interval)):
            tforms_intval = _tforms[:,interval[str(num_inter)],...]
            prev_tform = torch.eye(4)[None,...].expand(tforms_intval.shape[0], -1, -1)
            for i in range (tforms_intval.shape[1]):
                preds, prev_tform = self.accumulate_prediction(prev_tform.to(self.device), tforms_intval[:,i,...])
            preds_all[str(num_inter)], prev_tform_all[str(num_inter)] = preds, prev_tform
        return preds_all, prev_tform_all

    def transform_to_point_dist_2(self, _tforms,interval):
        preds_all, prev_tform_all = [], []
        # prev_tform = torch.eye(4)[None, ...].expand(_tforms.shape[0], -1, -1)
        for i in range(_tforms.shape[1]):
            prev_tform = _tforms[:, i, ...]
            for j in range(i+1,_tforms.shape[1]):
                preds, prev_tform = self.accumulate_prediction(prev_tform.to(self.device), _tforms[:, j, ...])
                if self.weighted == 'assigned_weight':
                    preds = preds/((j-i+1)*self.single_interval)
                preds_all.append(preds)
        return preds_all
