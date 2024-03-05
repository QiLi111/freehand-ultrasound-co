
import numpy as np
import SimpleITK as sitk
import cv2
import openpyxl
# import torch
# from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

def read_frame_transform(filename, scan_crop_indices, resample_factor=1, delay_tform=0):

    get_transform = lambda ii, image : list(map(float,image.GetMetaData('Seq_Frame{:04d}_ProbeToTrackerTransform'.format(ii)).split(' ')))
    get_transform_status = lambda ii, image : image.GetMetaData('Seq_Frame{:04d}_ProbeToTrackerTransformStatus'.format(ii))=="OK"

    image = sitk.ReadImage(filename)  # imageIO="MetaImageIO"    
    frames = sitk.GetArrayFromImage(image)  # [no_frames,h,w] - nb. size = image.GetSize()  # [w,h,no_frames]

    if (scan_crop_indices[1]+delay_tform) > frames.shape[0]:
        scan_crop_indices[1] = frames.shape[0]
        print("WARNING: scan_crop_indices has been reduced due to delayed transform.")

    frames = frames[scan_crop_indices[0]:scan_crop_indices[1],...] 
    tforms = [get_transform(ii,image) for ii in range(scan_crop_indices[0]+delay_tform,scan_crop_indices[1]+delay_tform)]
    tforms = np.stack([np.array(t,dtype=np.float32).reshape(4,4) for t in tforms],axis=0)

    tform_status = [get_transform_status(ii,image) for ii in range(scan_crop_indices[0]+delay_tform,scan_crop_indices[1]+delay_tform)]
    if not all(tform_status):
        frames = frames[tform_status,:,:]
        tforms = tforms[tform_status,:,:]
    
    if resample_factor != 1:
        frames = np.stack(
            [frame_resize(frames[ii,...], resample_factor) for ii in range(frames.shape[0])], 
            axis=0
            )  # resample on 2D frames
    
    return frames, tforms, tform_status


def frame_resize(image, resample_factor):
    # frame_resize = lambda im : cv2.resize(im, None, fx=1/RESAMPLE_FACTOR, fy=1/RESAMPLE_FACTOR, interpolation = cv2.INTER_LINEAR)
    return cv2.resize(
        image, 
        dsize=None, 
        fx=1/resample_factor, 
        fy=1/resample_factor, 
        interpolation = cv2.INTER_LINEAR
        )


def read_scan_crop_indices_file(filename, num_scans):
    fid_xls = openpyxl.load_workbook(filename).active
    return list(fid_xls.iter_rows(values_only=True))[1:num_scans+1]


# def extract_frame_features(frames,pretrained_model,device):
#     # encode frames
#     frames = torch.unsqueeze(torch.unsqueeze(frames, 0),0).to(device)
#
#     # frame_frets = torch.empty(frames.shape[0], frames.shape[1], 1000)
#     # for i in range(frames.shape[0]):
#     frame_frets = pretrained_model(frames)
#
#     return frame_frets
#
#
# def Pretrained_model(in_frames):
#     model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT )
#     model.features[0][0] = torch.nn.Conv2d(
#             in_channels  = in_frames,
#             out_channels = model.features[0][0].out_channels,
#             kernel_size  = model.features[0][0].kernel_size,
#             stride       = model.features[0][0].stride,
#             padding      = model.features[0][0].padding,
#             bias         = model.features[0][0].bias
#         )
#


    # pretrained_model = pretrainedmodels.resnet50(weights=models.ResNet50_Weights.DEFAULT) 
    # pretrained_model.conv1 = torch.nn.Conv2d(
    #     in_channels=in_frames,
    #     out_channels=pretrained_model.conv1.out_channels,
    #     kernel_size=pretrained_model.conv1.kernel_size,
    #     stride=pretrained_model.conv1.stride,
    #     padding=pretrained_model.conv1.padding,
    #     bias=pretrained_model.conv1.bias
    # )
    # pretrained_model.fc = torch.nn.Linear(
    #     in_features=pretrained_model.fc.in_features,
    #     out_features=pred_dim
    # )

    return model

class ImageTransform:
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std

    def __call__(self,images):
        images = (torch.from_numpy(images).to(torch.float32) - self.mean) / self.std
        return images + torch.normal(mean=0, std=torch.ones_like(images)*0.01)
