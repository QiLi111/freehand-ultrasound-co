# chech the quality of scan
 

import numpy as np
import SimpleITK as sitk
import cv2,os,sys,csv

sys.path.append("/freehand-ultrasound")
from utilits_grid_data import *
from data.calib import read_calib_matrices
from transform import LabelTransform
from utils import pair_samples, reference_image_points

from utils_rec_reg import *
# load scan
PATH_SAVE = os.path.join(os.path.expanduser("~"), "/public_data/forearm_US_large_dataset/data_size59")
folder = '041'
file_name1 = 'TrackedImageSequence_20231128_091434.mha'
file_name2 = 'TrackedImageSequence_20231128_091756.mha'
file_name3 = 'TrackedImageSequence_20231128_091950.mha'
file_name4 = 'TrackedImageSequence_20231128_092248.mha'

scan1 = os.path.join(PATH_SAVE, folder,file_name1)
scan2 = os.path.join(PATH_SAVE, folder,file_name2)
scan3 = os.path.join(PATH_SAVE, folder,file_name3)
scan4 = os.path.join(PATH_SAVE, folder,file_name4)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')



def read_frame_transform_check(filename, resample_factor=1, delay_tform=0):


    get_transform = lambda ii, image : list(map(float,image.GetMetaData('Seq_Frame{:04d}_ProbeToTrackerTransform'.format(ii)).split(' ')))
    get_transform_status = lambda ii, image : image.GetMetaData('Seq_Frame{:04d}_ProbeToTrackerTransformStatus'.format(ii))=="OK"

    image = sitk.ReadImage(filename)  # imageIO="MetaImageIO"    
    frames = sitk.GetArrayFromImage(image)  # [no_frames,h,w] - nb. size = image.GetSize()  # [w,h,no_frames]

    num_frames=frames.shape[0]

    frames = frames[0:num_frames-delay_tform,...] 
    tforms = [get_transform(ii,image) for ii in range(delay_tform,num_frames)]
    tforms = np.stack([np.array(t,dtype=np.float32).reshape(4,4) for t in tforms],axis=0)

    tform_status = [get_transform_status(ii,image) for ii in range(delay_tform,num_frames)]
    
    if not all(tform_status):
        frames = frames[tform_status,:,:]
        tforms = tforms[tform_status,:,:]
    
    if resample_factor != 1:
        frames = np.stack(
            [frame_resize(frames[ii,...], resample_factor) for ii in range(frames.shape[0])], 
            axis=0
            )  # resample on 2D frames
    
    # find the first missing status
    false_indices = []
    for i in range(len(tform_status)):
        if tform_status[i] == False:
            false_indices.append(i)
    
    if len(false_indices)==0:
        first_missing_status_idx = 'All OK'
    else:
        first_missing_status_idx = false_indices[0]


    return frames, tforms, tform_status, first_missing_status_idx, len(false_indices)/len(tform_status)

def frame_resize(image, resample_factor):
    # frame_resize = lambda im : cv2.resize(im, None, fx=1/RESAMPLE_FACTOR, fy=1/RESAMPLE_FACTOR, interpolation = cv2.INTER_LINEAR)
    return cv2.resize(
        image, 
        dsize=None, 
        fx=1/resample_factor, 
        fy=1/resample_factor, 
        interpolation = cv2.INTER_LINEAR
        )



    
                    
    





def generate_volume_3D(frames,tforms,device):
    data_pairs_all = data_pairs_adjacent(frames.shape[0])
    data_pairs_all=torch.tensor(data_pairs_all)
    all_points = reference_image_points((120,160), (120,160)).to(device)
    tform_calib_scale, tform_calib_R_T, tform_calib = read_calib_matrices(
                filename_calib='data/calib_matrix.csv',
                resample_factor=4,
                device=device#self.device''
            )

    transform_label_all = LabelTransform(
                label_type='transform',
                pairs=data_pairs_all,  #
                image_points=all_points ,
                in_image_coords=True,
                tform_image_to_tool=tform_calib,
                tform_image_mm_to_tool=tform_calib_R_T
                )
    tforms_inv = torch.linalg.inv(tforms)
    tforms_each_frame2frame0_gt_all = transform_label_all(tforms.unsqueeze(0), tforms_inv.unsqueeze(0)).to(device)
    labels = torch.matmul(tforms_each_frame2frame0_gt_all,torch.matmul(tform_calib,all_points))[:,:,0:3,...]


    gt_volume, gt_volume_position = interpolation_3D_pytorch_batched(scatter_pts = labels,
                                                                frames = frames,
                                                                time_log=None,
                                                                saved_folder_test = None,
                                                                scan_name=None,
                                                                device = device,
                                                                option = 'bilinear',
                                                                volume_size = 'fixed_interval',
                                                                volume_position = None
                                                                )
    return gt_volume


frames_rotate1, tforms_rotate1, _,first_missing_idx_rotate1,missing_rate1 = read_frame_transform_check(
            filename = scan1,
            resample_factor = 4,
            delay_tform = 4
        )

frames_rotate2, tforms_rotate2, _,first_missing_idx_rotate2,missing_rate2 = read_frame_transform_check(
            filename = scan2,
            resample_factor = 4,
            delay_tform = 4
        )

frames_rotate3, tforms_rotate3, _,first_missing_idx_rotate3,missing_rate3 = read_frame_transform_check(
            filename = scan3,
            resample_factor = 4,
            delay_tform = 4
        )

frames_rotate4, tforms_rotate4, _,first_missing_idx_rotate4,missing_rate4 = read_frame_transform_check(
            filename = scan4,
            resample_factor = 4,
            delay_tform = 4
        )

frames_rotate1 = torch.tensor(frames_rotate1[:-100,...]).to(device)/255
frames_rotate2 = torch.tensor(frames_rotate2[:-100,...]).to(device)/255
frames_rotate3 = torch.tensor(frames_rotate3[:-100,...]).to(device)/255
frames_rotate4 = torch.tensor(frames_rotate4[:-100,...]).to(device)/255

tforms_rotate1 = torch.tensor(tforms_rotate1[:-100,...])
tforms_rotate2 = torch.tensor(tforms_rotate2[:-100,...])
tforms_rotate3 = torch.tensor(tforms_rotate3[:-100,...])
tforms_rotate4 = torch.tensor(tforms_rotate4[:-100,...])


volume1 = generate_volume_3D(frames_rotate1,tforms_rotate1,device)
volume2 = generate_volume_3D(frames_rotate2,tforms_rotate2,device)
volume3 = generate_volume_3D(frames_rotate3,tforms_rotate3,device)
volume4 = generate_volume_3D(frames_rotate4,tforms_rotate4,device)


save2mha(volume1[0,...].cpu().numpy(),sx = 1,sy=1,sz=1,
             save_folder='volume1.mha'
             )
save2mha(volume2[0,...].cpu().numpy(),sx = 1,sy=1,sz=1,
             save_folder='volume2.mha'
             )
save2mha(volume3[0,...].cpu().numpy(),sx = 1,sy=1,sz=1,
             save_folder='volume3.mha'
             )
save2mha(volume4[0,...].cpu().numpy(),sx = 1,sy=1,sz=1,
             save_folder='volume4.mha'
             )

print('done')