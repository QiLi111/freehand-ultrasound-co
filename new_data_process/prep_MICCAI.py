
# genrate data with classification of anatomy and protocol, from .mha file

import os

import h5py,csv
import matplotlib.pyplot as plt

import numpy as np
# import torch
from utils_4_data import read_frame_transform
from utils_4_data import ImageTransform,PlotScan,read_frame_transform_croped


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



FLAG_SCANFILE = False

DIR_RAW = os.path.join(os.path.expanduser("~"), "/raid/candi/Qi/public_data/forearm_US_large_dataset")
NUM_SCANS = 12
RESAMPLE_FACTOR = 4
PATH_SAVE = os.path.join(os.path.expanduser("~"), "/raid/candi/Qi/public_data/forearm_US_large_dataset")
DELAY_TFORM = 4  # delayed tform from temporal calibration
start_idx = 100
FILENAME_CALIB= "/raid/candi/Qi/public_data/forearm_US_large_dataset/calib_matrix.csv"


fh5_frames = h5py.File(os.path.join(PATH_SAVE,'frames_res{}_cls'.format(RESAMPLE_FACTOR)+".h5"),'a')
if FLAG_SCANFILE:
    fh5_scans = h5py.File(os.path.join(PATH_SAVE,'scans_res{}_cls'.format(RESAMPLE_FACTOR)+".h5"),'a')


folders_subject = [f for f in os.listdir(DIR_RAW) if os.path.isdir(os.path.join(DIR_RAW, f))]
num_frames = np.zeros((len(folders_subject),NUM_SCANS),dtype=np.uint16)
cls_scans_protocol = np.zeros((len(folders_subject),NUM_SCANS),dtype=np.uint16)
cls_scans_anatomy = np.zeros((len(folders_subject),NUM_SCANS),dtype=np.uint16)


temp = list(range(len(folders_subject)))
cls_anatomy = np.array([i * 2 for i in temp])
cls_protocol = np.array([0,1,2])

name_scan = [[None for i in range(NUM_SCANS)] for j in range(len(folders_subject))]


transform_image = ImageTransform(mean=30.873100930319428, std=31.349069347795712)
# pretrained_model = Pretrained_model(1).to(device) # pretrained model for extracting features for each frame
scan_names = ['RH_Ver_L','RH_Ver_C','RH_Ver_S','RH_Par_L','RH_Par_C','RH_Par_S','LH_Ver_L','LH_Ver_C','LH_Ver_S','LH_Par_L','LH_Par_C','LH_Par_S']


for i_sub, folder in enumerate(folders_subject):

    fn_mha = [f for f in os.listdir(os.path.join(DIR_RAW, folder)) if f.endswith(".mha")]
    # ordered in acquisition time
    fn_mha=sorted(fn_mha, key=lambda x: int(''.join(filter(str.isdigit, x))))
    if len(fn_mha) != NUM_SCANS: raise('Should contain 12 mha files in folder "{}"'.format(DIR_RAW))    

    for i_scan, fn in enumerate(fn_mha):

        # get image and transformations-all images in a scan
        frames, tforms, _,first_missing_idx = read_frame_transform(
            filename = os.path.join(os.path.join(DIR_RAW, folder),fn),
            resample_factor = RESAMPLE_FACTOR,
            delay_tform = DELAY_TFORM
        )

        tforms_inv = np.linalg.inv(tforms)  # pre-compute the iverse

        # plot scan
        plot_scan_init = PlotScan(FILENAME_CALIB,RESAMPLE_FACTOR, Image_Shape=(120,160))
        
        # plot the trajectory of each scan and compute the axis range for x, y, and z
        px_all, py_all, pz_all = plot_scan_init.plot_scan(frames, tforms, tforms_inv,PATH_SAVE+'/'+folder+'/'+fn.split(".")[0])
        # plot_scan_init.Plot_Video(frames, tforms, tforms_inv,fn.split(".")[0],px_all, py_all, pz_all)
        # plot_scan_init.Save_Video(fn.split(".")[0])
        # plot_scan_init.plot_img_in_2d(frames,fn.split(".")[0])

        # detect the frame indexes for start (which is always fixed at 100), loop, and end
        index_loop,final_idx = plot_scan_init.detect_loop_points(px_all,py_all,pz_all,PATH_SAVE+'/'+folder+'/'+fn.split(".")[0])


        # get image and transformations- croped images in a scan
        frames_croped, tforms_croped, _ = read_frame_transform_croped(
            filename = os.path.join(os.path.join(DIR_RAW, folder),fn),
            scan_crop_indices = [start_idx,index_loop,final_idx],
            resample_factor = RESAMPLE_FACTOR,
            delay_tform = DELAY_TFORM
        )
        
        tforms_inv_croped = np.linalg.inv(tforms_croped) # pre-compute the iverse

        # plot the trajectory of each scan and compute the axis range for x, y, and z
        px_all_croped, py_all_croped, pz_all_croped = plot_scan_init.plot_scan(frames_croped, tforms_croped, tforms_inv_croped,PATH_SAVE+'/'+folder+'/'+fn.split(".")[0],croped=True)
        # plot_scan_init.Plot_Video(frames_croped, tforms_croped, tforms_inv_croped,fn.split(".")[0],px_all_croped, py_all_croped, pz_all_croped,croped=True)
        # plot_scan_init.Save_Video(fn.split(".")[0],croped=True)
        # plot_scan_init.plot_img_in_2d(frames_croped,fn.split(".")[0],croped=True)
        
        index_loop_croped,final_idx_croped = plot_scan_init.detect_loop_points(px_all_croped,py_all_croped,pz_all_croped,PATH_SAVE+'/'+folder+'/'+fn.split(".")[0],croped=True)



        num_frames[i_sub,i_scan] = frames_croped.shape[0]
        name_scan[i_sub][i_scan] = fn

        # generate classification labels
        if 'L' in name_scan[i_sub][i_scan]:
            protocol_cls = cls_protocol[0]

        elif 'C' in name_scan[i_sub][i_scan]:
            protocol_cls = cls_protocol[1]
        elif 'S' in name_scan[i_sub][i_scan]:
            protocol_cls = cls_protocol[2]

        else:
            raise('wrong protocol')

        idx_name = folders_subject.index(folder)
        if 'RH' in name_scan[i_sub][i_scan]:
            anatomy_cls = cls_anatomy[idx_name]
        elif 'LH' in name_scan[i_sub][i_scan]:
            anatomy_cls = cls_anatomy[idx_name]+1
        else:
            raise('wrong anatomy')

        cls_scans_protocol[i_sub][i_scan] = protocol_cls
        cls_scans_anatomy[i_sub][i_scan] = anatomy_cls




        for i_frame in range(frames_croped.shape[0]):
            fh5_frames.create_dataset('/sub%03d_scan%02d_frame%04d' % (i_sub,i_scan,i_frame), frames_croped.shape[1:3], dtype=frames_croped.dtype, data=frames_croped[i_frame,...])
            fh5_frames.create_dataset('/sub%03d_scan%02d_tform%04d' % (i_sub,i_scan,i_frame), tforms_croped.shape[1:3], dtype=tforms_croped.dtype, data=tforms_croped[i_frame,...])
            fh5_frames.create_dataset('/sub%03d_scan%02d_tform_inv%04d' % (i_sub,i_scan,i_frame), tforms_inv_croped.shape[1:3], dtype=tforms_inv_croped.dtype, data=tforms_inv_croped[i_frame,...])
            fh5_frames.create_dataset('/sub%03d_scan%02d_cls_protocol%04d' % (i_sub,i_scan,i_frame), protocol_cls.shape, dtype=protocol_cls.dtype, data=protocol_cls)
            fh5_frames.create_dataset('/sub%03d_scan%02d_cls_anatomy%04d' % (i_sub,i_scan,i_frame), anatomy_cls.shape, dtype=anatomy_cls.dtype, data=anatomy_cls)


            
        if FLAG_SCANFILE:
            fh5_scans.create_dataset('/sub%03d_frames%02d' % (i_sub,i_scan), frames_croped.shape, dtype=frames_croped.dtype, data=frames_croped)
            fh5_scans.create_dataset('/sub%03d_tforms%02d' % (i_sub,i_scan), tforms_croped.shape, dtype=tforms_croped.dtype, data=tforms_croped)
            fh5_scans.create_dataset('/sub%03d_tforms_inv%02d' % (i_sub,i_scan), tforms_inv_croped.shape, dtype=tforms_inv_croped.dtype, data=tforms_inv_croped)

fh5_frames.create_dataset('num_frames', num_frames.shape, dtype=num_frames.dtype, data=num_frames)
fh5_frames.create_dataset('sub_folders', len(folders_subject), data=folders_subject)
fh5_frames.create_dataset('frame_size', 2, data=frames_croped.shape[1:3])
fh5_frames.create_dataset('name_scan', tuple((len(name_scan),len(name_scan[0]))), data=name_scan)

fh5_frames.create_dataset('cls_scans_protocol', cls_scans_protocol.shape, dtype=cls_scans_protocol.dtype, data=cls_scans_protocol)
fh5_frames.create_dataset('cls_scans_anatomy', cls_scans_anatomy.shape, dtype=cls_scans_anatomy.dtype, data=cls_scans_anatomy)


fh5_frames.flush()
fh5_frames.close()

if FLAG_SCANFILE:
    fh5_scans.create_dataset('num_frames', num_frames.shape, dtype=num_frames.dtype, data=num_frames)
    fh5_scans.create_dataset('sub_folders', len(folders_subject), data=folders_subject)
    fh5_scans.create_dataset('frame_size', 2, data=frames_croped.shape[1:3])
    fh5_scans.create_dataset('name_scan', tuple((len(name_scan), len(name_scan[0]))), data=name_scan)

    fh5_scans.flush()
    fh5_scans.close()



print('Saved at: %s' % os.path.join(PATH_SAVE,'frames_res{}'.format(RESAMPLE_FACTOR)+".h5"))
if FLAG_SCANFILE:
    print('Saved at: %s' % os.path.join(PATH_SAVE,'scans_res{}'.format(RESAMPLE_FACTOR)+".h5"))
