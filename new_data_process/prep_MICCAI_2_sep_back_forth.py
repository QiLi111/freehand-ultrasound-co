
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


fh5_frames_forth = h5py.File(os.path.join(PATH_SAVE,'frames_res{}_cls_6_forth'.format(RESAMPLE_FACTOR)+".h5"),'a')
if FLAG_SCANFILE:
    fh5_scans_forth = h5py.File(os.path.join(PATH_SAVE,'scans_res{}_cls_6_forth'.format(RESAMPLE_FACTOR)+".h5"),'a')

fh5_frames_back = h5py.File(os.path.join(PATH_SAVE,'frames_res{}_cls_6_back'.format(RESAMPLE_FACTOR)+".h5"),'a')
if FLAG_SCANFILE:
    fh5_scans_back = h5py.File(os.path.join(PATH_SAVE,'scans_res{}_cls_6_back'.format(RESAMPLE_FACTOR)+".h5"),'a')


folders_subject = [f for f in os.listdir(DIR_RAW) if os.path.isdir(os.path.join(DIR_RAW, f))]

num_frames_forth = np.zeros((len(folders_subject),NUM_SCANS),dtype=np.uint16)
num_frames_back = np.zeros((len(folders_subject),NUM_SCANS),dtype=np.uint16)


name_scan_forth = [[None for i in range(NUM_SCANS)] for j in range(len(folders_subject))]
name_scan_back = [[None for i in range(NUM_SCANS)] for j in range(len(folders_subject))]

cls_scans_protocol_forth = np.zeros((len(folders_subject),NUM_SCANS),dtype=np.uint16)
cls_scans_anatomy_forth = np.zeros((len(folders_subject),NUM_SCANS),dtype=np.uint16)


temp = list(range(len(folders_subject)))
cls_anatomy_forth = np.array([i * 2 for i in temp])
cls_protocol_forth = np.array([0,1,2,3,4,5])

cls_scans_protocol_back = np.zeros((len(folders_subject),NUM_SCANS),dtype=np.uint16)
cls_scans_anatomy_back = np.zeros((len(folders_subject),NUM_SCANS),dtype=np.uint16)


temp = list(range(len(folders_subject)))
cls_anatomy_back = np.array([i * 2 for i in temp])
cls_protocol_back = np.array([0,1,2,3,4,5])

transform_image = ImageTransform(mean=30.873100930319428, std=31.349069347795712)

scan_names = ['RH_Ver_L','RH_Ver_C','RH_Ver_S','RH_Par_L','RH_Par_C','RH_Par_S','LH_Ver_L','LH_Ver_C','LH_Ver_S','LH_Par_L','LH_Par_C','LH_Par_S']
# scan_names = ['RH_Par_L','RH_Par_C','RH_Par_S','RH_Ver_L','RH_Ver_C','RH_Ver_S','LH_Par_L','LH_Par_C','LH_Par_S','LH_Ver_L','LH_Ver_C','LH_Ver_S']


# # rename each scan
# for i_sub, folder in enumerate(folders_subject):
#     fn_mha = [f for f in os.listdir(os.path.join(DIR_RAW, folder)) if f.endswith(".mha")]
#     fn_mha.sort(reverse=False)  # ordered in acquisition time
    
#     scan_names_csv=[]
#     for i_scan, fn in enumerate(fn_mha):

#         if fn.endswith("TrackedImageSequence"):
#             os.rename(os.path.join(os.path.join(DIR_RAW, folder),fn), os.path.join(os.path.join(DIR_RAW, folder),scan_names[i_scan]+'_'+fn[21:]))
#             scan_names_csv.append(scan_names[i_scan] + '_' + fn[21:])



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

        plot_scan_init.plot_scan_different_color_4_loop(frames, tforms, tforms_inv,PATH_SAVE+'/'+folder+'/'+fn.split(".")[0],index_loop,croped=False)

        # get image and transformations- croped images in a scan
        frames_croped, tforms_croped, _ = read_frame_transform_croped(
            filename = os.path.join(os.path.join(DIR_RAW, folder),fn),
            scan_crop_indices = [start_idx,index_loop,final_idx],
            resample_factor = RESAMPLE_FACTOR,
            delay_tform = DELAY_TFORM
        )
        
        tforms_inv_croped = np.linalg.inv(tforms_croped) # pre-compute the iverse

        frames_forth = frames_croped[0:index_loop-start_idx,...]
        tforms_forth = tforms_croped[0:index_loop-start_idx,...]
        tforms_inv_forth = tforms_inv_croped[0:index_loop-start_idx,...]

        frames_back = frames_croped[index_loop-start_idx:,...]
        tforms_back = tforms_croped[index_loop-start_idx:,...]
        tforms_inv_back = tforms_inv_croped[index_loop-start_idx:,...]

        num_frames_forth[i_sub,i_scan] = frames_forth.shape[0]
        name_scan_forth[i_sub][i_scan] = fn

        num_frames_back[i_sub,i_scan] = frames_back.shape[0]
        name_scan_back[i_sub][i_scan] = fn

        # generate classification labels
        if 'L' in name_scan_forth[i_sub][i_scan] and 'Par' in name_scan_forth[i_sub][i_scan]:
            protocol_cls_forth = cls_protocol_forth[0]
        elif 'L' in name_scan_forth[i_sub][i_scan] and 'Ver' in name_scan_forth[i_sub][i_scan]:
            protocol_cls_forth = cls_protocol_forth[1]

        elif 'C' in name_scan_forth[i_sub][i_scan] and 'Par' in name_scan_forth[i_sub][i_scan]:
            protocol_cls_forth = cls_protocol_forth[2]
        elif 'C' in name_scan_forth[i_sub][i_scan] and 'Ver' in name_scan_forth[i_sub][i_scan]:
            protocol_cls_forth = cls_protocol_forth[3]


        elif 'S' in name_scan_forth[i_sub][i_scan] and 'Par' in name_scan_forth[i_sub][i_scan]:
            protocol_cls_forth = cls_protocol_forth[4]

        elif 'S' in name_scan_forth[i_sub][i_scan] and 'Ver' in name_scan_forth[i_sub][i_scan]:
            protocol_cls_forth = cls_protocol_forth[5]


        else:
            raise('wrong protocol')


        idx_name_forth = folders_subject.index(folder)
        if 'RH' in name_scan_forth[i_sub][i_scan]:
            anatomy_cls_forth = cls_anatomy_forth[idx_name_forth]
        elif 'LH' in name_scan_forth[i_sub][i_scan]:
            anatomy_cls_forth = cls_anatomy_forth[idx_name_forth]+1
        else:
            raise('wrong anatomy')

        cls_scans_protocol_forth[i_sub][i_scan] = protocol_cls_forth
        cls_scans_anatomy_forth[i_sub][i_scan] = anatomy_cls_forth


        # generate classification labels
        if 'L' in name_scan_back[i_sub][i_scan] and 'Par' in name_scan_back[i_sub][i_scan]:
            protocol_cls_back = cls_protocol_back[0]
        elif 'L' in name_scan_back[i_sub][i_scan] and 'Ver' in name_scan_back[i_sub][i_scan]:
            protocol_cls_back = cls_protocol_back[1]

        elif 'C' in name_scan_back[i_sub][i_scan] and 'Par' in name_scan_back[i_sub][i_scan]:
            protocol_cls_back = cls_protocol_back[2]
        elif 'C' in name_scan_back[i_sub][i_scan] and 'Ver' in name_scan_back[i_sub][i_scan]:
            protocol_cls_back = cls_protocol_back[3]


        elif 'S' in name_scan_back[i_sub][i_scan] and 'Par' in name_scan_back[i_sub][i_scan]:
            protocol_cls_back = cls_protocol_back[4]

        elif 'S' in name_scan_back[i_sub][i_scan] and 'Ver' in name_scan_back[i_sub][i_scan]:
            protocol_cls_back = cls_protocol_back[5]


        else:
            raise('wrong protocol')

        idx_name_back = folders_subject.index(folder)
        if 'RH' in name_scan_back[i_sub][i_scan]:
            anatomy_cls_back = cls_anatomy_back[idx_name_back]
        elif 'LH' in name_scan_back[i_sub][i_scan]:
            anatomy_cls_back = cls_anatomy_back[idx_name_back]+1
        else:
            raise('wrong anatomy')

        cls_scans_protocol_back[i_sub][i_scan] = protocol_cls_back
        cls_scans_anatomy_back[i_sub][i_scan] = anatomy_cls_back




        for i_frame in range(frames_forth.shape[0]):
            fh5_frames_forth.create_dataset('/sub%03d_scan%02d_frame%04d' % (i_sub,i_scan,i_frame), frames_forth.shape[1:3], dtype=frames_forth.dtype, data=frames_forth[i_frame,...])
            fh5_frames_forth.create_dataset('/sub%03d_scan%02d_tform%04d' % (i_sub,i_scan,i_frame), tforms_forth.shape[1:3], dtype=tforms_forth.dtype, data=tforms_forth[i_frame,...])
            fh5_frames_forth.create_dataset('/sub%03d_scan%02d_tform_inv%04d' % (i_sub,i_scan,i_frame), tforms_inv_forth.shape[1:3], dtype=tforms_inv_forth.dtype, data=tforms_inv_forth[i_frame,...])
            fh5_frames_forth.create_dataset('/sub%03d_scan%02d_cls_protocol%04d' % (i_sub,i_scan,i_frame), protocol_cls_forth.shape, dtype=protocol_cls_forth.dtype, data=protocol_cls_forth)
            fh5_frames_forth.create_dataset('/sub%03d_scan%02d_cls_anatomy%04d' % (i_sub,i_scan,i_frame), anatomy_cls_forth.shape, dtype=anatomy_cls_forth.dtype, data=anatomy_cls_forth)

        if FLAG_SCANFILE:
            fh5_scans_forth.create_dataset('/sub%03d_frames%02d' % (i_sub,i_scan), frames_forth.shape, dtype=frames_forth.dtype, data=frames_forth)
            fh5_scans_forth.create_dataset('/sub%03d_tforms%02d' % (i_sub,i_scan), tforms_forth.shape, dtype=tforms_forth.dtype, data=tforms_forth)
            fh5_scans_forth.create_dataset('/sub%03d_tforms_inv%02d' % (i_sub,i_scan), tforms_inv_forth.shape, dtype=tforms_inv_forth.dtype, data=tforms_inv_forth)


        num_frames_back[i_sub,i_scan] = frames_back.shape[0]
        name_scan_back[i_sub][i_scan] = fn
        for i_frame in range(frames_back.shape[0]):
            fh5_frames_back.create_dataset('/sub%03d_scan%02d_frame%04d' % (i_sub,i_scan,i_frame), frames_back.shape[1:3], dtype=frames_back.dtype, data=frames_back[i_frame,...])
            fh5_frames_back.create_dataset('/sub%03d_scan%02d_tform%04d' % (i_sub,i_scan,i_frame), tforms_back.shape[1:3], dtype=tforms_back.dtype, data=tforms_back[i_frame,...])
            fh5_frames_back.create_dataset('/sub%03d_scan%02d_tform_inv%04d' % (i_sub,i_scan,i_frame), tforms_inv_back.shape[1:3], dtype=tforms_inv_back.dtype, data=tforms_inv_back[i_frame,...])
            fh5_frames_back.create_dataset('/sub%03d_scan%02d_cls_protocol%04d' % (i_sub,i_scan,i_frame), protocol_cls_back.shape, dtype=protocol_cls_back.dtype, data=protocol_cls_back)
            fh5_frames_back.create_dataset('/sub%03d_scan%02d_cls_anatomy%04d' % (i_sub,i_scan,i_frame), anatomy_cls_back.shape, dtype=anatomy_cls_back.dtype, data=anatomy_cls_back)

        if FLAG_SCANFILE:
            fh5_scans_back.create_dataset('/sub%03d_frames%02d' % (i_sub,i_scan), frames_back.shape, dtype=frames_back.dtype, data=frames_back)
            fh5_scans_back.create_dataset('/sub%03d_tforms%02d' % (i_sub,i_scan), tforms_back.shape, dtype=tforms_back.dtype, data=tforms_back)
            fh5_scans_back.create_dataset('/sub%03d_tforms_inv%02d' % (i_sub,i_scan), tforms_inv_back.shape, dtype=tforms_inv_back.dtype, data=tforms_inv_back)

        
fh5_frames_back.create_dataset('num_frames', num_frames_back.shape, dtype=num_frames_back.dtype, data=num_frames_back)
fh5_frames_back.create_dataset('sub_folders', len(folders_subject), data=folders_subject)
fh5_frames_back.create_dataset('frame_size', 2, data=frames_back.shape[1:3])
fh5_frames_back.create_dataset('name_scan', tuple((len(name_scan_back),len(name_scan_back[0]))), data=name_scan_back)
fh5_frames_back.create_dataset('cls_scans_protocol', cls_scans_protocol_back.shape, dtype=cls_scans_protocol_back.dtype, data=cls_scans_protocol_back)
fh5_frames_back.create_dataset('cls_scans_anatomy', cls_scans_anatomy_back.shape, dtype=cls_scans_anatomy_back.dtype, data=cls_scans_anatomy_back)

fh5_frames_back.flush()
fh5_frames_back.close()

if FLAG_SCANFILE:
    fh5_scans_back.create_dataset('num_frames', num_frames_back.shape, dtype=num_frames_back.dtype, data=num_frames_back)
    fh5_scans_back.create_dataset('sub_folders', len(folders_subject), data=folders_subject)
    fh5_scans_back.create_dataset('frame_size', 2, data=frames_back.shape[1:3])
    fh5_scans_back.create_dataset('name_scan', tuple((len(name_scan_back), len(name_scan_back[0]))), data=name_scan_back)

    fh5_scans_back.flush()
    fh5_scans_back.close()


fh5_frames_forth.create_dataset('num_frames', num_frames_forth.shape, dtype=num_frames_forth.dtype, data=num_frames_forth)
fh5_frames_forth.create_dataset('sub_folders', len(folders_subject), data=folders_subject)
fh5_frames_forth.create_dataset('frame_size', 2, data=frames_forth.shape[1:3])
fh5_frames_forth.create_dataset('name_scan', tuple((len(name_scan_forth),len(name_scan_forth[0]))), data=name_scan_forth)
fh5_frames_forth.create_dataset('cls_scans_protocol', cls_scans_protocol_forth.shape, dtype=cls_scans_protocol_forth.dtype, data=cls_scans_protocol_forth)
fh5_frames_forth.create_dataset('cls_scans_anatomy', cls_scans_anatomy_forth.shape, dtype=cls_scans_anatomy_forth.dtype, data=cls_scans_anatomy_forth)

fh5_frames_forth.flush()
fh5_frames_forth.close()

if FLAG_SCANFILE:
    fh5_scans_forth.create_dataset('num_frames', num_frames_forth.shape, dtype=num_frames_forth.dtype, data=num_frames_forth)
    fh5_scans_forth.create_dataset('sub_folders', len(folders_subject), data=folders_subject)
    fh5_scans_forth.create_dataset('frame_size', 2, data=frames_forth.shape[1:3])
    fh5_scans_forth.create_dataset('name_scan', tuple((len(name_scan_forth), len(name_scan_forth[0]))), data=name_scan_forth)

    fh5_scans_forth.flush()
    fh5_scans_forth.close()


print('Saved at: %s' % os.path.join(PATH_SAVE,'frames_res{}_forth'.format(RESAMPLE_FACTOR)+".h5"))
if FLAG_SCANFILE:
    print('Saved at: %s' % os.path.join(PATH_SAVE,'scans_res{}_forth'.format(RESAMPLE_FACTOR)+".h5"))

print('Saved at: %s' % os.path.join(PATH_SAVE,'frames_res{}_back'.format(RESAMPLE_FACTOR)+".h5"))
if FLAG_SCANFILE:
    print('Saved at: %s' % os.path.join(PATH_SAVE,'scans_res{}_back'.format(RESAMPLE_FACTOR)+".h5"))
