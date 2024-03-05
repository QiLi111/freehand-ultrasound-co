
import os

import h5py
import numpy as np
# import torch
from utils_4_data import read_frame_transform, read_scan_crop_indices_file
from utils_4_data import ImageTransform


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


Extract_Feats = False # use pretarined model to extract feature for each frame

FLAG_SCANFILE = False

DIR_RAW = os.path.join(os.path.expanduser("~"), "workspace/forearm_US")
NUM_SCANS = 12
RESAMPLE_FACTOR = 4
PATH_SAVE = os.path.join(os.path.expanduser("~"), "workspace/")
DELAY_TFORM = 4  # delayed tform from temporal calibration

fh5_frames = h5py.File(os.path.join(PATH_SAVE,'frames_res{}_cls'.format(RESAMPLE_FACTOR)+".h5"),'a')
if FLAG_SCANFILE:
    fh5_scans = h5py.File(os.path.join(PATH_SAVE,'scans_res{}_cls'.format(RESAMPLE_FACTOR)+".h5"),'a')
if Extract_Feats:
    fh5_feats = h5py.File(os.path.join(PATH_SAVE,'frame_feats_res{}_cls'.format(RESAMPLE_FACTOR)+".h5"),'a')


folders_subject = [f for f in os.listdir(DIR_RAW) if os.path.isdir(os.path.join(DIR_RAW, f))]
num_frames = np.zeros((len(folders_subject),NUM_SCANS),dtype=np.uint16)
cls_scans_protocol = np.zeros((len(folders_subject),NUM_SCANS),dtype=np.uint16)
cls_scans_anatomy = np.zeros((len(folders_subject),NUM_SCANS),dtype=np.uint16)


temp = list(range(19))
cls_anatomy = np.array([i * 2 for i in temp])
cls_protocol = np.array([0,1,2])

name_scan = [[None for i in range(NUM_SCANS)] for j in range(len(folders_subject))]


transform_image = ImageTransform(mean=30.873100930319428, std=31.349069347795712)
# pretrained_model = Pretrained_model(1).to(device) # pretrained model for extracting features for each frame


for i_sub, folder in enumerate(folders_subject):

    fn_xls = [f for f in os.listdir(os.path.join(DIR_RAW, folder)) if f.endswith(".xlsx")]
    if len(fn_xls) != 1: 
        raise('Should contain 1 xlsx file in folder "{}"'.format(folder))
    scan_crop_idx = read_scan_crop_indices_file(os.path.join(DIR_RAW, folder, fn_xls[0]), NUM_SCANS)  # TBA: checks for 1) item name/order and if the 12 scans are complete

    fn_mha = [f for f in os.listdir(os.path.join(DIR_RAW, folder)) if f.endswith(".mha")]
    fn_mha.sort(reverse=False)  # ordered in acquisition time
    if len(fn_mha) != NUM_SCANS: raise('Should contain 12 mha files in folder "{}"'.format(folder))    

    for i_scan, fn in enumerate(fn_mha):

        frames, tforms, _ = read_frame_transform(
            filename = os.path.join(DIR_RAW, folder, fn),
            scan_crop_indices = scan_crop_idx[i_scan][1:3],
            resample_factor = RESAMPLE_FACTOR,
            delay_tform = DELAY_TFORM
        )

        tforms_inv = np.linalg.inv(tforms)  # pre-compute the iverse

        num_frames[i_sub,i_scan] = frames.shape[0]
        name_scan[i_sub][i_scan] = scan_crop_idx[i_scan][0] + '_' + folder

        # generate classification labels
        if 'Linear' in name_scan[i_sub][i_scan]:
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




        for i_frame in range(frames.shape[0]):
            fh5_frames.create_dataset('/sub%03d_scan%02d_frame%04d' % (i_sub,i_scan,i_frame), frames.shape[1:3], dtype=frames.dtype, data=frames[i_frame,...])
            fh5_frames.create_dataset('/sub%03d_scan%02d_tform%04d' % (i_sub,i_scan,i_frame), tforms.shape[1:3], dtype=tforms.dtype, data=tforms[i_frame,...])
            fh5_frames.create_dataset('/sub%03d_scan%02d_tform_inv%04d' % (i_sub,i_scan,i_frame), tforms.shape[1:3], dtype=tforms.dtype, data=tforms_inv[i_frame,...])
            fh5_frames.create_dataset('/sub%03d_scan%02d_cls_protocol%04d' % (i_sub,i_scan,i_frame), protocol_cls.shape, dtype=protocol_cls.dtype, data=protocol_cls)
            fh5_frames.create_dataset('/sub%03d_scan%02d_cls_anatomy%04d' % (i_sub,i_scan,i_frame), anatomy_cls.shape, dtype=anatomy_cls.dtype, data=anatomy_cls)


            # if Extract_Feats:
            #     frames_norm = transform_image(frames[i_frame,...])
            #     # frames_feats = extract_frame_features(frames_norm,pretrained_model, device)
            #     # frames_feats = frames_feats.detach().cpu().numpy()
            #     fh5_feats.create_dataset('/sub%03d_scan%02d_frame%04d' % (i_sub, i_scan, i_frame), frames_feats.shape,
            #                              dtype=frames_feats.dtype, data=frames_feats)
            #     fh5_feats.create_dataset('/sub%03d_scan%02d_tform%04d' % (i_sub, i_scan, i_frame), tforms.shape[1:3],
            #                              dtype=tforms.dtype, data=tforms[i_frame, ...])
            #     fh5_feats.create_dataset('/sub%03d_scan%02d_tform_inv%04d' % (i_sub, i_scan, i_frame),
            #                              tforms.shape[1:3],
            #                              dtype=tforms.dtype, data=tforms_inv[i_frame, ...])

        if FLAG_SCANFILE:
            fh5_scans.create_dataset('/sub%03d_frames%02d' % (i_sub,i_scan), frames.shape, dtype=frames.dtype, data=frames)
            fh5_scans.create_dataset('/sub%03d_tforms%02d' % (i_sub,i_scan), tforms.shape, dtype=tforms.dtype, data=tforms)
            fh5_scans.create_dataset('/sub%03d_tforms_inv%02d' % (i_sub,i_scan), tforms.shape, dtype=tforms.dtype, data=tforms_inv)

fh5_frames.create_dataset('num_frames', num_frames.shape, dtype=num_frames.dtype, data=num_frames)
fh5_frames.create_dataset('sub_folders', len(folders_subject), data=folders_subject)
fh5_frames.create_dataset('frame_size', 2, data=frames.shape[1:3])
fh5_frames.create_dataset('name_scan', tuple((len(name_scan),len(name_scan[0]))), data=name_scan)

fh5_frames.create_dataset('cls_scans_protocol', cls_scans_protocol.shape, dtype=cls_scans_protocol.dtype, data=cls_scans_protocol)
fh5_frames.create_dataset('cls_scans_anatomy', cls_scans_anatomy.shape, dtype=cls_scans_anatomy.dtype, data=cls_scans_anatomy)


fh5_frames.flush()
fh5_frames.close()

if FLAG_SCANFILE:
    fh5_scans.create_dataset('num_frames', num_frames.shape, dtype=num_frames.dtype, data=num_frames)
    fh5_scans.create_dataset('sub_folders', len(folders_subject), data=folders_subject)
    fh5_scans.create_dataset('frame_size', 2, data=frames.shape[1:3])
    fh5_frames.create_dataset('name_scan', tuple((len(name_scan), len(name_scan[0]))), data=name_scan)

    fh5_scans.flush()
    fh5_scans.close()


if Extract_Feats:
    fh5_feats.create_dataset('num_frames', num_frames.shape, dtype=num_frames.dtype, data=num_frames)
    fh5_feats.create_dataset('sub_folders', len(folders_subject), data=folders_subject)
    fh5_feats.create_dataset('frame_size', 2, data=frames.shape[1:3])
    fh5_feats.create_dataset('name_scan', tuple((len(name_scan), len(name_scan[0]))), data=name_scan)

    fh5_feats.flush()
    fh5_feats.close()

print('Saved at: %s' % os.path.join(PATH_SAVE,'frames_res{}'.format(RESAMPLE_FACTOR)+".h5"))
if FLAG_SCANFILE:
    print('Saved at: %s' % os.path.join(PATH_SAVE,'scans_res{}'.format(RESAMPLE_FACTOR)+".h5"))
if Extract_Feats:
    print('Saved at: %s' % os.path.join(PATH_SAVE,'feats_res{}'.format(RESAMPLE_FACTOR)+".h5"))
