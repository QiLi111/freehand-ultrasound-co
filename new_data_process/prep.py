# sort out the original .mha file
# 1) rename the .mha file by using shape of this scan;
# 2) detect the loop point and the final image, and crop out the unvalid frame,
#    and save the valid frames into .h5 file

import os

import h5py,csv
import matplotlib.pyplot as plt

import numpy as np
# import torch
from utils_4_data import read_frame_transform
from utils_4_data import ImageTransform,PlotScan,read_frame_transform_croped

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



FLAG_SCANFILE = True

DIR_RAW = os.path.join(os.path.expanduser("~"), "/public_data/forearm_US_large_dataset/data_size60")
NUM_SCANS = 12
NUM_SCANS_rotate = 2
RESAMPLE_FACTOR = 4
PATH_SAVE = os.path.join(os.path.expanduser("~"), "/public_data/forearm_US_large_dataset/data_size60")
DELAY_TFORM = 4  # delayed tform from temporal calibration
start_idx = 100
FILENAME_CALIB= "/public_data/forearm_US_large_dataset/calib_matrix.csv"


fh5_frames = h5py.File(os.path.join(PATH_SAVE,'frames_res{}'.format(RESAMPLE_FACTOR)+".h5"),'a')
if FLAG_SCANFILE:
    fh5_scans = h5py.File(os.path.join(PATH_SAVE,'scans_res{}'.format(RESAMPLE_FACTOR)+".h5"),'a')
    fh5_scans_rotate = h5py.File(os.path.join(PATH_SAVE,'scans_res{}_rotate'.format(RESAMPLE_FACTOR)+".h5"),'a')

fh5_frames_rotate = h5py.File(os.path.join(PATH_SAVE,'frames_res{}_rotate'.format(RESAMPLE_FACTOR)+".h5"),'a')


folders_subject = [f for f in os.listdir(DIR_RAW) if os.path.isdir(os.path.join(DIR_RAW, f))]
folders_subject = sorted(folders_subject, key=lambda x: int(x), reverse=False)

num_frames = np.zeros((len(folders_subject),NUM_SCANS),dtype=np.uint16)
num_frames_rotate = np.zeros((len(folders_subject),NUM_SCANS_rotate),dtype=np.uint16)


name_scan = [[None for i in range(NUM_SCANS)] for j in range(len(folders_subject))]
name_scan_rotate = [[None for i in range(NUM_SCANS_rotate)] for j in range(len(folders_subject))]


# transform_image = ImageTransform(mean=30.873100930319428, std=31.349069347795712)

scan_names = ['RH_Ver_L','RH_Ver_C','RH_Ver_S','RH_Par_L','RH_Par_C','RH_Par_S','LH_Ver_L','LH_Ver_C','LH_Ver_S','LH_Par_L','LH_Par_C','LH_Par_S']
scan_names_rotate = ['RH_rotate','LH_rotate']
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
    fn_mha.sort(reverse=False)  # ordered in acquisition time
    if len(fn_mha) != NUM_SCANS+ NUM_SCANS_rotate: raise('Should contain 14 mha files in folder "{}"'.format(folder))    

    fn_mha_rotate = [fn_mha[6],fn_mha[13]]
    fn_mha = fn_mha[0:6]+fn_mha[7:13]

    scan_names_csv=[]
    scan_names_rotate_csv=[]

    header = ['scan_name','start','loop','end']
   

    # rename scan
    for i_scan_1, fn_1 in enumerate(fn_mha):

        if fn_1.startswith("TrackedImageSequence"):
            
            os.rename(os.path.join(os.path.join(DIR_RAW, folder),fn_1), os.path.join(os.path.join(DIR_RAW, folder),scan_names[i_scan_1]+'_'+fn_1[21:]))
            scan_names_csv.append(scan_names[i_scan_1] + '_' + fn_1[21:])

    for i_scan, fn in enumerate(fn_mha_rotate):

        if fn.startswith("TrackedImageSequence"):
            
            os.rename(os.path.join(os.path.join(DIR_RAW, folder),fn), os.path.join(os.path.join(DIR_RAW, folder),scan_names_rotate[i_scan]+'_'+fn[21:]))
            scan_names_rotate_csv.append(scan_names_rotate[i_scan] + '_' + fn[21:])



    fn_mha = [f for f in os.listdir(os.path.join(DIR_RAW, folder)) if f.endswith(".mha")]
    # ordered in acquisition timed
    fn_mha=sorted(fn_mha, key=lambda x: int(''.join(filter(str.isdigit, x))))
    if len(fn_mha) != NUM_SCANS+NUM_SCANS_rotate: raise('Should contain 14 mha files in folder "{}"'.format(DIR_RAW))    

    fn_mha_rotate = [fn_mha[6],fn_mha[13]]
    fn_mha = fn_mha[0:6]+fn_mha[7:13] 

    with open(os.path.join(os.path.join(DIR_RAW,folder),'valid_frames.csv'), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        row = ['Scan_name','start_idx','loop','end','first_missing_status_in_scan']
        # write the header
        writer.writerow(row)

    with open(os.path.join(os.path.join(DIR_RAW,folder),'valid_frames_rotate.csv'), 'w', encoding='UTF8') as f_rotate:
        writer_rotate = csv.writer(f_rotate)
        row_rotate = ['Scan_name','start_idx','loop','end','first_missing_status_in_scan']
        # write the header
        writer_rotate.writerow(row_rotate)



    for i_scan_rotate, fn_rotate in enumerate(fn_mha_rotate):

        
        
        # get image and transformations-all images in a scan
        frames_rotate, tforms_rotate, _,first_missing_idx_rotate = read_frame_transform(
            filename = os.path.join(os.path.join(DIR_RAW, folder),fn_rotate),
            resample_factor = RESAMPLE_FACTOR,
            delay_tform = DELAY_TFORM
        )

        tforms_inv_rotate = np.linalg.inv(tforms_rotate)  # pre-compute the iverse

        # plot scan
        plot_scan_init_rotate = PlotScan(FILENAME_CALIB,RESAMPLE_FACTOR, Image_Shape=(120,160))
        
        # plot the trajectory of each scan and compute the axis range for x, y, and z
        px_all_rotate, py_all_rotate, pz_all_rotate= plot_scan_init_rotate.plot_scan(frames_rotate, tforms_rotate, tforms_inv_rotate,PATH_SAVE+'/'+folder+'/'+fn_rotate.split(".")[0])
        # plot_scan_init_rotate.Plot_Video(frames_rotate, tforms_rotate, tforms_inv_rotate,PATH_SAVE+'/'+folder+'/'+fn_rotate.split(".")[0],px_all_rotate, py_all_rotate, pz_all_rotate)
        # plot_scan_init_rotate.Save_Video(PATH_SAVE+'/'+folder+'/'+fn_rotate.split(".")[0])
        # plot_scan_init_rotate.plot_img_in_2d(frames_rotate,PATH_SAVE+'/'+folder+'/'+fn_rotate.split(".")[0])

        # detect the frame indexes for start (which is always fixed at 100), loop, and end
        index_loop_rotate,final_idx_rotate = plot_scan_init_rotate.detect_loop_points(px_all_rotate,py_all_rotate,pz_all_rotate,PATH_SAVE+'/'+folder+'/'+fn_rotate.split(".")[0])

        plot_scan_init_rotate.plot_scan_different_color_4_loop(frames_rotate, tforms_rotate, tforms_inv_rotate,PATH_SAVE+'/'+folder+'/'+fn_rotate.split(".")[0],index_loop_rotate,croped=False)

        # get image and transformations- croped images in a scan
        frames_croped_rotate, tforms_croped_rotate, _ = read_frame_transform_croped(
            filename = os.path.join(os.path.join(DIR_RAW, folder),fn_rotate),
            scan_crop_indices = [start_idx,index_loop_rotate,final_idx_rotate],
            resample_factor = RESAMPLE_FACTOR,
            delay_tform = DELAY_TFORM
        )
        
        tforms_inv_croped_rotate = np.linalg.inv(tforms_croped_rotate) # pre-compute the iverse

        # plot the trajectory of each scan and compute the axis range for x, y, and z
        px_all_croped_rotate, py_all_croped_rotate, pz_all_croped_rotate = plot_scan_init_rotate.plot_scan(frames_croped_rotate, tforms_croped_rotate, tforms_inv_croped_rotate,PATH_SAVE+'/'+folder+'/'+fn_rotate.split(".")[0],croped=True)
        # plot_scan_init_rotate.Plot_Video(frames_croped_rotate, tforms_croped_rotate, tforms_inv_croped_rotate,PATH_SAVE+'/'+folder+'/'+fn_rotate.split(".")[0],px_all_croped_rotate, py_all_croped_rotate, pz_all_croped_rotate,croped=True)
        # plot_scan_init_rotate.Save_Video(PATH_SAVE+'/'+folder+'/'+fn_rotate.split(".")[0],croped=True)
        # plot_scan_init_rotate.plot_img_in_2d(frames_croped_rotate,PATH_SAVE+'/'+folder+'/'+fn_rotate.split(".")[0],croped=True)
        
        index_loop_croped_rotate,final_idx_croped_rotate = plot_scan_init_rotate.detect_loop_points(px_all_croped_rotate,py_all_croped_rotate,pz_all_croped_rotate,PATH_SAVE+'/'+folder+'/'+fn_rotate.split(".")[0],croped=True)
        plot_scan_init_rotate.plot_scan_different_color_4_loop(frames_croped_rotate, tforms_croped_rotate, tforms_inv_croped_rotate,PATH_SAVE+'/'+folder+'/'+fn_rotate.split(".")[0],index_loop_croped_rotate, croped=True)

        relative_transf_6_DOF_each_scan_rotate,relative_transf_6_DOF_to_1st_img_each_scan_rotate = plot_scan_init_rotate.get_transf_para_each_scan(tforms_croped_rotate, tforms_inv_croped_rotate,PATH_SAVE+'/'+folder+'/'+fn_rotate.split(".")[0])
        
        if 'relative_transf_6_DOF_all_scan_rotate' not in globals():
            relative_transf_6_DOF_all_scan_rotate = relative_transf_6_DOF_each_scan_rotate
        else:
            relative_transf_6_DOF_all_scan_rotate = np.concatenate((relative_transf_6_DOF_all_scan_rotate, relative_transf_6_DOF_each_scan_rotate), axis=0)

        if 'relative_transf_6_DOF_to_1st_img_all_scan_rotate' not in globals():
            relative_transf_6_DOF_to_1st_img_all_scan_rotate = relative_transf_6_DOF_to_1st_img_each_scan_rotate
        else:
            relative_transf_6_DOF_to_1st_img_all_scan_rotate = np.concatenate((relative_transf_6_DOF_to_1st_img_all_scan_rotate, relative_transf_6_DOF_to_1st_img_each_scan_rotate), axis=0)

    

        with open(os.path.join(os.path.join(DIR_RAW,folder),'valid_frames_rotate.csv'), 'a', encoding='UTF8') as f_rotate:
            writer_rotate = csv.writer(f_rotate)
            row_rotate = [fn_rotate,start_idx,index_loop_rotate,final_idx_rotate, first_missing_idx_rotate]
            # write the header
            writer_rotate.writerow(row_rotate)

        

        num_frames_rotate[i_sub,i_scan_rotate] = frames_croped_rotate.shape[0]
        name_scan_rotate[i_sub][i_scan_rotate] = fn_rotate
        for i_frame_rotate in range(frames_croped_rotate.shape[0]):
            fh5_frames_rotate.create_dataset('/sub%03d_scan%02d_frame%04d' % (i_sub,i_scan_rotate,i_frame_rotate), frames_croped_rotate.shape[1:3], dtype=frames_croped_rotate.dtype, data=frames_croped_rotate[i_frame_rotate,...])
            fh5_frames_rotate.create_dataset('/sub%03d_scan%02d_tform%04d' % (i_sub,i_scan_rotate,i_frame_rotate), tforms_croped_rotate.shape[1:3], dtype=tforms_croped_rotate.dtype, data=tforms_croped_rotate[i_frame_rotate,...])
            fh5_frames_rotate.create_dataset('/sub%03d_scan%02d_tform_inv%04d' % (i_sub,i_scan_rotate,i_frame_rotate), tforms_inv_croped_rotate.shape[1:3], dtype=tforms_inv_croped_rotate.dtype, data=tforms_inv_croped_rotate[i_frame_rotate,...])
            # if Extract_Feats:

        if FLAG_SCANFILE:
            fh5_scans_rotate.create_dataset('/sub%03d_frames%02d' % (i_sub,i_scan_rotate), frames_croped_rotate.shape, dtype=frames_croped_rotate.dtype, data=frames_croped_rotate)
            fh5_scans_rotate.create_dataset('/sub%03d_tforms%02d' % (i_sub,i_scan_rotate), tforms_croped_rotate.shape, dtype=tforms_croped_rotate.dtype, data=tforms_croped_rotate)
            fh5_scans_rotate.create_dataset('/sub%03d_tforms_inv%02d' % (i_sub,i_scan_rotate), tforms_inv_croped_rotate.shape, dtype=tforms_inv_croped_rotate.dtype, data=tforms_inv_croped_rotate)




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
        # plot_scan_init.Plot_Video(frames, tforms, tforms_inv,PATH_SAVE+'/'+folder+'/'+fn.split(".")[0],px_all, py_all, pz_all)
        # plot_scan_init.Save_Video(PATH_SAVE+'/'+folder+'/'+fn.split(".")[0])
        # plot_scan_init.plot_img_in_2d(frames,PATH_SAVE+'/'+folder+'/'+fn.split(".")[0])

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

        # plot the trajectory of each scan and compute the axis range for x, y, and z
        px_all_croped, py_all_croped, pz_all_croped = plot_scan_init.plot_scan(frames_croped, tforms_croped, tforms_inv_croped,PATH_SAVE+'/'+folder+'/'+fn.split(".")[0],croped=True)
        # plot_scan_init.Plot_Video(frames_croped, tforms_croped, tforms_inv_croped,PATH_SAVE+'/'+folder+'/'+fn.split(".")[0],px_all_croped, py_all_croped, pz_all_croped,croped=True)
        # plot_scan_init.Save_Video(PATH_SAVE+'/'+folder+'/'+fn.split(".")[0],croped=True)
        # plot_scan_init.plot_img_in_2d(frames_croped,PATH_SAVE+'/'+folder+'/'+fn.split(".")[0],croped=True)
        
        index_loop_croped,final_idx_croped = plot_scan_init.detect_loop_points(px_all_croped,py_all_croped,pz_all_croped,PATH_SAVE+'/'+folder+'/'+fn.split(".")[0],croped=True)
        plot_scan_init.plot_scan_different_color_4_loop(frames_croped, tforms_croped, tforms_inv_croped,PATH_SAVE+'/'+folder+'/'+fn.split(".")[0],index_loop_croped, croped=True)

        relative_transf_6_DOF_each_scan,relative_transf_6_DOF_to_1st_img_each_scan = plot_scan_init.get_transf_para_each_scan(tforms_croped, tforms_inv_croped,PATH_SAVE+'/'+folder+'/'+fn.split(".")[0])
        
        if 'relative_transf_6_DOF_all_scan' not in globals():
            relative_transf_6_DOF_all_scan = relative_transf_6_DOF_each_scan
        else:
            relative_transf_6_DOF_all_scan = np.concatenate((relative_transf_6_DOF_all_scan, relative_transf_6_DOF_each_scan), axis=0)

        if 'relative_transf_6_DOF_to_1st_img_all_scan' not in globals():
            relative_transf_6_DOF_to_1st_img_all_scan = relative_transf_6_DOF_to_1st_img_each_scan
        else:
            relative_transf_6_DOF_to_1st_img_all_scan = np.concatenate((relative_transf_6_DOF_to_1st_img_all_scan, relative_transf_6_DOF_to_1st_img_each_scan), axis=0)

        

        with open(os.path.join(os.path.join(DIR_RAW,folder),'valid_frames.csv'), 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            row = [fn,start_idx,index_loop,final_idx, first_missing_idx]
            # write the header
            writer.writerow(row)

        num_frames[i_sub,i_scan] = frames_croped.shape[0]
        name_scan[i_sub][i_scan] = fn
        for i_frame in range(frames_croped.shape[0]):
            fh5_frames.create_dataset('/sub%03d_scan%02d_frame%04d' % (i_sub,i_scan,i_frame), frames_croped.shape[1:3], dtype=frames_croped.dtype, data=frames_croped[i_frame,...])
            fh5_frames.create_dataset('/sub%03d_scan%02d_tform%04d' % (i_sub,i_scan,i_frame), tforms_croped.shape[1:3], dtype=tforms_croped.dtype, data=tforms_croped[i_frame,...])
            fh5_frames.create_dataset('/sub%03d_scan%02d_tform_inv%04d' % (i_sub,i_scan,i_frame), tforms_inv_croped.shape[1:3], dtype=tforms_inv_croped.dtype, data=tforms_inv_croped[i_frame,...])
            # if Extract_Feats:

        if FLAG_SCANFILE:
            fh5_scans.create_dataset('/sub%03d_frames%02d' % (i_sub,i_scan), frames_croped.shape, dtype=frames_croped.dtype, data=frames_croped)
            fh5_scans.create_dataset('/sub%03d_tforms%02d' % (i_sub,i_scan), tforms_croped.shape, dtype=tforms_croped.dtype, data=tforms_croped)
            fh5_scans.create_dataset('/sub%03d_tforms_inv%02d' % (i_sub,i_scan), tforms_inv_croped.shape, dtype=tforms_inv_croped.dtype, data=tforms_inv_croped)


    


    fig, ax = plt.subplots()
    ax.boxplot(relative_transf_6_DOF_all_scan)
    fig.savefig(os.path.join(DIR_RAW,folder)+'/'+'6DOF_between_imgs_all_scan.png')
    plt.close()

    fig1, ax1 = plt.subplots()
    ax1.boxplot(relative_transf_6_DOF_to_1st_img_all_scan)
    fig1.savefig(os.path.join(DIR_RAW,folder)+'/'+'6DOF_to_1st_img_all_scan.png')
    plt.close()


    if 'relative_transf_6_DOF_all_scan_all_sub' not in globals():
        relative_transf_6_DOF_all_scan_all_sub = relative_transf_6_DOF_all_scan
    else:
        relative_transf_6_DOF_all_scan_all_sub = np.concatenate((relative_transf_6_DOF_all_scan_all_sub, relative_transf_6_DOF_all_scan), axis=0)

    if 'relative_transf_6_DOF_to_1st_img_all_scan_all_sub' not in globals():
        relative_transf_6_DOF_to_1st_img_all_scan_all_sub = relative_transf_6_DOF_to_1st_img_all_scan
    else:
        relative_transf_6_DOF_to_1st_img_all_scan_all_sub = np.concatenate((relative_transf_6_DOF_to_1st_img_all_scan_all_sub, relative_transf_6_DOF_to_1st_img_all_scan), axis=0)



    fig_rotate, ax_rotate = plt.subplots()
    ax_rotate.boxplot(relative_transf_6_DOF_all_scan_rotate)
    fig_rotate.savefig(os.path.join(DIR_RAW,folder)+'/'+'6DOF_between_imgs_all_scan_rotate.png')
    plt.close()

    fig1_rotate, ax1_rotate = plt.subplots()
    ax1_rotate.boxplot(relative_transf_6_DOF_to_1st_img_all_scan_rotate)
    fig1_rotate.savefig(os.path.join(DIR_RAW,folder)+'/'+'6DOF_to_1st_img_all_scan_rotate.png')
    plt.close()


    if 'relative_transf_6_DOF_all_scan_all_sub_rotate' not in globals():
        relative_transf_6_DOF_all_scan_all_sub_rotate = relative_transf_6_DOF_all_scan_rotate
    else:
        relative_transf_6_DOF_all_scan_all_sub_rotate = np.concatenate((relative_transf_6_DOF_all_scan_all_sub_rotate, relative_transf_6_DOF_all_scan_rotate), axis=0)

    if 'relative_transf_6_DOF_to_1st_img_all_scan_all_sub_rotate' not in globals():
        relative_transf_6_DOF_to_1st_img_all_scan_all_sub_rotate = relative_transf_6_DOF_to_1st_img_all_scan_rotate
    else:
        relative_transf_6_DOF_to_1st_img_all_scan_all_sub_rotate = np.concatenate((relative_transf_6_DOF_to_1st_img_all_scan_all_sub_rotate, relative_transf_6_DOF_to_1st_img_all_scan_rotate), axis=0)



fig2, ax2 = plt.subplots()
ax2.boxplot(relative_transf_6_DOF_all_scan_all_sub)
fig2.savefig(DIR_RAW+'/'+'6DOF_between_imgs_all_scan_all_sub.png')
plt.close()

fig3, ax3 = plt.subplots()
ax3.boxplot(relative_transf_6_DOF_to_1st_img_all_scan_all_sub)
fig3.savefig(DIR_RAW+'/'+'6DOF_to_1st_img_all_scan_all_sub.png')
plt.close()


        
fh5_frames.create_dataset('num_frames', num_frames.shape, dtype=num_frames.dtype, data=num_frames)
fh5_frames.create_dataset('sub_folders', len(folders_subject), data=folders_subject)
fh5_frames.create_dataset('frame_size', 2, data=frames_croped.shape[1:3])
fh5_frames.create_dataset('name_scan', tuple((len(name_scan),len(name_scan[0]))), data=name_scan)

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



fig2_rotate, ax2_rotate = plt.subplots()
ax2_rotate.boxplot(relative_transf_6_DOF_all_scan_all_sub_rotate)
fig2_rotate.savefig(DIR_RAW+'/'+'6DOF_between_imgs_all_scan_all_sub_rotate.png')
plt.close()

fig3_rotate, ax3_rotate = plt.subplots()
ax3_rotate.boxplot(relative_transf_6_DOF_to_1st_img_all_scan_all_sub_rotate)
fig3_rotate.savefig(DIR_RAW+'/'+'6DOF_to_1st_img_all_scan_all_sub_rotate.png')
plt.close()


        
fh5_frames_rotate.create_dataset('num_frames', num_frames_rotate.shape, dtype=num_frames_rotate.dtype, data=num_frames_rotate)
fh5_frames_rotate.create_dataset('sub_folders', len(folders_subject), data=folders_subject)
fh5_frames_rotate.create_dataset('frame_size', 2, data=frames_croped_rotate.shape[1:3])
fh5_frames_rotate.create_dataset('name_scan', tuple((len(name_scan_rotate),len(name_scan_rotate[0]))), data=name_scan_rotate)

fh5_frames_rotate.flush()
fh5_frames_rotate.close()

if FLAG_SCANFILE:
    fh5_scans_rotate.create_dataset('num_frames', num_frames_rotate.shape, dtype=num_frames_rotate.dtype, data=num_frames_rotate)
    fh5_scans_rotate.create_dataset('sub_folders', len(folders_subject), data=folders_subject)
    fh5_scans_rotate.create_dataset('frame_size', 2, data=frames_croped_rotate.shape[1:3])
    fh5_scans_rotate.create_dataset('name_scan', tuple((len(name_scan_rotate), len(name_scan_rotate[0]))), data=name_scan_rotate)

    fh5_scans_rotate.flush()
    fh5_scans_rotate.close()



print('Saved at: %s' % os.path.join(PATH_SAVE,'frames_res{}_rotate'.format(RESAMPLE_FACTOR)+".h5"))
if FLAG_SCANFILE:
    print('Saved at: %s' % os.path.join(PATH_SAVE,'scans_res{}_rotate'.format(RESAMPLE_FACTOR)+".h5"))
