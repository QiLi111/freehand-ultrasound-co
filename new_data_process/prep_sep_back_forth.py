# sort out the original .mha file
# 1) remane the .mha file by using shape of this scan;
# 2) detect the loop point and the final image, and crop out the unvalid frame,
#    and save the valid frames into .h5 file
# 3) Seperate the whole loop scan into forth and back scan for future training use



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

DIR_RAW = os.path.join(os.path.expanduser("~"), "/raid/Qi/public_data/forearm_US_large_dataset/data_size59")
NUM_SCANS = 12
NUM_SCANS_rotate=2
RESAMPLE_FACTOR = 4
PATH_SAVE = os.path.join(os.path.expanduser("~"), "/raid/Qi/public_data/forearm_US_large_dataset/data_size59")
DELAY_TFORM = 4  # delayed tform from temporal calibration
start_idx = 100
FILENAME_CALIB= "/raid/Qi/public_data/forearm_US_large_dataset/calib_matrix.csv"


fh5_frames_forth = h5py.File(os.path.join(PATH_SAVE,'frames_res{}_forth'.format(RESAMPLE_FACTOR)+".h5"),'a')
fh5_frames_forth_back = h5py.File(os.path.join(PATH_SAVE,'frames_res{}_forth_back'.format(RESAMPLE_FACTOR)+".h5"),'a')
fh5_frames_back = h5py.File(os.path.join(PATH_SAVE,'frames_res{}_back'.format(RESAMPLE_FACTOR)+".h5"),'a')

if FLAG_SCANFILE:
    fh5_scans_forth = h5py.File(os.path.join(PATH_SAVE,'scans_res{}_forth'.format(RESAMPLE_FACTOR)+".h5"),'a')
    fh5_scans_back = h5py.File(os.path.join(PATH_SAVE,'scans_res{}_back'.format(RESAMPLE_FACTOR)+".h5"),'a')
    fh5_scans_forth_back = h5py.File(os.path.join(PATH_SAVE,'scans_res{}_forth_back'.format(RESAMPLE_FACTOR)+".h5"),'a')


folders_subject = [f for f in os.listdir(DIR_RAW) if os.path.isdir(os.path.join(DIR_RAW, f))]
folders_subject = sorted(folders_subject, key=lambda x: int(x), reverse=False)

num_frames_forth = np.zeros((len(folders_subject),NUM_SCANS),dtype=np.uint16)
num_frames_back = np.zeros((len(folders_subject),NUM_SCANS),dtype=np.uint16)
num_frames_forth_back = np.zeros((len(folders_subject),NUM_SCANS*2),dtype=np.uint16)



name_scan_forth = [[None for i in range(NUM_SCANS)] for j in range(len(folders_subject))]
name_scan_back = [[None for i in range(NUM_SCANS)] for j in range(len(folders_subject))]
name_scan_forth_back = [[None for i in range(NUM_SCANS*2)] for j in range(len(folders_subject))]



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
    fn_mha.sort(reverse=False)  # ordered in acquisition time

    scan_names_csv=[]
    header = ['scan_name','start','loop','end']
   

    if len(fn_mha) != NUM_SCANS+ NUM_SCANS_rotate: raise('Should contain 14 mha files in folder "{}"'.format(folder))    

    # rename scan
    for i_scan_1, fn_1 in enumerate(fn_mha):

        if fn_1.startswith("TrackedImageSequence"):
            raise("should run prep.py to rename the .mha files")
            
            # os.rename(os.path.join(os.path.join(DIR_RAW, folder),fn_1), os.path.join(os.path.join(DIR_RAW, folder),scan_names[i_scan_1]+'_'+fn_1[21:]))
            # scan_names_csv.append(scan_names[i_scan_1] + '_' + fn_1[21:])


    fn_mha = [f for f in os.listdir(os.path.join(DIR_RAW, folder)) if f.endswith(".mha")]
    # ordered in acquisition time
    fn_mha=sorted(fn_mha, key=lambda x: int(''.join(filter(str.isdigit, x))))
    if len(fn_mha) != NUM_SCANS+NUM_SCANS_rotate: raise('Should contain 12 mha files in folder "{}"'.format(DIR_RAW))    

    fn_mha_rotate = [fn_mha[6],fn_mha[13]]
    fn_mha = fn_mha[0:6]+fn_mha[7:13] 


    with open(os.path.join(os.path.join(DIR_RAW,folder),'valid_frames_sep.csv'), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        row = ['Scan_name','start_idx','loop','end','first_missing_status_in_scan']
        # write the header
        writer.writerow(row)

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


        # plot the trajectory of each scan and compute the axis range for x, y, and z
        px_all_forth, py_all_forth, pz_all_forth = plot_scan_init.plot_scan(frames_forth, tforms_forth, tforms_inv_forth,PATH_SAVE+'/'+folder+'/'+fn.split(".")[0]+'_forth',croped=True)
        px_all_back, py_all_back, pz_all_back = plot_scan_init.plot_scan(frames_back, tforms_back, tforms_inv_back,PATH_SAVE+'/'+folder+'/'+fn.split(".")[0]+'_back',croped=True)

        # plot_scan_init.Plot_Video(frames_forth, tforms_forth, tforms_inv_forth,fn.split(".")[0]+'_forth',px_all_forth, py_all_forth, pz_all_forth,croped=True)
        # plot_scan_init.Save_Video(fn.split(".")[0]+'_forth',croped=True)
        # plot_scan_init.plot_img_in_2d(frames_forth,fn.split(".")[0]+'_forth',croped=True)
        
        # plot_scan_init.Plot_Video(frames_back, tforms_back, tforms_inv_back,fn.split(".")[0]+'_back',px_all_back, py_all_back, pz_all_back,croped=True)
        # plot_scan_init.Save_Video(fn.split(".")[0]+'_back',croped=True)
        # plot_scan_init.plot_img_in_2d(frames_back,fn.split(".")[0]+'_back',croped=True)
        

        # index_loop_forth,final_idx_forth = plot_scan_init.detect_loop_points(px_all_forth,py_all_forth,pz_all_forth,PATH_SAVE+'/'+folder+'/'+fn.split(".")[0]+'_forth',croped=True)
        relative_transf_6_DOF_each_scan_forth,relative_transf_6_DOF_to_1st_img_each_scan_forth = plot_scan_init.get_transf_para_each_scan(tforms_forth, tforms_inv_forth,PATH_SAVE+'/'+folder+'/'+fn.split(".")[0]+'_forth')
        
        # index_loop_back,final_idx_back = plot_scan_init.detect_loop_points(px_all_back,py_all_back,pz_all_back,PATH_SAVE+'/'+folder+'/'+fn.split(".")[0]+'_back',croped=True)
        relative_transf_6_DOF_each_scan_back,relative_transf_6_DOF_to_1st_img_each_scan_back = plot_scan_init.get_transf_para_each_scan(tforms_back, tforms_inv_back,PATH_SAVE+'/'+folder+'/'+fn.split(".")[0]+'_back')
        

        if 'relative_transf_6_DOF_all_scan_forth' not in globals():
            relative_transf_6_DOF_all_scan_forth = relative_transf_6_DOF_each_scan_forth
        else:
            relative_transf_6_DOF_all_scan_forth = np.concatenate((relative_transf_6_DOF_all_scan_forth, relative_transf_6_DOF_each_scan_forth), axis=0)

        if 'relative_transf_6_DOF_to_1st_img_all_scan_forth' not in globals():
            relative_transf_6_DOF_to_1st_img_all_scan_forth = relative_transf_6_DOF_to_1st_img_each_scan_forth
        else:
            relative_transf_6_DOF_to_1st_img_all_scan_forth = np.concatenate((relative_transf_6_DOF_to_1st_img_all_scan_forth, relative_transf_6_DOF_to_1st_img_each_scan_forth), axis=0)

        
        if 'relative_transf_6_DOF_all_scan_back' not in globals():
            relative_transf_6_DOF_all_scan_back = relative_transf_6_DOF_each_scan_back
        else:
            relative_transf_6_DOF_all_scan_back = np.concatenate((relative_transf_6_DOF_all_scan_back, relative_transf_6_DOF_each_scan_back), axis=0)

        if 'relative_transf_6_DOF_to_1st_img_all_scan_back' not in globals():
            relative_transf_6_DOF_to_1st_img_all_scan_back = relative_transf_6_DOF_to_1st_img_each_scan_back
        else:
            relative_transf_6_DOF_to_1st_img_all_scan_back = np.concatenate((relative_transf_6_DOF_to_1st_img_all_scan_back, relative_transf_6_DOF_to_1st_img_each_scan_back), axis=0)




        with open(os.path.join(os.path.join(DIR_RAW,folder),'valid_frames_sep.csv'), 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            row = [fn,start_idx,index_loop,final_idx, first_missing_idx]
            # write the header
            writer.writerow(row)

        

        num_frames_forth[i_sub,i_scan] = frames_forth.shape[0]
        name_scan_forth[i_sub][i_scan] = fn+'_forth'

        for i_frame in range(frames_forth.shape[0]):
            fh5_frames_forth.create_dataset('/sub%03d_scan%02d_frame%04d' % (i_sub,i_scan,i_frame), frames_forth.shape[1:3], dtype=frames_forth.dtype, data=frames_forth[i_frame,...])
            fh5_frames_forth.create_dataset('/sub%03d_scan%02d_tform%04d' % (i_sub,i_scan,i_frame), tforms_forth.shape[1:3], dtype=tforms_forth.dtype, data=tforms_forth[i_frame,...])
            fh5_frames_forth.create_dataset('/sub%03d_scan%02d_tform_inv%04d' % (i_sub,i_scan,i_frame), tforms_inv_forth.shape[1:3], dtype=tforms_inv_forth.dtype, data=tforms_inv_forth[i_frame,...])
            # if Extract_Feats:

        if FLAG_SCANFILE:
            fh5_scans_forth.create_dataset('/sub%03d_frames%02d' % (i_sub,i_scan), frames_forth.shape, dtype=frames_forth.dtype, data=frames_forth)
            fh5_scans_forth.create_dataset('/sub%03d_tforms%02d' % (i_sub,i_scan), tforms_forth.shape, dtype=tforms_forth.dtype, data=tforms_forth)
            fh5_scans_forth.create_dataset('/sub%03d_tforms_inv%02d' % (i_sub,i_scan), tforms_inv_forth.shape, dtype=tforms_inv_forth.dtype, data=tforms_inv_forth)


        num_frames_back[i_sub,i_scan] = frames_back.shape[0]
        name_scan_back[i_sub][i_scan] = fn+'_back'
        for i_frame in range(frames_back.shape[0]):
            fh5_frames_back.create_dataset('/sub%03d_scan%02d_frame%04d' % (i_sub,i_scan,i_frame), frames_back.shape[1:3], dtype=frames_back.dtype, data=frames_back[i_frame,...])
            fh5_frames_back.create_dataset('/sub%03d_scan%02d_tform%04d' % (i_sub,i_scan,i_frame), tforms_back.shape[1:3], dtype=tforms_back.dtype, data=tforms_back[i_frame,...])
            fh5_frames_back.create_dataset('/sub%03d_scan%02d_tform_inv%04d' % (i_sub,i_scan,i_frame), tforms_inv_back.shape[1:3], dtype=tforms_inv_back.dtype, data=tforms_inv_back[i_frame,...])
            # if Extract_Feats:

        if FLAG_SCANFILE:
            fh5_scans_back.create_dataset('/sub%03d_frames%02d' % (i_sub,i_scan), frames_back.shape, dtype=frames_back.dtype, data=frames_back)
            fh5_scans_back.create_dataset('/sub%03d_tforms%02d' % (i_sub,i_scan), tforms_back.shape, dtype=tforms_back.dtype, data=tforms_back)
            fh5_scans_back.create_dataset('/sub%03d_tforms_inv%02d' % (i_sub,i_scan), tforms_inv_back.shape, dtype=tforms_inv_back.dtype, data=tforms_inv_back)
        
        
        num_frames_forth_back[i_sub,i_scan*2] = frames_forth.shape[0]
        name_scan_forth_back[i_sub][i_scan*2] = fn+'_forth'

        for i_frame in range(frames_forth.shape[0]):
            fh5_frames_forth_back.create_dataset('/sub%03d_scan%02d_frame%04d' % (i_sub,i_scan*2,i_frame), frames_forth.shape[1:3], dtype=frames_forth.dtype, data=frames_forth[i_frame,...])
            fh5_frames_forth_back.create_dataset('/sub%03d_scan%02d_tform%04d' % (i_sub,i_scan*2,i_frame), tforms_forth.shape[1:3], dtype=tforms_forth.dtype, data=tforms_forth[i_frame,...])
            fh5_frames_forth_back.create_dataset('/sub%03d_scan%02d_tform_inv%04d' % (i_sub,i_scan*2,i_frame), tforms_inv_forth.shape[1:3], dtype=tforms_inv_forth.dtype, data=tforms_inv_forth[i_frame,...])
            # if Extract_Feats:

        if FLAG_SCANFILE:
            fh5_scans_forth_back.create_dataset('/sub%03d_frames%02d' % (i_sub,i_scan*2), frames_forth.shape, dtype=frames_forth.dtype, data=frames_forth)
            fh5_scans_forth_back.create_dataset('/sub%03d_tforms%02d' % (i_sub,i_scan*2), tforms_forth.shape, dtype=tforms_forth.dtype, data=tforms_forth)
            fh5_scans_forth_back.create_dataset('/sub%03d_tforms_inv%02d' % (i_sub,i_scan*2), tforms_inv_forth.shape, dtype=tforms_inv_forth.dtype, data=tforms_inv_forth)

        num_frames_forth_back[i_sub,i_scan*2+1] = frames_back.shape[0]
        name_scan_forth_back[i_sub][i_scan*2+1] = fn+'_back'
        for i_frame in range(frames_back.shape[0]):
            fh5_frames_forth_back.create_dataset('/sub%03d_scan%02d_frame%04d' % (i_sub,i_scan*2+1,i_frame), frames_back.shape[1:3], dtype=frames_back.dtype, data=frames_back[i_frame,...])
            fh5_frames_forth_back.create_dataset('/sub%03d_scan%02d_tform%04d' % (i_sub,i_scan*2+1,i_frame), tforms_back.shape[1:3], dtype=tforms_back.dtype, data=tforms_back[i_frame,...])
            fh5_frames_forth_back.create_dataset('/sub%03d_scan%02d_tform_inv%04d' % (i_sub,i_scan*2+1,i_frame), tforms_inv_back.shape[1:3], dtype=tforms_inv_back.dtype, data=tforms_inv_back[i_frame,...])
            # if Extract_Feats:

        if FLAG_SCANFILE:
            fh5_scans_forth_back.create_dataset('/sub%03d_frames%02d' % (i_sub,i_scan*2+1), frames_back.shape, dtype=frames_back.dtype, data=frames_back)
            fh5_scans_forth_back.create_dataset('/sub%03d_tforms%02d' % (i_sub,i_scan*2+1), tforms_back.shape, dtype=tforms_back.dtype, data=tforms_back)
            fh5_scans_forth_back.create_dataset('/sub%03d_tforms_inv%02d' % (i_sub,i_scan*2+1), tforms_inv_back.shape, dtype=tforms_inv_back.dtype, data=tforms_inv_back)



    fig, ax = plt.subplots()
    ax.boxplot(relative_transf_6_DOF_all_scan_forth)
    fig.savefig(os.path.join(DIR_RAW,folder)+'/'+'6DOF_between_imgs_all_scan_forth.png')
    plt.close()

    fig1, ax1 = plt.subplots()
    ax1.boxplot(relative_transf_6_DOF_to_1st_img_all_scan_forth)
    fig1.savefig(os.path.join(DIR_RAW,folder)+'/'+'6DOF_to_1st_img_all_scan_forth.png')
    plt.close()

    fig0, ax0 = plt.subplots()
    ax0.boxplot(relative_transf_6_DOF_all_scan_back)
    fig0.savefig(os.path.join(DIR_RAW,folder)+'/'+'6DOF_between_imgs_all_scan_back.png')
    plt.close()

    fig11, ax11 = plt.subplots()
    ax11.boxplot(relative_transf_6_DOF_to_1st_img_all_scan_back)
    fig11.savefig(os.path.join(DIR_RAW,folder)+'/'+'6DOF_to_1st_img_all_scan_back.png')
    plt.close()


    if 'relative_transf_6_DOF_all_scan_all_sub_forth' not in globals():
        relative_transf_6_DOF_all_scan_all_sub_forth = relative_transf_6_DOF_all_scan_forth
    else:
        relative_transf_6_DOF_all_scan_all_sub_forth = np.concatenate((relative_transf_6_DOF_all_scan_all_sub_forth, relative_transf_6_DOF_all_scan_forth), axis=0)

    if 'relative_transf_6_DOF_to_1st_img_all_scan_all_sub_forth' not in globals():
        relative_transf_6_DOF_to_1st_img_all_scan_all_sub_forth = relative_transf_6_DOF_to_1st_img_all_scan_forth
    else:
        relative_transf_6_DOF_to_1st_img_all_scan_all_sub_forth = np.concatenate((relative_transf_6_DOF_to_1st_img_all_scan_all_sub_forth, relative_transf_6_DOF_to_1st_img_all_scan_forth), axis=0)

    if 'relative_transf_6_DOF_all_scan_all_sub_back' not in globals():
        relative_transf_6_DOF_all_scan_all_sub_back = relative_transf_6_DOF_all_scan_back
    else:
        relative_transf_6_DOF_all_scan_all_sub_back = np.concatenate((relative_transf_6_DOF_all_scan_all_sub_back, relative_transf_6_DOF_all_scan_back), axis=0)

    if 'relative_transf_6_DOF_to_1st_img_all_scan_all_sub_back' not in globals():
        relative_transf_6_DOF_to_1st_img_all_scan_all_sub_back = relative_transf_6_DOF_to_1st_img_all_scan_back
    else:
        relative_transf_6_DOF_to_1st_img_all_scan_all_sub_back = np.concatenate((relative_transf_6_DOF_to_1st_img_all_scan_all_sub_back, relative_transf_6_DOF_to_1st_img_all_scan_back), axis=0)




fig2, ax2 = plt.subplots()
ax2.boxplot(relative_transf_6_DOF_all_scan_all_sub_forth)
fig2.savefig(DIR_RAW+'/'+'6DOF_between_imgs_all_scan_all_sub_forth.png')
plt.close()

fig3, ax3 = plt.subplots()
ax3.boxplot(relative_transf_6_DOF_to_1st_img_all_scan_all_sub_forth)
fig3.savefig(DIR_RAW+'/'+'6DOF_to_1st_img_all_scan_all_sub_forth.png')
plt.close()

fig20, ax20 = plt.subplots()
ax20.boxplot(relative_transf_6_DOF_all_scan_all_sub_back)
fig20.savefig(DIR_RAW+'/'+'6DOF_between_imgs_all_scan_all_sub_back.png')
plt.close()

fig30, ax30 = plt.subplots()
ax30.boxplot(relative_transf_6_DOF_to_1st_img_all_scan_all_sub_back)
fig30.savefig(DIR_RAW+'/'+'6DOF_to_1st_img_all_scan_all_sub_back.png')
plt.close()
        
fh5_frames_back.create_dataset('num_frames', num_frames_back.shape, dtype=num_frames_back.dtype, data=num_frames_back)
fh5_frames_back.create_dataset('sub_folders', len(folders_subject), data=folders_subject)
fh5_frames_back.create_dataset('frame_size', 2, data=frames_back.shape[1:3])
fh5_frames_back.create_dataset('name_scan', tuple((len(name_scan_back),len(name_scan_back[0]))), data=name_scan_back)

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

fh5_frames_forth.flush()
fh5_frames_forth.close()

if FLAG_SCANFILE:
    fh5_scans_forth.create_dataset('num_frames', num_frames_forth.shape, dtype=num_frames_forth.dtype, data=num_frames_forth)
    fh5_scans_forth.create_dataset('sub_folders', len(folders_subject), data=folders_subject)
    fh5_scans_forth.create_dataset('frame_size', 2, data=frames_forth.shape[1:3])
    fh5_scans_forth.create_dataset('name_scan', tuple((len(name_scan_forth), len(name_scan_forth[0]))), data=name_scan_forth)

    fh5_scans_forth.flush()
    fh5_scans_forth.close()


fh5_frames_forth_back.create_dataset('num_frames', num_frames_forth_back.shape, dtype=num_frames_forth_back.dtype, data=num_frames_forth_back)
fh5_frames_forth_back.create_dataset('sub_folders', len(folders_subject), data=folders_subject)
fh5_frames_forth_back.create_dataset('frame_size', 2, data=frames_forth.shape[1:3])
fh5_frames_forth_back.create_dataset('name_scan', tuple((len(name_scan_forth_back),len(name_scan_forth_back[0]))), data=name_scan_forth_back)

fh5_frames_forth_back.flush()
fh5_frames_forth_back.close()

if FLAG_SCANFILE:
    fh5_scans_forth_back.create_dataset('num_frames', num_frames_forth_back.shape, dtype=num_frames_forth_back.dtype, data=num_frames_forth_back)
    fh5_scans_forth_back.create_dataset('sub_folders', len(folders_subject), data=folders_subject)
    fh5_scans_forth_back.create_dataset('frame_size', 2, data=frames_forth.shape[1:3])
    fh5_scans_forth_back.create_dataset('name_scan', tuple((len(name_scan_forth_back), len(name_scan_forth_back[0]))), data=name_scan_forth_back)

    fh5_scans_forth_back.flush()
    fh5_scans_forth_back.close()



print('Saved at: %s' % os.path.join(PATH_SAVE,'frames_res{}_forth'.format(RESAMPLE_FACTOR)+".h5"))
print('Saved at: %s' % os.path.join(PATH_SAVE,'frames_res{}_forth_back'.format(RESAMPLE_FACTOR)+".h5"))
print('Saved at: %s' % os.path.join(PATH_SAVE,'frames_res{}_back'.format(RESAMPLE_FACTOR)+".h5"))
if FLAG_SCANFILE:
    print('Saved at: %s' % os.path.join(PATH_SAVE,'scans_res{}_back'.format(RESAMPLE_FACTOR)+".h5"))
    print('Saved at: %s' % os.path.join(PATH_SAVE,'scans_res{}_forth_back'.format(RESAMPLE_FACTOR)+".h5"))
    print('Saved at: %s' % os.path.join(PATH_SAVE,'scans_res{}_forth'.format(RESAMPLE_FACTOR)+".h5"))
