
import numpy as np
import SimpleITK as sitk
import cv2,os,sys,csv
import torch
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import re,math
from PIL import Image
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
import torch
from scipy.stats.stats import pearsonr   
from scipy.stats import linregress

sys.path.append("/raid/candi/Qi/freehand-ultrasound")
sys.path.append("/raid/candi/Qi/freehand-ultrasound")

from transform import LabelTransform



def frame_resize(image, resample_factor):
    # frame_resize = lambda im : cv2.resize(im, None, fx=1/RESAMPLE_FACTOR, fy=1/RESAMPLE_FACTOR, interpolation = cv2.INTER_LINEAR)
    return cv2.resize(
        image, 
        dsize=None, 
        fx=1/resample_factor, 
        fy=1/resample_factor, 
        interpolation = cv2.INTER_LINEAR
        )


# def read_scan_crop_indices_file(filename, num_scans):
#     fid_xls = openpyxl.load_workbook(filename).active
#     return list(fid_xls.iter_rows(values_only=True))[1:num_scans+1]


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

def read_frame_transform_croped(filename, scan_crop_indices, resample_factor=1, delay_tform=0):
    scan_crop_indices=list(map(int, scan_crop_indices))

    get_transform = lambda ii, image : list(map(float,image.GetMetaData('Seq_Frame{:04d}_ProbeToTrackerTransform'.format(ii)).split(' ')))
    get_transform_status = lambda ii, image : image.GetMetaData('Seq_Frame{:04d}_ProbeToTrackerTransformStatus'.format(ii))=="OK"

    image = sitk.ReadImage(filename)  # imageIO="MetaImageIO"    
    frames = sitk.GetArrayFromImage(image)  # [no_frames,h,w] - nb. size = image.GetSize()  # [w,h,no_frames]

    if (scan_crop_indices[2]+delay_tform) > frames.shape[0]:
        scan_crop_indices[2] = frames.shape[0]
        print("WARNING: scan_crop_indices has been reduced due to delayed transform.")

    frames = frames[scan_crop_indices[0]:scan_crop_indices[2],...] 
    tforms = [get_transform(ii,image) for ii in range(scan_crop_indices[0]+delay_tform,scan_crop_indices[2]+delay_tform)]
    tforms = np.stack([np.array(t,dtype=np.float32).reshape(4,4) for t in tforms],axis=0)

    tform_status = [get_transform_status(ii,image) for ii in range(scan_crop_indices[0]+delay_tform,scan_crop_indices[2]+delay_tform)]
    if not all(tform_status):
        frames = frames[tform_status,:,:]
        tforms = tforms[tform_status,:,:]
    
    if resample_factor != 1:
        frames = np.stack(
            [frame_resize(frames[ii,...], resample_factor) for ii in range(frames.shape[0])], 
            axis=0
            )  # resample on 2D frames
    
    return frames, tforms, tform_status

def read_frame_transform(filename, resample_factor=1, delay_tform=0):


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


    return frames, tforms, tform_status, first_missing_status_idx


class PlotScan():
    def __init__(self, FILENAME_CALIB,RESAMPLE_FACTOR, Image_Shape=(120,160)):
        
        self.RESAMPLE_FACTOR=RESAMPLE_FACTOR
        self.Image_Shape=Image_Shape
        self.pixel_points = reference_image_points(Image_Shape, 2)#.to(self.device)
        self.all_points_in_img = reference_image_points(Image_Shape, Image_Shape)
        self.tform_calib_scale, self.tform_calib_R_T, self.tform_calib = read_calib_matrices(
            filename_calib=FILENAME_CALIB,
            resample_factor=RESAMPLE_FACTOR,
            device='cpu'#self.device''
        )


        self.transform_label = LabelTransform(
                label_type="point",  # for plotting
                pairs=torch.tensor([0, 1])[None,],
                image_points=self.pixel_points,
                in_image_coords=True,  # for plotting
                tform_image_to_tool=self.tform_calib,
                tform_image_mm_to_tool=self.tform_calib_R_T
            )
        
        self.transform_label_all_pixel = LabelTransform(
                label_type="point",  # for plotting
                pairs=torch.tensor([0, 1])[None,],
                image_points=self.all_points_in_img,
                in_image_coords=True,  # for plotting
                tform_image_to_tool=self.tform_calib,
                tform_image_mm_to_tool=self.tform_calib_R_T
            )
        self.fps=30 # frame per second



    def plot_scan(self,frames, tforms, tforms_inv,saved_folder_name,croped=False):
        # plot scan moving of a scan

        if not os.path.exists(saved_folder_name):
            os.makedirs(saved_folder_name)

        if croped:   
            scan_name = saved_folder_name+'/'+get_last_folder(saved_folder_name)+'_first_img_red_last_img_green_croped.png'
        
        else:
            scan_name = saved_folder_name+'/'+get_last_folder(saved_folder_name)+'_first_img_red_last_img_green.png'


        
        # output_video_path = saved_folder_name+'/'+saved_folder_name+'.mp4'
        
        
        # # Create a video writer object
        # fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Specify the codec for the output video
        # video_writer = cv2.VideoWriter(output_video_path, fourcc, self.fps, self.Image_Shape)
        
        # compute the range of x,y,z axis
        px_all,py_all,pz_all=[],[],[]
           

        #  plot the trajectory of a scan

        idx_f0 = 0  #   # this is the reference starting frame for network prediction

        idx_p0 = idx_f0 # + torch.squeeze(self.data_pairs[PAIR_INDEX])[0]  # this is the reference frame for transformaing others to
        idx_p1 = idx_f0 + 1

        # plot the frame 0
        px, py, pz = [np.matmul(self.tform_calib_scale, np.matmul(np.array([[self.RESAMPLE_FACTOR, 0, 0, 0], [0, self.RESAMPLE_FACTOR, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],np.float32), self.all_points_in_img))[ii,].reshape(self.Image_Shape) for ii in range(3)]
        pix_intensities = (torch.from_numpy(frames)[idx_p0, ..., None].float() / 255).cpu().expand(-1, -1, 3).numpy()
        fx, fy, fz = [np.matmul(self.tform_calib_scale, np.matmul(torch.from_numpy(np.array([[self.RESAMPLE_FACTOR, 0, 0, 0], [0, self.RESAMPLE_FACTOR, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],np.float32)), self.pixel_points.cpu()))[ii,].reshape(2, 2) for ii in range(3)]

        px_all.append(fx.flatten().tolist())
        py_all.append(fy.flatten().tolist())
        pz_all.append(fz.flatten().tolist())


        fig = plt.figure() # figsize=(10, 10)
        ax = fig.add_subplot(projection='3d')

        # fig_video = plt.figure()
        # ax_video = fig_video.add_subplot(projection='3d')

        # ax = plt.axes(projection='3d')
        # image_num=1
        ax.plot_surface(px, py, pz, facecolors=pix_intensities, linewidth=0,  antialiased=True, alpha=0.7)
        ax.plot_surface(fx, fy, fz, edgecolor='r', linewidth=1, alpha=0.5, antialiased=True)

        # ax_video.plot_surface(px, py, pz, facecolors=pix_intensities, linewidth=0, edgecolors=None, antialiased=True, alpha=0.2)
        # ax_video.savefig(image_num)
        # plot the first number of frames to make sure different intervals have the same start frames
        tforms_val, tforms_inv_val = (t[[0, idx_p0], ...] for t in [tforms, tforms_inv])
        label = self.transform_label(torch.from_numpy(tforms_val).unsqueeze(0).cpu(), torch.from_numpy(tforms_inv_val).unsqueeze(0).cpu())
        px, py, pz = [label[:, :, ii, :] for ii in range(3)]
        ax.scatter(px, py, pz, c='r', alpha=0.2, s=2)  #

        fx, fy, fz = [np.matmul(self.tform_calib_scale, np.matmul(np.array(
            [[self.RESAMPLE_FACTOR, 0, 0, 0], [0, self.RESAMPLE_FACTOR, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32), self.pixel_points))[ii,].reshape(2, 2).cpu() for ii in range(3)]

        while 1:
            if (idx_f0 + 1) >= frames.shape[0]:
                break


            # label -> points in image coords, wrt. the "unchanged" reference starting frame, idx_p0
            tforms_val, tforms_inv_val = (t[[idx_p0, idx_p1], ...] for t in [tforms, tforms_inv])
            label = self.transform_label(torch.from_numpy(tforms_val).unsqueeze(0).cpu(), torch.from_numpy(tforms_inv_val).unsqueeze(0).cpu())
            px, py, pz = [label[:, :, ii, :] for ii in range(3)]
            ax.scatter(px, py, pz, c='r', alpha=0.2, s=2)

            px_all.append(torch.squeeze(px).tolist())
            py_all.append(torch.squeeze(py).tolist())
            pz_all.append(torch.squeeze(pz).tolist())

            # pix_intensities = (torch.from_numpy(frames)[idx_p1, ..., None].float() / 255).cpu().expand(-1, -1, 3).numpy()
            
            
            # ax_video.plot_surface(torch.squeeze(px).reshape(2,2), torch.squeeze(py).reshape(2,2), torch.squeeze(pz).reshape(2,2), facecolors=pix_intensities, linewidth=0, edgecolors=None, antialiased=True, alpha=0.2)
            # image_num += 1
            # ax_video.savefig(image_num)
            # video_writer.write(ax_video)



          
            idx_f0 += 1
            idx_p1 += 1
            

        # plot the last image
        
        px, py, pz = px.reshape(2,2), py.reshape(2,2), pz.reshape(2,2)
        ax.plot_surface(px, py, pz, edgecolor='g', linewidth=1, alpha=0.2, antialiased=True)
        # image_num += 1
        # ax_video.savefig(image_num)

        # ax.legend()


        # plt.show()
        
        plt.savefig(scan_name)
        if croped:
            plt.savefig('/raid/Qi/public_data/forearm_US_large_dataset/check_volume/'+("_").join(saved_folder_name[-35:].split("/"))+'_first_img_red_last_img_green_croped.png')

        plt.close(fig)

        return px_all, py_all, pz_all

        # video_writer.release()

    def plot_scan_different_color_4_loop(self,frames, tforms, tforms_inv,saved_folder_name,loop_idx,croped=False):
        # plot scan moving of a scan

        if not os.path.exists(saved_folder_name):
            os.makedirs(saved_folder_name)

        if croped:   
            scan_name = saved_folder_name+'/'+get_last_folder(saved_folder_name)+'_1_half_red_2_half_green_croped.png'
        else:
            scan_name = saved_folder_name+'/'+get_last_folder(saved_folder_name)+'_1_half_red_2_half_green.png'


        
           

        #  plot the trajectory of a scan

        idx_f0 = 0  #   # this is the reference starting frame for network prediction

        idx_p0 = idx_f0 # + torch.squeeze(self.data_pairs[PAIR_INDEX])[0]  # this is the reference frame for transformaing others to
        idx_p1 = idx_f0 + 1

        # plot the frame 0
        px, py, pz = [np.matmul(self.tform_calib_scale, np.matmul(np.array([[self.RESAMPLE_FACTOR, 0, 0, 0], [0, self.RESAMPLE_FACTOR, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],np.float32), self.all_points_in_img))[ii,].reshape(self.Image_Shape) for ii in range(3)]
        pix_intensities = (torch.from_numpy(frames)[idx_p0, ..., None].float() / 255).cpu().expand(-1, -1, 3).numpy()
        fx, fy, fz = [np.matmul(self.tform_calib_scale, np.matmul(torch.from_numpy(np.array([[self.RESAMPLE_FACTOR, 0, 0, 0], [0, self.RESAMPLE_FACTOR, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],np.float32)), self.pixel_points.cpu()))[ii,].reshape(2, 2) for ii in range(3)]



        fig = plt.figure() # figsize=(10, 10)
        ax = fig.add_subplot(projection='3d')

        
        ax.plot_surface(px, py, pz, facecolors=pix_intensities, linewidth=0,  antialiased=True, alpha=0.7)
        ax.plot_surface(fx, fy, fz, edgecolor='r', linewidth=1, alpha=0.5, antialiased=True)

        
        # plot the first number of frames to make sure different intervals have the same start frames
        tforms_val, tforms_inv_val = (t[[0, idx_p0], ...] for t in [tforms, tforms_inv])
        label = self.transform_label(torch.from_numpy(tforms_val).unsqueeze(0).cpu(), torch.from_numpy(tforms_inv_val).unsqueeze(0).cpu())
        px, py, pz = [label[:, :, ii, :] for ii in range(3)]
        ax.scatter(px, py, pz, c='r', alpha=0.2, s=2)  #

        fx, fy, fz = [np.matmul(self.tform_calib_scale, np.matmul(np.array(
            [[self.RESAMPLE_FACTOR, 0, 0, 0], [0, self.RESAMPLE_FACTOR, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32), self.pixel_points))[ii,].reshape(2, 2).cpu() for ii in range(3)]

        while 1:
            if (idx_f0 + 1) >= frames.shape[0]:
                break

            if idx_p1 < loop_idx:
                color = 'r'
            else:
                color = 'g'


            # label -> points in image coords, wrt. the "unchanged" reference starting frame, idx_p0
            tforms_val, tforms_inv_val = (t[[idx_p0, idx_p1], ...] for t in [tforms, tforms_inv])
            label = self.transform_label(torch.from_numpy(tforms_val).unsqueeze(0).cpu(), torch.from_numpy(tforms_inv_val).unsqueeze(0).cpu())
            px, py, pz = [label[:, :, ii, :] for ii in range(3)]
            ax.scatter(px, py, pz, c=color, alpha=0.2, s=2)

    

          
            idx_f0 += 1
            idx_p1 += 1
            

        # plot the last image
        pix_intensities = (torch.from_numpy(frames)[idx_p1-1, ..., None].float() / 255).cpu().expand(-1, -1, 3).numpy()
        label_all_pixel = self.transform_label_all_pixel(torch.from_numpy(tforms_val).unsqueeze(0).cpu(), torch.from_numpy(tforms_inv_val).unsqueeze(0).cpu())
        px_all_pixel, py_all_pixel, pz_all_pixel = [label_all_pixel[:, :, ii, :] for ii in range(3)]
            
        
        px, py, pz = px.reshape(2,2), py.reshape(2,2), pz.reshape(2,2)
        ax.plot_surface(torch.squeeze(px_all_pixel).reshape(self.Image_Shape), torch.squeeze(py_all_pixel).reshape(self.Image_Shape), torch.squeeze(pz_all_pixel).reshape(self.Image_Shape), facecolors=pix_intensities, linewidth=0,  antialiased=True, alpha=0.5)
       


        # plt.show()
        
        plt.savefig(scan_name)
        plt.close(fig)

    

    def Plot_Video(self,frames, tforms, tforms_inv,saved_folder_name, px_all, py_all, pz_all,croped=False):
        # plot scan moving of a scan

        if not os.path.exists(saved_folder_name):
            os.makedirs(saved_folder_name)
        
        if croped:
            if not os.path.exists(saved_folder_name+'/'+'img_in_3D_croped'):
                os.makedirs(saved_folder_name+'/'+'img_in_3D_croped')

        else:
            if not os.path.exists(saved_folder_name+'/'+'img_in_3D'):
                os.makedirs(saved_folder_name+'/'+'img_in_3D')

        px_current,py_current,pz_current=[],[],[]
        
        #  plot the trajectory of a scan

        idx_f0 = 0  #   # this is the reference starting frame for network prediction

        idx_p0 = idx_f0 # + torch.squeeze(self.data_pairs[PAIR_INDEX])[0]  # this is the reference frame for transformaing others to
        idx_p1 = idx_f0 + 1

        # plot the frame 0
        px, py, pz = [np.matmul(self.tform_calib_scale, np.matmul(np.array([[self.RESAMPLE_FACTOR, 0, 0, 0], [0, self.RESAMPLE_FACTOR, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],np.float32), self.all_points_in_img))[ii,].reshape(self.Image_Shape) for ii in range(3)]
        pix_intensities = ((torch.from_numpy(frames)[idx_p0, ..., None].float() /255)).cpu().expand(-1, -1, 3).numpy()
        fx, fy, fz = [np.matmul(self.tform_calib_scale, np.matmul(torch.from_numpy(np.array([[self.RESAMPLE_FACTOR, 0, 0, 0], [0, self.RESAMPLE_FACTOR, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],np.float32)), self.pixel_points.cpu()))[ii,].reshape(2, 2) for ii in range(3)]

        px_current.append(fx.flatten().tolist())
        py_current.append(fy.flatten().tolist())
        pz_current.append(fz.flatten().tolist())
        px_1st, py_1st,pz_1st = px, py, pz 

        fig = plt.figure() # figsize=(10, 10)
        ax = fig.add_subplot(projection='3d')

        
        image_num=1
        # px_mesh, py_mesh = np.meshgrid(range(px.shape[0]), range(px.shape[1]),indexing='xy')

        ax.plot_surface(px, py, pz, facecolors=pix_intensities, linewidth=0, antialiased=True, alpha=0.5)
        
        ax.set_zlim3d(min(np.array(pz_all).flatten()),max(np.array(pz_all).flatten()))                   
        ax.set_ylim3d(min(np.array(py_all).flatten()),max(np.array(py_all).flatten()))                      
        ax.set_xlim3d(min(np.array(px_all).flatten()),max(np.array(px_all).flatten()))   
        
        if croped:
            plt.savefig(saved_folder_name+'/'+'img_in_3D_croped'+'/'+f'{image_num:04d}.png')
        else:
            plt.savefig(saved_folder_name+'/'+'img_in_3D'+'/'+f'{image_num:04d}.png')


        plt.close()
        # ax.plot_surface(fx, fy, fz, edgecolor='r', linewidth=1, alpha=0.2, antialiased=True)

        # ax_video.plot_surface(px, py, pz, facecolors=pix_intensities, linewidth=0, edgecolors=None, antialiased=True, alpha=0.2)
        # ax_video.savefig(image_num)
        # plot the first number of frames to make sure different intervals have the same start frames
        tforms_val, tforms_inv_val = (t[[0, idx_p0], ...] for t in [tforms, tforms_inv])
        label = self.transform_label(torch.from_numpy(tforms_val).unsqueeze(0).cpu(), torch.from_numpy(tforms_inv_val).unsqueeze(0).cpu())
        px, py, pz = [label[:, :, ii, :] for ii in range(3)]
        # ax.scatter(px, py, pz, c='r', alpha=0.2, s=2)  #

        fx, fy, fz = [np.matmul(self.tform_calib_scale, np.matmul(np.array(
            [[self.RESAMPLE_FACTOR, 0, 0, 0], [0, self.RESAMPLE_FACTOR, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32), self.pixel_points))[ii,].reshape(2, 2).cpu() for ii in range(3)]



        while 1:
            if (idx_f0 + 1) >= frames.shape[0]:
                break


            # label -> points in image coords, wrt. the "unchanged" reference starting frame, idx_p0
            tforms_val, tforms_inv_val = (t[[idx_p0, idx_p1], ...] for t in [tforms, tforms_inv])
            label = self.transform_label(torch.from_numpy(tforms_val).unsqueeze(0).cpu(), torch.from_numpy(tforms_inv_val).unsqueeze(0).cpu())
            px, py, pz = [label[:, :, ii, :] for ii in range(3)]
            # ax.scatter(px, py, pz, c='r', alpha=0.5, s=2)

            fig = plt.figure() # figsize=(10, 10)
            ax = fig.add_subplot(projection='3d')

            px_current.append(torch.squeeze(px).tolist())
            py_current.append(torch.squeeze(py).tolist())
            pz_current.append(torch.squeeze(pz).tolist())

            ax.scatter(px_current, py_current, pz_current, c='r', alpha=0.5, s=2)


            label_all_pixel = self.transform_label_all_pixel(torch.from_numpy(tforms_val).unsqueeze(0).cpu(), torch.from_numpy(tforms_inv_val).unsqueeze(0).cpu())
            px_all_pixel, py_all_pixel, pz_all_pixel = [label_all_pixel[:, :, ii, :] for ii in range(3)]
            
            ax.plot_surface(px_1st, py_1st, pz_1st, facecolors=pix_intensities, linewidth=0,  antialiased=True, alpha=0.5)

            surface = ax.plot_surface(torch.squeeze(px_all_pixel).reshape(self.Image_Shape), torch.squeeze(py_all_pixel).reshape(self.Image_Shape), torch.squeeze(pz_all_pixel).reshape(self.Image_Shape), facecolors=pix_intensities, linewidth=0,  antialiased=True, alpha=0.8)

            ax.set_zlim3d(min(np.array(pz_all).flatten()),max(np.array(pz_all).flatten()))                   
            ax.set_ylim3d(min(np.array(py_all).flatten()),max(np.array(py_all).flatten()))                      
            ax.set_xlim3d(min(np.array(px_all).flatten()),max(np.array(px_all).flatten()))   
            
            image_num += 1

            if croped:
                plt.savefig(saved_folder_name+'/'+'img_in_3D_croped'+'/'+f'{(image_num):04d}.png')
            else:
                plt.savefig(saved_folder_name+'/'+'img_in_3D'+'/'+f'{(image_num):04d}.png')


            # surface.set_alpha(0.01) 
            
            plt.close()
            # pix_intensities = (torch.from_numpy(frames)[idx_p1, ..., None].float() / 255).cpu().expand(-1, -1, 3).numpy()
            
            
            # ax_video.plot_surface(torch.squeeze(px).reshape(2,2), torch.squeeze(py).reshape(2,2), torch.squeeze(pz).reshape(2,2), facecolors=pix_intensities, linewidth=0, edgecolors=None, antialiased=True, alpha=0.2)
            # image_num += 1
            # ax_video.savefig(image_num)
            # video_writer.write(ax_video)



          
            idx_f0 += 1
            idx_p1 += 1
            

        # # plot the last image
        # px, py, pz = px.reshape(2, 2), py.reshape(2, 2), pz.reshape(2, 2)
        # ax.plot_surface(px, py, pz, edgecolor='g', linewidth=1, alpha=0.5, antialiased=True)
        # # image_num += 1
        # ax_video.savefig(image_num)

        # ax.legend()

    
        # plt.show()
        # image_num += 1
        # plt.savefig(saved_folder_name+'/'+'img_in_3D'+'/'+f'{str(image_num):04d}.png')
        # plt.close()

    def Save_Video(self,saved_folder_name,croped=False):
        # save into video

        # Video settings
        if croped:
            output_video_path = saved_folder_name+'/'+get_last_folder(saved_folder_name)+"_croped.mp4"  # Output video path
        else:
            output_video_path = saved_folder_name+'/'+get_last_folder(saved_folder_name)+".mp4"  # Output video path

        fps = 30  # Frames per second (FPS)

        # Get the dimensions of the first image
        if croped:

            video_imgs = [f for f in os.listdir(saved_folder_name+'/'+'img_in_3D_croped') if f.endswith('.png') if re.match(r'^\d', f)]
        else:
            video_imgs = [f for f in os.listdir(saved_folder_name+'/'+'img_in_3D') if f.endswith('.png') if re.match(r'^\d', f)]

        video_imgs = sorted(video_imgs, key=lambda x: int(os.path.splitext(x)[0]))

        if croped:
            first_image = cv2.imread(saved_folder_name+'/'+'img_in_3D_croped'+'/'+video_imgs[0])
        else:
            first_image = cv2.imread(saved_folder_name+'/'+'img_in_3D'+'/'+video_imgs[0])

        height, width, _ = first_image.shape

        # Create a video writer object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Specify the codec for the output video
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Iterate over the image paths and write each image to the video
        for image_path in video_imgs:
            if croped:
                image = cv2.imread(saved_folder_name+'/'+'img_in_3D_croped'+'/'+image_path)
            else:
                image = cv2.imread(saved_folder_name+'/'+'img_in_3D'+'/'+image_path)

            # Ensure the image was read successfully
            if image is not None:
                video_writer.write(image)
            else:
                print(f"Error reading image: {image_path}")

        # Release the video writer and close the output video file
        video_writer.release()

        print("Video generation completed!")


    def plot_img_in_2d(self,frames,saved_folder_name,croped=False):
        # plot imag in 2D
        if croped:
            if not os.path.exists(saved_folder_name+'/'+'img_in_2D_croped'):
                os.makedirs(saved_folder_name+'/'+'img_in_2D_croped')
        else:
            if not os.path.exists(saved_folder_name+'/'+'img_in_2D'):
                os.makedirs(saved_folder_name+'/'+'img_in_2D')

        
        for i in range(frames.shape[0]):
            image = Image.fromarray(frames[i,...])
            if croped:
                image.save(saved_folder_name+'/'+'img_in_2D_croped'+'/'+f'{i:04d}.png')
            else:
                image.save(saved_folder_name+'/'+'img_in_2D'+'/'+f'{i:04d}.png')

    def detect_loop_points(self,frames,px_all,py_all,pz_all,saved_folder_name,start_idx=100,croped=False):
        # detect the points in the loop, by using the largest distance from the first image
        px_all = np.array(px_all)
        py_all = np.array(py_all)
        pz_all = np.array(pz_all)

        px_all_1 = px_all[1:,:]
        py_all_1 = py_all[1:,:]
        pz_all_1 = pz_all[1:,:]

        mse_x = np.sum((px_all_1-px_all[0:-1,:])**2,1)
        mse_y = np.sum((py_all_1-py_all[0:-1,:])**2,1)
        mse_z = np.sum((pz_all_1-pz_all[0:-1,:])**2,1)
        mse = np.sqrt((mse_x+mse_y+mse_z)/px_all.shape[1])

        fig, ax = plt.subplots()
        ax.plot(range(len(mse)), mse)

        if not croped:
            # detect the final image
            # diff = abs(mse[start_idx+1:]-mse[start_idx:-1])

            index = None

            # window_len = 2 # can be changed to another number
            # for i in range(start_idx+1+window_len, len(mse)-1):
            #     # check the slope of each segment, which can be computed using corrolation of value with regard to x-axis
            #     # print(linregress(range(window_len),mse[i-window_len:i]).slope)
            #     if linregress(range(window_len),mse[i-window_len:i])[0] > 5:
            #         index = i
            #         # print(index)
            #         break

            #     # previous_mean = np.mean(mse[start_idx:i])
            #     # if mse[i+1] > 20* previous_mean:
            #     #     index = i
            #     #     break


            # check the intensity of each frame, to detect the point when the probe is away from arm
            # check the last 400 frames, find the first intensity that smaller than 15
            mean_inten = np.mean(frames, axis=(1,2))
            index = np.where(mean_inten[-300:]<14)[0][0]
            # fig4, ax4 = plt.subplots()
            # ax4.plot(range(len(mean_inten)),mean_inten,'.-')
            
            # fig4.savefig('mean_inten')



            try:
                
                final_idx = len(mse)-(300-index) -50  # -50 frame give a relatively conservative solution
            except:
                raise('Note: incorrect case')

                # final_idx = len(px_all)-100
            
            # throw less than 200 frames, this is not correct
            # if final_idx< len(px_all)-200:
            #     final_idx = len(px_all)-100

            ax.axvline(x=start_idx, color='r', linestyle='--', alpha=0.3)
            ax.axvline(x=final_idx, color='r', linestyle='--', alpha=0.3)
        else:
            final_idx=None

        ax.set_xlabel('Image index')
        ax.set_ylabel('Distance between two images')
        if croped:
            fig.savefig(saved_folder_name+'/'+'Distance_between_two_imgs_croped.png')

        else:
            fig.savefig(saved_folder_name+'/'+'Distance_between_two_imgs.png')
            fig.savefig('/raid/Qi/public_data/forearm_US_large_dataset/check_volume/'+("_").join(saved_folder_name[-35:].split("/"))+'Distance_between_two_imgs.png')


        plt.close(fig)

        accumulated_dist = np.cumsum(mse)

        fig1, ax1 = plt.subplots()
        ax1.plot(range(len(accumulated_dist)), accumulated_dist)

        if not croped:
            ax1.axvline(x=start_idx, color='r', linestyle='--', alpha=0.3)
            ax1.axvline(x=final_idx, color='r', linestyle='--', alpha=0.3)

        ax1.set_xlabel('Image index')
        ax1.set_ylabel('Accumulated_distance')

        if croped:
            fig1.savefig(saved_folder_name+'/'+'Accumulated_distance_croped.png')
        else:
            fig1.savefig(saved_folder_name+'/'+'Accumulated_distance.png')

        plt.close(fig1)

        # distance from the first image
        if not croped:
            expanded_arr_x = np.tile(px_all[start_idx,:], (px_all.shape[0]-start_idx,1))
            expanded_arr_y = np.tile(py_all[start_idx,:], (py_all.shape[0]-start_idx,1))
            expanded_arr_z = np.tile(pz_all[start_idx,:], (pz_all.shape[0]-start_idx,1))

            mse_x_2 = np.sum((px_all[start_idx:]-expanded_arr_x)**2,1) 
            mse_y_2 = np.sum((py_all[start_idx:]-expanded_arr_y)**2,1)
            mse_z_2 = np.sum((pz_all[start_idx:]-expanded_arr_z)**2,1)
            mse_2 = np.sqrt((mse_x_2+mse_y_2+mse_z_2)/px_all.shape[1])

            index_loop = np.argmax(mse_2[:final_idx-start_idx])+start_idx
        else:

            expanded_arr_x = np.tile(px_all[0,:], (px_all.shape[0],1))
            expanded_arr_y = np.tile(py_all[0,:], (py_all.shape[0],1))
            expanded_arr_z = np.tile(pz_all[0,:], (pz_all.shape[0],1))

            mse_x_2 = np.sum((px_all[0:]-expanded_arr_x)**2,1) 
            mse_y_2 = np.sum((py_all[0:]-expanded_arr_y)**2,1)
            mse_z_2 = np.sum((pz_all[0:]-expanded_arr_z)**2,1)
            mse_2 = np.sqrt((mse_x_2+mse_y_2+mse_z_2)/px_all.shape[1])

            index_loop = np.argmax(mse_2)
            

        fig2, ax2 = plt.subplots()
        ax2.plot(range(start_idx,len(mse_2)+start_idx), mse_2)
        if not croped:
            ax2.axvline(x=start_idx, color='r', linestyle='--', alpha=0.3)
            ax2.axvline(x=final_idx, color='r', linestyle='--', alpha=0.3)
            ax2.axvline(x=index_loop, color='r', linestyle='--', alpha=0.3)
        else:
            ax2.axvline(x=start_idx, color='r', linestyle='--', alpha=0.3)
            ax2.axvline(x=pz_all.shape[0]+start_idx, color='r', linestyle='--', alpha=0.3)
            ax2.axvline(x=index_loop+start_idx, color='r', linestyle='--', alpha=0.3)


        ax2.set_xlabel('Image index')
        ax2.set_ylabel('Distance from the first image')
        if croped:

            fig2.savefig(saved_folder_name+'/'+'Distance_from_the_first_img_croped.png')
        else:
            fig2.savefig(saved_folder_name+'/'+'Distance_from_the_first_img.png')
            fig2.savefig('/raid/Qi/public_data/forearm_US_large_dataset/check_volume/'+("_").join(saved_folder_name[-35:].split("/"))+'Distance_from_the_first_img.png')

        plt.close(fig2)

       
        return index_loop,final_idx
    
    def get_transf_para_each_scan(self, tforms_croped, tforms_inv_croped,saved_dir):
        relative_transf=np.empty((tforms_croped.shape[0]-1, tforms_croped.shape[1],tforms_croped.shape[2]))
        relative_transf_6_DOF=np.empty((tforms_croped.shape[0]-1,6))
        
        relative_transf_to_1st_img=np.empty((tforms_croped.shape[0]-1, tforms_croped.shape[1],tforms_croped.shape[2]))
        relative_transf_6_DOF_to_1st_img=np.empty((tforms_croped.shape[0]-1,6))
        
        for i in range(tforms_croped.shape[0]-1):
            relative_transf[i,...] = np.matmul(tforms_inv_croped[i+1,...], tforms_croped[i,...])
            r = R.from_matrix(relative_transf[i,0:3,0:3])
            relative_transf_6_DOF[i,0:3] = r.as_euler('zyx', degrees=True)
            relative_transf_6_DOF[i,3:6] = relative_transf[i,...][0:3,3]

            relative_transf_to_1st_img[i,...] = np.matmul(tforms_inv_croped[i+1,...], tforms_croped[0,...])
            r = R.from_matrix(relative_transf_to_1st_img[i,0:3,0:3])
            relative_transf_6_DOF_to_1st_img[i,0:3] = r.as_euler('zyx', degrees=True)
            relative_transf_6_DOF_to_1st_img[i,3:6] = relative_transf_to_1st_img[i,...][0:3,3]




        fig, ax = plt.subplots()
        ax.boxplot(relative_transf_6_DOF)
        fig.savefig(saved_dir+'/'+'6DOF_between_imgs_each_scan.png')
        plt.close(fig)

        fig1, ax1 = plt.subplots()
        ax1.boxplot(relative_transf_6_DOF_to_1st_img)
        fig1.savefig(saved_dir+'/'+'6DOF_to_1st_img_each_scan.png')
        plt.close(fig1)


        return relative_transf_6_DOF,relative_transf_6_DOF_to_1st_img





def reference_image_points(image_size, density=2):
    """
    :param image_size: (x, y), used for defining default grid image_points
    :param density: (x, y), point sample density in each of x and y, default n=2
    """
    if isinstance(density,int):
        density=(density,density)

    # image_points = torch.cartesian_prod(
    #     torch.linspace(-image_size[0]/2,image_size[0]/2,density[0]),
    #     torch.linspace(-image_size[1]/2,image_size[1]/2,density[1])
    #     ).t()  # transpose to 2-by-n

    image_points = torch.cartesian_prod(
        torch.linspace(0, image_size[0] , density[0]),
        torch.linspace(0, image_size[1], density[1])
    ).t()
    
    image_points = torch.cat([
        image_points, 
        torch.zeros(1,image_points.shape[1])*image_size[0]/2,
        torch.ones(1,image_points.shape[1])
        ], axis=0)
    
    return image_points


def read_calib_matrices(filename_calib, resample_factor,device):
    # T{image->tool} = T{image_mm -> tool} * T{image_pix -> image_mm} * T{resampled_image_pix -> image_pix}
    tform_calib = np.empty((8,4), np.float32)
    with open(os.path.join(os.getcwd(),filename_calib)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')    
        for ii, row in enumerate(csv_reader):
            tform_calib[ii,:] = (list(map(float,row)))
    return torch.tensor(tform_calib[0:4,:],device=device),torch.tensor(tform_calib[4:8,:],device=device), torch.tensor(tform_calib[4:8,:] @ tform_calib[0:4,:] @ np.array([[resample_factor,0,0,0], [0,resample_factor,0,0], [0,0,1,0], [0,0,0,1]], np.float32),device=device)



def get_last_folder(path):
    normalized_path = os.path.normpath(path)
    return os.path.basename(normalized_path)


