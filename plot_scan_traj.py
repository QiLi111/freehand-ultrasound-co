# plot scan trajactory using different methiods
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import pickle

Path = '/raid/candi/Qi/freehand-ultrasound/plot_scan_trajactory'
gt_fd = 'gt_all'
frame_fd = 'frame_all'
figs = 'figs'
viridis = cm.get_cmap('tab10',10)
width = 4
scatter = 8
step = 10
legend_size=20
elev=30 
azim=4 
roll=15

if not os.path.exists(Path + '/' + figs):
    os.makedirs(Path + '/' +figs)

pred_fds = [f for f in os.listdir(Path) if f.startswith('seq_len')  and not os.path.isfile(os.path.join(Path, f))]

gt_all = sorted(os.listdir(Path+'/'+gt_fd))
frame_all = sorted(os.listdir(Path+'/'+frame_fd))



for i_scan in range(len(gt_all)):
    scan_name_gt = gt_all[i_scan]
    scan_name_frame = frame_all[i_scan]
    with open(Path + '/' + gt_fd + '/' + scan_name_gt, 'rb') as f:
        gt = np.load(f)
    with open(Path+ '/' + frame_fd + '/' + scan_name_frame, 'rb') as f:
        frame = np.load(f)
    
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(projection='3d')
    plt.tight_layout()
    # plot surface
    ysize, xsize = frame.shape[-2:]
    grid=np.meshgrid(np.linspace(0,1,ysize),np.linspace(0,1,xsize),indexing='ij')
    coord = np.zeros((3,ysize,xsize))
    for i_frame in range(0,gt.shape[1],step): 
        gx, gy, gz = [gt[0, i_frame, ii, :] for ii in range(3)]
        gx, gy, gz = gx.reshape(2, 2), gy.reshape(2, 2), gz.reshape(2, 2)
        coord[0]=gx[0,0]+(gx[1,0]-gx[0,0])*grid[1]+(gx[0,1]-gx[0,0])*grid[0]
        coord[1]=gy[0,0]+(gy[1,0]-gy[0,0])*grid[1]+(gy[0,1]-gy[0,0])*grid[0]
        coord[2]=gz[0,0]+(gz[1,0]-gz[0,0])*grid[1]+(gz[0,1]-gz[0,0])*grid[0]
        
        # for i in range(ysize):
        #     for j in range(xsize):
        #         coord[0,i,j]=gx[0,0]+(gx[1,0]-gx[0,0])*j/(xsize-1)+(gx[0,1]-gx[0,0])*i/(ysize-1)
        #         coord[1,i,j]=gy[0,0]+(gy[1,0]-gy[0,0])*j/(xsize-1)+(gy[0,1]-gy[0,0])*i/(ysize-1)
        #         coord[2,i,j]=gz[0,0]+(gz[1,0]-gz[0,0])*j/(xsize-1)+(gz[0,1]-gz[0,0])*i/(ysize-1)
        
        # pix_intensities = (torch.tensor(frame[0,i_frame, ..., None]/frame[0,i_frame, ...].max())).expand(-1, -1, 3).numpy()
        pix_intensities = (torch.tensor(frame[0,i_frame, ...]/frame[0,i_frame, ...].max()))

        # ax.plot_surface(gx, gy, gz, facecolors=plt.cm.gray(pix_intensities), shade=False,linewidth=0, antialiased=True, alpha=0.5,rstride=1, cstride=1)# linewidth=0, antialiased=False, alpha=0.05,rstride=1, cstride=1)#
        ax.plot_surface(coord[0], coord[1], coord[2], facecolors=plt.cm.gray(pix_intensities.cpu().numpy()[::-1]), shade=False,linewidth=0, antialiased=True, alpha=0.5)#,rstride=1, cstride=1)
    # plot gt
    gx_all, gy_all, gz_all = [gt[:, :, ii, :] for ii in range(3)]
    ax.scatter(gx_all, gy_all, gz_all, c=viridis.colors[0], alpha=0.5, s=scatter)
    # plot the first frame and the last frame
    plt.plot(gt[0,0,0,0:2], gt[0,0,1,0:2], gt[0,0,2,0:2], c=viridis.colors[0], linewidth = width)
    plt.plot(gt[0,0,0,[1,3]], gt[0,0,1,[1,3]], gt[0,0,2,[1,3]], c=viridis.colors[0], linewidth = width, label='GT') 
    plt.plot(gt[0,0,0,[3,2]], gt[0,0,1,[3,2]], gt[0,0,2,[3,2]], c=viridis.colors[0], linewidth = width) 
    plt.plot(gt[0,0,0,[2,0]], gt[0,0,1,[2,0]], gt[0,0,2,[2,0]], c=viridis.colors[0], linewidth = width)
    plt.plot(gt[0,-1,0,0:2], gt[0,-1,1,0:2], gt[0,-1,2,0:2], c=viridis.colors[0], linewidth = width)
    plt.plot(gt[0,-1,0,[1,3]], gt[0,-1,1,[1,3]], gt[0,-1,2,[1,3]], c=viridis.colors[0], linewidth = width) 
    plt.plot(gt[0,-1,0,[3,2]], gt[0,-1,1,[3,2]], gt[0,-1,2,[3,2]], c=viridis.colors[0], linewidth = width) 
    plt.plot(gt[0,-1,0,[2,0]], gt[0,-1,1,[2,0]], gt[0,-1,2,[2,0]], c=viridis.colors[0], linewidth = width)

    
    
            
    color_idx = 1
    for pred_sub in pred_fds:
        pred_all_model = sorted(os.listdir(Path+'/'+pred_sub))
        
        
        for i_model in pred_all_model:
            pred_all = sorted([f for f in os.listdir(Path+'/'+pred_sub+'/'+i_model) if f.endswith('.npy')])

            if pred_sub == 'seq_len2__efficientnet_b1__lr0.0001_0.0001__scan_len108__output_parameter__Loss_MSE_points__forth__ori_coords__fixed_interval__noninitial__BNoff__bs_32__pro_coord__inc_reg_1__Move__comp_crop_311':
                saved_name = 'seq2_MSE_32'

        
            elif pred_sub == 'seq_len100__efficientnet_b1__lr0.0001_0.0001__scan_len108__output_parameter__Loss_MSE_points__forth__ori_coords__fixed_interval__noninitial__BNoff__bs_4__pro_coord__inc_reg_1__Move__comp_crop_311':
                saved_name = 'seq100_MSE_4'
            elif pred_sub == 'seq_len100__efficientnet_b1__lr0.0001_0.0001__scan_len108__output_parameter__Loss_MSE_points__forth__ori_coords__fixed_interval__noninitial__BNoff__bs_32__pro_coord__inc_reg_1__Move__comp_crop_311':
                saved_name = 'seq100_MSE_32'

            if 'loss' in i_model: 
                saved_name = saved_name + '_loss'
            elif 'dist' in i_model: 
                saved_name = saved_name + '_dist'
            else:
                raise('incorrect file name')
            if 'R_T' in i_model:
                saved_name = saved_name + 'R_T'
            elif 'R_R' in i_model:
                saved_name = saved_name + 'R_R'
            elif 'T_T' in i_model:
                saved_name = saved_name + 'T_T'
            elif 'T_R' in i_model:
                saved_name = saved_name + 'T_R'
                
        
            
            scan_name_pred = pred_all[i_scan]
            

            if scan_name_gt[:51] != scan_name_pred[:51]:
                raise('scan is not the same for gt and pred')
            if scan_name_gt[:51] != scan_name_frame[:51]:
                raise('scan is not the same for gt and frame name')

            # load .npy
            
            with open(Path+'/'+pred_sub +'/'+i_model + '/' + scan_name_pred, 'rb') as f:
                pred = np.load(f)

            prex_all, prey_all, prez_all = [pred[:, :, ii, :] for ii in range(3)]
            ax.scatter(prex_all, prey_all, prez_all, color=viridis.colors[color_idx], alpha=0.2, s=scatter)

            
            # plt.plot(pred[0,0,0,0:2], pred[0,0,1,0:2], pred[0,0,2,0:2], color=viridis.colors[color_idx])
            # plt.plot(pred[0,0,0,[1,3]], pred[0,0,1,[1,3]], pred[0,0,2,[1,3]], color=viridis.colors[color_idx], label=saved_name) 
            # plt.plot(pred[0,0,0,[3,2]], pred[0,0,1,[3,2]], pred[0,0,2,[3,2]], color=viridis.colors[color_idx]) 
            # plt.plot(pred[0,0,0,[2,0]], pred[0,0,1,[2,0]], pred[0,0,2,[2,0]], color=viridis.colors[color_idx])

            
            plt.plot(pred[0,-1,0,0:2], pred[0,-1,1,0:2], pred[0,-1,2,0:2], color=viridis.colors[color_idx], linewidth = width, label=saved_name)
            plt.plot(pred[0,-1,0,[1,3]], pred[0,-1,1,[1,3]], pred[0,-1,2,[1,3]], linewidth = width, color=viridis.colors[color_idx]) 
            plt.plot(pred[0,-1,0,[3,2]], pred[0,-1,1,[3,2]], pred[0,-1,2,[3,2]], linewidth = width, color=viridis.colors[color_idx]) 
            plt.plot(pred[0,-1,0,[2,0]], pred[0,-1,1,[2,0]], pred[0,-1,2,[2,0]], linewidth = width, color=viridis.colors[color_idx])

            color_idx += 1


    ax.axis('equal')
    ax.legend()
    ax.grid(False)
    ax.axis('off')
    plt.legend(fontsize=legend_size)
    # ax.view_init(elev, azim)



    plt.savefig(Path + '/' + figs+ '/' +scan_name_pred+'.png')
    plt.savefig(Path+ '/' + figs+ '/'+scan_name_pred +'.pdf')
    pickle.dump(fig, open(Path+ '/' + figs+ '/'+scan_name_pred +'.pickle', 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`

    plt.close()

    # figx = pickle.load(open(Path+ '/' + figs+ '/'+scan_name_pred +'.pickle', 'rb'))
    # figx.show()
    # plt.show()
            




