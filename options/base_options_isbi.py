import argparse
import os
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--FILENAME_CALIB', type=str, default="data/calib_matrix.csv",help='dataroot of calibration matrix')
        self.parser.add_argument('--multi_gpu', type=bool,default=False,help='whether use multi gpus')
        self.parser.add_argument('--gpu_ids',type=str,default='1',help='gpu id: e.g., 0,1,2...')
        self.parser.add_argument('--RESAMPLE_FACTOR', type=int,default=4,help='resize of the original image')
        # self.parser.add_argument('--FILENAME_FRAMES', type=str, default='/public_data/forearm_US_large_dataset/frames_res4.h5',help='dataroot of training')

        # self.parser.add_argument('--FILENAME_FRAME_FEATS', type=str, default=os.path.join(os.path.expanduser("~"), "workspace", 'frame_feats_res{}'.format(4)+".h5"),help='dataroot of features of frames extracted by using pretrained model')

        self.parser.add_argument('--SAVE_PATH', type=str, default='results',help='foldername of saving path')
        self.parser.add_argument('--DATA_PATH', type=str, default='/public_data/forearm_US_large_dataset', help='foldername of saving path')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain
        self.opt.h5_file_name=None

        # str_ids = self.opt.gpu_ids.split(',')
        # self.opt.gpu_ids =[]
        # for str_id in str_ids:
        #     id = int(str_id)
        #     if id>=0:
        #         self.opt.gpu_ids.append(id)
        #
        # if len(self.opt.gpu_ids) >0:
        #     torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('----------Option----------')
        for k,v in sorted(args.items()):
            print('%s, %s' %(str(k),str(v)))
            print('\n')
        print('----------Option----------')

        # create saved result path
        if self.opt.CONSIATENT_LOSS == True or self.opt.ACCUMULAT_LOSS == True:
            saved_results = 'seq_len' + str(self.opt.NUM_SAMPLES) + '_' + self.opt.model_name + '_' + 'lr' + str(
                self.opt.LEARNING_RATE) + '_scan_len' + str(self.opt.MIN_SCAN_LEN)+'_CONSIATENT_LOSS_'+ str(self.opt.CONSIATENT_LOSS)+'_ACCUMULAT_LOSS_'+ str(self.opt.ACCUMULAT_LOSS)

        else:
            # if not self.opt.train_set_type:
            #     saved_results = 'seq_len' + str(self.opt.NUM_SAMPLES) + '_' + self.opt.model_name +'_'+ 'lr' + str(self.opt.LEARNING_RATE) + '_scan_len' + str(self.opt.MIN_SCAN_LEN)+'_'+str(self.opt.train_set)#+'/'+'isbi'
            # else:
            saved_results = 'seq_len' + str(self.opt.NUM_SAMPLES) + '__' + self.opt.model_name +'__'+ 'lr' + str(self.opt.LEARNING_RATE) + '__scan_len' + str(self.opt.MIN_SCAN_LEN)+'__output_'+str(self.opt.PRED_TYPE)+'__Loss_'+str(self.opt.Loss_type)+'__'+str(self.opt.train_set)#+'/'+'isbi'




        if self.opt.train_val_folder == 'train': # used when training
            self.opt.SAVE_PATH = os.path.join(os.getcwd(),saved_results)
        else:
            os.chdir("..") # used when ploting
            self.opt.SAVE_PATH = os.path.join(os.path.abspath(os.curdir),saved_results)

        if self.opt.train_set == 'loop':
            self.opt.h5_file_name = 'frames_res4.h5'
        elif self.opt.train_set == 'forth':
            self.opt.h5_file_name = 'frames_res4_forth.h5'
        elif self.opt.train_set == 'back':
            self.opt.h5_file_name = 'frames_res4_back.h5'
        elif self.opt.train_set == 'forth_back':
            self.opt.h5_file_name = 'frames_res4_forth_back.h5'
        

        if not os.path.exists(self.opt.SAVE_PATH):
            os.makedirs(self.opt.SAVE_PATH)
        if not os.path.exists(os.path.join(self.opt.SAVE_PATH,'saved_model')):
            os.makedirs(os.path.join(self.opt.SAVE_PATH,'saved_model'))
        if not os.path.exists(os.path.join(self.opt.SAVE_PATH,'train_results')):
            os.makedirs(os.path.join(self.opt.SAVE_PATH,'train_results'))
        if not os.path.exists(os.path.join(self.opt.SAVE_PATH, 'val_results')):
            os.makedirs(os.path.join(self.opt.SAVE_PATH, 'val_results'))

        file_name = os.path.join(self.opt.SAVE_PATH,'config.txt')
        with open(file_name,'a') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k,v in sorted(args.items()):
                opt_file.write('%s,%s'%(str(k),str(v)))
                opt_file.write('\n')
            opt_file.write('------------ Options -------------\n')
        return self.opt

