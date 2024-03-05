import argparse
import os
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--FILENAME_CALIB', type=str, default="data/calib_matrix.csv",
                                 help='dataroot of calibration matrix')
        self.parser.add_argument('--multi_gpu', type=bool, default=False, help='whether use multi gpus')
        self.parser.add_argument('--gpu_ids', type=str, default='2', help='gpu id: e.g., 0,1,2...')
        self.parser.add_argument('--RESAMPLE_FACTOR', type=int, default=4, help='resize of the original image')
        # self.parser.add_argument('--FILENAME_FRAMES', type=str,default='frames_res4_cls', help='dataroot of training')
        # self.parser.add_argument('--FILENAME_FRAME_FEATS', type=str, default=os.path.join(os.path.expanduser("~"), "workspace", 'frame_feats_res{}'.format(4)+".h5"),help='dataroot of features of frames extracted by using pretrained model')

        self.parser.add_argument('--SAVE_PATH', type=str, default='results', help='foldername of saving path')
        self.parser.add_argument('--DATA_PATH', type=str, default='/public_data', help='foldername of saving path')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain

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
        for k, v in sorted(args.items()):
            print('%s, %s' % (str(k), str(v)))
            print('\n')
        print('----------Option----------')

        # create saved result path

        if self.opt.class_protocol == 3:
            self.opt.FILENAME_FRAMES = os.path.join(os.path.expanduser("~"), "workspace","frames_res4_cls" + ".h5")
        elif self.opt.class_protocol == 6:
            self.opt.FILENAME_FRAMES = os.path.join(os.path.expanduser("~"), "workspace","frames_res4_cls_6" + ".h5")


        if self.opt.CONSIATENT_LOSS == True or self.opt.ACCUMULAT_LOSS == True:
            saved_results = 'seq_len' + str(self.opt.NUM_SAMPLES) + '_' + self.opt.model_name + '_' + 'lr' + str(
                self.opt.LEARNING_RATE) + '_scan_len' + str(self.opt.MIN_SCAN_LEN) + '_CONSIATENT_LOSS_' + str(
                self.opt.CONSIATENT_LOSS) + '_ACCUMULAT_LOSS_' + str(self.opt.ACCUMULAT_LOSS)

        else:
            if not self.opt.train_set_type:
                saved_results = 'seq_len' + str(self.opt.NUM_SAMPLES) + '_' + self.opt.model_name + '_' + 'lr' + str(
                    self.opt.LEARNING_RATE)+ '_' + str(self.opt.LEARNING_RATE_cls) + '_scan_len' + str(self.opt.MIN_SCAN_LEN)  # +'/'+'isbi'
            else:
                saved_results = 'seq_len' + str(self.opt.NUM_SAMPLES) + '_' + self.opt.model_name + '_' + 'lr' + str(
                    self.opt.LEARNING_RATE)+'_' + str(self.opt.LEARNING_RATE_cls) + '_scan_len' + str(self.opt.MIN_SCAN_LEN) + '_' + str(
                    self.opt.train_set_type)+ '_optser_' + str(self.opt.optimiser_num)+ '_protocol_' + str(self.opt.class_protocol)+'_weight_'+str(self.opt.task_weight)  # +'/'+'isbi'

        if self.opt.train_val_folder == 'train':  # used when training
            self.opt.SAVE_PATH = os.path.join(os.getcwd(), saved_results)
        else:
            # os.chdir("..")  # used when ploting
            # self.opt.SAVE_PATH = os.path.join(os.path.abspath(os.curdir), saved_results)
            
            self.opt.SAVE_PATH = os.path.join(os.path.dirname(os.getcwd()), saved_results)

            



        if not os.path.exists(self.opt.SAVE_PATH):
            os.makedirs(self.opt.SAVE_PATH)
        if not os.path.exists(os.path.join(self.opt.SAVE_PATH, 'saved_model')):
            os.makedirs(os.path.join(self.opt.SAVE_PATH, 'saved_model'))
        if not os.path.exists(os.path.join(self.opt.SAVE_PATH, 'train_results')):
            os.makedirs(os.path.join(self.opt.SAVE_PATH, 'train_results'))
        if not os.path.exists(os.path.join(self.opt.SAVE_PATH, 'val_results')):
            os.makedirs(os.path.join(self.opt.SAVE_PATH, 'val_results'))

        file_name = os.path.join(self.opt.SAVE_PATH, 'config.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s,%s' % (str(k), str(v)))
                opt_file.write('\n')
            opt_file.write('------------ Options -------------\n')
        return self.opt
