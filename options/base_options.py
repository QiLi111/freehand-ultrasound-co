import argparse
import os
import json


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--FILENAME_CALIB', type=str, default="data/calib_matrix.csv",help='dataroot of calibration matrix')
        self.parser.add_argument('--multi_gpu', type=bool,default=False,help='whether use multi gpus')
        self.parser.add_argument('--gpu_ids',type=str,default='1',help='gpu id: e.g., 0,1,2...')
        self.parser.add_argument('--RESAMPLE_FACTOR', type=int,default=4,help='resize of the original image')
        self.parser.add_argument('--config', type=str,default='config/config_ete.json',help='config file')


        self.parser.add_argument('--SAVE_PATH', type=str, default='results',help='foldername of saving path')
        self.parser.add_argument('--DATA_PATH', type=str, default='/data', help='foldername of saving path')

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
        # update from config json
        with open(self.opt.config) as f:
            json_dict = json.load(f)

        args.update(json_dict)

        print('----------Option----------')
        for k,v in sorted(args.items()):
            print('%s, %s' %(str(k),str(v)))
            print('\n')
        print('----------Option----------')

        # create saved result path
        
        if self.opt.inter=='nointer' and self.opt.meta=='nonmeta':
            if self.opt.multi_gpu:
                saved_results = 'seq_len' + str(self.opt.NUM_SAMPLES)\
                + '__' + self.opt.model_name +'__'+ 'lr' + str(self.opt.LEARNING_RATE_rec)\
                +'_'+ str(self.opt.LEARNING_RATE_reg)+ '__scan_len'\
                + str(self.opt.MIN_SCAN_LEN)+'__output_'+str(self.opt.PRED_TYPE)\
                +'__Loss_'+str(self.opt.Loss_type)+'__'+str(self.opt.train_set)\
                +'__'+str(self.opt.Conv_Coords)+'__'+str(self.opt.intepoletion_volume)\
                +'__'+str(self.opt.initial)+'__'+str(self.opt.BatchNorm)+'__bs_'+str(self.opt.MINIBATCH_SIZE_rec)\
                +'__'+str(self.opt.img_pro_coord)+'__inc_reg_'+str(self.opt.in_ch_reg)\
                +'__'+str(self.opt.ddf_dirc)+'__baseline_311__multiGPU__M'
            else:
                saved_results = 'seq_len' + str(self.opt.NUM_SAMPLES)\
                + '__' + self.opt.model_name +'__'+ 'lr' + str(self.opt.LEARNING_RATE_rec)\
                +'_'+ str(self.opt.LEARNING_RATE_reg)+ '__scan_len'\
                + str(self.opt.MIN_SCAN_LEN)+'__output_'+str(self.opt.PRED_TYPE)\
                +'__Loss_'+str(self.opt.Loss_type)+'__'+str(self.opt.train_set)\
                +'__'+str(self.opt.Conv_Coords)+'__'+str(self.opt.intepoletion_volume)\
                +'__'+str(self.opt.initial)+'__'+str(self.opt.BatchNorm)+'__bs_'+str(self.opt.MINIBATCH_SIZE_rec)\
                +'__'+str(self.opt.img_pro_coord)+'__inc_reg_'+str(self.opt.in_ch_reg)\
                +'__'+str(self.opt.ddf_dirc)+'__baseline_311__M'#+'__comp_crop_311'#+'__baseline_311__M'#+'__baseline_corp_311'#+'__baseline_311__M'+'__corp_221_rerun'#+'_HalfConverge_cropped_bs1_debug'#+'_HalfBestModel_bs1_uncrop'#+'_BS1'#+'_HalfBestModel_bs4'#+'_batchsize1'
                #+'_one_scan'#+'__weightloss1000'#+'__iteratively'#+'/'+'isbi'. 
        elif self.opt.inter=='nointer' and self.opt.meta=='meta':
            saved_results = 'seq_len' + str(self.opt.NUM_SAMPLES)\
            + '__' + self.opt.model_name +'__'+ 'lr' + str(self.opt.LEARNING_RATE_rec)\
            +'_'+ str(self.opt.LEARNING_RATE_reg)+ '__scan_len' + str(self.opt.MIN_SCAN_LEN)\
            +'__output_'+str(self.opt.PRED_TYPE)+'__Loss_'+str(self.opt.Loss_type)\
            +'__'+str(self.opt.train_set)+'__'+str(self.opt.Conv_Coords)\
            +'__'+str(self.opt.intepoletion_volume)+'__meta'+'__'+str(self.opt.initial)\
            +'__'+str(self.opt.BatchNorm)+'__bs_'+str(self.opt.MINIBATCH_SIZE_rec)\
            +'__'+str(self.opt.img_pro_coord)+'__inc_reg_'+str(self.opt.in_ch_reg)\
            +'__'+str(self.opt.ddf_dirc)+'__M'#+'_one_scan'#+'__weightloss1000'#+'__iteratively'#+'/'+'isbi'. 

        elif self.opt.inter=='iteratively' and self.opt.meta=='nonmeta':
            saved_results = 'seq_len' + str(self.opt.NUM_SAMPLES) + '__' + self.opt.model_name\
            +'__'+ 'lr' + str(self.opt.LEARNING_RATE_rec)+'_'+ str(self.opt.LEARNING_RATE_reg)\
            + '__scan_len' + str(self.opt.MIN_SCAN_LEN)+'__output_'+str(self.opt.PRED_TYPE)\
            +'__Loss_'+str(self.opt.Loss_type)+'__'+str(self.opt.train_set)\
            +'__'+str(self.opt.Conv_Coords)+'__'+str(self.opt.intepoletion_volume)\
            +'__iteratively'+'__'+str(self.opt.initial)+'__'+str(self.opt.BatchNorm)\
            +'__bs_'+str(self.opt.MINIBATCH_SIZE_rec)+'__'+str(self.opt.img_pro_coord)\
            +'__inc_reg_'+str(self.opt.in_ch_reg)+'__'+str(self.opt.ddf_dirc)+'__M'#+'/'+'isbi'. 
        elif self.opt.inter=='iteratively' and self.opt.meta=='meta':
            if self.opt.multi_gpu:
                saved_results = 'seq_len' + str(self.opt.NUM_SAMPLES) + '__' + self.opt.model_name\
                +'__'+ 'lr' + str(self.opt.LEARNING_RATE_rec)+'_'+ str(self.opt.LEARNING_RATE_reg)\
                + '__scan_len' + str(self.opt.MIN_SCAN_LEN)+'__output_'+str(self.opt.PRED_TYPE)\
                +'__Loss_'+str(self.opt.Loss_type)+'__'+str(self.opt.train_set)\
                +'__'+str(self.opt.Conv_Coords)+'__'+str(self.opt.intepoletion_volume)\
                +'__iteratively__meta'+'__'+str(self.opt.initial)+'__'+str(self.opt.BatchNorm)\
                +'__bs_'+str(self.opt.MINIBATCH_SIZE_rec)+'__'+str(self.opt.img_pro_coord)\
                +'__inc_reg_'+str(self.opt.in_ch_reg)+'__'+str(self.opt.ddf_dirc)+'__multiGPU__M'#+'/'+'isbi'. 
            else:
                saved_results = 'seq_len' + str(self.opt.NUM_SAMPLES) + '__' + self.opt.model_name\
                +'__'+ 'lr' + str(self.opt.LEARNING_RATE_rec)+'_'+ str(self.opt.LEARNING_RATE_reg)\
                + '__scan_len' + str(self.opt.MIN_SCAN_LEN)+'__output_'+str(self.opt.PRED_TYPE)\
                +'__Loss_'+str(self.opt.Loss_type)+'__'+str(self.opt.train_set)\
                +'__'+str(self.opt.Conv_Coords)+'__'+str(self.opt.intepoletion_volume)\
                +'__iteratively__meta'+'__'+str(self.opt.initial)+'__'+str(self.opt.BatchNorm)\
                +'__bs_'+str(self.opt.MINIBATCH_SIZE_rec)+'__'+str(self.opt.img_pro_coord)\
                +'__inc_reg_'+str(self.opt.in_ch_reg)+'__'+str(self.opt.ddf_dirc)+'__M'



        self.opt.SAVE_PATH = os.path.join(os.getcwd(),'models_all/'+saved_results)
        
        # if self.opt.split_type == 'scan':
        # if self.opt.train_set == 'loop':
        #     self.opt.h5_file_name = 'frames_res4.h5'
        # elif self.opt.train_set == 'forth':
        #     self.opt.h5_file_name = 'frames_res4_forth.h5'
        # elif self.opt.train_set == 'back':
        #     self.opt.h5_file_name = 'frames_res4_back.h5'
        # elif self.opt.train_set == 'forth_back':
        #     self.opt.h5_file_name = 'frames_res4_forth_back.h5'
        # elif self.opt.split_type == 'sub':
        if self.opt.train_set == 'loop':
            self.opt.h5_file_name = 'scans_res4.h5'
        elif self.opt.train_set == 'forth':
            self.opt.h5_file_name = 'scans_res4_forth.h5'
        elif self.opt.train_set == 'back':
            self.opt.h5_file_name = 'scans_res4_back.h5'
        elif self.opt.train_set == 'forth_back':
            self.opt.h5_file_name = 'scans_res4_forth_back.h5'
        

        if not os.path.exists(self.opt.SAVE_PATH):
            os.makedirs(self.opt.SAVE_PATH)
        if not os.path.exists(os.path.join(self.opt.SAVE_PATH,'saved_model')):
            os.makedirs(os.path.join(self.opt.SAVE_PATH,'saved_model'))
        # if not os.path.exists(os.path.join(self.opt.SAVE_PATH,'train_results')):
        #     os.makedirs(os.path.join(self.opt.SAVE_PATH,'train_results'))
        # if not os.path.exists(os.path.join(self.opt.SAVE_PATH, 'val_results')):
        #     os.makedirs(os.path.join(self.opt.SAVE_PATH, 'val_results'))


        file_name = os.path.join(self.opt.SAVE_PATH,'config.txt')
        with open(file_name,'a') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k,v in sorted(args.items()):
                opt_file.write('%s,%s'%(str(k),str(v)))
                opt_file.write('\n')
            opt_file.write('------------ Options -------------\n')
        return self.opt

