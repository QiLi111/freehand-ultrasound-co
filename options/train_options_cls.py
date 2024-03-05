from .base_options_cls import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--PRED_TYPE', type=str,default='parameter',help='network output type: {"transform", "parameter", "point"}')
        self.parser.add_argument('--LABEL_TYPE', type=str,default='point',help='label type: {"point", "parameter"}')
        self.parser.add_argument('--weight_option', type=str,default='none',help='whether use weighted loss: {"assigned_weight", "trained_weight", "none"}')

        self.parser.add_argument('--NUM_SAMPLES', type=int,default=100,help='number of input frames/imgs')
        self.parser.add_argument('--SAMPLE_RANGE', type=int,default=100,help='from which the input frames/imgs are selected from')
        self.parser.add_argument('--NUM_PRED', type=int,default=99,help='to those frames/imgs, transformation matrix are predicted ')
        self.parser.add_argument('--sample', type=bool,default=True,help='False - use all data pairs for training; True - use only sampled data pairs for training')
        self.parser.add_argument('--MIN_SCAN_LEN', type=int, default=108,help='scan length that greater than this value can be used in training and val')
        self.parser.add_argument('--train_set_type', type=str, default='None',help='None; linear;c_s,remain_ind_in_use_25,remain_ind_in_use_50,remain_ind_in_use_75,')

        self.parser.add_argument('--single_interval', type=int,default=0,help='0 - use all interval predictions; 1,2,3,... - use only specific intervals')
        self.parser.add_argument('--single_interval_ACCUMULAT_LOSS', type=bool,default=False,help='whether add accumulated loss')
        self.parser.add_argument('--single_weight_option', type=str,default='none',help='whether use weighted loss: {"assigned_weight", "trained_weight", "none"}')
        self.parser.add_argument('--model_name', type=str,default='classicifation_b1',help='network name:{"efficientnet_b1", "resnet", "LSTM_0", "LSTM", "LSTM_GT","classicifation_b1"}')
        self.parser.add_argument('--train_val_folder', type=str,default='train',help='the saved foler when training and testing')

        self.parser.add_argument('--CONSIATENT_LOSS', type=bool,default=False,help='compute consistent loss')
        self.parser.add_argument('--ACCUMULAT_LOSS', type=bool,default=False,help='compute accumulate loss')

        self.parser.add_argument('--MAXNUM_PAIRS', type=int,default=50,help='maximum pairs of transformations to save to scalar')
        self.parser.add_argument('--retain', type=bool,default=False,help='whether load a pretrained model')
        self.parser.add_argument('--retain_epoch', type=str,default='00000000',help='whether load a pretrained model: {0: train from sctrach; a number, e.g., 1000, train from epoch 1000}')
        self.parser.add_argument('--MINIBATCH_SIZE', type=int,default=32,help='input batch size')
        self.parser.add_argument('--LEARNING_RATE',type=float,default=1e-4,help='learing rate')
        self.parser.add_argument('--NUM_EPOCHS',type =int,default=int(1e6),help='# of iter to lin')
        self.parser.add_argument('--FREQ_INFO', type=int, default=10,help='frequency of print info')
        self.parser.add_argument('--FREQ_SAVE', type=int, default=100,help='frequency of save model')
        self.parser.add_argument('--val_fre', type=int, default=1,help='frequency of validation')
        self.parser.add_argument('--class_protocol', type=int, default=6,help='the number of classes of protocols')
        self.parser.add_argument('--class_anatomy', type=int, default=38,help='the number of classes of anatomies')
        self.parser.add_argument('--protocol_anatomy', type=int, default=1,help='1: classification for protocol; classification for anatomy')



        #######used in testing##################################################
        self.parser.add_argument('--PAIR_INDEX', type=list, default=[4],#8, 17, 12, 18
                                 help='sample_dists4plot(opt,data_pairs) used predicted pair for ploting [0,1,3,15] [0,1,3,6,15,28,66,120,276,496,1128,4560] ')
        self.parser.add_argument('--MAX_INTERVAL', type=int, default=6,
                                 help='use max interval for camputing evaluation matrix')
        # self.parser.add_argument('--INTERVAL_LIST', type=list,default=range(1,7),help='interval that used in this setting[1,2,3,4,6,8,12,16,24,32,48,96]')

        self.parser.add_argument('--FILENAME_VAL', type=str, default="fold_03", help='validation json file')
        self.parser.add_argument('--FILENAME_TEST', type=str, default="fold_04", help='test json file')

        self.parser.add_argument('--FILENAME_TRAIN', type=list,
                                 default=["fold_00", "fold_01", "fold_02"],
                                 help='train json file')
        self.parser.add_argument('--MODEL_FN', type=str, default="saved_model/", help='model path for visulize')
        self.parser.add_argument('--plot_line', type=bool, default=True,
                                 help='True: only plot 4 points of one frame; False: plot all pixels (surface) in a frame')
        self.parser.add_argument('--START_FRAME_INDEX', type=int, default=0, help='starting frame - the reference')
        self.parser.add_argument('--use_bash_shell', type=bool, default=False,
                                 help='True: use bash shell script for testing')

        self.isTrain= True
