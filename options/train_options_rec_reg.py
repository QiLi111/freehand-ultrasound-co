from .base_options_rec_reg import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--PRED_TYPE', type=str,default='parameter',help='network output type: {"transform", "parameter", "point","quaternion"}')
        self.parser.add_argument('--LABEL_TYPE', type=str,default='transform',help='label type: {"point", "parameter"}')
        self.parser.add_argument('--weight_option', type=str,default='none',help='whether use weighted loss: {"assigned_weight", "trained_weight", "none"}')

        self.parser.add_argument('--NUM_SAMPLES', type=int,default=100,help='number of input frames/imgs')
        self.parser.add_argument('--SAMPLE_RANGE', type=int,default=100,help='from which the input frames/imgs are selected from')
        self.parser.add_argument('--NUM_PRED', type=int,default=99,help='to those frames/imgs, transformation matrix are predicted ')
        self.parser.add_argument('--sample', type=bool,default=True,help='False - use all data pairs for training; True - use only sampled data pairs for training')
        self.parser.add_argument('--MIN_SCAN_LEN', type=int, default=108,help='scan length that greater than this value can be used in training and val')
        self.parser.add_argument('--train_set_type', type=str, default='None',help='None; linear;c_s,remain_ind_in_use_25,remain_ind_in_use_50,remain_ind_in_use_75,')
        self.parser.add_argument('--train_set', type=str, default='forth',help='loop: all data in a h5 file; forth: only forth data; back: only back data; forth_back: forth and back data seperatly')
        self.parser.add_argument('--split_type', type=str, default='sub',help='sub: split dataset on subject level; scan: split dataset on scan level')
        self.parser.add_argument('--Loss_type', type=str, default='wraped',help='MSE_points: MSE loss on points;\
                                  Plane_norm: MSE loss and Loss over the norm of plane;\
                                  reg: only regietsrtion loss; rec_reg: reconstruction loss and registration loss;\
                                  rec_volume: reconstruction loss and volume loss; \
                                 rec_volume10000: weigh volume loss by 10000; volume_only: only use volume loss\
                                 wraped: MSE between wraped prediction and ground truth' )
        self.parser.add_argument('--intepoletion_method', type=str, default='bilinear',help='bilinear: mulyiply 3 axis difference ; IDW: inverse distance weighted;')
        self.parser.add_argument('--Conv_Coords', type=str, default='optimised_coord',help='optimised_coord: convert the volume into a optimised coordinates ; ori_coords: donnot convert coordinates' )
        self.parser.add_argument('--intepoletion_volume', type=str, default='fixed_interval',help='fixed_volume_size: set volume dimention as 128*128*128 ; fixed_interval: set the volume interal as 1 mm' )
        self.parser.add_argument('--img_pro_coord', type=str, default='pro_coord',help='img_coord: transform coordinaets system into image coordinares of the first image ; pro_coord: transform coordinaets into pro coordinares system of the first image' )
        self.parser.add_argument('--in_ch_reg', type=int,default=1,help='the input channel of registartion network')
        self.parser.add_argument('--ddf_dirc', type=str, default='Move',help='Move: based on moving image and then generate wraped fixed; Fix: based on fixed image and then generate wraped moving')


        self.parser.add_argument('--single_interval', type=int,default=0,help='0 - use all interval predictions; 1,2,3,... - use only specific intervals')
        self.parser.add_argument('--single_interval_ACCUMULAT_LOSS', type=bool,default=False,help='whether add accumulated loss')
        self.parser.add_argument('--single_weight_option', type=str,default='none',help='whether use weighted loss: {"assigned_weight", "trained_weight", "none"}')
        self.parser.add_argument('--model_name', type=str,default='efficientnet_b1',help='network name:{"efficientnet_b1", "resnet", "LSTM_0", "LSTM", "LSTM_GT"}')
        self.parser.add_argument('--train_val_folder', type=str,default='train',help='the saved foler when training and testing')

        self.parser.add_argument('--CONSIATENT_LOSS', type=bool,default=False,help='compute consistent loss')
        self.parser.add_argument('--ACCUMULAT_LOSS', type=bool,default=False,help='compute accumulate loss')

        self.parser.add_argument('--MAXNUM_PAIRS', type=int,default=50,help='maximum pairs of transformations to save to scalar')
        self.parser.add_argument('--retain', type=bool,default=False,help='whether load a pretrained model')
        self.parser.add_argument('--retain_epoch', type=str,default='00000000',help='whether load a pretrained model: {0: train from sctrach; a number, e.g., 1000, train from epoch 1000}')
        self.parser.add_argument('--MINIBATCH_SIZE_rec', type=int,default=32,help='input batch size for reconstruction network')
        self.parser.add_argument('--MINIBATCH_SIZE_reg', type=int,default=32,help='input batch size for registartion network')

        self.parser.add_argument('--LEARNING_RATE_rec',type=float,default=1e-4,help='learing rate for reconstruction network')
        self.parser.add_argument('--LEARNING_RATE_reg',type=float,default=1e-4,help='learing rate for registartion network')

        self.parser.add_argument('--NUM_EPOCHS',type =int,default=int(1e6),help='# of iter to lin')
        self.parser.add_argument('--max_rec_epoch_each_interation',type =int,default=int(2),help='# maxmum epoch to train rec or reg in each interation')
        self.parser.add_argument('--max_inter_rec_reg',type =int,default=int(500),help='# maxmum interation for training rec-reg models')
        self.parser.add_argument('--inter',type =str,default='nointer',help='nointer/iteratively: iteratively or not ')
        self.parser.add_argument('--meta',type =str,default='nonmeta',help='meta: use validation to train registration')
        self.parser.add_argument('--initial',type =str,default='InitialHalf',help='noninitial/InitialBest/InitialHalf: ')
        self.parser.add_argument('--BatchNorm',type =str,default='BNoff',help='BNoff/BNon: turn off batchnorm or not ')
        # self.parser.add_argument('--crop',type =str,default='uncrop',help='croped/uncrop: crop image or not ')

        
        self.parser.add_argument('--FREQ_INFO', type=int, default=1,help='frequency of print info')
        self.parser.add_argument('--FREQ_SAVE', type=int, default=100,help='frequency of save model')
        self.parser.add_argument('--val_fre', type=int, default=1,help='frequency of validation')
        #######used in testing##################################################
        self.parser.add_argument('--PAIR_INDEX', type=list, default=[0],#8, 17, 12, 18
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
