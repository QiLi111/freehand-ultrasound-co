from .base_options import BaseOptions
from utils import sample_dists4plot
from options.train_options import TrainOptions
from utils import pair_samples
opt = TrainOptions().parse()
data_pairs = pair_samples(opt.NUM_SAMPLES, opt.NUM_PRED, opt.single_interval)

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # self.parser.add_argument('--PRED_TYPE', type=str,default='parameter',help='network output type: {"transform", "parameter", "point"}')
        # self.parser.add_argument('--LABEL_TYPE', type=str,default='point',help='label type: {"point", "parameter"}')
        # self.parser.add_argument('--NUM_SAMPLES', type=int,default=7,help='number of input frames/imgs')
        # self.parser.add_argument('--SAMPLE_RANGE', type=int,default=7,help='from which the input frames/imgs are selected from')
        # self.parser.add_argument('--NUM_PRED', type=int,default=6,help='to those frames/imgs, transformation matrix are predicted ')
        self.parser.add_argument('--PAIR_INDEX', type=list,default=[3,7,12,18],help='sample_dists4plot(opt,data_pairs) used predicted pair for ploting [0,1,3,15] [0,1,3,6,15,28,66,120,276,496,1128,4560] ')
        self.parser.add_argument('--MAX_INTERVAL', type=int,default=6,help='use max interval for camputing evaluation matrix')
        # self.parser.add_argument('--INTERVAL_LIST', type=list,default=range(1,7),help='interval that used in this setting[1,2,3,4,6,8,12,16,24,32,48,96]')
        self.parser.add_argument('--MIN_SCAN_LEN', type=int,default=0,help='scan length that greater than this value can be used in training and val')

        self.parser.add_argument('--FILENAME_VAL', type=str,default="fold_04.json",help='validation json file')
        self.parser.add_argument('--FILENAME_TRAIN', type=list,default=["fold_00.json", "fold_01.json","fold_02.json","fold_03.json"],help='train json file')
        self.parser.add_argument('--MODEL_FN', type=str,default="saved_model/",help='model path for visulize')
        self.parser.add_argument('--plot_line', type=bool,default=True,help='True: only plot 4 points of one frame; False: plot all pixels (surface) in a frame')
        self.parser.add_argument('--START_FRAME_INDEX', type=int,default=0,help='starting frame - the reference')

        self.isTrain= True
