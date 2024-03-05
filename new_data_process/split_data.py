# split dataset into train, val, test, on scan and subject level respectively
import sys,os
from pathlib import Path
from split_data_utils import split_data
# add path to index script outside current folder
sys.path.append(Path(os.getcwd()).as_posix())#.parent.absolute())


from options.train_options_rec_reg import TrainOptions

opt = TrainOptions().parse()

dataset_size = 60

NUM_SAMPLES = 100
SAMPLE_RANGE = 100
MIN_SCAN_LEN = 108
saved_path = Path(os.getcwd()).parent.absolute().as_posix()+'/public_data/forearm_US_large_dataset/data_size%2d'%dataset_size

# filename_h5_all = saved_path+'/scans_res4.h5'
# dataset_type_all = 'all'

filename_h5_forth = saved_path+'/scans_res4_forth.h5'
dataset_type_forth = 'forth'

filename_h5_forth_back = saved_path+'/scans_res4_forth_back.h5'
dataset_type_forth_back = 'forth_back'

filename_h5_back = saved_path+'/scans_res4_back.h5'
dataset_type_back = 'back'


# split data in subject level
# split_data(filename_h5_all,NUM_SAMPLES,SAMPLE_RANGE,MIN_SCAN_LEN,saved_path,'all',sub_scan=True)
split_data(filename_h5_forth,NUM_SAMPLES,SAMPLE_RANGE,MIN_SCAN_LEN,saved_path,'forth',sub_scan=True)
split_data(filename_h5_back,NUM_SAMPLES,SAMPLE_RANGE,MIN_SCAN_LEN,saved_path,'back',sub_scan=True)
split_data(filename_h5_forth_back,NUM_SAMPLES,SAMPLE_RANGE,MIN_SCAN_LEN,saved_path,'forth_back',sub_scan=True)

# split data in scan level
# split_data(filename_h5_all,NUM_SAMPLES,SAMPLE_RANGE,MIN_SCAN_LEN,saved_path,'all',sub_scan=False)
split_data(filename_h5_forth,NUM_SAMPLES,SAMPLE_RANGE,MIN_SCAN_LEN,saved_path,'forth',sub_scan=False)
split_data(filename_h5_back,NUM_SAMPLES,SAMPLE_RANGE,MIN_SCAN_LEN,saved_path,'back',sub_scan=False)
split_data(filename_h5_forth_back,NUM_SAMPLES,SAMPLE_RANGE,MIN_SCAN_LEN,saved_path,'forth_back',sub_scan=False)





# filename_h5_all = saved_path+'/frames_res4.h5'
# dataset_type_all = 'all'

# filename_h5_forth = saved_path+'/frames_res4_forth.h5'
# dataset_type_forth = 'forth'

# filename_h5_forth_back = saved_path+'/frames_res4_forth_back.h5'
# dataset_type_forth_back = 'forth_back'

# filename_h5_back = saved_path+'/frames_res4_back.h5'
# dataset_type_back = 'back'

# # split data in scan level
# split_data(filename_h5_all,NUM_SAMPLES,SAMPLE_RANGE,MIN_SCAN_LEN,saved_path,'all',sub_scan=False)
# split_data(filename_h5_forth,NUM_SAMPLES,SAMPLE_RANGE,MIN_SCAN_LEN,saved_path,'forth',sub_scan=False)
# split_data(filename_h5_back,NUM_SAMPLES,SAMPLE_RANGE,MIN_SCAN_LEN,saved_path,'back',sub_scan=False)
# split_data(filename_h5_forth_back,NUM_SAMPLES,SAMPLE_RANGE,MIN_SCAN_LEN,saved_path,'forth_back',sub_scan=False)

# # split data in subject level
# split_data(filename_h5_all,NUM_SAMPLES,SAMPLE_RANGE,MIN_SCAN_LEN,saved_path,'all',sub_scan=True)
# split_data(filename_h5_forth,NUM_SAMPLES,SAMPLE_RANGE,MIN_SCAN_LEN,saved_path,'forth',sub_scan=True)
# split_data(filename_h5_back,NUM_SAMPLES,SAMPLE_RANGE,MIN_SCAN_LEN,saved_path,'back',sub_scan=True)
# split_data(filename_h5_forth_back,NUM_SAMPLES,SAMPLE_RANGE,MIN_SCAN_LEN,saved_path,'forth_back',sub_scan=True)





# # check if two json file is same or not
import json
with open(saved_path+'/'+'fold_02_seqlen100_sub_forth.json', 'r', encoding='utf-8') as f:
    obj1 = json.load(f)

with open(saved_path+'/'+'fold_04_seqlen100_sub_forth.json', 'r', encoding='utf-8') as f:
    obj2 = json.load(f)

print(obj1['indices_in_use']==obj2['indices_in_use'])
print(obj1['min_scan_len']==obj2['min_scan_len'])
print(obj1['num_samples']==obj2['num_samples'])
print(obj1['sample_range']==obj2['sample_range'])

print('done')







print('done')






