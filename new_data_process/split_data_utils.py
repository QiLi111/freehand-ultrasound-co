
import sys,os
from pathlib import Path
# add path to index script outside current folder
sys.path.append(Path(os.getcwd()).as_posix())#.parent.absolute())
from loader import SSFrameDataset


def split_data(filename_h5,NUM_SAMPLES,SAMPLE_RANGE,MIN_SCAN_LEN,saved_path,dataset_type,sub_scan):
    dataset_all = SSFrameDataset(
        min_scan_len = MIN_SCAN_LEN,
        filename_h5=filename_h5,
        num_samples=NUM_SAMPLES,
        sample_range=SAMPLE_RANGE
        )
    ## setup for cross-validation
    dset_folds = dataset_all.partition_by_ratio(
        ratios = [1]*5, 
        randomise=True, 
        subject_level=sub_scan
        )
    if sub_scan:
        for (idx, ds) in enumerate(dset_folds):
            ds.write_json(os.path.join(saved_path,"fold_{:02d}_seqlen{:d}_sub_{:s}.json".format(idx,NUM_SAMPLES,dataset_type)))  # see test.py for file reading

    
    else:
        for (idx, ds) in enumerate(dset_folds):
            ds.write_json(os.path.join(saved_path,"fold_{:02d}_seqlen{:d}_scan_{:s}.json".format(idx,NUM_SAMPLES,dataset_type)))  # see test.py for file reading

# seperate the dataset into train, validation, and test
# the validation set is use to tune the hyper-parameter, and the test set is used to evaluate the model performance
# dset_train = dset_folds[0]+dset_folds[1]+dset_folds[2]
# dset_val = dset_folds[3]
# dset_test = dset_folds[4]