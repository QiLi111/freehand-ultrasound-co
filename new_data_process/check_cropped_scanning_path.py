# feach the image of each scan and check if the scan trajectory make sence
import os

import h5py,csv
import matplotlib.pyplot as plt


DIR_RAW = os.path.join(os.path.expanduser("~"), "/raid/Qi/public_data/forearm_US_large_dataset/data_size60")

folders_subject = [f for f in os.listdir(DIR_RAW) if os.path.isdir(os.path.join(DIR_RAW, f))]
folders_subject = sorted(folders_subject, key=lambda x: int(x), reverse=False)


# for i_sub, folder in enumerate(folders_subject):
#     scan_fd = [f for f in folder if os.path.isdir(os.path.join(DIR_RAW, folders_subject, f))]

#     if scan_fd.endswith("TrackedImageSequence"):
