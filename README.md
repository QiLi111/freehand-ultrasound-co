# freehand-ultrasound-co
# Co-Optimisation for Trackless Freehand Ultrasound Reconstruction with Non-Rigid Deformation

This repository contains algorithms for freehand ultrasound reocnstruction, using deformation from rigistration-like network, trained in an end-to-end manner.

# Install conda environment
``` bash
conda create -n tracked-train python=3.9.13 && conda activate tracked-train &&
pip install torch==1.11.0+cu102 torchvision==0.12.0+cu102 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu102 &&
onda config --add channels conda-forge && conda install mdanalysis && conda install -c pytorch3d pytorch3d && conda install tensorboard && conda install h5py && conda install -c https://conda.anaconda.org/simpleitk SimpleITK && conda install scipy && conda install matplotlib
``` 
# train
``` bash
python train_rec_reg_baseline.py
``` 
# Test
``` bash
python test_rec_reg_large_dataset_non_common.py
``` 



