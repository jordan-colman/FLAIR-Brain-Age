import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import NiftiDataset
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, ToTensor, SqueezeDim, AsChannelLast
## you will additionally need to execute in BASH command line and  have niftkBiasFieldCorrection and NiftyReg installed onto system
## it is assumed all files are stored and will be kept in another directory called 'Healthy_Datasets'

def main():
    
    ## get list of subject IDs in file name to iterate through all images assuming the file has multiple columns and that the subject ID in the files names is titled 'Subject' and images named by ID and suffixes as seen below
    IDe = pd.read_csv('data_info.txt',sep='\t')
    IDe_S = IDe['Subject']
    
    
    for i in range(0,len(IDe_S)):
        ## build file names in format they are stored
        File_flair = os.sep.join(['./Dataset', str(IDe_S[i]) + '_flair.nii.gz'])
        File_t1 = os.sep.join(['./Dataset', str(IDe_S[i]) + '_t1w.nii.gz'])
        
        ## location of standard MNI brain for registration to mast be added to working folder
        File_MNI = 'MNI152_T1_1mm.nii.gz'
        
        ## build names for processed images
        file_out_flair_t1s = os.sep.join(['./Dataset', str(IDe_S[i]) + '_FLAIR_T1space.nii.gz'])
        file_out_flair_mni = os.sep.join(['./Dataset', str(IDe_S[i]) + '_FLAIR_MNIspace.nii.gz'])
        file_out_t1_mni = os.sep.join(['./Dataset', str(IDe_S[i]) + '_T1w_MNIspace.nii.gz'])
        file_out_flair_bc = os.sep.join(['./Dataset', str(IDe_S[i]) + '_FLAIR_BC.nii.gz'])
        file_out_t1_bc = os.sep.join(['./Dataset', str(IDe_S[i]) + '_T1w_BC.nii.gz'])
        
        ## define files names of matrix files
        mat_out_flair_t1s = os.sep.join(['./Dataset', str(IDe_S[i]) + '_flair_to_t1.txt'])
        mat_out_t1_mni = os.sep.join(['./Dataset', str(IDe_S[i]) + '_t1_to_mni.txt'])
        mat_out_flair_mni = os.sep.join(['./Dataset', str(IDe_S[i]) + '_flair_to_mni.txt'])

        ## perform processing on bash terminal in linux
        ### N4 bias correction
        os.system('niftkBiasFieldCorrection.py -in ' + File_flair + ' -out ' + file_out_flair_bc + ' -n4')
        os.system('niftkBiasFieldCorrection.py -in ' + File_t1 + ' -out ' + file_out_t1_bc + ' -n4')
        ## align images
        os.system('reg_aladin -ref ' + file_out_t1_bc + ' -flo ' + file_out_flair_bc + ' -aff ' + mat_out_flair_t1s + ' -res ' + file_out_flair_t1s)
        os.system('reg_aladin -ref ' + File_MNI + ' -flo ' + file_out_t1_bc + ' -aff ' + mat_out_t1_mni + ' -res ' + file_out_t1_mni)
        os.system('reg_transform -comp ' + mat_out_t1_mni + ' ' + mat_out_flair_t1s + ' ' + mat_out_flair_mni)
        os.system('reg_resample -ref ' + File_MNI + ' -flo ' + file_out_flair_bc + ' -trans ' + mat_out_flair_mni + ' -res ' + file_out_flair_mni)
    
if __name__ == "__main__":
    main()
