from SFCN_model.model_files.sfcn import SFCN
from SFCN_model import dp_loss as dpl
from SFCN_model import dp_utils as dpu
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import monai
from torch.utils.data import DataLoader
from monai.data import NiftiDataset
from monai.transforms import AddChannel, CenterSpatialCrop, Compose, RandFlip, Resize, ScaleIntensity, ToTensor, SqueezeDim, AsChannelFirst, NormalizeIntensity, RandScaleIntensity, RandShiftIntensity

def main():

  OUT_FILE = 'model_data_testing_output/SFCN_flair_'

  model = SFCN()
  model = torch.nn.DataParallel(model)
  fp_ = 'model_weights/best_SFCN_model_flair.pth'
  model.load_state_dict(torch.load(fp_))
  model.cuda()
  MAE_metric = torch.nn.L1Loss() 
  bin_range = [15,95]
  bin_step = 2
  sigma = 1
  
  #File_list_in = pd.read_csv('../final_traning_imgs_T1w_brain.txt',header=None)
  #Age_list_in = pd.read_csv('../final_traning_imgs_ages.txt',header=None)
  
  File_list_in_2 = pd.read_csv('testing_imgs_flair_brain.txt',header=None)
  Age_list_in_2 = pd.read_csv('testing_imgs_ages.txt',header=None)
  
  #File_list_1 = np.array(File_list_in[0])
  #Age_list_1 = np.array(Age_list_in[0]).astype(float)
  
  File_list = np.array(File_list_in_2[0])
  Age_list = np.array(Age_list_in_2[0]).astype(float)
  
  #File_list = np.concatenate((File_list_1,File_list_2))
  #Age_list = np.concatenate((Age_list_1,Age_list_2))
  
  os.system('echo predicted_age true_age >> ' + OUT_FILE)
  val_transforms = Compose([AddChannel(), NormalizeIntensity(nonzero=True), ScaleIntensity(), CenterSpatialCrop([160,192,160]), ToTensor()]) 
  
  # create a validation data loader
  val_ds = NiftiDataset(image_files=File_list, labels=Age_list, transform=val_transforms)
  val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # Evaluation
  model.eval() # Don't forget this. BatchNorm will be affected if not in eval mode.
  with torch.no_grad():
   num_correct = 0.0
   metric_count = 0
  
   for val_data in val_loader:
   
      val_images, val_labels = val_data[0].to(device), val_data[1]
      y, bc = dpu.num2vect(val_labels, bin_range, bin_step, sigma)
      val_output = model(val_images)
      
      y = torch.tensor(y, dtype=torch.float32).to(device)
      
      x = val_output[0].reshape([val_output[0].shape[0], -1])
      #loss = dpl.my_KLDivLoss(x, y)
      
    
      x = x.cpu().detach().numpy()
      y = y.cpu().detach().numpy()
    
      prob = np.exp(x)
      pred = prob@bc
      metric_val = MAE_metric(torch.tensor(pred, dtype=torch.float32), val_labels)
     
      metric_count += 1
      num_correct += metric_val.item()
      
      prob = np.exp(x)
      pred = prob@bc
      print(str(metric_val.item()),str(pred[0]),str(val_labels))
      os.system('echo ' + str(pred[0]) + ' >> ' + OUT_FILE + 'ages.txt')
      os.system('echo ' + str(metric_val.item()) + ' >> ' + OUT_FILE + 'metric.txt')
     
   metric_av = num_correct / metric_count
   print('overall MAE: ',str(metric_av))
if __name__ == "__main__":
    main()

      
      
