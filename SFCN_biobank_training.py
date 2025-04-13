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
from monai.transforms import AddChannel, Compose, RandFlip, Resize, ScaleIntensity, ToTensor, SqueezeDim, AsChannelFirst, NormalizeIntensity, RandScaleIntensity, RandShiftIntensity, RandAdjustContrast, RandGaussianSmooth, RandHistogramShift, RandGaussianNoise, RandSpatialCrop, CenterSpatialCrop

def main():

  OUT_FILE = 'model_data_training_output/SFCN_flair_'

  model = SFCN()
  model = torch.nn.DataParallel(model)
  fp_ = 'SFCN_T1_pretrained_brain_age_weights/run_20190719_00_epoch_best_mae.p'
  model.load_state_dict(torch.load(fp_))
  model.cuda()
  
  bin_range = [15,95]
  bin_step = 2
  sigma = 1
  
  File_list_in = pd.read_csv('traning_imgs_flair_brain.txt',header=None)
  Age_list_in = pd.read_csv('traning_imgs_ages_.txt',header=None)
  
  File_list = np.array(File_list_in[0])
  Age_list = np.array(Age_list_in[0]).astype(float)


  train_transforms = Compose([AddChannel(), ScaleIntensity(), RandScaleIntensity(factors=0.2, prob=0.25), RandShiftIntensity(offsets=0.2, prob=0.25), RandFlip(prob=0.5,spatial_axis=0), RandAdjustContrast(), RandGaussianSmooth(), RandHistogramShift(), RandGaussianNoise(), CenterSpatialCrop([163, 195, 163]), RandSpatialCrop([160,192,160],random_size=False), ToTensor()])
  val_transforms = Compose([AddChannel(), NormalizeIntensity(nonzero=True), ScaleIntensity(), CenterSpatialCrop([160,192,160]), ToTensor()]) 
    
  # Define image dataset, data loader
  check_ds = NiftiDataset(image_files=File_list, labels=Age_list, transform=train_transforms)
  check_loader = DataLoader(check_ds, batch_size=3, num_workers=1, pin_memory=torch.cuda.is_available())
  im, label = monai.utils.misc.first(check_loader)
  print(type(im), im.shape, label)
  
  # create a training data loader
  train_ds = NiftiDataset(image_files=File_list[:-174], labels=Age_list[:-174], transform=train_transforms)
  train_loader = DataLoader(train_ds, batch_size=3, shuffle=True, num_workers=1, pin_memory=torch.cuda.is_available())

  # create a validation data loader
  val_ds = NiftiDataset(image_files=File_list[-174:], labels=Age_list[-174:], transform=val_transforms)
  val_loader = DataLoader(val_ds, batch_size=3, num_workers=1, pin_memory=torch.cuda.is_available())
    
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  val_interval = 2
  best_metric = 100000
  best_metric_epoch = -1
  epoch_loss_values = list()
  metric_values = list()
  MAE_metric = torch.nn.L1Loss()  
  
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)

  for epoch in range(130):
  
    model.train()
    epoch_loss = 0
    epoch_metric = 0 
    step = 0
  
    for batch_data in train_loader:
      step += 1
      inputs, labels = batch_data[0].to(device), batch_data[1]
      #print(inputs.shape)
      optimizer.zero_grad()
  
      y, bc = dpu.num2vect(labels, bin_range, bin_step, sigma)
      #print(bc)
      y = torch.tensor(y, dtype=torch.float32).to(device)
      #print(y)
      
      outputs = model(inputs)
      #print(outputs)
      x = outputs[0].reshape([outputs[0].shape[0], -1])
      #print(x)
      loss = dpl.my_KLDivLoss(x, y)
      
      loss.backward()
      optimizer.step()
        
      x = x.cpu().detach().numpy()#.reshape(-1)
      y = y.cpu().detach().numpy()#.reshape(-1)
      
      #for i in range(0,train_loader.batch_size):
      
      prob = np.exp(x)
      pred = prob@bc
      print(pred,labels)
      metric = MAE_metric(torch.tensor(pred, dtype=torch.float32), labels)
      print(metric)
      
      epoch_loss += loss.item()
      epoch_metric += metric.item()
      epoch_len = len(train_ds) // train_loader.batch_size
      

      print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
    
    scheduler.step()
    #os.system('echo ' + str(step) + ' ' + str(metric.item()) + ' ' + str(loss.item()) + ' >> ' + OUT_FILE + 'train.txt') 
    
    epoch_loss /= step
    epoch_metric /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    os.system('echo ' + str(epoch) + ' ' + str(epoch_metric) + ' ' + str(epoch_loss) + ' >> ' + OUT_FILE + 'train_av.txt')
      
    # Evaluation
    model.eval() # Don't forget this. BatchNorm will be affected if not in eval mode.
    with torch.no_grad():
      num_correct = 0.0
      metric_count = 0
      loss_all = 0
      
      for val_data in val_loader:
        val_images, val_labels = val_data[0].to(device), val_data[1]
        y, bc = dpu.num2vect(val_labels, bin_range, bin_step, sigma)
        val_output = model(val_images)
        
        y = torch.tensor(y, dtype=torch.float32).to(device)
        
        x = val_output[0].reshape([val_output[0].shape[0], -1])
        loss = dpl.my_KLDivLoss(x, y)
        
      
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
      
        prob = np.exp(x)
        pred = prob@bc
        metric_val = MAE_metric(torch.tensor(pred, dtype=torch.float32), val_labels)
        
        metric_count += 1
        num_correct += metric_val.item()
        loss_all += loss.item()
    
    metric_av = num_correct / metric_count
    loss_av = loss_all / metric_count
    print('VAL LOSS',metric_av)
    os.system('echo ' + str(epoch) + ' ' + str(metric_av) + ' ' + str(loss_av) + ' >> ' + OUT_FILE + 'val.txt')
                
    metric_values.append(metric)
    if metric_av < best_metric:
      best_metric = metric_av
      best_metric_epoch = epoch + 1
      torch.save(model.state_dict(), "model_weights/best_SFCN_model_flair.pth")
      print("saved new best metric model")
      print("current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(epoch + 1, metric_av, best_metric, best_metric_epoch))
      #writer.add_scalar("val_accuracy", metric, epoch + 1)
  print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    
if __name__ == "__main__":
    main()

      
      
