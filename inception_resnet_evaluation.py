import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import monai
from monai.data import NiftiDataset
from monai.transforms import AddChannel, Compose, RandFlip, Resize, ScaleIntensity, ToTensor, SqueezeDim, AsChannelLast, NormalizeIntensity, CenterSpatialCrop

## load inception resnet v2 model
from inception_resnet_model import Inception_ResNetv2

def main():

    ## load image files and ages
    File_list_in = pd.read_csv('testing_imgs_flair.txt',header=None)
    Age_list_in = pd.read_csv('testing_imgs_ages.txt',header=None)
    File_list = np.array(File_list_in[0])
    Age_list = np.array(Age_list_in[0]).astype(float)
    print(len(File_list))
    print(len(Age_list))

    ## define input image transforms with no data augmentation
    val_transforms = Compose([AddChannel(), NormalizeIntensity(nonzero=True), Resize((121,145,121),mode="trilinear"), ScaleIntensity(), CenterSpatialCrop([118,142,118]), ToTensor()])

    # Define image dataset, data loader
    check_ds = NiftiDataset(image_files=File_list, labels=Age_list, transform=val_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
    im, label = monai.utils.misc.first(check_loader)
    print(type(im), im.shape, label)

    # create a validation data loader
    val_ds = NiftiDataset(image_files=File_list, labels=Age_list, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

    # load inception model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Inception_ResNetv2(dropout=0).to(device)
    loss_function = torch.nn.MSELoss()
    MAE_metric = torch.nn.L1Loss()
    
    model.load_state_dict(torch.load("best_MSE_model_inception_flair_r2_e3.pth"))
    model.eval()
    #optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    
    
    ## define output file names
    txt_file_age_out_name = 'testing_inception_flair_e1_ages.txt'
    txt_file_loss_out_name = 'testing_inception_flair_e1_metric.txt'
    
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        loss_sum = 0
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            val_outputs = torch.squeeze(model(val_images))
            print('True age: ', val_labels.item(), '| Model Output:', val_outputs.item())
            txt_file_train = open(txt_file_age_out_name, 'a')
            txt_file_train.write(str(val_outputs.item()))
            txt_file_train.write("\n")
            txt_file_train.close()
            
            loss = loss_function(val_outputs.float(), val_labels.float())
            metric_val = MAE_metric(val_outputs.float(), val_labels.float())
            print('MSE: ', loss.item(), '| MAE: ', metric_val.item(), '\n')
            txt_file_train = open(txt_file_loss_out_name, 'a')
            txt_file_train.write(str(loss.item()))
            txt_file_train.write(" ")
            txt_file_train.write(str(metric_val.item()))
            txt_file_train.write("\n")
            txt_file_train.close()
            
            metric_count += 1
            num_correct += metric_val.item()
            loss_sum += loss.item()
        metric = num_correct / metric_count
        final_loss = loss_sum / metric_count
        print("evaluation metric:", metric)
        print("evaluation loss:", final_loss)
        txt_file_train = open(txt_file_loss_out_name, 'a')
        txt_file_train.write("mean")
        txt_file_train.write("\n")
        txt_file_train.write(str(final_loss))
        txt_file_train.write(" ")
        txt_file_train.write(str(metric))
        txt_file_train.write("\n")
        txt_file_train.close()
    
if __name__ == "__main__":
    main()

