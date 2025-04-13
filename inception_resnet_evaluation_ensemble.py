import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import numpy as np
import monai
from monai.data import NiftiDataset
from monai.transforms import AddChannel, Compose, RandFlip, Resize, ScaleIntensity, ToTensor, SqueezeDim, AsChannelLast, NormalizeIntensity, CenterSpatialCrop

## load inception resenet V2 model in seperate file
from inception_resnet_model import Inception_ResNetv2

def main():

    File_list_in = pd.read_csv('testing_imgs_flair.txt',header=None)
    Age_list_in = pd.read_csv('testing_imgs_ages.txt',header=None)
    File_list = np.array(File_list_in[0])
    Age_list = np.array(Age_list_in[0]).astype(float)
    print(len(File_list))
    print(len(Age_list))

    val_transforms = Compose([AddChannel(), NormalizeIntensity(nonzero=True), Resize((121,145,121),mode="trilinear"), ScaleIntensity(), CenterSpatialCrop([118,142,118]), ToTensor()])

    # Define image dataset, data loader
    check_ds = NiftiDataset(image_files=File_list, transform=val_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
    im = monai.utils.misc.first(check_loader)

    # create a validation data loader
    val_ds = NiftiDataset(image_files=File_list, labels=Age_list, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

    ## set up 5 models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = Inception_ResNetv2(dropout=0).to(device)
    model2 = Inception_ResNetv2(dropout=0).to(device)
    model3 = Inception_ResNetv2(dropout=0).to(device)
    model4 = Inception_ResNetv2(dropout=0).to(device)
    model5 = Inception_ResNetv2(dropout=0).to(device)
    loss_function = torch.nn.MSELoss()
    MAE_metric = torch.nn.L1Loss()
    
    ## load the 5 sets of modle weights
    model1.load_state_dict(torch.load("best_MSE_model_inception_flair_r2_e3.pth"))
    model1.eval()
    model2.load_state_dict(torch.load("best_MSE_model_inception_flair_r2_e4.pth"))
    model2.eval()
    model3.load_state_dict(torch.load("best_MSE_model_inception_flair_r2_e6.pth"))
    model3.eval()
    model4.load_state_dict(torch.load("best_MSE_model_inception_flair_r2_e7.pth"))
    model4.eval()
    model5.load_state_dict(torch.load("best_MSE_model_inception_flair_r2_e10.pth"))
    model5.eval()  
    
    ## define output file names
    txt_file_age_out_name = 'testing_inception_flair_ages_top5_test.txt'
    txt_file_loss_out_name = 'testing_inception_flair_metric_top5_test.txt'
    
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        loss_sum = 0
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            ## run model for all 5 epoch
            val_outputs1 = torch.squeeze(model1(val_images))
            val_outputs2 = torch.squeeze(model2(val_images))
            val_outputs3 = torch.squeeze(model3(val_images))
            val_outputs4 = torch.squeeze(model4(val_images))
            val_outputs5 = torch.squeeze(model5(val_images))
            print('True age: ', val_labels.item(), '| age outputs: ', val_outputs1.item(),val_outputs2.item(),val_outputs3.item(),val_outputs4.item(),val_outputs5.item(), '| Mean output: ', (val_outputs1.item()+val_outputs2.item()+val_outputs3.item()+val_outputs4.item()+val_outputs5.item())/5)
            txt_file_train = open(txt_file_age_out_name, 'a')

            txt_file_train.write(str(val_outputs1.item())+" "+str(val_outputs2.item())+" "+str(val_outputs3.item())+" "+str(val_outputs4.item())+" "+str(val_outputs5.item())+" "+str((val_outputs1.item()+val_outputs2.item()+val_outputs3.item()+val_outputs4.item()+val_outputs5.item())/5))
            txt_file_train.write("\n")
            txt_file_train.close()
            
            #loss = loss_function(val_outputs.float(), val_labels.float())
            loss = (((val_outputs1.item() + val_outputs2.item() + val_outputs3.item() + val_outputs4.item() + val_outputs5.item() )/5) - val_labels.float())**2
            loss = loss.cpu().detach().numpy()
            metric_val = np.sqrt(loss)
            #metric_val = MAE_metric((val_outputs1.item()+val_outputs2.item()+val_outputs3.item()+val_outputs4.item()+val_outputs5.item())/5.0, val_labels.float())
            print('MSE: ', loss.item(), '| MAE: ', metric_val.item(), '\n')
            txt_file_train = open(txt_file_loss_out_name, 'a')
            txt_file_train.write(str(loss))
            txt_file_train.write(" ")
            txt_file_train.write(str(metric_val))
            txt_file_train.write("\n")
            txt_file_train.close()
            
            metric_count += 1
            num_correct += metric_val
            loss_sum += loss
            #saver.save_batch(val_outputs, val_data[2])
        print('Mean Absolute error: ', str(num_correct/metric_count))
        #saver.finalize()
    
    
if __name__ == "__main__":
    main()


