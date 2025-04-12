import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import monai
from monai.data import NiftiDataset
from monai.transforms import AddChannel, Compose, RandFlip, Resize, ScaleIntensity, ToTensor, SqueezeDim, AsChannelFirst, NormalizeIntensity, RandScaleIntensity, RandShiftIntensity, RandAdjustContrast, RandGaussianSmooth, RandHistogramShift, RandGaussianNoise, RandSpatialCrop, CenterSpatialCrop

## import inception_resnet_v2 model in seperate file
from inception_resnet_model import Inception_ResNetv2


def main():

    ## load image file list and ages. must be a text file with no header and a unique image file name on each row with ages in corresponding file in the exact same order.
    File_list_in = pd.read_csv('traning_imgs_flair.txt',header=None)
    Age_list_in = pd.read_csv('traning_imgs_ages.txt',header=None)
    File_list = np.array(File_list_in[0])
    Age_list = np.array(Age_list_in[0]).astype(float)
    print(len(File_list))
    print(len(Age_list))
    
    ## set up image transfomrations and data augmentation in this case the 1mm3 images are resized to 1.45mm isotropic 
    ## various data agmentations used which are outlined in monai documentation. The image is randomly cropped to 3 voxels smaller in each dimension to allow for random shift of voxels within convolutional kernels
    train_transforms = Compose([AddChannel(), NormalizeIntensity(nonzero=True), Resize((121,145,121),mode="trilinear"), ScaleIntensity(), RandScaleIntensity(factors=0.2, prob=0.25), RandShiftIntensity(offsets=0.2, prob=0.25), RandFlip(prob=0.5,spatial_axis=0), RandAdjustContrast(), RandGaussianSmooth(), RandHistogramShift(), RandGaussianNoise(), RandSpatialCrop([118,142,118],random_size=False), ToTensor()])
    
    ## set up validation input trasforms without data augmentation. centre croped used instead of random crop
    val_transforms = Compose([AddChannel(), NormalizeIntensity(nonzero=True), Resize((121,145,121),mode="trilinear"), ScaleIntensity(), CenterSpatialCrop([118,142,118]), ToTensor()]) 

    # Define image dataset, data loader
    check_ds = NiftiDataset(image_files=File_list, labels=Age_list, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=10, num_workers=1, pin_memory=torch.cuda.is_available())
    im, label = monai.utils.misc.first(check_loader)
    
    # create a training data loader. leave a set number of images out from the end of the list out for validation in this case 174 images. batch size set at 10 for a 12Gb GPU.
    train_ds = NiftiDataset(image_files=File_list[:-174], labels=Age_list[:-174], transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=10, shuffle=True, num_workers=1, pin_memory=torch.cuda.is_available())

    # create a validation data loader. 
    val_ds = NiftiDataset(image_files=File_list[-174:], labels=Age_list[-174:], transform=val_transforms) 
    val_loader = DataLoader(val_ds, batch_size=10, num_workers=1, pin_memory=torch.cuda.is_available())

    # Load 3D inception-ResNet-V2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Inception_ResNetv2(in_channels=1).to(device)
    ## define loss function and metric of the model in this case mean square error loss and mean absolute error.
    loss_function = torch.nn.MSELoss()
    MAE_metric = torch.nn.L1Loss()
    ## define the optimiser, ADAM is is used in this case with the commonly used learning rate of 1e-4 with a weight decay of 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,weight_decay=1e-5)
    ## define max number of epoch
    max_epochs = 200
    ## additional learning rate decay schduling is added for further regulization of the model. Schduling used as recomended by DeepLab and commonly used in the liturature.
    lambda1 = lambda epoch: (1 - (epoch / max_epochs))**0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    ## define and make text files to write traning progress monitoring to.
    txt_file_train_av_name = 'inception_flair_train_epoch_average_e1.txt'
    txt_file_val_name = 'inception_flair_validation_e1.txt'
    
    txt_file_train = open(txt_file_val_name, 'a')
    txt_file_train.write("inception_net_flair")
    txt_file_train.write("\n")
    txt_file_train.close()
    
    txt_file_train = open(txt_file_train_av_name, 'a')
    txt_file_train.write("inception_net_flair")
    txt_file_train.write("\n")
    txt_file_train.close()
    
    
    # start a typical PyTorch training
    # early stopping is utilized with a patients of 20 epochs defined here.
    val_interval = 1
    best_metric = 100000
    wait_num = 0
    patience = 20
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    #writer = SummaryWriter()
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{200}")
        
        model.train()
        epoch_loss = 0
        epoch_metric = 0

         
        step = 0
        for batch_data in train_loader:
            step += 1
            ##get iput tensor and lables
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            ## run model
            outputs = torch.squeeze(model(inputs))
            print(outputs,labels)
            ## calculate loss and back propogate
            loss = loss_function(outputs.float(), labels.float()) 
            metric = MAE_metric(outputs.float(), labels.float())
            loss.backward()
            optimizer.step()
            ## add batch loss and metric to running total
            epoch_loss += loss.item()
            epoch_metric += metric.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, metric: {metric} train_loss: {loss.item():.4f}")
        ## calc epoch mean loss and metric and write to monitoring file
        epoch_loss /= step
        epoch_metric /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        txt_file_train_av = open(txt_file_train_av_name, 'a')
        txt_file_train_av.write(f"epoch {epoch + 1} average metric: {epoch_metric} average loss: {epoch_loss:.4f}")
        txt_file_train_av.write("\n")
        txt_file_train_av.close()
        
        ## step forward chedular to decay LR
        scheduler.step()

        ## run evaluation images to monitor progress
        model.eval()
        with torch.no_grad():
            num_correct = 0.0
            metric_count = 0
            loss_all = 0
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                val_outputs = torch.squeeze(model(val_images))
                loss = loss_function(val_outputs.float(), val_labels.float())
                metric_val = MAE_metric(val_outputs.float(), val_labels.float())
                #value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                metric_count += 1
                num_correct += metric_val.item()
                loss_all += loss.item()
            metric_av = num_correct / metric_count
            loss_av = loss_all / metric_count
            print('VAL LOSS',metric_av)
            
            ## write mean loss and metric to file
            txt_file_val = open(txt_file_val_name, 'a')
            txt_file_val.write(f"epoch {epoch + 1}, Val_metric: {metric_av} Val_loss: {loss_av:.4f}")
            txt_file_val.write("\n")
            txt_file_val.close()
            
            metric_values.append(metric)
            ## early stopping monitoring if current epoch validation metric not improved will add 1 to wait, with patience of 20 to stop the traning early.
            if metric_av > best_metric:
                wait_num += 1
                if wait_num >= patience:
                    txt_file_val = open(txt_file_val_name, 'a')
                    txt_file_val.write(f"stopped early at epoch {epoch + 1}, best val metric: {best_metric} at epoch {best_metric_epoch}")
                    txt_file_val.write("\n")
                    txt_file_val.close()
                    break
            
            else:
                wait_num = 0
            
            ## update saved weights if current validation epoch metric is best performing so far
            if metric_av < best_metric:
                best_metric = metric_av
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_model_inception_net_flair_e1.pth")
                print("saved new best metric model")
            print(
                "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                    epoch + 1, metric_av, best_metric, best_metric_epoch
                )
            )
            
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

if __name__ == "__main__":
    main()

