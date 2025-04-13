import numpy as np
import argparse
import torch
import monai
from torch.utils.data import DataLoader
from monai.data import ImageDataset
from monai.transforms import AddChannel, Compose, ScaleIntensity, ToTensor, NormalizeIntensity, CenterSpatialCrop, Resize

## load inception resenet V2 model in seperate file
from inception_resnet_model import Inception_ResNetv2

def main(input,output,age):
    # Sanity check
    print("Input filename: "+input)
    print("Output filename: "+output)
    print("Biological: "+str(age))

    File_list = [input]
    Age_list = [float(age)]

    # Create a validation data loader
    val_transforms = Compose([AddChannel(), NormalizeIntensity(nonzero=True), Resize((121,145,121),mode="trilinear"), ScaleIntensity(), CenterSpatialCrop([118,142,118]), ToTensor()])
    val_ds = ImageDataset(image_files=File_list, labels=Age_list, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=False)
    
    # Set up 5 models
    device = torch.device('cpu')
    model1 = Inception_ResNetv2(dropout=0).to(device)
    model2 = Inception_ResNetv2(dropout=0).to(device)
    model3 = Inception_ResNetv2(dropout=0).to(device)
    model4 = Inception_ResNetv2(dropout=0).to(device)
    model5 = Inception_ResNetv2(dropout=0).to(device)
    
    # Load the 5 sets of model weights
    model1.load_state_dict(torch.load("/weights/best_MSE_model_inception_flair_r2_e3.pth",map_location=device))
    model1.eval()
    model2.load_state_dict(torch.load("/weights/best_MSE_model_inception_flair_r2_e4.pth",map_location=device))
    model2.eval()
    model3.load_state_dict(torch.load("/weights/best_MSE_model_inception_flair_r2_e6.pth",map_location=device))
    model3.eval()
    model4.load_state_dict(torch.load("/weights/best_MSE_model_inception_flair_r2_e7.pth",map_location=device))
    model4.eval()
    model5.load_state_dict(torch.load("/weights/best_MSE_model_inception_flair_r2_e10.pth",map_location=device))
    model5.eval()  
    
    with torch.no_grad():
        im = monai.utils.misc.first(val_loader)
        val_images, val_labels = im[0].to(device), im[1].to(device)

        ## run model for all 5 epoch
        val_outputs1 = torch.squeeze(model1(val_images))
        val_outputs2 = torch.squeeze(model2(val_images))
        val_outputs3 = torch.squeeze(model3(val_images))
        val_outputs4 = torch.squeeze(model4(val_images))
        val_outputs5 = torch.squeeze(model5(val_images))

        # We output the results
        print('Biological age: ', val_labels.item(), '| Age outputs: ', val_outputs1.item(),val_outputs2.item(),val_outputs3.item(),val_outputs4.item(),val_outputs5.item(), '| Brain Age: ', (val_outputs1.item()+val_outputs2.item()+val_outputs3.item()+val_outputs4.item()+val_outputs5.item())/5)
            
        # We save the brainAge
        txt_file = open(output, 'a')
        txt_file.write(str((val_outputs1.item()+val_outputs2.item()+val_outputs3.item()+val_outputs4.item()+val_outputs5.item())/5))
        txt_file.write("\n")
        txt_file.close()
   
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", dest='input', help="Input nifti filename", required=True)
    parser.add_argument("-output", dest='output', help="Output text filename", required=True)
    parser.add_argument("-age", dest='age', help="Biological age", required=True)
    args = parser.parse_args()
    main(args.input,args.output,args.age)


