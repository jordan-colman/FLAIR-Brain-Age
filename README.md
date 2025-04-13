# FLAIR-Brain-Age
3D T2-FLAIR brain age prediction model 

Seperate models trained on 1,394 healthy control 3D FLAIR MRIs and 3D T1w MRIs. 3D-Inception-ResNet was compared to DenseNet. Details of training and results under review for puplication and further infomration to follow here. Code for model architecture, training and evaluation included here for the 3D-Inseption_ResNet and DenseNet models. 

Trained model weights can be found at: 
https://drive.google.com/drive/folders/1yp8r0ssYllz11cT-fDTNmC2vSXR5f7II?usp=sharing

Model was trained on inputed imaged aligned to MNI space 1mm^3. Images resampled to lower resolution in programme.

The code can also be run in a Docker of the 3D Inspetion-ResNet ensemble model to evaluate 3D FLAIR images and can be dowloaded at: 
https://drive.google.com/file/d/1QHBC0mVXoOBRtZEDbVI-V0S45hMFgbnY/view?usp=sharing

Run Docker as docker run --rm \ -v `pwd`:/data \ flair-brainage-v1.0 \ compute_brainAge.sh \ /data/flair.nii.gz \ "chronological age"
