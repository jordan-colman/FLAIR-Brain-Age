# FLAIR-Brain-Age
3D T2-FLAIR brain age prediction model. 

Seperate models trained on 1,394 healthy control 3D FLAIR and 3D T1w MRIs. 3D-Inception-ResNet was compared to DenseNet169 and Simple Fully Convolutional Network (SFCN) models. Details of training and results under review for publication and further infomration to follow here. Code for model architecture, training and evaluation included here for the 3D-Inseption_ResNet model and the DenseNet and SFCN models. 

SFCN model code taken and adpted from: https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain

Trained model weights can be found at: 
https://drive.google.com/drive/folders/1yp8r0ssYllz11cT-fDTNmC2vSXR5f7II?usp=sharing

Model was trained on inputed imaged aligned to MNI space 1mm^3. Images resampled to lower resolution in programme.

Model perfomance on testing set of 174 scans is as follows:

|      Model          | Modality | Weights                       | Mean Absolute Error (MAE years) |
| :-----------------: | :------: | :---------------------------: | :-----------------------------: |
| 3D Inception-ResNet | FLAIR    | x5 Ensemble                   |  2.81                           |
| 3D Inception-ResNet | T1w      | x5 Ensemble                   |  2.84                           |
| DenseNet169         | FLAIR    | best_model_densenet_flair.pth |  3.74                           |
| DenseNet169         | T1w      | best_model_densenet_t1.pth    |  3.43                           |
| SFCN                | FLAIR    | best_SFCN_model_flair.pth     |  4.12                           |
| SFCN                | T1w      | best_SFCN_model_t1.pth        |  3.45                           |


The code can also be run in a Docker of the 3D Inspetion-ResNet ensemble model to evaluate 3D FLAIR images and can be dowloaded at: 
https://drive.google.com/file/d/1QHBC0mVXoOBRtZEDbVI-V0S45hMFgbnY/view?usp=sharing

To run docker use following example code (where "flair.nii.gz" is scan to be analysied and "chronological age" is true age at time of scan):
Run Docker as docker run --rm \ -v `pwd`:/data \ flair-brainage-v1.0 \ compute_brainAge.sh \ /data/flair.nii.gz \ "chronological age"
