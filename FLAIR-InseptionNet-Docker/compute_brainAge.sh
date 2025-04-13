#!/bin/bash

echo ""
echo "This script is the main tool for computing brainAge. We are expecting as input a 3D FLAIR in NIFTI format (nii or nii.gz) and the biological age."
echo ""
echo "Execution: $0 main_filename.nii.gz 35"
echo " "

source /opt/conda-p3.9/etc/profile.d/conda.sh
source ${FSLDIR}/etc/fslconf/fsl.sh
export SCRIPTSDIR="/usr/bin/scripts"
export PATH="${SCRIPTSDIR}:${PATH}"

directory=/data
main_filename=`readlink -e ${1}`
age=${2}

echo "-------------------------------"
echo " Working directory: $directory "
echo " Input data: $main_filename - ${1}"
echo " Biological age: ${2} years"
echo "-------------------------------"

mkdir -p ${directory}
cd ${directory}
rm -rf log.txt results.json /data/error.txt
fslchfiletype NIFTI_GZ ${main_filename} flair.nii.gz
seg_maths flair.nii.gz -range -scl flair.nii.gz
echo ""
echo "-------------------------------"
echo " Bias field correction      "
echo "-------------------------------"
echo ""
if [ ! -f "flair_bfc.nii.gz" ]; then
	/usr/bin/python2 ${SCRIPTSDIR}/niftkBiasFieldCorrection.py -in flair.nii.gz -out flair_bfc.nii.gz -n4
	rm -rf nifTK*
fi
if [ ! -f "flair_bfc.nii.gz" ]; then
	echo "[compute_brainAge.sh] ERROR computing the bias field correction" > error.txt
	exit -1
fi
echo "--------------------------------------"
echo " Bias field correction - COMPLETED      "
echo "--------------------------------------"
echo ""
echo "---------------------------------"
echo " Registration to MNI   "
echo "---------------------------------"
echo ""

if [ ! -f "flair_brainMNI.nii.gz" ]; then
	reg_aladin  \
		-ref ${FSLDIR}/data/standard/MNI152_T1_1mm.nii.gz \
		-flo flair_bfc.nii.gz \
		-aff affine_flair_to_mni.txt \
		-res flair_brainMNI.nii.gz
fi
if [ ! -f "flair_brainMNI.nii.gz" ]; then
	echo "[compute_brainAge.sh.sh] ERROR computing the registration to MNI" > error.txt
	exit -1
fi
echo "---------------------------------"
echo " Registration to MNI - COMPLETED   "
echo "---------------------------------"
echo ""
echo "---------------------------------"
echo " Computing brainAge   "
echo "---------------------------------"
echo ""
if [ ! -f "brainAge.txt" ]; then
	# For running brainAge we need python 3.9 with specific libraries installed in the brainAge environment
	seg_maths flair_brainMNI.nii.gz -removenan -range -scl flair_brainMNI.nii.gz
	. /opt/conda-p3.9/etc/profile.d/conda.sh && conda activate env_brainAge
		python3.9 /usr/bin/scripts/inception_resnet_evaluation_ensemble.py \
				-input /data/flair_brainMNI.nii.gz \
				-output /data/brainAge.txt \
				-age ${age} 2>&1 | grep -v arnin

	# For avoiding that crashes the brainAge graph, if we don't have 
	# brainAge available we assing the biological age
	brainage=`cat /data/brainAge.txt`
	if [ "${brainage}" == "nan" ]; then
		echo "${age}" > /data/brainAge.txt
		echo "${age} assigning biological age because brainAge gives NaN" > /data/brainAge-ERROR.txt
	fi
	
fi
if [ ! -f "brainAge.txt" ]; then
	echo "[compute_brainAge.sh] ERROR computing brainAge" > error.txt
	exit -1
fi
echo "---------------------------------"
echo " BrainAge computation - COMPLETED   "
echo "---------------------------------"
echo ""

echo "-------------------------------"
echo " Building final JSON file     "
echo "-------------------------------"

brainAge=`cat brainAge.txt | awk -F: '{print $1}'`
echo "{ " > results.json
echo "         \"brainAge\": ${brainAge}," >> results.json
echo "         \"biologicalAge\": ${age}" >> results.json
echo "} " >> results.json
echo ""
if [ ! -f "results.json" ]; then
	echo "[compute_brainAge.sh] ERROR computing the final results in the json file" > error.txt
	exit -1
fi
#cat results.json
echo ""
echo "-------------------------------"
echo " Building final JSON file - COMPLETED   "
echo "-------------------------------"
echo ""
rm -rf  flair.nii.gz \
	    flair_bfc.nii.gz \
		tmp* \
		output \
		flair_image.nii.gz \
		*biasfield* \
		*_1mm.nii.gz \
		mapn3 \
		affine*
