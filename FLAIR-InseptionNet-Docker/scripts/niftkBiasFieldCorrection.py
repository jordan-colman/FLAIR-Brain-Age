#! /usr/bin/env python
# -*- coding: UTF-8 -*-

#/*============================================================================
#
#  NifTK: A software platform for medical image computing.
#
#  Copyright (c) University College London (UCL). All rights reserved.
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.
#
#  See LICENSE.txt in the top level directory for details.
#
#============================================================================*/

#
# This script is for Boyes et al. Neuroimage 2008 normalization process. 
#
# The basic method is:
# If the user runs niftkNormalization.py --xml we respond with the XML function contained herein.
# All other command line invocations, we pass the parameters onto the underlying program.

# Import needed libraries
import atexit
import tempfile
from datetime import datetime, date, time
from _niftkCommon import *

###### DEFAULT OPTIONS #######

# This options for n3 has been extracted from Boyes et al. Neuroimage 2008 page 1756
n3_options='-clobber -stop 0.000100 -fwhm 0.050000 -distance 150 -iterations 1000 -shrink 2 -tmpdir /data/tmpn3 -mapping_dir /data/mapn3 '
n4_options='--convergence 0.000100 --FWHM 0.050000 --niters 1000'
dir_output='bfc/'

# Directory configurations for change the default
dir_niftyreg=""
mnidir=""
n3dir=""

###########################################################################
###########################################################################
# Function definition
###########################################################################
###########################################################################

# Begin of cleanup function
def cleanup():
	"""
	###########################################################################
	# Function  : Clean temp working directory 
	###########################################################################
	"""
	global 	dir_output
	global	debug_mode
	
	# Check if we are in the directory that we want to remove
	if os.path.isdir(os.path.join(os.pardir,dir_output)):
		os.chdir(os.pardir)

	#if os.path.isdir(dir_output):
	#	if (not debug_mode) and (len(dir_output)>0) :
	#		shutil.rmtree(dir_output) 

	return
# End of cleanup function

# Begin of bias_field_correction_n3 function
def bias_field_correction_N3(img,dir_output,output_image):
	"""
	###########################################################################
	# def bias_field_correction_n3(img,dir_output,orientation) 
	# Function  : Data normalization
	# Param	    : img, input image 
	# Param	    : dir_output, dir_output where we are working
	# Param     : output_image, output filename	
	###########################################################################
	"""
	global 	MASK_IMAGE
	global	mnidir
	global  dir_niftyreg
	global  n3_options
	
	# Normalization
	if not os.path.isfile('input_image_normalized.nii'):
		# Delete temporally files
		remove_files('*.mnc')
		remove_files(img+'_n3*')
		remove_files(img+'_ATLAS_mask_register*') 
		
		progress_xml(0.10,"Register atlas to target space")
		
		# Register atlas to target space
		if not os.path.isfile('source-to-target-12dof.txt'):	
			execute_command_or_else_stop(dir_niftyreg+'reg_aladin \
				-ref '+img+'.nii.gz \
				-flo '+ATLAS_IMAGE+' \
				-aff source-to-target-12dof.txt \
				-res '+img+'_register.nii.gz \
				-lp 5')
		else:
			print "File source-to-target-12dof.txt exists, we don't repeat the calculation"
		
		progress_xml(0.35,"Moving tissues mask to target space")
		
		# Move tissues mask to target space
		execute_command_or_else_stop(dir_niftyreg+'reg_resample \
			-ref '+img+'.nii.gz \
			-flo '+MASK_IMAGE+' \
			-aff source-to-target-12dof.txt \
			-res '+img+'_ATLAS_mask_register.nii.gz \
			-NN')
				
		# Set tissues mask from float to char
		execute_command_or_else_stop('fslmaths \
			'+img+'_ATLAS_mask_register.nii.gz \
			-bin \
			'+img+'_ATLAS_mask_register-bin.nii.gz \
			-odt char')

		progress_xml(0.40,"Set tissues mask from float to char")

		# Change format to NIFTI
		remove_files(img+'_ATLAS_mask_register.nii.gz') 
		execute_command_or_else_stop('fslchfiletype NIFTI '+img+'_ATLAS_mask_register-bin '+img+'_ATLAS_mask_register.nii')
		remove_files(img+'_ATLAS_mask_register-bin.*') 
		execute_command_or_else_stop('fslchfiletype NIFTI '+img+'.nii.gz '+img+'_2.nii')
		
		# Change format to mnc
		execute_command_or_else_stop(mnidir+'nii2mnc '+img+'_2.nii '+img+'.mnc')
		execute_command_or_else_stop(mnidir+'nii2mnc '+img+'_ATLAS_mask_register.nii '+img+'_ATLAS_mask_register.mnc')
		
		progress_xml(0.45,'Image ready to be normalized')
		
		mask=os.path.join(dir_output,img+'_ATLAS_mask_register.mnc')
		input_file=os.path.join(dir_output,img+'.mnc')
		output_file=os.path.join(dir_output,img+'_n3.mnc')
		output_file_tmp=os.path.join(dir_output,img+'_n3_temp.nii')
		
		progress_xml(0.50,"Starting N3")
		
		# This options for n3 has been extracted from Boyes et al. Neuroimage 2008 page 1756
		if not os.path.isfile(output_file):
			execute_command_or_else_stop(n3dir+'nu_correct '+n3_options+' \
			-mask '+mask+' \
			'+input_file+' \
			'+output_file+'')
				
		progress_xml(0.90,'Image normalized. Change format from mnc to NIFTI')

		# Change format from mnc to NIFTI
		execute_command_or_else_stop(mnidir+'mnc2nii -nii '+output_file+' '+output_file_tmp)
		copy_file_to_destination(output_file_tmp,output_image)		
		remove_files(output_file_tmp)
		remove_files('*.mnc')
	else:
		print "File input_image_normalized.nii exists, we don't repeat the calculation"

	return 
# End of bias_field_correction_n3 function

# Begin of bias_field_correction_n4 function
def bias_field_correction_N4(img,dir_output,output_image):
        """
        ###########################################################################
        # def bias_field_correction_n4(img,dir_output,output_image) 
        # Function  : Data normalization
        # Param     : img, input image 
        # Param     : dir_output, dir_output where we are working
        # Param     : output_image, output filename
        ###########################################################################
        """

        # Normalization
        if not os.path.isfile('input_image_normalized.nii'):
                # Delete temporally files
                remove_files(img+'_n4*')
                remove_files(img+'_ATLAS_mask_register*') 

                progress_xml(0.10,"Register atlas to target space")

                # Register atlas to target space
                if not os.path.isfile('source-to-target-12dof.txt'):
                        execute_command_or_else_stop('reg_aladin \
                                -ref '+img+'.nii.gz \
                                -flo '+ATLAS_IMAGE+' \
                                -aff source-to-target-12dof.txt \
                                -res '+img+'_register.nii.gz \
                                -lp 5')
                else:
                        print "File source-to-target-12dof.txt exists, we don't repeat the calculation"

                progress_xml(0.35,"Moving tissues mask to target space")

                # Move tissues mask to target space
                execute_command_or_else_stop('reg_resample \
                        -ref '+img+'.nii.gz \
                        -flo '+MASK_IMAGE+' \
                        -aff source-to-target-12dof.txt \
                        -res '+img+'_ATLAS_mask_register.nii.gz \
                        -NN')

                # Set tissues mask from float to char
                progress_xml(0.40,"Set tissues mask from float to char")
                execute_command_or_else_stop('fslmaths \
                        '+img+'_ATLAS_mask_register.nii.gz \
                        -bin \
                        '+img+'_ATLAS_mask_register-bin.nii.gz \
                        -odt char')

                progress_xml(0.50,"Starting N4")

                # This options for n4 are the same than n3 in Boyes et al. Neuroimage 2008 page 1756
                if not os.path.isfile(output_image):
                        execute_command_or_else_stop('niftkN4BiasFieldCorrection '+n4_options+' \
                        --inMask '+img+'_ATLAS_mask_register-bin.nii.gz \
                        -i '+img+'.nii.gz \
                        -o '+output_image+' --outBiasField '+output_image.replace('.nii','_biasfieldmap.nii')+' ')

                progress_xml(0.90,'Image bias corrected.')

        else:
                print "File input_image_normalized.nii exists, we don't repeat the calculation"

        return 
# End of bias_field_correction_n4 function


# XML Definition
xml="""<?xml version="1.0" encoding="utf-8"?>
<executable>
   <category>Multiple Sclerosis Tools.Bias field correction</category>
   <title>Bias Field Correction</title>
   <description><![CDATA[This script, provided within @NIFTK_PLATFORM@, is for doing Boyes et al. Neuroimage 2008 normalization process.<br>
   <ul>
   <li><i>Input image</i>, selects the file that you would like to normalize</li>
   <li><i>Orient</i>, indicates in which orientation is the input images: AXIAL, CORONAL or SAGITALL.</li>
   <li><i>Input atlas data</i>, selects the ATLAS data file, ex: ICBM-152/lin-1.0/icbm_avg_152_t1_tal_lin.nii, you need to be careful that it exists overlapping between this image and input image.</li>
   <li><i>Input mask atlas data</i>, selects the ATLAS mask data file, ex: ICBM-152/lin-1.0/icbm_avg_152_t1_tal_lin_mask.nii</li>
   <li><i>Output image</i>, select the name and the directory of the output file where the normalized image will be recorded.</li>
   </ul>
   <br>
    ]]></description>
   <version>@NIFTK_VERSION_MAJOR@.@NIFTK_VERSION_MINOR@.@NIFTK_VERSION_PATCH@</version>
   <documentation-url>http://www.sciencedirect.com/science/article/pii/S1053811907009494</documentation-url>
   <license>BSD</license>
   <contributor>Ferran Prados (UCL)</contributor>
   <parameters>
      <label>Mandatory arguments</label>
      <description>Input image to be normalized and the name of output image</description>
      <image fileExtensions="nii,nii.gz,img">
          <name>inputImageName</name>
          <longflag>in</longflag>
	  <description>Input image name</description>
	  <label>Input image</label>
	  <channel>input</channel>
      </image>
      <boolean>
          <name>n4</name>
          <longflag>n4</longflag>      
          <description>Bias field correction using N4 method</description>
          <label>N4 method</label>
      </boolean>
      <image fileExtensions="nii,nii.gz,img">
          <name>inputHead</name>
          <longflag>atlas</longflag>
	  <description>Input atlas data</description>
	  <label>Input atlas data</label>
	  <channel>input</channel>
      </image>
      <image fileExtensions="nii,nii.gz,img">
          <name>inputMask</name>
          <longflag>mask</longflag>
	  <description>Input atlas mask</description>
	  <label>Input atlas mask</label>
	  <channel>input</channel>
      </image>
      <image fileExtensions="nii,nii.gz,img">
          <name>outputImageName</name>
          <longflag>out</longflag>
	  <description>Output image name</description>
	  <label>Output image</label>
	  <default>output.nii.gz</default>
          <channel>output</channel>
      </image>
      <boolean>
          <name>debugMode</name>
          <longflag>debug</longflag>      
          <description>Debug mode doesn't delete temporary intermediate images</description>
          <label>Debug mode</label>
      </boolean>
      <boolean>
          <name>same</name>
          <longflag>same</longflag>      
          <description>Always use the same temp directory for computing (this option is useful mixed with -debug)</description>
          <label>Same temp directory</label>
      </boolean>
   </parameters>
</executable>"""

# Help usage message definition
help="""This script is for N3 or N4 bias field correction using Boyes et al. Neuroimage 2008 normalization process. 

Usage: niftkNormalization.py -in input_file -out output_file 

Mandatory Arguments:
 
  -in			: is the input file 
  -out			: is the output file 

Optional Arguments: 
  -mask			: is the mask atlas file
  -atlas		: is the atlas data file
  -n4			: runs N4 bias field correction method instead than N3, by default: N3
  -debug		: debug mode doesn't delete temporary intermediate images
  -same			: Always use the same temp directory for computing (this option is useful mixed with -debug)

"""

	
# Program start

# We register cleanup function as a function to be executed at termination 
atexit.register(cleanup)
os.environ['FSLOUTPUTTYPE']='NIFTI_GZ'
# We get the arguments
arg=len(sys.argv)
argv=sys.argv
orientation=''
debug_mode=False
same=False
INPUT_IMAGE=''
OUTPUT_IMAGE=''
MASK_IMAGE=os.path.join(os.getenv('FSLDIR','/usr/share/fsl'), 'data', 'standard', 'MNI152_T1_2mm_brain_mask_dil.nii.gz') # MNI space
ATLAS_IMAGE=os.path.join(os.getenv('FSLDIR','/usr/share/fsl'), 'data', 'standard', 'MNI152_T1_2mm.nii.gz') # MNI space
N3_bfc=True
# If no arguments, we print usage message
if arg <= 4: 
	usage(help)

i=1
# Parse remaining command line options
while i < arg:
    # Clean unnecessary whitespaces
    argv[i]=argv[i].strip()
    if argv[i].upper() in ['--XML','-XML']:
	usage(xml,0)
    
    elif argv[i].upper() in ['--H','--HELP','-H','-HELP']:
	usage(text)

    elif argv[i].upper() in ['--IN','-IN']:
	INPUT_IMAGE=argv[i+1]
	i=i+1

    elif argv[i].upper() in ['--OUT','-OUT']:
	OUTPUT_IMAGE=argv[i+1]
	i=i+1
	
    elif argv[i].upper() in ['--MASK','-MASK']:
	MASK_IMAGE=argv[i+1]
	i=i+1
	
    elif argv[i].upper() in ['--ATLAS','-ATLAS']:
	ATLAS_IMAGE=argv[i+1]
	i=i+1
	
    elif argv[i].upper() in ['--ORIENT','-ORIENT']:
	if argv[i+1].upper() in ['A']:
		orientation="-transverse"
		
	elif argv[i+1].upper() in ['C']: 
		orientation="-coronal"
		
	elif argv[i+1].upper() in ['S']: 
		orientation="-sagittal"
	i=i+1

    elif argv[i].upper() in ['--DEBUG','-DEBUG']:
	debug_mode=True

    elif argv[i].upper() in ['--SAME','-SAME']:
	same=True
	
    elif argv[i].upper() in ['--N4','-N4']:
	N3_bfc=False

    else:
	print "\n\nERROR: option ",argv[i]," not recognised\n\n"
	usage(help)
	
    i=i+1
# end while
	
# Start of the main program
open_progress_xml('Bias Field Correction Starts')

progress_xml(0.02,'Checking programs')

# Check that all programs exist
check_program_exists('fslcpgeom')
check_program_exists('fslchfiletype')
check_program_exists('fslmaths')
check_program_exists('reg_aladin')
check_program_exists('reg_resample')
check_program_exists('reg_tools')
if N3_bfc:
	check_program_exists('nii2mnc')
	check_program_exists('mnc2nii')
	check_program_exists('nu_correct')
else: 
	check_program_exists('niftkN4BiasFieldCorrection')


# Checking the correctness of the output file 
progress_xml(0.03,'Checking input and output file')

# We have an output file name
if OUTPUT_IMAGE == '':
	progress_xml(0.03,"Failed, specify an output filename is needed.")
	exit_program("Failed, specify an output filename is needed.")

# It isn't a directory
if os.path.isdir(OUTPUT_IMAGE) :
	progress_xml(0.04,OUTPUT_IMAGE+" is not a file, select a file")
	exit_program(OUTPUT_IMAGE+" is not a file, select a file")

# It hasn't any file with similar name in the final output directory
if len(glob.glob(os.path.basename(OUTPUT_IMAGE)+'*'))>0 :
	progress_xml (0.05,"There are files with the same name of the output file ("+OUTPUT_IMAGE+") in the output directory. It could be a source of conflicts. Please, change the name, or remove the files.")
	exit_program ("There are files with the same name of the output file ("+OUTPUT_IMAGE+") in the output directory. It could be a source of conflicts. Please, change the name, or remove the files.")

# We put all path in a normalized absolutized version of the pathname
INPUT_IMAGE=os.path.abspath(INPUT_IMAGE)
ATLAS_IMAGE=os.path.abspath(ATLAS_IMAGE)
MASK_IMAGE=os.path.abspath(MASK_IMAGE)
OUTPUT_IMAGE=os.path.abspath(OUTPUT_IMAGE)

# Check if all needed files exist
check_file_exists(INPUT_IMAGE) 
check_file_exists(ATLAS_IMAGE)
check_file_exists(MASK_IMAGE)

# Get specific information
patient='tmp'

# Create the work temp dir
dir_output=os.path.join("/data/",patient)
if not os.path.isdir(dir_output) :
	os.makedirs(dir_output)

# Copy data to work dir
copy_file_to_destination(INPUT_IMAGE,os.path.join(dir_output,"input_image.nii.gz"))

# Go to the output directory
current_dir=os.getcwd()
os.chdir(dir_output)

# Start process
if N3_bfc:
	bias_field_correction_N3("input_image",dir_output,OUTPUT_IMAGE)
else:
	bias_field_correction_N4("input_image",dir_output,OUTPUT_IMAGE)

# Go back to the corresponding directory
os.chdir(current_dir)

progress_xml(1,"Finish")
close_progress_xml(OUTPUT_IMAGE)

# End of the main program
exit(0)
