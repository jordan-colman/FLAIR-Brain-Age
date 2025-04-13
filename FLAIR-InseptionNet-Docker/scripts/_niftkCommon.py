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

####################################################
# Please try and keep exit codes in this file >= 950
####################################################

# Import needed libraries
import sys
import shlex
import subprocess
import time
import platform
import os
import shutil
import glob
import re
import zipfile
import tempfile

NIFTYREGDIR='/home/ferran/bin/niftyreg-ion-install/bin/'
os.environ['FSLOUTPUTTYPE']='NIFTI_GZ'
os.environ['PATH']+=os.pathsep+'/usr2/mrtools/niftyseg-20140602/bin/'+os.pathsep+'/home/ferran/bin/nifty_seg/seg-apps/'
try:    
	NIFTYREGDIR=os.environ['NIFTYREGDIR']
except KeyError:                
	NIFTYREGDIR='/home/ferran/bin/niftyreg-ion-install/bin/'


# Begin of check_for_help_arg function
def check_for_help_arg(argv=None):
	"""
	#####################################################################################
	# def check_for_help_arg(argv=None)
	# Function   : Will check the supplied array for -h, returning 1
	#              if present and zero otherwise.
	# Param      : argv an array, normally of command line arguments.
	#####################################################################################
	"""
  
	we_all_need_help_sometimes=False
	arg=len(argv)
	i=1
	while i < arg:
		if argv[i] in ['--h','--help','-h','-help']:
			we_all_need_help_sometimes=True
	    
		i=i+1
	# end while
	
	return we_all_need_help_sometimes
# End of check_for_help_arg function


# Begin of check_file_exists function
def check_file_exists(filename):
	"""
	#####################################################################################
	# def check_file_exists(filename)
	# Function   : Simply checks if a file exists, and exits if it doesn't.
	# Param      : filename, string with the file name
	#####################################################################################
	"""
  
	if os.path.isfile(filename):
		print 'File '+filename+' exists'
	else:
		exit_program('File '+filename+' does NOT exist!',951)

	if filename == '':
		exit_program('Empty filename supplied',950)
		
	return 
# End of check_file_exists function


# Begin of check_directory_exists function
def check_directory_exists(dirname):
	"""
	#####################################################################################
	# def check_directory_exists(dirname)
	# Function   : Simply checks if a directory exists, and exits if it doesn't.
	# Param      : dirname, directory name
	#####################################################################################
	"""

	if os.path.isdir(dirname):
		print 'Directory '+dirname+' does exist'
	else:
		exit_program('Directory '+dirname+' does NOT exist!',952)

	return
# End of check_directory_exists function	


# Begin of execute_command_or_else_stop function
def execute_command_or_else_stop(command_line,output='OFF',echo='OFF'):
	"""
	#####################################################################################
	# def execute_command_or_else_stop(command_line,echo='OFF')
	# Function   : This is a bit drastic, can be used to execute any command
	#              and stops if the exit code of the command is non-zero.
	# Param      : command_line, a string containing a command. We simply 'eval' it.
	# Param      : output, if 'on' the method will return the execution output.
	# Param      : echo, if 'on' we only print the instruction, it's like a dry run.
	# Return	 : the execution output
	#####################################################################################
	"""
	
	# Before we remove white spaces, newlines and tabs
	pat = re.compile(r'\s+')
	command_line=pat.sub(' ',command_line)
	out=''
	
	# We can run in a dry mode
	if  echo == 'ON':
		print 'Echoing: execute_command_or_else_stop ('+command_line+')'
	else:
		print 'Evaluating: execute_command_or_else_stop ('+command_line+')'
		args = shlex.split(command_line)
		
		if output == 'OFF':
			p = subprocess.Popen(args)							
			p.wait()
		else:
			p = subprocess.Popen(args,
							stdout=subprocess.PIPE,
							stderr=subprocess.PIPE)
			out, err = p.communicate()
			print out
			
		if p.returncode != 0 :
			exit_program('The command ('+command_line+') failed, so emergency stop. '+str(p.returncode),999)
	
	return	out	
# End of execute_command_or_else_stop function


# Begin of execute_command function
def execute_command(command_line,output='OFF'):
	"""
	#####################################################################################
	# def execute_command(command_line,output='OFF')
	# Function   : To Run a command, but print it out first.
	# Param      : command_line, a string containing a command. We simply 'eval' it.
	# Param      : output, if 'on' the method will return the execution output.
	# Return     : the execution output
	#####################################################################################
	"""
	
	# Before we remove white spaces, newlines and tabs
	pat = re.compile(r'\s+')
	command_line=pat.sub(' ',command_line)
	out=''

	# Print and run
	print 'Evaluating: ('+command_line+')'
	args = shlex.split(command_line)
	if output == 'OFF':
		p = subprocess.Popen(args)							
		p.wait()
	else:
		p = subprocess.Popen(args,
					stdout=subprocess.PIPE,
					stderr=subprocess.PIPE)
		out, err = p.communicate()
		print out

	if p.returncode != 0 :
		#exit_program('The command ('+command_line+') failed. '+str(p.returncode),999)
		print 'The command ('+command_line+') failed. '+str(p.returncode)
	
	return out
# End of execute_command function


# Begin of execute_command_or_echo_it function      
def execute_command_or_echo_it(command_line,echo='ON'):
	"""
	#####################################################################################
	# def execute_command_or_echo_it(command_line,echo='ON')
	# Function   : Can run a coomand or just echo it.
	# Param      : command_line, a string containing a command. We simply 'eval' it.
	# Param      : echo, if OFF, run the command, any argument for echo print the 
	#              instruction, it's like a dry run.
	#####################################################################################
	"""
	
	# Before we remove white spaces, newlines and tabs
	pat = re.compile(r'\s+')
	command_line=pat.sub(' ',command_line)
	  
	if echo == "OFF":
		execute_command(command_line)
	else:
		print command_line
	
	return
# End of execute_command_or_echo_it function


# Begin of check_program_exists function
def check_program_exists(program):
	"""
	#####################################################################################
	# def check_program_exists(program)
	# Function   : Checks if a command exists by exploring path directories
	# Param      : program, command name like 'ls' or 'cat' or 'echo' or anything.
	#####################################################################################
	"""
	
	fpath, fname = os.path.split(program)
	result=0
	if fpath:
		if os.path.isfile(program) and os.access(program, os.X_OK):
			result=1
	else:
		for path in os.environ.get('PATH').split(os.pathsep):
			path = path.strip('"')
			exe_file = os.path.join(path, program)
			if os.path.isfile(exe_file) and os.access(exe_file, os.X_OK):
				result=1

	if result == 0:
		exit_program("Didn't find "+program,999)
	
	return
# End of check_program_exists function


# Begin of check_and_run_file function
def check_and_run_file(filename):
	"""
	#####################################################################################
	# def check_and_run_file(filename)
	# Function   : Checks if file filename exists, and if so tries to run it.
	# Param      : filename, is the filename
	#####################################################################################
	"""
	
	check_file_exists(filename)
	execute_command(filename)

	return
# End of check_and_run_file function


# Begin of check_all_files_in_string_exists function
def check_all_files_in_string_exists(filenames_list):
	"""
	#####################################################################################
	# def check_all_files_in_string_exists(filenames_list)
	# Function   : Checks a whole load of files from command line to make sure they exist.
	# Param      : filenames_list, is the string containing file names.
	#####################################################################################
	"""
	
	flist=filenames_list.split()
	for name in flist:
		check_file_exists(name)

	return
# End of check_all_files_in_string_exists function


# Begin of copy_and_unzip_analyze_image function
def copy_and_unzip_analyze_image(from_dir,to_dir,image_name):
	"""
	#####################################################################################
	# def copy_and_unzip_analyze_image(from_dir,to_dir,image_name)
	# Function   : Copies an analyze image to a given directory.
	# Param      : from_dir, from directory
	# Param      : to_dir, to directory
	# Param      : image_name, base file name, no .img or .hdr extension
	#####################################################################################
	"""

	print "Copying "+from_dir+"/"+image_name+" to "+to_dir+"/"+image_name
	
	sourcehdr=os.path.join(from_dir,image_name+".hdr")
	desthdr=os.path.join(to_dir,image_name+".hdr")
	sourceimg=os.path.join(from_dir,image_name+".img.gz")
	destimg=os.path.join(to_dir,image_name+".img.gz")
	
	shutil.copyfile(sourcehdr,desthdr)
	
	if os.path.isfile(sourceimg):
		shutil.copyfile(sourceimg,destimg)
		extractAll(destimg)
	else:
		sourceimg=os.path.join(from_dir,image_name+".img")
		destimg=os.path.join(to_dir,image_name+".img")
		shutil.copyfile(sourcehdr,desthdr)

	return
# End of copy_and_unzip_analyze_image function


# Begin of make_mask function
def make_mask(filename,region,output,dil='',extra=''):
	"""
	#####################################################################################
	# def make_mask(filename,region,output,dil='')
	# Function   : Converts a segmented region into a masked image
	# Param      : filename, image.img (ie. with img extension)
	# Param      : region, file defining the region (no file extension)
	# Param      : output, output file name (no file extension)
	# Param      : dil, number of dilations
	# Param      : extra, extra parameters, if we want to add
	#####################################################################################
	"""
	
	if not dil == '':
		dil='-d '+dil

	if os.environ.get('MIDAS_BIN')!=None:
		command=os.path.join(os.environ.get('MIDAS_BIN'),'makemask')
	else:
		command='makemask'
		
	check_program_exists(command)
	execute_command_or_else_stop(command+' '+filename+' '+region+' '+output+' '+dil+' '+extra)

	return
# End of make_mask function


# Begin of check_int function
def check_int(val):
	"""
	#####################################################################################
	# def check_int(val)
	# Function   : Check that the string input is an integer
	# Param      : val, input to be checked
	#####################################################################################
	"""
	
	result=True
	try: 
		int(val)
	except ValueError:
		result=False
		
	return result
# End of check_int function


# Begin of checkf_writeable function
def checkf_writeable(filename):
	"""
	#####################################################################################
	# def checkf_writeable(filename)
	# Function   : Check that we can create/write to a specified file
	# Param      : filename, path to the file to write
	#####################################################################################
        """
	
	result=0
	try:
		if not os.path.exists(filename):
			#no existing file, so try to create a temp file:
			tempFile = open(filename, 'w')
			tempFile.close()
			os.remove(filename)
		else:
			#existing file:
			#try to get a write lock on the filepath:
			file = open(filename, 'r+')
	except:
		result=2
		print filename+' is not writable'
		
	return result
# End of checkf_writeable function


# Begin of exit_program function
def exit_program(text,val=950):
	"""
	#####################################################################################
	# def exit_program(text,val=100)
	# Function   : Exit with a message, status.
	# Param      : text, message before exit, it will be displayed throught stderr.
	# Param      : val, termination value
	#####################################################################################
	"""
	
	sys.stderr.write(os.linesep+text+os.linesep+os.linesep)
	if isinstance(val,int):
		exit(val)
	else:
		exit(950)

	return
# End of exit_program function


# Begin of message function
def message(text):
	"""
	#####################################################################################
	# def message(text)
	# Function   : Write out a pretty message
	# Param      : text, text to be displayed
	#####################################################################################
	"""
	
	print "=============="
	print text
	print "=============="
	
	return
# End of message function


# Begin of copyimg function
def copyimg(source,dest): 
	"""
	#####################################################################################
	# def copyimg(source,dest)
	# Function   : Requires documentation, and possibly a usage analysis.
	# Param      : source, path from original image
	# Param      : dest, path to the destination
	#####################################################################################
	"""
	
	if os.environ.get('ANCHANGE')!=None: 
		execute_command(os.environ.get('ANCHANGE')+' '+source+' '+dest+' -sex m')
	else:
		print 'ANCHANGED is not installed, we copy images in other way.'
		copy_file_to_destination(source,dest)

	return
# End of copyimg function


# Begin of make_midas_mask function
def make_midas_mask(filename,region,output,dilations):
	"""
	#####################################################################################
	# def make_midas_mask(img,region,output,dilations)
	# Function   : Requires documentation, and possibly a usage analysis.
	# Param      : filename, image.img (ie. with img extension)
	# Param      : region, file defining the region (no file extension)
	# Param      : output, output file name (no file extension)
	# Param      : dilations, number of dilations
	#####################################################################################
	"""
	
	dilationArg=''
	if dilations > 0:
		dilationArg=str(dilations)
	
	make_mask (img,region,output,dilations,'-bpp 16 -val 255')

	return
# End of make_midas_mask function


# Begin of open_progress_xml function
def dos_2_unix(filename):
	"""
	#####################################################################################
	# def dos_2_unix(filename)
	# Function   : Checks if a file is file is writable, and if it is runs dos2unix on it.
	#              If not writable, method will print a message, but soldier on regardless.
	# Param      : filename, string with the file name
	#####################################################################################
	"""

	if checkf_writeable(filename) > 0:
		print "Can't run dos2unix on file="+filename+", but will continue anyway"
	else:
		print "Running dos2unix on file="+filename
		with tempfile.NamedTemporaryFile(delete=False) as fh:
			for line in open(filename):
				line = line.rstrip()
				fh.write(line + '\n')
			os.rename(filename, filename + '.bak')
			os.rename(fh.name, filename)
	
	return
# End of dos_2_unix function


# Begin of open_progress_xml function
def open_progress_xml(text):
	"""
	#####################################################################################
	# def open_progress_xml(text)
	# Function   : this function open the log progress for Command Line Module plugin
	# Param      : text, it is an information message, usually the program name
	#####################################################################################
	"""
	
	print '<filter-start>' 
	print '<filter-name>'+text+'</filter-name>'
	print '<filter-comment>'+text+'</filter-comment>'
	print '</filter-start>' 
	time.sleep(1) # sleep 1000ms to avoid squashing the last progress event with the finished event

	return
# End of open_progress_xml function


# Begin of close_progress_xml function
def close_progress_xml(text):
	"""
	#####################################################################################
	# def close_progress_xml(text)
	# Function   : this function close the log progress for Command Line Module plugin
	# Param      : text, it is an information message with name of output image
	#####################################################################################
	"""
	
	print '<filter-result name="outputImageName">'+text+'</filter-result>'
	print '<filter-result name="exitStatusOutput">Normal exit</filter-result>'
	print '<filter-progress>1</filter-progress>'
	print '<filter-end><filter-comment>Finished successfully.</filter-comment></filter-end>'

	return
# End of close_progress_xml function


# Begin of progress_xml function
def progress_xml(percentage,text):
	"""
	#####################################################################################
	# def progress_xml(percentage,text)
	# Function   : this function output a log of the progress for Command Line Module plugin
	# Param      : percentage, float [0..1], it is the progression percentage
	# Param      : text, it is an information message
	#####################################################################################
	"""
	
	print '<filter-progress-text progress="'+str(percentage)+'">'+text+'</filter-progress-text>'
	time.sleep(1) # sleep 1000ms to avoid squashing the last progress event with the finished event
	
	return
# End of progress_xml function


# Begin of copy_file_to_destination function	
def copy_file_to_destination(source,destination):
	"""
	#####################################################################################
	# def copy_file_to_destination(source,destination)
	# Function  : Copy a file to a specific destination file, depending on the format 
	#             it will be changed or not
	# Param	    : source, path to the source image
	# Param     : destination, path to the destination image
	# Examples of it uses:
	# copy_file_to_destination ("img/foo.nii.gz","/tmp/foo.nii.gz") it copies it
	# copy_file_to_destination ("img/foo.nii.gz" "/tmp/foo.nii") it copies it and it changes it of format
	# copy_file_to_destination ("foo.hdr","/tmp/foo2.img") and it copies foo.hdr and foo.img 
	#                          to /tmp/foo.hdr and /tmp/foo.img
	#
	#####################################################################################
	"""

	type1 = get_output_file_type(source)
	type2 = get_output_file_type(destination)

	if type1 == type2: 
		if type1 == 'ANALYZE_GZ' or  type1 == 'ANALYZE':
			sourcehdr=source.str('.img','.hdr')
			sourceimg=source.str('.hdr','.img')
			desthdr=destination.str('.img','.hdr')
			destimg=destination.str('.hdr','.img')
			shutil.copyfile(sourcehdr,desthdr)
			shutil.copyfile(sourceimg,destimg)
		else:
			shutil.copyfile(source,destination)
	else:
		execute_command_or_else_stop('fslchfiletype '+type2+' '+source+' '+destination)
		
	return
# End of copy_file_to_destination function		


# Begin of get_output_file_type function
def get_output_file_type(image):
	"""
	#####################################################################################
	# def get_output_file_type(image)
	# Function  : Get the file type
	# Param	    : image, input image
	# Return    : image type
	#####################################################################################
	"""
	
	output_type='NIFTI_GZ'
	
	if image.endswith('.nii.gz'):
		output_type='NIFTI_GZ'
	elif image.endswith('.nii'):
		output_type='NIFTI'
	elif image.endswith('.img.gz') or image.endswith('.hdr.gz'):
		output_type='ANALYZE_GZ'
	elif image.endswith('.img') or image.endswith('.hdr'):
		output_type='ANALYZE'
	else:
		output_type=''

	return output_type
# End of get_output_file_type function


# Begin of get_output_file_extension function
def get_output_file_extension(image):
	"""
	#####################################################################################
	# def get_output_file_extension(image)
	# Function  : Get the file extension
	# Param	    : image, input image
	# Return    : image extension
	#####################################################################################
	"""
	
	output_type=get_output_file_type(image)
	extension=''
	if output_type in  ['NIFTI_GZ']:
		extension='.nii.gz'
	elif output_type in  ['NIFTI']: 
		extension='.nii'
	elif output_type in ['ANALYZE_GZ']: 
		extension='.hdr.gz'
	elif output_type in ['ANALYZE']:
		extension='.hdr'

	return extension
# End of get_output_file_extension function


# Begin of get_file_name function
def get_file_name(image):
	"""
	#####################################################################################
	# def get_file_name(image)
	# Function  : Get the file name without extension
	# Param	    : image, input image
	# Return    : file name
	#####################################################################################
	"""
	
	image=os.path.basename(image)
	extension=get_output_file_extension(image)
	name=image.replace(extension,'')

	return name
# End of get_output_file_extension function

# Begin of extractAll function
def extractAll(zip_name):
	"""
	#####################################################################################
	# def extractAll(zipName)
	# Function  : extract all the files of a zip file taken into account OS
	# Param	    : zip_name, name of the file to be unzipped
	#####################################################################################
	"""
	
	z = ZipFile(zip_name)
	for f in z.namelist():
		if f.endswith('/'):
			os.makedirs(f)
		else:
			z.extract(f)

	return
# End of extractAll function


# Begin of remove_files function
def remove_files(pattern):
	"""
	#####################################################################################
	# def remove_files(pattern)
	# Function  : removes all files that follow the pattern Ex: *.txt, file*,...
	# Param	    : pattern, names of the files to be removed
	#####################################################################################
	"""
	
	files=glob.glob(pattern)
	for filename in files:
		removeFile(filename)
		
	return
# End of removesFiles function


# Begin of removeFile function
def removeFile(filename):
	if os.path.isfile(filename):
		os.unlink (filename)

	return 
# End of removeFile function


# Begin of usage function
def usage(text,val=127):
	"""
	#####################################################################################
	# def usage(text,val=127)
	# Function   : We respond with the text.
	# Param      : text, a string containing the text to print out.
	#####################################################################################
	"""
	
	print text
	exit(val)
	
	return
# End of usage function	


# Begin of reg_resample function
def reg_resample(ref,flo,transform,result,flags=''):
	"""
	#####################################################################################
	# def reg_resample(ref,flo,type,transform,result)
	# Function   : Resample the image
	#####################################################################################
	"""
  	global NIFTYREGDIR

	if not os.path.isfile(result):
		execute_command_or_else_stop(NIFTYREGDIR+'reg_resample \
				-ref '+ref+' \
				-flo '+flo+' \
				-trans '+transform+' \
				-res '+result+' \
				'+flags+' ')
	else:
		print 'File '+result+' does exist, we don\'t repeat the calculation' 	
	
	return 
# End of reg_resample function


# Begin of reg_resample_2_space function
def reg_resample_2_space(ref,mid,flo,trans1,trans2,result,interpolation=''):
	"""
	#####################################################################################
	# def reg_resample_2_space(ref,mid,flo,trans1,trans2,result)
	# Function   : Resample the image
	#####################################################################################
	"""
  	global NIFTYREGDIR

	if not os.path.isfile(result):
		execute_command_or_else_stop(NIFTYREGDIR+'reg_resample \
				-ref '+mid+' \
				-flo '+flo+' \
				-def '+trans1+' \
				-res temp1.nii \
				'+interpolation+' -voff ')
		
		execute_command_or_else_stop(NIFTYREGDIR+'reg_resample \
				-ref '+ref+' \
				-flo temp1.nii \
				-def '+trans2+' \
				-res '+result+' \
				'+interpolation+' -voff ')

		removeFile ('temp1.nii')
	else:
		print 'File '+result+' does exist, we don\'t repeat the calculation' 	
	
	return 
# End of reg_resample_2_space function


# Begin of update_sform function
def update_sform(ref,transform,result):
	"""
	#####################################################################################
	# def update_sform(ref,transformation,result)
	# Function   : update the sform header of a file
	#####################################################################################
	"""
	global NIFTYREGDIR

	if not os.path.isfile(result):
		execute_command_or_else_stop(NIFTYREGDIR+'reg_transform \
				-updSform \
				'+ref+' \
				'+transform+' \
				'+result+' ')
# End of update_sform function


# Begin of reset_scale function
def reset_scale(input_file,flag=''):
	"""
	#####################################################################################
	# def reset_scale(input_file)
	# Function   : update the sform header of a file
	#####################################################################################
	"""
	execute_command_or_else_stop('seg_maths \
			'+input_file+' \
			'+flag+' \
			-range -scl \
			'+input_file+' ')

# End of reset_scale function


# Begin of register_images function
def register_images(ref,flo,transformation,result='',flags=''):
	"""
	#####################################################################################
	# def register_images(ref,flo,transformation,result,flags)
	# Function   : Register two images using reg_aladin
	#####################################################################################
	"""

	register_images_mask(ref,'',flo,'',transformation,result,flags)

# End of register_images function


# Begin of register_images_mask function
def register_images_mask(ref,rmask,flo,fmask,transformation,result='',flags=''):
	"""
	#####################################################################################
	# def register_images_mask(ref,flo,transformation,result,flags)
	# Function   : Register two images using reg_aladin
	#####################################################################################
	"""
	global NIFTYREGDIR

	if result=='':
		result_file='remove.nii.gz'
	else:
		result_file=result

	if rmask=='':
		rmask=''
	else:
		rmask='-rmask '+rmask

	if fmask=='':
		fmask=''
	else:
		fmask='-fmask '+fmask

	if not os.path.isfile(transformation):
		execute_command_or_else_stop(NIFTYREGDIR+'reg_aladin \
				-ref '+ref+' \
				'+rmask+' \
				-flo '+flo+' \
				'+fmask+' \
				-aff '+transformation+' \
				-res '+result_file+' \
				'+flags+' ')
	if result=='':
		removeFile (result_file)

# End of register_images_mask function
