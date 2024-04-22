
# coding: utf-8

# #### Post analysis of time-series linking: get measurements

import sys, traceback
import cv2
import numpy as np
import argparse
import string
import pickle as pkl
from plantcv import plantcv as pcv
import matplotlib
import utilities as u
from skimage import img_as_ubyte
import os

def options():
	parser = argparse.ArgumentParser(description="Imaging processing with PlantCV.")
	parser.add_argument("-i", "--image", help="Input image file.", required=True)
	parser.add_argument("-r","--result", help="Result file.", required= True)
	parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
	parser.add_argument("-w","--writeimg", help="Write out images.", default=False, action="store_true")
	parser.add_argument("-D", "--debug", help="Turn on debug, prints intermediate images.")
	args = parser.parse_args()
	return args

def main():

	args = options()
	file_name = args.image
	# Read in thermal csv data
	thermal_data,path,filename = pcv.readimage(filename=file_name, mode="csv")
	#print(thermal_data.shape)
	# Create RGB img path from csv path
	thermal_path, image_name = os.path.split(path)
	rgb_imgpath = thermal_path + "/RGB/" + filename[:-4] + "_RGB.pkl"
 	# Read in the corresponding, registered RGB image
	unregistered = pkl.load(open(rgb_imgpath, 'rb'))
	# Extract the already registered RGB image
	rgb_unregistered = unregistered['rgb_fullres']
	# Read in the model for registration
	model = pkl.load(open("/shares/mgehan_share/kmurphy/thermal/03112022/workflows/main_EDIT/model_pkl/model.pkl" , 'rb'))
	rgb_registered = u.regist_rgb_therm(rgb_unregistered, thermal_data, model, debug_mode = 'plot')
	rgb_registered = img_as_ubyte(rgb_registered)
	# Create binary mask of plants
	a = pcv.rgb2gray_lab(rgb_registered,'a')
	bin_img = pcv.threshold.binary(a, 118, 255, "dark")
	er_img = pcv.erode(gray_img=bin_img,ksize=3, i=4)
	m_blur = pcv.median_blur(er_img, 12)
	mask   = pcv.fill_holes(bin_img=m_blur)
	#mask = pcv.fill_holes(bin_img=mask1)
	#select ROIs using the rescale image
	rois, roi_hierarchy = pcv.roi.multi(img=rgb_registered, coord=[(130,118), (331,152), (495,139), (135,300), (305,330), (490,316)], radius=85)
	#name the rois
	all_masks = []
	labels = ["HLP_1", "WT_1", "HLP_2", "WT_2","HLP_3","WT_3"]
	id_objects, obj_hierarchy = pcv.find_objects(img=rgb_registered, mask=mask)
	#loop over the grid of ROIs to analyze plants separately
	for i in range(0, len(rois)):
		roi = rois[i]
		hierarchy = roi_hierarchy[i]
		label=labels[i]
    	# Find objects
		filtered_contours, filtered_hierarchy, filtered_mask, filtered_area = pcv.roi_objects(
			img=rgb_registered, roi_type="cutto", roi_contour=roi, roi_hierarchy=hierarchy, object_contour=id_objects,
			obj_hierarchy=obj_hierarchy)
		if filtered_area > 0: # If no plant detected, then skip analysis
			analysis_img = pcv.analyze_thermal_values(thermal_array=thermal_data, mask=filtered_mask, histplot=True, label=label)

	# Create pseudocolored image on entire mask
	matplotlib.rcParams["image.interpolation"] = "nearest"
	x=50
	y=0
	h=450
	w=510
	cropped_thermal_data = thermal_data[y:y + h, x:x + w]
	cropped_mask = mask[y:y + h, x:x + w]
	pseudo_img = pcv.visualize.pseudocolor(gray_img = cropped_thermal_data, mask=cropped_mask, cmap='jet',
                                           min_value=20, max_value=45)

	pcv.print_image(pseudo_img,os.path.join(args.outdir,filename[:-4]+'_pseudo.jpg'))

	pcv.print_results(filename=args.result)



if __name__ == '__main__':
    main()
