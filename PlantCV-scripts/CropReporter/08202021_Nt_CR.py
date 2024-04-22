#!/usr/bin/env python
import os
import argparse
from plantcv import plantcv as pcv
from skimage.util import img_as_ubyte

### Parse command-line arguments
def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i", "--image", help="Input INF file.", required=True)
    parser.add_argument("-r", "--result", help="Result file", required=True)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=True)
    parser.add_argument("-w", "--writeimg", help="Save output images.", action="store_true")
    parser.add_argument("-D", "--debug", help="Turn on debug, prints intermediate images.", action="store_true")
    args = parser.parse_args()
    return args

### Main workflow
def main():
    # Get options
    args = options()

    pcv.params.debug = args.debug #set debug mode
    pcv.params.debug_outdir = args.outdir #set output directory

    # Read fluorescence image data
    ps = pcv.photosynthesis.read_cropreporter(filename=args.image)
    
    
    # Threshold the chlorophyll image

    # Inputs:
    #   gray_img        - Grayscale image data
    #   max_value       - Value to apply above threshold (255 = white)
    #   object_type     - 'light' (default) or 'dark'. If the object is lighter than the
    #                       background then standard threshold is done. If the object is
    #                       darker than the background then inverse thresholding is done.
    mask = pcv.threshold.otsu(gray_img=img_as_ubyte(ps.chlorophyll.sel(frame_label="Chl").data.squeeze()), 
                              max_value=255, object_type="light")
    # Fill small objects

    # Inputs:
    #   bin_img         - Binary image data
    mask = pcv.fill_holes(bin_img=mask)
    
    chl_img = img_as_ubyte(ps.chlorophyll.sel(frame_label="Chl").data.squeeze())
    mask = pcv.erode(gray_img=mask, ksize=3, i=1)
    
    id_objects, obj_hierarchy = pcv.find_objects(img=chl_img, mask=mask)
    pcv.params.line_thickness = 5
    roi_contour_1, roi_hierarchy_1 = pcv.roi.rectangle(img=chl_img, x=400, y=300, h=400, w = 400)
    roi_objects_1, hierarchy_1, kept_mask_1, obj_area_1 = pcv.roi_objects(chl_img, roi_contour_1, roi_hierarchy_1, id_objects, obj_hierarchy, 'partial')
     # Object combine kept objects
    # Inputs:
    #   img - RGB or grayscale image data for plotting 
    #   contours - Contour list 
    #   hierarchy - Contour hierarchy array 
    
    obj1, mask1 = pcv.object_composition(img=chl_img, contours=roi_objects_1, hierarchy=hierarchy_1)
    
    # Find shape properties, output shape image (optional)

    # Inputs:
    #   img - RGB or grayscale image data 
    #   obj- Single or grouped contour object
    #   mask - Binary image mask to use as mask for moments analysis  

    shape_img_1 = pcv.analyze_object(img=chl_img, obj=obj1, mask=mask1)
    
    # Dark-adapted fluorescence induction curve

    # Inputs:
    # ps_da       = photosynthesis xarray DataArray
    # mask        = mask of plant (binary, single channel)
    # 
    # Returns:
    # ps_da       = dataarray with updated frame_label coordinate
    # ind_fig     = ggplot induction curve of fluorescence
    # ind_df      = data frame of mean fluorescence in the masked region at each timepoint

    dark_da, dark_fig, dark_df = pcv.photosynthesis.reassign_frame_labels(ps_da=ps.darkadapted, mask=mask)
    
    # Light-adapted fluorescence induction curve

    # Inputs:
    # ps_da       = photosynthesis xarray DataArray
    # mask        = mask of plant (binary, single channel)
    # 
    # Returns:
    # ps_da       = dataarray with updated frame_label coordinate
    # ind_fig     = ggplot induction curve of fluorescence
    # ind_df      = data frame of mean fluorescence in the masked region at each timepoint

    light_da, light_fig, light_df = pcv.photosynthesis.reassign_frame_labels(ps_da=ps.lightadapted, mask=mask)

    # Analyze Fv/Fm

    # Inputs:
    # ps_da               = photosynthesis xarray DataArray
    # mask                = mask of plant (binary, single channel)
    # bins                = number of bins for the histogram (1 to 256 for 8-bit; 1 to 65,536 for 16-bit; default is 256)
    # measurement_labels  = labels for each measurement, modifies the variable name of observations recorded
    # label               = optional label parameter, modifies the variable name of observations recorded
    # 
    # Returns:
    # yii       = DataArray of efficiency estimate values
    # hist_fig  = Histogram of efficiency estimate

    fvfm, fvfm_hist = pcv.photosynthesis.analyze_yii(ps_da=dark_da, mask=mask, measurement_labels=["Fv/Fm"])
    
    # Analyze Fq'/Fm'

    # Inputs:
    # ps_da               = photosynthesis xarray DataArray
    # mask                = mask of plant (binary, single channel)
    # bins                = number of bins for the histogram (1 to 256 for 8-bit; 1 to 65,536 for 16-bit; default is 256)
    # measurement_labels  = labels for each measurement, modifies the variable name of observations recorded
    # label               = optional label parameter, modifies the variable name of observations recorded
    # 
    # Returns:
    # yii       = DataArray of efficiency estimate values
    # hist_fig  = Histogram of efficiency estimate

    fqfm, fqfm_hist = pcv.photosynthesis.analyze_yii(ps_da=light_da, mask=mask, measurement_labels=["Fq'/Fm'"])
    
    # Analyze NPQ

    # Inputs:
    # ps_da_light        = photosynthesis xarray DataArray for which to compute npq
    # ps_da_dark         = photosynthesis xarray DataArray that contains frame_label `Fm`
    # mask               = mask of plant (binary, single channel)
    # bins               = number of bins for the histogram (1 to 256 for 8-bit; 1 to 65,536 for 16-bit; default is 256)
    # measurement_labels = labels for each measurement in ps_da_light, modifies the variable name of observations recorded
    # label              = optional label parameter, modifies the entity name of observations recorded
    # 
    # Returns:
    # npq       = DataArray of npq values
    # hist_fig  = Histogram of npq estimate

    npq, npq_hist = pcv.photosynthesis.analyze_npq(ps_da_light=light_da, ps_da_dark=dark_da, 
                                                   mask=mask, measurement_labels=["NPQ"])
    
    # Pseudocolor the PSII metric images

    # Inputs:
    # gray_img    - grayscale image data
    # obj         - (optional) ROI or plant contour object. If provided, the pseudocolored image gets cropped
    #               down to the region of interest. default = None
    # mask        - (optional) binary mask
    # cmap        - (optional) colormap. default is the matplotlib default, viridis
    # background  - (optional) background color/type, options are "image" (gray_img), "white", or "black"
    #               (requires a mask). default = 'image'
    # min_value   - (optional) minimum value for range of interest. default = 0
    # max_value   - (optional) maximum value for range of interest. default = 255
    # axes        - (optional) if False then x- and y-axis won't be displayed, nor will the title. default = True
    # colorbar    - (optional) if False then colorbar won't be displayed. default = True
    # obj_padding - (optional) if "auto" (default) and an obj is supplied, then the image is cropped to an extent 20%
    #               larger in each dimension than the object. An single integer is also accepted to define the padding
    #               in pixels
    # title       - (optional) custom title for the plot gets drawn if title is not None. default = None
    # bad_mask    - (optional) binary mask of pixels with "bad" values, e.g. nan or inf or any other values considered
    #               to be not informative and to be excluded from analysis. default = None
    # bad_color   - (optional) desired color to show "bad" pixels. default = "red"
    fvfm_cmap = pcv.visualize.pseudocolor(gray_img=fvfm.data.squeeze(), mask=mask, cmap="viridis", 
                                          min_value=0, max_value=1, title="Fv/Fm")
    fqfm_cmap = pcv.visualize.pseudocolor(gray_img=fqfm.data.squeeze(), mask=mask, cmap="viridis", 
                                          min_value=0, max_value=1, title="Fq'/Fm'")
    npq_cmap = pcv.visualize.pseudocolor(gray_img=npq.data.squeeze(), mask=mask, cmap="viridis", 
                                         min_value=0, max_value=1, title="NPQ")

    # Inputs:
    # hsi         = hyperspectral image (PlantCV Spectral_data instance)
    # distance    = how lenient to be if the required wavelengths are not available
    # 
    # Returns:
    # index_array = Index data as a Spectral_data instance
    ari = pcv.spectral_index.ari(hsi=ps.spectral)

    ari_ps = pcv.visualize.pseudocolor(gray_img=ari.array_data, min_value=0, max_value=10, 
                                       cmap="Purples", mask=mask, background="black", 
                                       title="Anthocyanin Reflectance Index")
    # Inputs:
    # index_array  = Instance of the Spectral_data class, usually the output from pcv.hyperspectral.extract_index
    # mask         = Binary mask made from selected contours
    # histplot     = if True plots histogram of intensity values
    # bins         = optional, number of classes to divide spectrum into
    # min_bin      = optional, minimum bin value ("auto" or user input minimum value)
    # max_bin      = optional, maximum bin value ("auto" or user input maximum value)
    # label        = optional label parameter, modifies the variable name of observations recorded
    ari_hist = pcv.hyperspectral.analyze_index(index_array=ari, mask=mask, min_bin=0, max_bin=10)


    # Inputs:
    # hsi         = hyperspectral image (PlantCV Spectral_data instance)
    # distance    = how lenient to be if the required wavelengths are not available
    # 
    # Returns:
    # index_array = Index data as a Spectral_data instance
    ndvi = pcv.spectral_index.ndvi(hsi=ps.spectral)

    ndvi_ps = pcv.visualize.pseudocolor(gray_img=ndvi.array_data, min_value=0, max_value=1, 
                                        cmap="jet", mask=mask, background="black", 
                                        title="Normalized Difference Vegetation Index")
    # Inputs:
    # index_array  = Instance of the Spectral_data class, usually the output from pcv.hyperspectral.extract_index
    # mask         = Binary mask made from selected contours
    # histplot     = if True plots histogram of intensity values
    # bins         = optional, number of classes to divide spectrum into
    # min_bin      = optional, minimum bin value ("auto" or user input minimum value)
    # max_bin      = optional, maximum bin value ("auto" or user input maximum value)
    # label        = optional label parameter, modifies the variable name of observations recorded
    ndvi_hist = pcv.hyperspectral.analyze_index(index_array=ndvi, mask=mask, min_bin=0, max_bin=1)
    
    # Inputs:
    # hsi         = hyperspectral image (PlantCV Spectral_data instance)
    # distance    = how lenient to be if the required wavelengths are not available
    # 
    # Returns:
    # index_array = Index data as a Spectral_data instance
    ci = pcv.spectral_index.ci_rededge(hsi=ps.spectral)

    ci_ps = pcv.visualize.pseudocolor(gray_img=ci.array_data, min_value=0, max_value=5, 
                                      cmap="Greens", mask=mask, background="black", 
                                      title="Chlorophyll Index Red Edge")
    # Inputs:
    # index_array  = Instance of the Spectral_data class, usually the output from pcv.hyperspectral.extract_index
    # mask         = Binary mask made from selected contours
    # histplot     = if True plots histogram of intensity values
    # bins         = optional, number of classes to divide spectrum into
    # min_bin      = optional, minimum bin value ("auto" or user input minimum value)
    # max_bin      = optional, maximum bin value ("auto" or user input maximum value)
    # label        = optional label parameter, modifies the variable name of observations recorded
    ci_hist = pcv.hyperspectral.analyze_index(index_array=ci, mask=mask, min_bin=0, max_bin=5)
    
     # Save Fv/Fm, Fq'/Fm', and NPQ results
    pcv.outputs.save_results(filename=args.result)

    if args.writeimg:
        pcv.print_image(img=dark_fig, filename=os.path.join(args.outdir, ps.filename[:-4] + "_fvfm_induction.png"))
        pcv.print_image(img=light_fig, filename=os.path.join(args.outdir, ps.filename[:-4] + "_fqfm_induction.png"))
        pcv.print_image(img=fvfm_hist, filename=os.path.join(args.outdir, ps.filename[:-4] + "_fvfm_histogram.png"))
        pcv.print_image(img=fqfm_hist, filename=os.path.join(args.outdir, ps.filename[:-4] + "_fqfm_histogram.png"))
        pcv.print_image(img=npq_hist, filename=os.path.join(args.outdir, ps.filename[:-4] + "_npq_histogram.png"))
        pcv.print_image(img=fvfm_cmap, filename=os.path.join(args.outdir, ps.filename[:-4] + "_fvfm_cmap.png"))
        pcv.print_image(img=fqfm_cmap, filename=os.path.join(args.outdir, ps.filename[:-4] + "_fqfm_cmap.png"))
        pcv.print_image(img=npq_cmap, filename=os.path.join(args.outdir, ps.filename[:-4] + "_npq_cmap.png"))
        pcv.print_image(img=ari_ps, filename=os.path.join(args.outdir, ps.filename[:-4] + "_ari_cmap.png"))
        pcv.print_image(img=ari_hist, filename=os.path.join(args.outdir, ps.filename[:-4] + "_ari_hist.png"))
        pcv.print_image(img=ci_ps, filename=os.path.join(args.outdir, ps.filename[:-4] + "_ci_cmap.png"))
        pcv.print_image(img=ci_hist, filename=os.path.join(args.outdir, ps.filename[:-4] + "_ci_hist.png"))
        pcv.print_image(img=ndvi_ps, filename=os.path.join(args.outdir, ps.filename[:-4] + "_ndvi_cmap.png"))
        pcv.print_image(img=ndvi_hist, filename=os.path.join(args.outdir, ps.filename[:-4] + "_ndvi_hist.png"))

if __name__ == '__main__':
    main()
