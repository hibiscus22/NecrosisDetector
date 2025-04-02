# Necrosis Detector Additional Tools

Besides from Necrosis Prediction Functions, we add in this repo some additional Python tools that may result useful:

## Color Map TTC

For color mapping grayscale images. TTC stands for a red dye, but it is easy to replace by others.

## Calculate Necrotic Area

To calculate and plot necrotic area.

## Change Format

To have every image in the same format (TIFF, JPG, PNG...). Include all the images in the folder `2format` and run the code 

```powershell
python change_format.py
```
The formatted images will appear in the `formatted` folder.

## Reduce Image Quality

To make images lightweight (for high-consumption models or lite reports). Input the images in the folder `2reduce` and run the python script. The new images will appear in the `reduced` folder.

## Put Scalebar in Images

Very useful in microscopy images. As before, you will find the folders `2 scalebar` and `scalebard`, as well as several scalebar files, that you can read in the python file, as well as selecting the postion to place the scalebar.

## Jaccard Similarity

Caluclate the Similarity of two binary images
