#!/usr/bin/python

from osgeo import gdal
from rios import applier
from rios import fileinfo
from scipy.signal import savgol_filter
import sys
import numpy as np

image = sys.argv[1]

infiles = applier.FilenameAssociations()
outfiles = applier.FilenameAssociations()
otherargs = applier.OtherInputs()

infiles.image = image + '.tif'
infiles.labels = image + '_label.tif'

outfiles.normalized_stacked = image + '.img'

def stats(filename, nodata, i):

	gtif = gdal.Open(filename)
	band = gtif.GetRasterBand(i)
	band.SetNoDataValue(nodata)
	stats = band.ComputeStatistics(False)
	
	return {
		'min':stats[0],
		'max': stats[1],
		'mean': stats[2],
		'stddev': stats[3],
		'nodata': nodata
	}

def normalize(block, stats, outputNodata):
	block[block != stats['nodata']] = 2 * (( block[block != stats['nodata']] - stats['min']) / (stats['max'] - stats['min'])) - 1
	block[block == stats['nodata']] = outputNodata
	return block

def filter(info, inputs, outputs, otherargs):
	
	print("Processing status " + str(info.getPercent()) + "%")

	norm_blue = normalize(inputs.image[0:1,:,:].astype('Float32'), otherargs.blue_stats, otherargs.output_nodata)
	norm_green = normalize(inputs.image[1:2,:,:].astype('Float32'), otherargs.green_stats, otherargs.output_nodata)
	norm_red = normalize(inputs.image[2:3,:,:].astype('Float32'), otherargs.red_stats, otherargs.output_nodata)
	norm_nir = normalize(inputs.image[3:4,:,:].astype('Float32'), otherargs.nir_stats, otherargs.output_nodata)

	labels = inputs.labels.astype('Float32')
	labels[labels == 0] = -2
	labels[labels == 3] = 1
	labels[labels == 12] = 2
	labels[labels == 25] = 3
	labels[labels == 26] = 4
	
	outputs.normalized_stacked = np.concatenate((norm_blue, norm_green, norm_red, norm_nir, labels))

otherargs.output_nodata = -2
otherargs.blue_stats = stats(infiles.image, 0, 1)
otherargs.green_stats = stats(infiles.image, 0, 2)
otherargs.red_stats = stats(infiles.image, 0, 3)
otherargs.nir_stats = stats(infiles.image, 0, 4)

controls = applier.ApplierControls()
controls.setNumThreads(8);
controls.setJobManagerType('multiprocessing')

applier.apply(filter, infiles, outfiles, otherargs, controls=controls)