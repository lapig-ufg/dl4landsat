#!/usr/bin/python

from osgeo import gdal
from rios import applier
from rios import fileinfo
from scipy.signal import savgol_filter
import sys
import numpy as np

basename = sys.argv[1]
mask = sys.argv[2]
reference = sys.argv[3]

infiles = applier.FilenameAssociations()
outfiles = applier.FilenameAssociations()
otherargs = applier.OtherInputs()

infiles.blue = basename + '_B2_TOA.tif'
infiles.green = basename + '_B3_TOA.tif'
infiles.red = basename + '_B4_TOA.tif'
infiles.nir = basename + '_B5_TOA.tif'
infiles.swir1 = basename + '_B6_TOA.tif'
infiles.swir2 = basename + '_B7_TOA.tif'
infiles.ndvi = basename + '_NDVI_TOA.tif'
infiles.thermal = basename + '_THERMAL.tif'
infiles.cloud = basename + '_cloud_bqa_mask.tif'
infiles.mask = mask
infiles.reference = reference

outfiles.normalized_stacked = basename + '.img'

def stats(filename, nodata):

	gtif = gdal.Open(filename)
	band = gtif.GetRasterBand(1)
	band.SetNoDataValue(nodata)
	stats = band.ComputeStatistics(False)
	
	return {
		'min':stats[0],
		'max': stats[1],
		'mean': stats[2],
		'stddev': stats[3],
		'nodata': nodata
	}

def normalize(block, stats, outputNodata, mask):
	block[block != stats['nodata']] = 2 * (( block[block != stats['nodata']] - stats['min']) / (stats['max'] - stats['min'])) - 1
	block[block == stats['nodata']] = outputNodata
	block[mask == 0] = outputNodata
	return block

def filter(info, inputs, outputs, otherargs):
	
	print("Processing status " + str(info.getPercent()) + "%")

	norm_blue = normalize(inputs.blue.astype('Float32'), otherargs.blue_stats, otherargs.output_nodata, inputs.mask)
	norm_green = normalize(inputs.green.astype('Float32'), otherargs.green_stats, otherargs.output_nodata, inputs.mask)
	norm_red = normalize(inputs.red.astype('Float32'), otherargs.red_stats, otherargs.output_nodata, inputs.mask)
	norm_nir = normalize(inputs.nir.astype('Float32'), otherargs.nir_stats, otherargs.output_nodata, inputs.mask)
	norm_swir1 = normalize(inputs.swir1.astype('Float32'), otherargs.swir1_stats, otherargs.output_nodata, inputs.mask)
	norm_swir2 = normalize(inputs.swir2.astype('Float32'), otherargs.swir2_stats, otherargs.output_nodata, inputs.mask)
	norm_ndvi = normalize(inputs.ndvi.astype('Float32'), otherargs.ndvi_stats, otherargs.output_nodata, inputs.mask)
	norm_thermal = normalize(inputs.thermal.astype('Float32'), otherargs.thermal_stats, otherargs.output_nodata, inputs.mask)

	reference = inputs.reference.astype('Float32')
	reference[np.logical_and( (reference == 1.0), (inputs.cloud == 1) )] = 0
	reference[inputs.mask == 0] = otherargs.output_nodata
	
	outputs.normalized_stacked = np.concatenate((norm_blue,norm_green,norm_red,norm_nir,norm_swir1,norm_swir2,norm_ndvi,norm_thermal,reference))

otherargs.output_nodata = -2
otherargs.blue_stats = stats(infiles.blue, 32767.0)
otherargs.green_stats = stats(infiles.green, 32767.0)
otherargs.red_stats = stats(infiles.red, 32767.0)
otherargs.nir_stats = stats(infiles.nir, 32767.0)
otherargs.swir1_stats = stats(infiles.swir1, 32767.0)
otherargs.swir2_stats = stats(infiles.swir2, 32767.0)
otherargs.ndvi_stats = stats(infiles.ndvi, 32767.0)
otherargs.thermal_stats = stats(infiles.thermal, 0)

controls = applier.ApplierControls()
controls.setNumThreads(4);
controls.referenceImage = infiles.mask
controls.setJobManagerType('multiprocessing')
applier.apply(filter, infiles, outfiles, otherargs, controls=controls)