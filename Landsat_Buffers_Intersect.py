# The following script determines the bounding box of a raster and creates based on the bounding box a geometry.
import ogr, gdal
from osgeo.gdalconst import *
import numpy as np
import sys
from pandas import DataFrame
import pandas as pd 

gdal.PushErrorHandler('CPLQuietErrorHandler')


def bbox_to_pixel_offsets(gt, bbox):
    originX = gt[0]
    originY = gt[3]
    pixel_width = gt[1]
    pixel_height = gt[5]
    x1 = int((bbox[0] - originX) / pixel_width)
    x2 = int((bbox[1] - originX) / pixel_width) + 1

    y1 = int((bbox[3] - originY) / pixel_height)
    y2 = int((bbox[2] - originY) / pixel_height) + 1

    xsize = x2 - x1
    ysize = y2 - y1
    return (x1, y1, xsize, ysize)

def zonal_stats(vector_path, raster_path, nodata_value='-999.0', global_src_extent=False):
    rds = gdal.Open(raster_path, GA_ReadOnly)
    assert(rds)
    rb = rds.GetRasterBand(1)
    rgt = rds.GetGeoTransform()

    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)

    vds = ogr.Open(vector_path, GA_ReadOnly)  # TODO maybe open update if we want to write stats
    assert(vds)
    vlyr = vds.GetLayer(0)

    # create an in-memory numpy array of the source raster data
    # covering the whole extent of the vector layer
    if global_src_extent:
        # use global source extent
        # useful only when disk IO or raster scanning inefficiencies are your limiting factor
        # advantage: reads raster data in one pass
        # disadvantage: large vector extents may have big memory requirements
        src_offset = bbox_to_pixel_offsets(rgt, vlyr.GetExtent())
        src_array = rb.ReadAsArray(*src_offset)

        # calculate new geotransform of the layer subset
        new_gt = (
            (rgt[0] + (src_offset[0] * rgt[1])),
            rgt[1],
            0.0,
            (rgt[3] + (src_offset[1] * rgt[5])),
            0.0,
            rgt[5]
        )

    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')

    # Loop through vectors
    stats = []
    feat = vlyr.GetNextFeature()
    counter = 0
    errorCount = 0
    while feat is not None:
        counter += 1
        print(counter)
        


        if not global_src_extent:
            # use local source extent
            # fastest option when you have fast disks and well indexed raster (ie tiled Geotiff)
            # advantage: each feature uses the smallest raster chunk
            # disadvantage: lots of reads on the source raster
            src_offset = bbox_to_pixel_offsets(rgt, feat.geometry().GetEnvelope())
            src_array = rb.ReadAsArray(*src_offset)

            # calculate new geotransform of the feature subset
            new_gt = (
                (rgt[0] + (src_offset[0] * rgt[1])),
                rgt[1],
                0.0,
                (rgt[3] + (src_offset[1] * rgt[5])),
                0.0,
                rgt[5]
            )

        # Create a temporary vector layer in memory
        mem_ds = mem_drv.CreateDataSource('out')
        mem_layer = mem_ds.CreateLayer('poly', None, ogr.wkbPolygon)
        mem_layer.CreateFeature(feat.Clone())

        # Rasterize it
        rvds = driver.Create('', src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
        rvds.SetGeoTransform(new_gt)
        gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
        rv_array = rvds.ReadAsArray()


        tempDict = dict()
        for i in range(1, feat.GetFieldCount()):
            key = feat.GetDefnRef().GetFieldDefn(i).GetName()
            value = feat.GetFieldAsString(i)
            # print('Key: '+key+', Value: '+value)
            tempDict[key] = value
        # exit()

        try:
        # Mask the source data array with our current feature
        # we take the logical_not to flip 0<->1 to get the correct mask effect
        # we also mask out nodata values explictly
            masked = np.ma.MaskedArray(
            src_array,
            mask=np.logical_or(
                src_array == nodata_value,
                np.logical_not(rv_array)
                )
            )

            feature_stats = {
                'min': float(masked.min()),
                'mean': float(masked.mean()),
                'max': float(masked.max()),
                'std': float(masked.std()),
                'sum': float(masked.sum()),
                'count': int(masked.count()),
                'fid': int(feat.GetFID()),
                'PC': tempDict['POSTALCODE'],
                'LONG': tempDict['LONG'],
                'LAT': tempDict['LAT']}
        except:
            print('fID: '+str(feat.GetFID()))
            errorCount += 1

            feature_stats = {
	            'min': np.nan,
	            'mean': np.nan,
	            'max': np.nan,
	            'std': np.nan,
	            'sum': np.nan,
	            'count': int(masked.count()),
                'fid': int(feat.GetFID()),
                'PC': tempDict['POSTALCODE'],
                'LONG': tempDict['LONG'],
                'LAT': tempDict['LAT']}



        stats.append(feature_stats)

        rvds = None
        mem_ds = None
        feat = vlyr.GetNextFeature()

    vds = None
    rds = None
    print(errorCount)
    return stats


def getListOfPCs(year):
    yearStr = str(year)[2:4]
    PCsFolder = 'D:/CANUE_PROVIDED_DATA_Sorted/POSTALCODES/DMTI_SLI'
    PCsFile = PCsFolder + '/' + 'DMTI_SLI_' + yearStr + '.csv'
    PCsDF = pd.read_csv(PCsFile)
    PCsCol = PCsDF['POSTALCODE'+yearStr]
    return PCsCol




# This function will remove the count and fid Columns, add the west and east side together and will bring the PC column to the beginning and sort the values based on the PC column.
def attachPCsAndReorderCols(listOfAreas, year):
    canadaDataDF = pd.concat(listOfAreas)
    canadaDataDF = canadaDataDF.drop(['count', 'fid'], axis=1)
    canadaDataDFYearSpecific = canadaDataDF
    ##  For datasets without the postalcodes we do not the next few lines.
    ##  If comment the following lines, we should also comment anything related to postalcodes in the code if necessary.
    headerList = list(canadaDataDF)
    newHeaderList = ['PC'] + list(np.delete(headerList, 2))
    canadaDataDF = canadaDataDF[newHeaderList]
    canadaDataDF.sort_values(by=['PC'], inplace=True)
    actualPCsInThatYear = getListOfPCs(year)
    canadaDataDFYearSpecific = canadaDataDF.loc[canadaDataDF['PC'].isin(actualPCsInThatYear)]

    return canadaDataDFYearSpecific



import os
import glob

bufferSizeList = ['1km', '500m', '250m', '100m']
outputFolder = 'Landsat_Intersection_Results'

if (not os.path.exists(outputFolder)):
	os.makedirs(outputFolder)

for bufferSize in bufferSizeList:
	print(bufferSize)
	mainBufferFolder = 'Buffers'
	eastShapeFileName = mainBufferFolder + '/' + 'East_'+bufferSize
	westShapeFileName = mainBufferFolder + '/' + 'West_'+bufferSize
	# OntShapeFileName = mainBufferFolder + '/' + 'MPL_Ont_'+bufferSize
	eastVectorPath = eastShapeFileName + '.shp'
	westVectorPath = westShapeFileName + '.shp'
	# OntVectorPath = OntShapeFileName + '.shp'


	inputFolder = 'NDVI_Landsat_Images'

	for year in range (2015, 2016):
		print(year)

		canadaInputFile = 'Landsat_2015.tif'
		raster_path = canadaInputFile

		eastStats = zonal_stats(eastVectorPath, raster_path)
		westStats = zonal_stats(westVectorPath, raster_path)
		# OntStats = zonal_stats(OntVectorPath, raster_path)

		eastDataDF = DataFrame(eastStats)
		westDataDF = DataFrame(westStats)
		# OntDataDF = DataFrame(OntStats)

		listOfAreas = [eastDataDF, westDataDF] #, OntDataDF]

		canadaDataDF = attachPCsAndReorderCols(listOfAreas, year)

		print(canadaDataDF.shape)
		canadaDataDF.to_csv(outputFolder + '/' + 'Landsat_'+bufferSize+'_'+str(year)+'.csv', index=False)
		print('****************************************')






# os.system('gdal_merge.py -init 255 -o out.tif in1.tif in2.tif')