# Landsat_GDAL_Intersection

Here is the script for intersecting Landsat NDVAI GeoTiff images with the postal codes point shapefiles using GDAL in python in order to get the NDVI values for all the postal codes in Canada.

In order to get the NDVI values for all the postal codes in Canada, the first step is to "create" the cloud-free, NDVI-calculated Landsat images in Google Earth Engine. There might be algorithms in GEE to upload the postal codes as point shapefiles and calculate the buffer inside the GEE itself. But we tried something different. The link for the GEE code is provided: https://code.earthengine.google.com/cb15521b2bb30450807fccb1ba69b8ba

We first export the cloud-free NDVI images in GeoTIFF format in the Google Drive (we could not upload/download enough number of points effeciently), then download them from the Google Drive. Since the resolution of the Landsat is 30 meter, and in order to cover the whole Canada, GEE exports the images as smaller Tiles, instead of one huge image. After downloading the smaller tiles, we would attach them together again on a local computer. The size of the attached image will be around 120 to 130 GB! The python script Merge_Annual_tile.py is for attaching the tiles together.

At this point, we have the Canada NDVI image and we also have the postal codes with their buffers as a shapefile with many small circles in it (for each buffer size we need a new shapefile). The intersecting code which is the main one, requires two inputs, one raster image and a vector which can be any shapefile. The address to the raster image and the vector can be set in the code. Since we have around 850,000 postal codes in Canada, we needed to have a shapefile with that amount of circles inside.  ArcGIS shapefiles can only be 2GB so you might have to create multiple files to get all points of interest. We created two shapefiles, one having the postal codes in the "West" of Canada (everywhere except ON) and one for the "East" of Canada which was basically postal codes in Ontario. Then we intersected the Canada image with the West and East shapefiles separately and merge the results to have full Canada again. You can find the code for this part in the Landsat_Buffers_Intersect.py.