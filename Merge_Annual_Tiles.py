# After downloading the GeoTIFF image tiles from the GEE, we need to attach them together
# to have the full Canada coverage. Here is the script to do so. This may take a while to run.

for year in range (2015, 2016):
	print(year)
	inputFilesList = glob.glob(inputFolder + '/Canada_NDVI_'+str(year)+'-*.tif')
	inputString = ''
	for inputFile in inputFilesList:
		inputString += inputFile + ' '

	commandInput = 'python gdal_merge.py -o out.tif ' + inputString
	os.system(commandInput)
	print(commandInput)
