import os, getopt, sys
from osgeo import gdal
from osgeo import osr
import numpy
import glob
from insitu_reader import read_value

# config
GDAL_DATA_TYPE = gdal.GDT_Float32
GEOTIFF_DRIVER_NAME = r'GTiff'
NO_DATA = -9999
SPATIAL_REFERENCE_SYSTEM_WKID = 4326

def do_parse(id):
    # S2A_MSIL1C_20171221T112501_N0206_R037_T29SNC_20171221T114356
    print('parsing ' + id)
    string = id
    y = string[11:15]
    m = string[15:17]
    d = string[17:19]
    return y, m, d

def create_raster(output_path,
                  columns,
                  rows,
                  nband=1,
                  gdal_data_type=GDAL_DATA_TYPE,
                  driver=GEOTIFF_DRIVER_NAME):
    ''' returns gdal data source raster object

    '''
    # create driver
    driver = gdal.GetDriverByName(driver)

    output_raster = driver.Create(output_path,
                                  int(columns),
                                  int(rows),
                                  nband,
                                  eType=gdal_data_type)
    return output_raster


def numpy_array_to_raster(output_path,
                          numpy_array,
                          upper_left_tuple,
                          cell_resolution,
                          nband=1,
                          no_data=NO_DATA,
                          gdal_data_type=GDAL_DATA_TYPE,
                          spatial_reference_system_wkid=SPATIAL_REFERENCE_SYSTEM_WKID,
                          driver=GEOTIFF_DRIVER_NAME):
    ''' returns a gdal raster data source

    keyword arguments:

    output_path -- full path to the raster to be written to disk
    numpy_array -- numpy array containing data to write to raster
    upper_left_tuple -- the upper left point of the numpy array (should be a tuple structured as (x, y))
    cell_resolution -- the cell resolution of the output raster
    nband -- the band to write to in the output raster
    no_data -- value in numpy array that should be treated as no data
    gdal_data_type -- gdal data type of raster (see gdal documentation for list of values)
    spatial_reference_system_wkid -- well known id (wkid) of the spatial reference of the data
    driver -- string value of the gdal driver to use

    '''

    print('UL: (%s, %s)' % (upper_left_tuple[0],
                            upper_left_tuple[1]))

    rows, columns = numpy_array.shape
    print('ROWS: %s\n COLUMNS: %s\n' % (rows,
                                        columns))

    # create output raster
    output_raster = create_raster(output_path,
                                  int(columns),
                                  int(rows),
                                  nband,
                                  gdal_data_type)

    # geotransform = (upper_left_tuple[0],
    #                 cell_resolution,
    #                 upper_left_tuple[1] + cell_resolution,
    #                 -1 * (cell_resolution),
    #                 0,
    #                 0)

    geotransform = (upper_left_tuple[0],
                    cell_resolution,
                    0,
                    upper_left_tuple[1],
                    0,
                    -cell_resolution)

    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(spatial_reference_system_wkid)
    output_raster.SetProjection(spatial_reference.ExportToWkt())
    output_raster.SetGeoTransform(geotransform)
    output_band = output_raster.GetRasterBand(1)
    output_band.SetNoDataValue(no_data)
    output_band.WriteArray(numpy_array)
    output_band.FlushCache()
    output_band.ComputeStatistics(False)

    if os.path.exists(output_path) == False:
        raise Exception('Failed to create raster: %s' % output_path)

    return output_raster


def generate(infolder, datafile, fname, coordinates, shape):
    
    in_dir = infolder
    product_folders = glob.glob(in_dir + '/*')

    for pf in product_folders:
        product_id = pf.split('/')
        product_id = product_id[len(product_id) - 1]
        print(pf)
        print(product_id)

        y, m, d = do_parse(product_id)

        dest = infolder + '/' + product_id + '/' + fname + '.tif'

        value = read_value(year=y, month=m, day=d, datafile=datafile, obs=fname)

        print('generating ' + dest + ' with value ' + str(value))

        numpy_array_to_raster(dest, numpy.ones([shape[0], shape[1]], dtype=numpy.float32) * value, (coordinates[0], coordinates[3]),
                              0.000091903317710, 1, NO_DATA,
                              GDAL_DATA_TYPE, SPATIAL_REFERENCE_SYSTEM_WKID, GEOTIFF_DRIVER_NAME)


def main(argv):
    in_folder = None
    data_file = None
    observation = None

    try:
        opts, args = getopt.getopt(argv, "hi:d:o:", ["help", "ifolder=", "dfile=", 'observation='])
    except getopt.GetoptError:
        print('generate_raster.py -i <inputfolder> -d <datafileinsitu> -o <observation>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('generate_raster.py -i <inputfolder> -d <datafileinsitu> -o <observation>')
            sys.exit()
        elif opt in ("-i", "--ifolder"):
            in_folder = arg
        elif opt in ("-d", "--dfile"):
            data_file = arg
        elif opt in ("-o", "--observation"):
            observation = arg

    if in_folder is None:
        print('Missing input folder')
        sys.exit(1)
    if data_file is None:
        print('Missing output file name')
        sys.exit(1)
    if observation is None:
        print('Missing observation name')
        sys.exit(1)


    print('Input folder is ' + in_folder)
    print('Insitu data file name is ' + data_file)
    print('Observation name is ' + observation)

    generate(infolder=in_folder, datafile=data_file, fname=observation)


if __name__ == "__main__":
    main(sys.argv[1:])
