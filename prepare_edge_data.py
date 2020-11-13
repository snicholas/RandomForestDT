import ftplib
import os
import sys, getopt
import glob
from prepare_sentinel_product import do_transform
import numpy as np
from numpy import genfromtxt
from generate_raster import numpy_array_to_raster, NO_DATA, GDAL_DATA_TYPE, SPATIAL_REFERENCE_SYSTEM_WKID, GEOTIFF_DRIVER_NAME, do_parse


def invoke_prepare(infolder, outfolder,y,m,d, coordinates, shape):
    in_dir = infolder
    product_folders = glob.glob(in_dir + '/*.csv')

    for pf in product_folders:
        product_id = pf.split('/')
        product_id = product_id[len(product_id) - 1]
        print(pf)
        print(product_id)
        product_out = outfolder + product_id
        product_out = product_out.replace('.csv', '.tif')
        data = genfromtxt(pf, delimiter=',', names=True)
        print(data)
        v = -9999
        for dt in data:
           if int(dt['YYYY'])==int(y) and int(dt['MM'])==int(m) and int(dt['DD'])==int(d):
               v = dt['value']
        
        print('v: ', v)
        image = np.ones([shape[0], shape[1]])*v
        
        numpy_array_to_raster(product_out, image, (float(coordinates[0]), float(coordinates[3])),
                          0.000091903317710, 1, NO_DATA,
                          GDAL_DATA_TYPE, SPATIAL_REFERENCE_SYSTEM_WKID, GEOTIFF_DRIVER_NAME)

        # os.remove(pf)