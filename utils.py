import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import numpy as np
from osgeo import gdal

def Setcmap(D, inv=1):
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = (.0, .0, .0, 1.0)
    if inv:
        cmap = LinearSegmentedColormap.from_list('Custom_cmap', cmaplist, cmap.N)
    else:
        cmap = LinearSegmentedColormap.from_list('Custom_cmap', cmaplist[::-1], cmap.N)
    plt.register_cmap(name='Custom_cmap', cmap=cmap)
    bounds = np.linspace(0, D, D+1)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm

def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
            .swapaxes(1,2)
            .reshape(-1, nrows, ncols))

def unblockshaped(arr, h, w):    
    n, nrows, ncols = arr.shape
    a =arr.reshape(h//nrows, -1, nrows, ncols)
    a = a.swapaxes(1,2)
    a = a.reshape(h, w)
    return a

def ReadImage(fname, readflag=1):
    src = gdal.Open(fname)
    projection = src.GetProjection()
    geotransform = src.GetGeoTransform()
    datatype = src.GetRasterBand(1).DataType
    datatype = gdal.GetDataTypeName(datatype)
    ulx, xres, xskew, uly, yskew, yres = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)
    cols = src.RasterXSize
    rows = src.RasterYSize
    Image = 0
    if readflag:
        Image = src.GetRasterBand(1).ReadAsArray()
        print('Image shape: %d %d' % (Image.shape))
    print('Spatial resolution: %f %f' % (xres, yres))
    norm_factor = np.linalg.norm(Image)
    Image = Image / norm_factor
    return Image, projection, geotransform, (ulx, uly), (lrx, lry), src