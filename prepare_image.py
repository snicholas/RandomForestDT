import os, sys
import numpy as np
import fnmatch

from PIL import Image

import numpy as np
from osgeo import gdal

def ReadImage(fname, windowx, windowy):
    ds = gdal.Open(fname)
    im = np.array(ds.GetRasterBand(1).ReadAsArray())
    if im.shape[0] % windowx != 0 or im.shape[1] % windowy != 0:
        # im.resize(im.shape[0] + (windowx - im.shape[0] % windowx), im.shape[1] + (windowx - im.shape[1] % windowy))
        z = np.zeros((im.shape[0] + (windowx - im.shape[0] % windowx), im.shape[1] + (windowx - im.shape[1] % windowy)), dtype=im.dtype)
        z[0:im.shape[0], 0:im.shape[1]] = im[:, : ]
        im = z
        del z
    return im


def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


def unblockshaped(arr, h, w):
    n, nrows, ncols = arr.shape
    return (arr.reshape(h // nrows, -1, nrows, ncols)
            .swapaxes(1, 2)
            .reshape(h, w))


window = 32

norm_divisor = {
    "default": 65535,
    "HH": 100,
    "SOIL": 100,
    "AIRTEMP": 100,
    "PREC": 1000
}

def getInputDataArray(folder, suffixes, modifiers={}):
    Stack = np.array([])
    origshape = None
    for suffix in suffixes:
        print(folder, suffix)
        infile = os.path.join(folder, fnmatch.filter(os.listdir(folder), '*' + suffix + '*.tif')[0])
        print('data array from ' + infile)
        data = ReadImage(infile, window, window)
        if suffix in modifiers.keys():
            print("Modifying {} by {}".format(suffix, modifiers[suffix]))
            data = data + modifiers[suffix]
        origshape = data.shape
        if suffix in norm_divisor:
            data = data.astype('float32')/norm_divisor[suffix]
        else:
            data = data.astype('float32')/65535
            print('final type band ' + str(data.dtype))
        print("data.mean()", data.mean())
        data[np.isnan(data)]=data.mean()
        A = blockshaped(data, window, window)
        if Stack.size == 0:
            Stack = np.expand_dims(A, axis=0)
        else:
            Stack = np.concatenate((Stack, np.expand_dims(A, axis=0)), axis=0)
        del A, data
    return Stack, origshape


def stackImageArrays(images):
    stack = None
    for img in images:
        if stack is None:
            stack = np.expand_dims(img, axis=0)
        else:
            a = np.expand_dims(img, axis=0)
            stack = np.concatenate((stack, a), axis=2)
            del a
        del img
    stack = stack.swapaxes(0, 1)
    return stack


def loadTrainData(folder, features=['b02', 'b03', 'b04', 'b08'], targets=['SOIL']):
    flds = list(os.walk(folder))[0][1]
    images = []
    refer = []
    origshape = (1, 1)
    for fld in flds:
        data, origshape = getInputDataArray(folder + fld, features)
        images.append(data)
        r, _ = getInputDataArray(folder + fld, targets)
        refer.append(r)

    # images = sorted(list(images), reverse=True, key=lambda x: x.shape[0])
    print("stacking images")
    images = stackImageArrays(images)
    print("stacking refs")
    refer = sorted(list(refer), key=lambda x: x.shape[1])
    refer = stackImageArrays(refer)
    refer = np.squeeze(refer, axis=0)
    refer = np.squeeze(refer, axis=0)
    print("refer.shape", refer.shape)
    images = np.squeeze(images, axis=1)
    print("images.shape", images.shape)
    return images, refer, origshape


def loadClassifyData(folder, features=['b02', 'b03', 'b04', 'b08'], modifiers={}):
    print("modifiers", modifiers)
    print("features", features)
    dirs = [d for r, d, f in os.walk(folder)]
    if any("R10m" in x for x in dirs):
        folder += "/R10m/"
    images = []
    data, origshape = getInputDataArray(folder+'/', features, modifiers=modifiers)
    images.append(data)
    images = sorted(list(images), reverse=True, key=lambda x: x.shape[1])
    print("stacking images")
    images = stackImageArrays(images)
    images = np.squeeze(images, axis=1)
    print("images.shape", images.shape)
    return images, origshape


features = ['b02', 'b03', 'b04', 'b08']  # , 'b08', 'b11', 'b12']

targets = ['SOIL']
# X,Y = loadData('/home/snicholas/work/test1/')
# print("X.shape", X.shape)
# print("Y.shape", Y.shape)
