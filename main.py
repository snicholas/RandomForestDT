"""
Title: Random forest digital twin template
Authors: Blagoj Delipetrev, Mattia Santoro, Nicholas Spadaro
Date created: 2020/11/09
Last modified: 2020/11/09
Description: Templates for creation and execution through the VLAB on DestinationEarth VirtualCloud of random forest based digital twins.
Version: 0.1
"""

import json, csv
import os, sys, getopt, math
import numpy as np
import math, copy, time
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from joblib import dump, load

from prepare_image import loadTrainData, loadClassifyData
from utils import blockshaped, unblockshaped, Setcmap
from generate_raster import numpy_array_to_raster, NO_DATA, GDAL_DATA_TYPE, SPATIAL_REFERENCE_SYSTEM_WKID, GEOTIFF_DRIVER_NAME, do_parse
from prepare_sentinel_product import subset_and_resample
from soil_moisture_ftp_client import download_to
from prepare_edge_data import invoke_prepare
# csv edgestream
# west,south,east,north,YYYY,MM,DD,value

def getVlabParams():
    # bbox i west,south,east,north
    with open('vlabparams.json') as json_file:
        return json.load(json_file)
        

def train():
    vlabparams=getVlabParams()
    bbox = vlabparams['bbox'].split(',') 
    features = vlabparams['features'].split(',')
    targets = vlabparams['targets'].split(',')

    X, Y, origshape = loadTrainData('data/inputs/', features = features, targets=targets)
    X = np.rollaxis(X, 0, 4)
    X = X.reshape(X.shape[0]*X.shape[1]*X.shape[2],X.shape[3])
    Y = Y.flatten()
    Y = np.digitize(Y,bins=[0.1, 0.2, 0.3, 0.4, 0.5]) #, 0.6, 0.7, 0.8, 0.9, 1.0])

    print("X.shape: ", X.shape)
    print("Y.shape: ", Y.shape)
    X[np.isnan(X)]=0
    Y[np.isnan(Y)]=0
    print("Train/Test split")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)
    del X,Y

    regr = RandomForestRegressor(max_depth=100, min_samples_leaf=3,min_samples_split=8, bootstrap=True,  random_state=0, n_estimators = 100, verbose=2, n_jobs=-1, warm_start=True)
    print("Train")
    regr.fit(X_train, y_train)
    print("train score: ", regr.score(X_train,y_train))
    print("Dump")
    dump(regr, 'data/outputs/model.joblib')
    print("Test")
    print(regr.score(X_test,y_test))
    del X_train, X_test, y_train, y_test
    
    flds = list(os.walk('data/satproduct/'))[0][1] 
    X,origshape   = loadClassifyData('data/inputs/{}'.format(flds[0]), features = features, modifiers={})
    X = np.rollaxis(X, 0, 4)
    X = X.reshape(X.shape[0]*X.shape[1]*X.shape[2],X.shape[3])
    Y = (regr.predict(X))
    Y=Y.reshape(-1,32, 32)
    Y=unblockshaped(Y,int(origshape[0]), int(origshape[1]))
    numpy_array_to_raster('data/outputs/prediction.tif', Y, (2.2500000, 41.6800000),
                          0.000091903317710, 1, NO_DATA,
                          GDAL_DATA_TYPE, SPATIAL_REFERENCE_SYSTEM_WKID, GEOTIFF_DRIVER_NAME)

def run():
    vlabparams=getVlabParams()
    bbox = vlabparams['bbox'].split(',')
    
    features = vlabparams['features'].split(',')
    modifiers = json.loads(vlabparams['modifiers'])
    
    X, origshape = loadClassifyData('data/inputs/', features = features, modifiers=modifiers)
    X = np.rollaxis(X, 0, 4)
    X=X.reshape(X.shape[0]*X.shape[1]*X.shape[2],X.shape[3])
    regr = load('data/model.joblib')
    Y = (regr.predict(X))
    Y=Y.reshape(-1,32, 32)
    Y=unblockshaped(Y,int(origshape[0]), int(origshape[1]))

    numpy_array_to_raster('data/outputs/prediction.tif', Y, (2.2500000, 41.6800000),
                          0.000091903317710, 1, NO_DATA,
                          GDAL_DATA_TYPE, SPATIAL_REFERENCE_SYSTEM_WKID, GEOTIFF_DRIVER_NAME)


def main(argv):
    #data/satproduct/
    vlabparams=getVlabParams()
    bbox = vlabparams['bbox'].split(',')
    Path("data/inputs/").mkdir(parents=True, exist_ok=True)
    flds = list(os.walk('data/satproduct/'))[0][1] 
    mode = int(sys.argv[1])
    if mode==0:
        shape = None
        for fld in flds:
            shape = subset_and_resample("data/satproduct/{}/".format(fld), 'data/inputs/', bbox )
            y, m, d = do_parse(fld)
            if 'SOIL' in vlabparams['features'].split(',') or 'SOIL' in vlabparams['targets'].split(','):
                download_to('data/edge/','data/inputs/'.format(fld),y,m,d, bbox)
            print(y, m, d)
            invoke_prepare('data/edge/','data/inputs/'.format(fld),y,m,d, bbox, shape) 
        run()
    else:
        shape = None
        for fld in flds:
            Path('data/inputs/{}/'.format(fld)).mkdir(parents=True, exist_ok=True)
            Path('data/edge/{}/'.format(fld)).mkdir(parents=True, exist_ok=True)
            shape = subset_and_resample("data/satproduct/{}/".format(fld), 'data/inputs/{}/'.format(fld), bbox )
            y, m, d = do_parse(fld)
            if 'SOIL' in vlabparams['features'].split(',') or 'SOIL' in vlabparams['targets'].split(','):
                download_to('data/edge/{}/'.format(fld),'data/inputs/{}/'.format(fld),y,m,d, bbox)
            
            invoke_prepare('data/edge/'.format(fld),'data/inputs/{}/'.format(fld),y,m,d, bbox, shape) 
        train()

if __name__ == "__main__":
    main(sys.argv[1:])
    


