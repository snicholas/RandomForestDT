import os
import sys, getopt
import glob
import json
from prepare_image import ReadImage


def subset_and_resample(infolder, outfolder, coordinates):
    
    ios = []
    if infolder.find('S2') != -1:
        in_dir = infolder + 'GRANULE/*/IMG_DATA/'

        # Search directory for desired bands
        file_b1 = glob.glob(in_dir + '**B01**.jp2')
        file_b2 = glob.glob(in_dir + '**B02**.jp2')
        file_b3 = glob.glob(in_dir + '**B03**.jp2')
        file_b4 = glob.glob(in_dir + '**B04**.jp2')
        file_b5 = glob.glob(in_dir + '**B05**.jp2')
        file_b7 = glob.glob(in_dir + '**B07**.jp2')
        file_b8 = glob.glob(in_dir + '**B08**.jp2')
        file_b9 = glob.glob(in_dir + '**B09**.jp2')
        file_b10 = glob.glob(in_dir + '**B10**.jp2')

        dir_20m = infolder + 'GRANULE/*/IMG_DATA/'
        file_b6 = glob.glob(dir_20m + '**B06**.jp2')
        file_b11 = glob.glob(dir_20m + '**B11**.jp2')
        file_b12 = glob.glob(dir_20m + '**B12**.jp2')
        file_b8a = glob.glob(dir_20m + '**B8A**.jp2')
        ios.append({
            'in': file_b10[0],
            'out': outfolder + 'b10.tif'
        })
        ios.append({
            'in': file_b9[0],
            'out': outfolder + 'b09.tif'
        })
        ios.append({
            'in': file_b7[0],
            'out': outfolder + 'b07.tif'
        })
        ios.append({
            'in': file_b1[0],
            'out': outfolder + 'b01.tif'
        })
        ios.append({
            'in': file_b5[0],
            'out': outfolder + 'b05.tif'
        })
        ios.append({
            'in': file_b2[0],
            'out': outfolder + 'b02.tif'
        })
        ios.append({
            'in': file_b3[0],
            'out': outfolder + 'b03.tif'
        })
        ios.append({
            'in': file_b4[0],
            'out': outfolder + 'b04.tif'
        })
        ios.append({
            'in': file_b8[0],
            'out': outfolder + 'b08.tif'
        })
        ios.append({
            'in': file_b6[0],
            'out': outfolder + 'b06.tif'
        })
        ios.append({
            'in': file_b11[0],
            'out': outfolder + 'b11.tif'
        })
        ios.append({
            'in': file_b12[0],
            'out': outfolder + 'b12.tif'
        })
        ios.append({
            'in': file_b8a[0],
            'out': outfolder + 'b8a.tif'
        })

    if infolder.find('S1') != -1:
        in_dir = infolder + 'measurement/'

        # Search directory for desired bands
        file_vv = glob.glob(in_dir + '**-vv-**.tiff')
        file_vh = glob.glob(in_dir + '**-vh-**.tiff')
        ios.append({
            'in': file_vv[0],
            'out': outfolder + 'vv.tif'
        })
        ios.append({
            'in': file_vh[0],
            'out': outfolder + 'vh.tif'
        })
    shape = None
    for io in ios:
        do_transform(infile=io['in'], outfile=io['out'], coordinates=coordinates)
        if not shape:
            shape = ReadImage(io['out'], 1, 1).shape
            print(shape)
    
    return shape

def do_transform(infile, outfile, coordinates, res='-r near'):
    teopt = ' '
    c = None
    if coordinates is None:
        c = [' ', ' ', ' ', ' ']
        teopt = ' '
    else:
        c = coordinates
        teopt = '-te'

    ulx = c[0]  # west
    uly = c[1]  # north
    lrx = c[2]  # east
    lry = c[3]  # south

    translate = 'gdalwarp -tr 0.000091903317710 -0.000091903317710 -t_srs EPSG:4326 %s %s %s %s %s %s %s %s' % (
        res, teopt, ulx, uly, lrx, lry, infile, outfile)
    print("GDAL command " + translate)
    os.system(translate)
    


def main(argv):
    in_folder = None
    out_folder = None
    coordinates = None
    try:
        opts, args = getopt.getopt(argv, "hi:o:b:", ["help", "ifolder=", "ofolder=", "bbox="])
    except getopt.GetoptError:
        print('prepare_sentinel_product.py -i <inputfolder> -o <outputfolder> -b <ulx,uly,lrx,lry>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('prepare_sentinel_product.py -i <inputfolder> -o <outputfolder> -b <ulx,uly,lrx,lry>')
            sys.exit()
        elif opt in ("-i", "--ifolder"):
            in_folder = arg
        elif opt in ("-o", "--ofolder"):
            out_folder = arg
        elif opt in ("-b", "--bbox"):
            coordinates = arg.split(",")

    if in_folder is None:
        print('Missing input folder')
        sys.exit(1)
    if out_folder is None:
        print('Missing output folder')
        sys.exit(1)

    print('Input folder is ' + in_folder)
    print('Output folder is ' + out_folder)
    print('Coordinates is ' + str(coordinates))

    subset_and_resample(infolder=in_folder, outfolder=out_folder, coordinates=coordinates)


if __name__ == "__main__":
    main(sys.argv[1:])
