import ftplib
import os
import sys, getopt
import glob
from prepare_sentinel_product import do_transform
def invoke_prepare(infolder, outfolder, coordinates):
    in_dir = infolder
    product_folders = glob.glob(in_dir + '/*.nc')

    for pf in product_folders:
        product_id = pf.split('/')
        product_id = product_id[len(product_id) - 1]
        print(pf)
        print(product_id)
        product_out = outfolder + product_id
        product_out = product_out.replace('.nc', '.tif')
        tiff_file_name = product_id.replace('.nc', '_tmp.tif')

        translate = 'gdal_translate -a_srs EPSG:4326 NETCDF:' + pf + ':sm -of GTiff ' + pf.replace(product_id, tiff_file_name)
        print("GDAL command " + translate)
        os.system(translate)

        do_transform(infile=pf.replace(product_id, tiff_file_name), outfile=product_out, coordinates=coordinates)
        # os.remove(pf.replace(product_id, tiff_file_name))
        # os.remove(pf)
        

def download_to(folder, outfolder, year, month, day, coordinates):
    ftp = ftplib.FTP("anon-ftp.ceda.ac.uk")
    ftp.login("anonymous", "ftplib-example-1")

    data = []
    ftp.cwd('/neodc/esacci/soil_moisture/data/daily_files/ACTIVE/v04.7/' + year + '/')

    ftp.dir(data.append)

    for line in data:
        # print("-", line)
        spl = line.split(' ')
        fname = spl[len(spl) - 1]
        if '{}{}{}'.format(year,month,day) in fname:
            ftp.retrbinary('RETR ' + fname, open(folder + fname, 'wb').write)
    
    ftp.quit()
    invoke_prepare(folder, outfolder, coordinates)


def main(argv):
    out_folder = None
    year = None

    try:
        opts, args = getopt.getopt(argv, "ho:y:", ["help", "ofolder=", "year="])
    except getopt.GetoptError:
        print('soil_moisture_ftp_client.py -o <outputfolder> -y <year>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('soil_moisture_ftp_client.py -o <outputfolder> -y <year>')
            sys.exit()
        elif opt in ("-o", "--ofolder"):
            out_folder = arg
        elif opt in ("-y", "--year"):
            year = arg

    if out_folder is None:
        print('Missing output folder')
        sys.exit(1)

    if year is None:
        print('Missing year')
        sys.exit(1)

    print('Output folder is ' + out_folder)
    print('Year is ' + year)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    download_to(folder=out_folder, year=year)


if __name__ == "__main__":
    main(sys.argv[1:])
