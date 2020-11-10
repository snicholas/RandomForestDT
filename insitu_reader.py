import csv


def read_value(year, month, day, datafile, obs):
    with open(datafile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        yearcol = 9
        monthcol = 10
        daycol = 11
        obscol = 20

        valcol = 23

        for row in csv_reader:
            line_count += 1
            # print(str(line_count) + ' --> length ' + str(len(row)) + ' ' + str(row))
            if line_count >= 8:

                try:
                    if int(year) == int(row[yearcol]) and int(month) == int(row[monthcol]) \
                            and int(day) == int(row[daycol]) and obs == row[obscol]:
                        v = float(row[valcol].replace(",", "."))

                        print('found!! ' + str(v))

                        return v
                except:
                    pass

                # print(str(row[daycol]) + '/' + str(row[monthcol]) + '/' + str(row[yearcol]))
                # break

    return float(-9999)


# read_value(2017, 3, 4,
#            '/Users/ilsanto/data/digitaltwin/original_data/insitu/montseny_temp_prec_hum/eLTER_VA_Data_Reporting_LTER_EU_ES_025_Montseny_LTER_1996-2017_METEO.csv',
#            'AIRTEMP')
