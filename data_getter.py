import pandas as pd
import os

def get_data(year,month,region='QLD', path='data'):
    # regions : QLD,NSW,VIC,SA
    # first date for data : January 1999
    available_regions = ['QLD','NSW','VIC','SA']
    available_months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    available_years = [str(i) for i in range(1999,2017)]


    if region not in available_regions:
        raise Exception('Region not found!')

    if year not in available_years:
        raise Exception('Year not found!')

    if month not in available_months:
        raise Exception('Month not found! Check 02-2 issue')


    url_csv = "https://www.aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{}{}_{}1.csv".format(year,month,region)
    filename = "{}_{}_{}.csv".format(year, month, region)
    path_region = path+'/'+region

    # if folder not found in current dir
    if region not in os.listdir(path):
        os.makedirs(path_region)

    # if not already exist
    if filename not in os.listdir(path_region):
        print('Fetching url...')
        print(url_csv)
        df = pd.read_csv(url_csv)

        print('Saving...', filename)
        df.to_csv("{}/{}/{}_{}_{}.csv".format(path,region,year, month, region))

    else:
        print(filename,' already exist.')





available_months = ['01','02','03','04','05','06','07','08','09','10','11','12']
available_years = [str(i) for i in range(1999,2017)]
available_regions = ['QLD','NSW','VIC','SA']

for r in available_regions:
    for y in available_years:
        for m in available_months:
            get_data(y, m, region=r)