import requests
import pandas as pd
from bs4 import BeautifulSoup

def get_month_data(year, month):
    """
    Gets the table content from idojarasbudapest.hu for a given month.
    :param year: the year of the month we want to download
    :param month: the month of the data we want to download
    :return: data of the month
    """
    r = requests.post('http://idojarasbudapest.hu/archivalt-idojaras',
                      data={'ev': year, 'ho': month, 'button': 'Mehet'}) # sends aPOST request
    soup = BeautifulSoup(r.content, features='html5lib')
    curr = pd.read_html(str(soup.table)) # parses the table from html
    return curr[0][1:]


def get_training_data():
    """
    Downloads all the available weather data from the the site.
    :return: pandas dataframe with the parsed table data
    """
    df = pd.DataFrame()

    for year in range(2012, 2018):
        for month in {'01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'}:
            month_data = get_month_data(year, month)
            df = df.append(month_data)
    year = 2018
    for month in {'01', '02', '03', '04', '05', '06', '07', '08', '09', '10'}:
        df = df.append(get_month_data(year, month))
    return df

if __name__ == '__main__':
    # we save all the data from idojarasbudapest.hu
    get_training_data().to_csv('data/weather_dataset.csv', index=False)
