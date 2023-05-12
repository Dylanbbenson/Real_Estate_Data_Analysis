import requests
import re
import json
import pandas as pd
import sys
import warnings
from datetime import date
warnings.filterwarnings('ignore')

try:
    
    # take in command line arguments
    city = sys.argv[1]
    state = sys.argv[2]
    current_date = date.today().strftime('%Y-%m-%d')

    # input checking and modification
    if city == '': 
        city = input("Enter the city name: ")

    if state == '':
        state = input("Enter the state in abbreviation format (eg: Minnesota = MN) : ")

    while len(state) != 2:
        print("Invalid state entry. Try again.")
        state = input("Please try again: ")

    city = city.replace(" ", "-")
    city = city.capitalize()
    city_state = city + "-" + state

    # grab the first 20 pages
    base_url = 'https://www.zillow.com/homes/for_sale/{}/{}_p/'
    urls = [base_url.format(city, i) for i in range(1, 11)]
    base_url2 = 'https://www.zillow.com/{}/apartments/'
    urls2 = [base_url2.format(city_state, i) for i in range(1, 11)]

    print("Scraping data...")

    # add headers for chromedrivers
    req_headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.8',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'
    }

    # scrape data into data frames
    with requests.Session() as s:
        data_list = []
        for url in urls:
            r = s.get(url, headers=req_headers)
            data_list.append(json.loads(re.search(r'!--(\{"queryState".*?)-->', r.text).group(1)))

        data_list2 = []
        for url2 in urls2:
            r = s.get(url2, headers=req_headers)
            data_list2.append(json.loads(re.search(r'!--(\{"queryState".*?)-->', r.text).group(1)))

    df = pd.DataFrame()
    df2 = pd.DataFrame()

    # make combine data frames
    def make_combined_frame(frame, data_list):
        for i in data_list:
            for item in i['cat1']['searchResults']['listResults']:
                frame = frame.append(item, ignore_index=True)
        return frame

    df = make_combined_frame(df, data_list)
    df2 = make_combined_frame(df2, data_list2)

    # clean data
    df = df.drop_duplicates(subset='zpid', keep="last")
    df2 = df2.drop_duplicates(subset='zpid', keep="last")

    df['zestimate'] = df['zestimate'].fillna(0)
    df['best_deal'] = df['unformattedPrice'] - df['zestimate']
    df = df.sort_values(by='best_deal', ascending=True)

    df2['zestimate'] = df2['zestimate'].fillna(0)
    df2['best_deal'] = df2['unformattedPrice'] - df2['zestimate']
    df2 = df2.sort_values(by='best_deal', ascending=True)

    # select certain fields and output (this portion can be changed)
    data = df[['zpid', 'imgSrc', 'statusType', 'price', 'unformattedPrice', 'zestimate', 'best_deal', 'address', 'addressZipcode', 'beds', 'baths', 'area', 'variableData', 'brokerName', 'builderName']]
    data.to_csv("./data/" + city + "_Homes_ForSale_"+current_date+".csv")

    data2 = df2[['zpid', 'imgSrc', 'statusType', 'price', 'unformattedPrice', 'zestimate', 'best_deal', 'address', 'addressZipcode', 'beds', 'baths', 'area', 'variableData']]
    data2.to_csv("./data/" + city + "_Apartments_ForRental_"+current_date+".csv")

    print('Done.\n'+city+"_Homes_ForSale_"+current_date+".csv is available for viewing.\n"+city+"_Apartments_ForRental_"+current_date+".csv is available for viewing.")

except:
    print("Error: The City or State entered could not be found.")