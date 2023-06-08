import requests
import re
import json
import pandas as pd
import sys
import warnings
from datetime import date
from IPython import get_ipython
warnings.filterwarnings('ignore')

while True:
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
        base_url2 = 'https://www.zillow.com/{}/apartments/{}_p/'
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
                match = re.search(r'!--(\{"queryState".*?)-->', r.text)
                if match:
                    data_list.append(json.loads(match.group(1)))

            data_list2 = []
            for url2 in urls2:
                r = s.get(url2, headers=req_headers)
                match2 = re.search(r'!--(\{"queryState".*?)-->', r.text)
                if match2:
                    data_list2.append(json.loads(match2.group(1)))

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
        df = df.sort_values(by='unformattedPrice', ascending=True)
        df2 = df2.sort_values(by='unformattedPrice', ascending=True)
        null_price = df[df["unformattedPrice"].isnull()]
        df.drop(null_price.index, inplace=True)
        ############################################################################################################

        pattern = r"\$(\d+(?:,\d+)?)\+"
        df2['units'] = df2['units'].astype(str)
        def extract_price(text):
            match = re.search(pattern, text)
            if match:
                return match.group(1)
            else:
                return None

        df2["units"] = df2["units"].apply(extract_price)
        df2['unformattedPrice'].fillna(df2['units'], inplace=True)
        null_price = df2['unformattedPrice'].isnull()
        not_null_units = ~df2['units'].isnull()
        df2.loc[null_price & not_null_units, 'unformattedPrice'] = df2.loc[null_price & not_null_units, 'units']
        null_price2 = df2[df2["unformattedPrice"].isnull()]
        df2.drop(null_price2.index, inplace=True)
        df2['unformattedPrice'] = pd.to_numeric(df2['unformattedPrice'], errors='coerce')
        df2['price'] = '$' + df2['unformattedPrice'].astype(str) + '/mo'
        df2 = df2.dropna(subset=['unformattedPrice'])
        df2 = df2.sort_values('unformattedPrice')

        ############################################################################################################

        # select certain fields and output (this portion can be changed)
        data = df[['zpid', 'imgSrc', 'statusType', 'price', 'unformattedPrice', 'zestimate', 'best_deal', 'address', 'addressZipcode', 'beds', 'baths', 'area', 'variableData', 'builderName']]
        data.to_csv("./data/" + city + "_Homes_ForSale_"+current_date+".csv")

        data2 = df2[['zpid', 'imgSrc', 'statusType', 'price', 'unformattedPrice', 'address', 'addressZipcode', 'beds', 'baths', 'area', 'variableData']]
        data2.to_csv("./data/" + city + "_Apartments_ForRental_"+current_date+".csv")

        print('Done.\n'+city+"_Homes_ForSale_"+current_date+".csv is available for viewing.\n"+city+"_Apartments_ForRental_"+current_date+".csv is available for viewing.")

        break  # exit loop if no errors
    except Exception as e:
        print(f"Error occurred: {e}")
        continue 