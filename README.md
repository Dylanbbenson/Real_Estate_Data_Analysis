# Real_Estate_Data_Analysis
This project is a data analysis of Real Estate listings in any given U.S. city by scraping data from Zillow.com. This repo includes the following:

- **zillow_data_scrape.py**: python script that scrape real estate data from Zillow's web interface for any given U.S. city, performs some data cleaning and transformation, then outputs it to a csv file.
  - syntax: python3 zillow_data_scrape.py --city <city> --state <state>
  - note: This script only works if you're able to bypass zillow's frontend captcha.

- **FM_Real_Estate_Analysis.ipynb**: jupyter notebook that performs analysis on the extracted data for the Fargo-Moorehead area.

- **Real_Estate_Analysis_Template.ipynb**: jupyter notebook that performs analysis on the extracted data for any given city-state.

- **FM_Real_Estate_Analysis.py**: a python script created as an alternative to the jupyter notebook. Performs analysis on the extracted data for the Fargo-Moorehead area

- **Real_Estate_Analysis_Template.py**: a python script created as an alternative to the jupyter notebook. Performs analysis for any given city-state.

- **FM_Real_Estate_Analysis.pdf**: an example pdf report of visualized data created by the notebook. The city I used was my hometown Fargo, ND.

- **dockerfile**: contains all the commands required to assemble the docker image for this project.
