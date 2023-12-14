# Dockerfile, Image, Container
FROM python:3.9-slim-buster
FROM continuumio/miniconda3:latest

WORKDIR /Fargo_Real_Estate_Analysis

ADD FM_Real_Estate_Analysis.py .
ADD zillow_data_scrape.py .
ADD FM_Real_Estate_Analysis.ipynb .
ADD credentials.json .

RUN pip install numpy pandas matplotlib
RUN pip install requests
RUN pip install simplejson
RUN pip install pandas 
RUN pip install numpy 
RUN pip install seaborn 
RUN pip install matplotlib
RUN pip install googlemaps
RUN pip install geopandas
RUN pip install requests
RUN pip install plotly
RUN pip install shapely
RUN pip install folium
RUN pip install contextily
RUN pip install shapely
RUN pip install ipython
RUN pip install jupyter
RUN pip install pandoc
RUN apt-get update -y
RUN apt-get install -y texlive-xetex
RUN mkdir /Fargo_Real_Estate_Analysis/data

SHELL ["/bin/bash", "-c"]
RUN conda install -y pandoc

WORKDIR /Fargo_Real_Estate_Analysis

COPY FM_Real_Estate_Analysis.py .
COPY zillow_data_scrape.py .
COPY FM_Real_Estate_Analysis.ipynb .
COPY credentials.json .

CMD ["python3", "./FM_Real_Estate_Analysis.py"]
RUN jupyter nbconvert --to pdf FM_Real_Estate_Analysis.ipynb
