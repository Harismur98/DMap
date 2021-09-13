from flask import Flask, render_template, request,send_file
from geopy.geocoders.nominatim import _DEFAULT_NOMINATIM_DOMAIN
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from io import BytesIO
import io
import base64
from sklearn.cluster import KMeans
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import plotly_express as px
import json

class Dmap:
    def __init__(self,cluster,lat,lng):
        self.cluster = cluster
        self.lat = lat
        self.lng= lng

coordinate = []

def clean():
    file = pd.read_csv("static/excel/output.csv")
    coor = file[['cluster','latitude','longitude']]
    coor_array = np.array(coor)
    index = len(coor_array)

    for i in range(index)[::-1]:
        if np.isnan(coor_array[i][0]) or np.isnan(coor_array[i][1]) == True:
            coor_array = np.delete(coor_array, i, 0)
             
    for i in range(len(coor_array)):
        cluster = coor_array[i][0]
        lat = coor_array[i][1]
        log = coor_array[i][2]
        coordinate.append(Dmap(cluster,lat,log))
    return coordinate

coorlist =  clean()
print(coorlist[2].lat)
