from flask import Flask, render_template, request,send_file
from geopy.geocoders.nominatim import _DEFAULT_NOMINATIM_DOMAIN
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import json
import io
# import Class.Kmean as k
import Class.Dmap as d
import requests
import logging
import time

app = Flask(__name__)

logger = logging.getLogger("root")
logger.setLevel(logging.DEBUG)
# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

API_KEY = 'AIzaSyCgkPuBV5PCq1CggJkB_rTPkUXEeNd2sno'
BACKOFF_TIME = 30
output_filename = 'static/excel/google_corrdinate.csv'
RETURN_FULL_RESULTS = False

@app.route('/')
def home():
    return render_template('upload.html')

color = []
l = []
cluster = []
coordinate = []
stats = []
total_nof_cluster = []
temp1 =[]
labelZone=[]
ns = []
negeri = ['Selangor','Melaka','Pahang','Negeri Sembilan','Johor','Kedah','Perak','Perlis','Kelantan','Terengganu','Sarawak','Sabah','Pulau Pinang']
class n_clus():
    def __init__(self,clus):
        self.clus = clus

def output(k):
    n = k
    total_data = len(cluster)
    cluster1 = 0
    total_nof_cluster.clear()
    for i in range(n):
        cluster1 = 0
        for x in range(total_data):
            if cluster[x] == str(i+1):
                cluster1 += 1
        total_nof_cluster.append(n_clus(cluster1))
    return total_nof_cluster

def clean():
    file = pd.read_csv(output_filename)
    coor = file[['cluster','latitude','longitude','labelZone']]
    coor_array = np.array(coor)
    index = len(coor_array)

    for i in range(index)[::-1]:
        if np.isnan(coor_array[i][0]) or np.isnan(coor_array[i][1]) == True:
            coor_array = np.delete(coor_array, i, 0)
             
    for i in range(len(coor_array)):
        clus =coor_array[i][3]
        lat = coor_array[i][1]
        log = coor_array[i][2]
        coordinate.append(d.Dmap(clus,lat,log))
    return coordinate
    
def nstat(k):
    file = pd.read_csv(output_filename)
    coor = file[['labelZone','Negeri']]
    coor_array = np.array(coor)
    index = len(coor_array)
    
    for i in range(len(negeri)):
        g = y =r =b =p =0
        Lo = 0
        for a in range(index):
            if coor_array[a][1] == negeri[i]:
                if coor_array[a][0] == 'Green Zone':
                    g += 1
                elif coor_array[a][0] == 'Yellow Zone':
                    y += 1
                elif coor_array[a][0] == 'Red Zone':
                    r += 1
                elif coor_array[a][0] == 'Blue Zone':
                    b += 1
                elif coor_array[a][0] == 'Purple Zone':
                    p += 1
                Lo = coor_array[a][1]
        ns.append(negerist(Lo,g,y,r,b,p))
    return ns


def get_google_results(address, api_key=None, return_full_response=False):
    """
    Get geocode results from Google Maps Geocoding API.
    
    Note, that in the case of multiple google geocode reuslts, this function returns details of the FIRST result.
    
    @param address: String address as accurate as possible. For Example "18 Grafton Street, Dublin, Ireland"
    @param api_key: String API key if present from google. 
                    If supplied, requests will use your allowance from the Google API. If not, you
                    will be limited to the free usage of 2500 requests per day.
    @param return_full_response: Boolean to indicate if you'd like to return the full response from google. This
                    is useful if you'd like additional location details for storage or parsing later.
    """
    # Set up your Geocoding url
    geocode_url = "https://maps.googleapis.com/maps/api/geocode/json?address={}".format(address)
    if api_key is not None:
        geocode_url = geocode_url + "&key={}".format(api_key)
        
    # Ping google for the reuslts:
    results = requests.get(geocode_url)
    # Results will be in JSON format - convert to dict using requests functionality
    results = results.json()
    
    # if there's no results or an error, return empty results.
    if len(results['results']) == 0:
        output = {
            "formatted_address" : None,
            "latitude": None,
            "longitude": None,
            "accuracy": None,
            "google_place_id": None,
            "type": None,
            "postcode": None
        }
    else:    
        answer = results['results'][0]
        output = {
            "formatted_address" : answer.get('formatted_address'),
            "latitude": answer.get('geometry').get('location').get('lat'),
            "longitude": answer.get('geometry').get('location').get('lng'),
            "accuracy": answer.get('geometry').get('location_type'),
            "google_place_id": answer.get("place_id"),
            "type": ",".join(answer.get('types')),
            "postcode": ",".join([x['long_name'] for x in answer.get('address_components') 
                                  if 'postal_code' in x.get('types')])
        }
        
    # Append some other details:    
    output['input_string'] = address
    output['number_of_results'] = len(results['results'])
    output['status'] = results.get('status')
    if return_full_response is True:
        output['response'] = results
    
    return output

class stat:
    def __init__(self,min_x,min_y,max_x,max_y,label):
        self.min_x = min_x
        self.min_y= min_y
        self.max_x= max_x
        self.max_y= max_y
        self.label = label

class negerist:
     def __init__(self,name,g,y,r,b,p):
         self.g = g
         self.y = y
         self.r = r
         self.b = b
         self.p = p
         self.name = name

class kmeans:
    """Apply kmeans algorithm"""
    def __init__(self, num_clusters, max_iter=1000):
        """Initialize number of clusters"""
        
        self.num_clusters = num_clusters
        self.max_iter = max_iter
    
    def initalize_centroids(self, X):
        """Choosing k centroids randomly from data X"""
        
        idx = np.random.permutation(X.shape[0])
        centroids = X[idx[:self.num_clusters]]
        return centroids

    def compute_centroid(self, X, labels):
        """Modify centroids by finding mean of all k partitions"""
        
        centroids = np.zeros((self.num_clusters, X.shape[1]))
        for k in range(self.num_clusters):
            centroids[k] = np.mean(X[labels == k], axis=0)
            
        return centroids

    def compute_distance(self, X, centroids):
        """Computing L2 norm between datapoints and centroids"""

        distances = np.zeros((X.shape[0], self.num_clusters))
        
        for k in range(self.num_clusters):
            dist = np.linalg.norm(X - centroids[k], axis=1)
            distances[:,k] = np.square(dist)
            
        return distances
    
    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)
    
    def fit(self, X):
        self.centroids = self.initalize_centroids(X)
        
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroid(X, self.labels)
            
            if np.all(old_centroids == self.centroids):
                break

@app.route("/result", methods = ['GET', 'POST'])
def fig():
    if request.method == 'POST':
        f = request.files['file']
        coor_select = request.form['has_coordinate']
        num_k= int(request.form['k'])
        stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
        data = pd.read_csv(stream)
        dt = data
        dt['address'] = data.Lokaliti + ' ' + data.Daerah + ' ' + data.Negeri
        addresses = dt['address'].tolist()
        cluster_data = data[['Jumlah Kes Terkumpul','Tempoh Wabak Berlaku (Hari)']]
        cluster_array = np.array(cluster_data)
        
        #my kmeans algo
        kmeansmodel = kmeans(num_clusters=num_k, max_iter=1000)
        kmeansmodel.fit(cluster_array)
        
        stats.clear()
        #get a bit of statistik
        def out():
            for i in range(num_k):
                minx1=int(min(cluster_array[kmeansmodel.labels == i, 0]))
                miny1=int(min(cluster_array[kmeansmodel.labels == i, 1]))
                maxx1=int(max(cluster_array[kmeansmodel.labels == i, 0]))
                maxy1=int(max(cluster_array[kmeansmodel.labels == i, 1]))

                if num_k==2:    
                    if miny1 == temp1[0]:
                        name = "Green Zone"               
                    elif miny1== temp1[1]:
                        name = "Red Zone"
                elif num_k==3:
                    if miny1 == temp1[0]:
                        name = "Green Zone"               
                    elif miny1== temp1[1]:
                        name = "Yellow Zone"
                    elif miny1== temp1[2]:
                        name = "Red Zone"
                elif num_k==4:
                    if miny1 == temp1[0]:
                        name = "Green Zone" 
                    elif miny1== temp1[1]:
                        name = "Blue Zone"                  
                    elif miny1== temp1[2]:
                        name = "Yellow Zone"
                    elif miny1== temp1[3]:
                        name = "Red Zone"
                elif num_k==5:
                    if miny1 == temp1[0]:
                        name = "Green Zone" 
                    elif miny1== temp1[1]:
                        name = "Blue Zone"                  
                    elif miny1== temp1[2]:
                        name = "Yellow Zone"
                    elif miny1== temp1[3]:
                        name = "Purple Zone"
                    elif miny1== temp1[4]:
                        name = "Red Zone"
                stats.append(stat(minx1,miny1,maxx1,maxy1,name))               
            return stats
          
        centroids = kmeansmodel.centroids
        centroids[0]
        for y in range(num_k):
            temp1.append(int(min(cluster_array[kmeansmodel.labels == y, 1])))


        temp1.sort()
        if num_k == 2:
            for y in range(num_k):
                if int(min(cluster_array[kmeansmodel.labels == y, 1])) == temp1[0]:
                    l.append("Green Zone")
                    color.append("green")
                elif int(min(cluster_array[kmeansmodel.labels == y, 1]))== temp1[1]:
                    l.append("Red Zone")
                    color.append("red")
        elif num_k == 3:
            for y in range(num_k):
                if int(min(cluster_array[kmeansmodel.labels == y, 1])) == temp1[0]:
                    l.append("Green Zone")
                    color.append("green")
                elif int(min(cluster_array[kmeansmodel.labels == y, 1]))== temp1[1]:
                    l.append("Yellow Zone")
                    color.append("yellow")
                elif int(min(cluster_array[kmeansmodel.labels == y, 1]))== temp1[2]:
                    l.append("Red Zone")
                    color.append("red")
        elif num_k == 4:
            for y in range(num_k):
                if int(min(cluster_array[kmeansmodel.labels == y, 1])) == temp1[0]:
                    l.append("Green Zone")
                    color.append("green")
                elif int(min(cluster_array[kmeansmodel.labels == y, 1])) == temp1[1]:
                    l.append("Blue Zone")
                    color.append("blue")
                elif int(min(cluster_array[kmeansmodel.labels == y, 1]))== temp1[2]:
                    l.append("Yellow Zone")
                    color.append("yellow")
                elif int(min(cluster_array[kmeansmodel.labels == y, 1]))== temp1[3]:
                    l.append("Red Zone")
                    color.append("red")
        elif num_k == 5:
            for y in range(num_k):
                if int(min(cluster_array[kmeansmodel.labels == y, 1])) == temp1[0]:
                    l.append("Green Zone")
                    color.append("green")
                elif int(min(cluster_array[kmeansmodel.labels == y, 1])) == temp1[1]:
                    l.append("Blue Zone")
                    color.append("blue")
                elif int(min(cluster_array[kmeansmodel.labels == y, 1]))== temp1[2]:
                    l.append("Yellow Zone")
                    color.append("yellow")
                elif int(min(cluster_array[kmeansmodel.labels == y, 1]))== temp1[3]:
                    l.append("Purple Zone")
                    color.append("purple")
                elif int(min(cluster_array[kmeansmodel.labels == y, 1]))== temp1[4]:
                    l.append("Red Zone")
                    color.append("red")
        

        # create label
        cluster.clear()
        labelZone.clear()
        for i in range(len(cluster_array)):
            if cluster_array[i][0] in cluster_array[kmeansmodel.labels == 0, 0] and cluster_array[i][1] in cluster_array[kmeansmodel.labels == 0, 1]:
                cluster.append('1')
                if num_k == 2:
                    if int(min(cluster_array[kmeansmodel.labels == 0, 1])) == temp1[0]:
                        labelZone.append("Green Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 0, 1]))== temp1[1]:
                        labelZone.append("Red Zone")
                elif num_k == 3:
                    if int(min(cluster_array[kmeansmodel.labels == 0, 1])) == temp1[0]:
                        labelZone.append("Green Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 0, 1]))== temp1[1]:
                        labelZone.append("Yellow Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 0, 1]))== temp1[2]:
                        labelZone.append("Red Zone")
                elif num_k == 4:
                    if int(min(cluster_array[kmeansmodel.labels == 0, 1])) == temp1[0]:
                        labelZone.append("Green Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 0, 1])) == temp1[1]:
                        labelZone.append("Blue Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 0, 1]))== temp1[2]:
                        labelZone.append("Yellow Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 0, 1]))== temp1[3]:
                        labelZone.append("Red Zone")
                elif num_k == 5:
                    if int(min(cluster_array[kmeansmodel.labels == 0, 1])) == temp1[0]:
                        labelZone.append("Green Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 0, 1])) == temp1[1]:
                        labelZone.append("Blue Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 0, 1]))== temp1[2]:
                        labelZone.append("Yellow Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 0, 1]))== temp1[3]:
                        labelZone.append("Purple Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 0, 1]))== temp1[4]:
                        labelZone.append("Red Zone")
            elif cluster_array[i][0] in cluster_array[kmeansmodel.labels == 1, 0] and cluster_array[i][1] in cluster_array[kmeansmodel.labels == 1, 1]:
                cluster.append('2')
                if num_k==2:
                    if int(min(cluster_array[kmeansmodel.labels == 1, 1])) == temp1[0]:
                        labelZone.append("Green Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 1, 1]))== temp1[1]:
                        labelZone.append("Red Zone")
                elif num_k == 3:
                    if int(min(cluster_array[kmeansmodel.labels == 1, 1])) == temp1[0]:
                        labelZone.append("Green Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 1, 1]))== temp1[1]:
                        labelZone.append("Yellow Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 1, 1]))== temp1[2]:
                        labelZone.append("Red Zone")
                elif num_k == 4:
                    if int(min(cluster_array[kmeansmodel.labels == 1, 1])) == temp1[0]:
                        labelZone.append("Green Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 1, 1])) == temp1[1]:
                        labelZone.append("Blue Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 1, 1]))== temp1[2]:
                        labelZone.append("Yellow Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 1, 1]))== temp1[3]:
                        labelZone.append("Red Zone")
                elif num_k == 5:
                    if int(min(cluster_array[kmeansmodel.labels == 1, 1])) == temp1[0]:
                        labelZone.append("Green Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 1, 1])) == temp1[1]:
                        labelZone.append("Blue Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 1, 1]))== temp1[2]:
                        labelZone.append("Yellow Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 1, 1]))== temp1[3]:
                        labelZone.append("Purple Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 1, 1]))== temp1[4]:
                        labelZone.append("Red Zone")
            elif cluster_array[i][0] in cluster_array[kmeansmodel.labels == 2, 0] and cluster_array[i][1] in cluster_array[kmeansmodel.labels == 2, 1]:
                cluster.append('3')    
                if num_k==2:
                    if int(min(cluster_array[kmeansmodel.labels == 2, 1])) == temp1[0]:
                        labelZone.append("Green Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 2, 1]))== temp1[1]:
                        labelZone.append("Red Zone")
                elif num_k == 3:
                    if int(min(cluster_array[kmeansmodel.labels == 2, 1])) == temp1[0]:
                        labelZone.append("Green Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 2, 1]))== temp1[1]:
                        labelZone.append("Yellow Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 2, 1]))== temp1[2]:
                        labelZone.append("Red Zone")
                elif num_k == 4:
                    if int(min(cluster_array[kmeansmodel.labels == 2, 1])) == temp1[0]:
                        labelZone.append("Green Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 2, 1])) == temp1[1]:
                        labelZone.append("Blue Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 2, 1]))== temp1[2]:
                        labelZone.append("Yellow Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 2, 1]))== temp1[3]:
                        labelZone.append("Red Zone")
                elif num_k == 5:
                    if int(min(cluster_array[kmeansmodel.labels == 2, 1])) == temp1[0]:
                        labelZone.append("Green Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 2, 1])) == temp1[1]:
                        labelZone.append("Blue Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 2, 1]))== temp1[2]:
                        labelZone.append("Yellow Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 2, 1]))== temp1[3]:
                        labelZone.append("Purple Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 2, 1]))== temp1[4]:
                        labelZone.append("Red Zone")
            elif cluster_array[i][0] in cluster_array[kmeansmodel.labels == 3, 0] and cluster_array[i][1] in cluster_array[kmeansmodel.labels == 3, 1]:
                cluster.append('4')
                if num_k==2:    
                    if int(min(cluster_array[kmeansmodel.labels == 3, 1])) == temp1[0]:
                        labelZone.append("Green Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 3, 1]))== temp1[1]:
                        labelZone.append("Red Zone")
                elif num_k == 3:
                    if int(min(cluster_array[kmeansmodel.labels == 3, 1])) == temp1[0]:
                        labelZone.append("Green Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 3, 1]))== temp1[1]:
                        labelZone.append("Yellow Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 3, 1]))== temp1[2]:
                        labelZone.append("Red Zone")
                elif num_k == 4:
                    if int(min(cluster_array[kmeansmodel.labels == 3, 1])) == temp1[0]:
                        labelZone.append("Green Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 3, 1])) == temp1[1]:
                        labelZone.append("Blue Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 3, 1]))== temp1[2]:
                        labelZone.append("Yellow Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 3, 1]))== temp1[3]:
                        labelZone.append("Red Zone")
                elif num_k == 5:
                    if int(min(cluster_array[kmeansmodel.labels == 3, 1])) == temp1[0]:
                        labelZone.append("Green Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 3, 1])) == temp1[1]:
                        labelZone.append("Blue Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 3, 1]))== temp1[2]:
                        labelZone.append("Yellow Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 3, 1]))== temp1[3]:
                        labelZone.append("Purple Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 3, 1]))== temp1[4]:
                        labelZone.append("Red Zone")
            elif cluster_array[i][0] in cluster_array[kmeansmodel.labels == 4, 0] and cluster_array[i][1] in cluster_array[kmeansmodel.labels == 4, 1]:
                cluster.append('5')
                if num_k==2:    
                    if int(min(cluster_array[kmeansmodel.labels == 4, 1])) == temp1[0]:
                        labelZone.append("Green Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 4, 1]))== temp1[1]:
                        labelZone.append("Red Zone")
                elif num_k == 3:
                    if int(min(cluster_array[kmeansmodel.labels == 4, 1])) == temp1[0]:
                        labelZone.append("Green Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 4, 1]))== temp1[1]:
                        labelZone.append("Yellow Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 4, 1]))== temp1[2]:
                        labelZone.append("Red Zone")
                elif num_k == 4:
                    if int(min(cluster_array[kmeansmodel.labels == 4, 1])) == temp1[0]:
                        labelZone.append("Green Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 4, 1])) == temp1[1]:
                        labelZone.append("Blue Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 4, 1]))== temp1[2]:
                        labelZone.append("Yellow Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 4, 1]))== temp1[3]:
                        labelZone.append("Red Zone")
                elif num_k == 5:
                    if int(min(cluster_array[kmeansmodel.labels == 4, 1])) == temp1[0]:
                        labelZone.append("Green Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 4, 1])) == temp1[1]:
                        labelZone.append("Blue Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 4, 1]))== temp1[2]:
                        labelZone.append("Yellow Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 4, 1]))== temp1[3]:
                        labelZone.append("Purple Zone")
                    elif int(min(cluster_array[kmeansmodel.labels == 4, 1]))== temp1[4]:
                        labelZone.append("Red Zone")
        data["cluster"] = cluster
        data["labelZone"] = labelZone

        # plotting the clustered data 
        plt.figure(figsize=(6,6))
        for i in range(num_k):
            plt.scatter(cluster_array[kmeansmodel.labels == i, 0], cluster_array[kmeansmodel.labels == i, 1], c = color[i], label = l[i])
        plt.xlabel('Total Case')
        plt.ylabel('Days')
        plt.legend()

        if coor_select == '1':
            #-----------------------------------Getting the Coordinate------------------------------------------
            # Ensure, before we start, that the API key is ok/valid, and internet access is ok
            test_result = get_google_results("London, England", API_KEY, RETURN_FULL_RESULTS)
            if (test_result['status'] != 'OK') or (test_result['formatted_address'] != 'London, UK'):
                logger.warning("There was an error when testing the Google Geocoder.")
                raise ConnectionError('Problem with test results from Google Geocode - check your API key and internet connection.')

            # Create a list to hold results
            results = []
            # Go through each address in turn
            for address in addresses:
                # While the address geocoding is not finished:
                geocoded = False
                while geocoded is not True:
                    # Geocode the address with google
                    try:
                        geocode_result = get_google_results(address, API_KEY, return_full_response=RETURN_FULL_RESULTS)
                    except Exception as e:
                        logger.exception(e)
                        logger.error("Major error with {}".format(address))
                        logger.error("Skipping!")
                        geocoded = True
                        
                    # If we're over the API limit, backoff for a while and try again later.
                    if geocode_result['status'] == 'OVER_QUERY_LIMIT':
                        logger.info("Hit Query Limit! Backing off for a bit.")
                        time.sleep(BACKOFF_TIME * 60) # sleep for 30 minutes
                        geocoded = False
                    else:
                        # If we're ok with API use, save the results
                        # Note that the results might be empty / non-ok - log this
                        if geocode_result['status'] != 'OK':
                            logger.warning("Error geocoding {}: {}".format(address, geocode_result['status']))
                        logger.debug("Geocoded: {}: {}".format(address, geocode_result['status']))
                        results.append(geocode_result)           
                        geocoded = True

                # Print status every 100 addresses
                if len(results) % 100 == 0:
                    logger.info("Completed {} of {} address".format(len(results), len(addresses)))
                        
                # Every 500 addresses, save progress to file(in case of a failure so you have something!)
                if len(results) % 500 == 0:
                    pd.DataFrame(results).to_csv("{}_bak".format(output_filename))

            # All done
            logger.info("Finished geocoding all addresses")
            # Write the full results to csv using the pandas library.
            r = pd.DataFrame(results)
            # r['cluster'] = data['cluster']
            last = pd.concat([r,data],axis=1,join='inner')
            last.to_csv(output_filename, encoding='utf8',index=False)
            #---------------------------------------------------------------------------------------------    
            plt.savefig('static/images/pic2.png')
            coorlist =  clean()  
            total = output(num_k)     
            clus_stat = out()       
            return render_template('result.html', url ='static/images/pic2.png', name = 'new_plot',coorlist=json.dumps([ob.__dict__ for ob in coorlist]) ,
            total = json.dumps([ob.__dict__ for ob in total]),clus_stat = json.dumps([ob.__dict__ for ob in clus_stat]),k = num_k )
        
        else:
            data.to_csv(output_filename,index=False)
            plt.savefig('static/images/pic2.png')
            coorlist =  clean()  
            total = output(num_k) 
            negeristat = nstat(num_k)
            clus_stat = out()    
            return render_template('result.html', url ='static/images/pic2.png', name = 'new_plot',coorlist=json.dumps([ob.__dict__ for ob in coorlist]),
            total = json.dumps([ob.__dict__ for ob in total]),clus_stat =json.dumps([ob.__dict__ for ob in clus_stat]),k = num_k,negeristat = json.dumps([ob.__dict__ for ob in negeristat]) )
if __name__ == '__main__':
    app.run(debug=True)