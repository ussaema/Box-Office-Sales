from django.shortcuts import render
from django.http import JsonResponse
from django.forms.models import model_to_dict
from django.http import HttpResponse
import numpy as np
import time
import datetime
import os
import json
import functools
import itertools
import operator
import urllib.parse
import urllib.request
import requests
#!pip install youtube_dl
import youtube_dl
#!pip install imdbpy
import imdb
from bs4 import BeautifulSoup
import cv2     # for capturing videos
import math   # for mathematical operations
from skimage.transform import resize   # for resizing images
import time
import random
import speech_recognition as sr
from bs4 import BeautifulSoup
import urllib.request
import urllib.parse
import re
import json
import sys
sys.path += ('../neural_network/',)
from metadata_model.predict import predict_oneMovie as metadata_predictor
from trailer_model.predict import predict_oneMovie as trailer_predictor

def urlopen(url, mobile = False):
    try:
        if mobile:
            urlheader =  {'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 5_0 like Mac OS X) AppleWebKit/534.46' ,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
            'Accept-Encoding': 'none',
            'Accept-Language': 'en-US,en;q=0.8',
            'Connection': 'keep-alive'}
        else:
            urlheader = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) ' 
                          'AppleWebKit/537.11 (KHTML, like Gecko) '
                          'Chrome/23.0.1271.64 Safari/537.11',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
            'Accept-Encoding': 'none',
            'Accept-Language': 'en-US,en;q=0.8',
            'Connection': 'keep-alive'
                        }
        #header2 = 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'
        return urllib.request.urlopen(urllib.request.Request(url=url, data=None, headers=urlheader)).read().decode('utf-8')
    except HTTPError as e:
        if (_WARNINGS):
            time.sleep(5);
            warnings.warn(str(e))
            return urlopen(url)
        else:
            raise e

class DatasetLoader():
    def __init__(self):
        self.Youtube_urlroot = "https://www.youtube.com"
        self.Imdb_urlroot = "https://www.imdb.com"
        pass
    
    def Load(self, filename, dataset_dir):
        self.filename = filename
        # go to repository
        with open(os.path.join(dataset_dir,filename+'.json'), "r+") as jsonFile:
            self.data = json.load(jsonFile)
            jsonFile.seek(0)  # rewind
            json.dump(self.data, jsonFile)
            jsonFile.truncate()
        # filter english movies only
        self.data = [item for id, item in self.data.items() if 'United Kingdom' in item['boxOffice']['country'] or  'United States' in item['boxOffice']['country']]
        # get size
        self.size = len(self.data)
    
    def ExtractMetaInfo(self):
        genre_color = {'Action':'yell', 'Adventure':'orange', 'Comedy': 'green'}
        # extract names
        self.name = [item['name'] for item in self.data]
        # extract year
        self.year = [item['boxOffice']['year'] for item in self.data if 'year' in item['boxOffice'].keys()]
        # extract and code genres
        self.genres = {}
        for id, item in enumerate(self.data):
            if 'genre' in item.keys():
                if isinstance(item['genre'], list):
                    self.genres[id] = []
                    for count, genre in enumerate(item['genre']):
                        self.genres[id] += [[genre, genre_color[genre] if genre in genre_color else 'blue'],]
                        if count == 2:
                            break
                else:
                    genre = item['genre']
                    self.genres[id] = [[genre, genre_color[genre] if genre in genre_color else 'blue']]
            else:
                pass
        self.genres = list(self.genres.values())
        # extract images
        self.image = [item['image'] if 'image' in item.keys() else '' for item in self.data]
        # extract total revenue
        self.revenue_total = [item['boxOffice']['revenue_total'] for item in self.data]
        # extract revenue first week
        self.revenue_week1 = [item['boxOffice']['revenue_week1'] for item in self.data]
        # extract rating
        self.rating = [float(item['aggregateRating']['ratingValue']) if 'aggregateRating' in item.keys() else -1. for item in self.data]
        # extract trailers
        self.youtube_trailer = [item['video']['videos']['videoMetadata']['youtube']['url'].replace('/watch?v=', '') for item in self.data]
        # extract trailers
        self.url = [item['url'].replace('/title/tt', '').replace('/', '') for item in self.data]
        # extract date
        self.date = []
        for item in self.data:
            try:
                self.date += [time.mktime(datetime.datetime.strptime(item['datePublished'], "%Y-%m-%d").timetuple()),]
            except:
                self.date += [-1.,]
        
        
# Create your views here.
movies = DatasetLoader()
movies.Load("Dataset", dataset_dir = '../neural_network/dataset')
# Extract and preprocess raw Data
movies.ExtractMetaInfo()
context = {}


# Create your views here.
def tumovies_index_view(request):
    while True:
        try:
             # list of random movies (1)
            movies_random_idx = list(range(len(movies.data)))
            movies_random_idx = random.sample(movies_random_idx, len(movies_random_idx))
            context["movies_random"] = [[movies.name[id], movies.genres[id], movies.image[id], movies.rating[id], movies.youtube_trailer[id], movies.year[id], movies.url[id]] for count, id in enumerate(movies_random_idx) if (len(movies.image[id]) and count<14 and movies.rating[id] >=0 and movies.date[id]>=0)]
            # list of random movies (2)
            movies_random_idx2 = list(range(len(movies.data)))
            movies_random_idx2 = random.sample(movies_random_idx, len(movies_random_idx))
            context["movies_random2"] = [[movies.name[id], movies.genres[id], movies.image[id], movies.rating[id], movies.youtube_trailer[id], movies.year[id], movies.url[id]] for count, id in enumerate(movies_random_idx2) if (len(movies.image[id]) and count<14 and movies.rating[id] >=0 and movies.date[id]>=0)]
            # list of most popular movies
            movies_rating = np.array(movies.rating)
            movies_rating_idx = np.argsort(-movies_rating).tolist()
            context["movies_rating"] = [[movies.name[id], movies.genres[id], movies.image[id], movies.rating[id]] for count, id in enumerate(movies_rating_idx) if (len(movies.image[id]) and count<14 and movies.rating[id] >=0)]
            # list of last published
            movies_date = np.array(movies.date)
            movies_date_idx = np.argsort(-movies_date).tolist()
            context["movies_date"] = [[movies.name[id], movies.genres[id], movies.image[id], movies.rating[id], movies.youtube_trailer[id], movies.year[id], movies.url[id]] for count, id in enumerate(movies_date_idx) if (len(movies.image[id]) and count<14 and movies.rating[id] >=0 and movies.date[id]>=0)]
            # list of last predicted
            with open('predicted_movies.json') as json_file:
                last_predicted = json.load(json_file)
                context["movies_predicted"] = []
                count = 0
                for item in reversed(last_predicted):
                    if 'image' in item.keys() and count<4 and 'aggregateRating' in item.keys():
                        context["movies_predicted"].append([item['name'], item['genre'], item['image'], item['aggregateRating']['ratingValue'], item['imdb_id']])
                        count+= 1
        except IndexError:
             continue
        else:
            return render(request, "index.html",context= context)
    
   

#---------------------------------------------------------------------

def YoutubeRetrieve(movie_name, movie_year):
    
    data = {}
    query = urllib.parse.quote(movie_name+ ' '+ str(movie_year)+' official trailer')
    url = 'https://www.google.com/search?biw=1620&bih=889&tbs=srcf%3AH4sIAAAAAAAAANOuzC8tKU1K1UvOz1UrSM0vyIEwSzJSy4sSC8DsssSizNSSSoiSnMTK5NS8kqLEHL2UVLX0zPREEA0AcHJbJEcAAAA&tbm=vid&q='+query
    #print(url)
    html = urlopen(url, mobile=True)
    soup = BeautifulSoup(html, 'html.parser')
    div = soup.findAll(attrs={'class':'mnr-c Tyz4ad'})
    if len(div):
        try:
            pos = 0
            while not('watch?v=' in str(div[pos])):
                pos += 1
            div = div[pos]

            href = div.find_all('a')[0]['href']
            #print(href)
            data['name'] = soup.findAll(attrs={'class':'lCT7Rc Rqb6rf'})[0].text
            data['url'] = '/watch?v='+str(href[href.index('watch?v=')+len('watch?v='):])
            return data
        except IndexError:
            return None
    else:
        return None
            
def IMDbRetrieve(id):
    
    data = {}
    if id:
        url = 'https://www.imdb.com/title/tt'+str(id)
        html = urlopen(url)
        soup = BeautifulSoup(html, 'html.parser')
        load = json.loads(soup.find('script', type='application/ld+json').text)
        data.update(load)
    return data
    
def YoutubeRetrieveTitle(id):
    
    title = None
    if id:
        url = "https://www.youtube.com/embed/"+str(id)
        print("url: ", url)
        html = urlopen(url, mobile=True)
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string
    return title

def IMDbSearch(movie_name):
    
    IMDb = imdb.IMDb()
    result = IMDb.search_movie(movie_name)
    #print(result)
    score = 0
    res = []
    for item in result:
        print(item.movieID)
        if (item['kind'] == 'movie'):
            if (len(set(list(str(item).lower().split(" "))).intersection(list(movie_name.lower().split(" "))))>score):
                movie = IMDbRetrieve(item.movieID)
                try:
                    movie.update({'year': item['year']})
                    print("rrrrrrrrrrrrrrrrrrrrrrr")
                except Exception as e:
                    pass
                    
                res +=[movie,]
            else:
                continue
    return res
def search(request):
    
    return render(request, "search.html", context= context)

def search_query(request):
    global context    
    
    if (request.POST['fct'] == "predict_value"):
        print("#################request.POST############", request.POST)
        print("predict triggered")
        resp = ""
        metadata_pediction = None
        trailer_pediction = None
        try:
            if (request.POST['metadata_pred_bool'] == 'true'):
                print("********************call metadata network***********************")
                print("network call")
                metadata_pediction = metadata_predictor(request.POST['imdb_id'], '../neural_network/metadata_model')
                print("meta-data predicted value =", metadata_pediction)
                resp += str(metadata_pediction)+","
            else:
                resp += "null,"
            trailer_pediction = None; trailer_features = None
            if (request.POST['trailer_pred_bool'] == 'true'):
                print("********************call trailer network***********************")
                print("network call")
                trailer_pediction, trailer_features = trailer_predictor("https://youtube.com/watch?v="+request.POST['trailer_url'], '../neural_network/trailer_model')
                #emotions = [visual_feats[0], visual_feats[2], visual_feats[4], visual_feats[6], visual_feats[8],visual_feats[10], visual_feats[12]]
                #argmax_emotion = np.argmax(np.array(emotions))
                #emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                #mle_emotion = emotion_label[argmax_emotion]
                #mle_r = visual_feats[14]
                #mle_g = visual_feats[16]
                #mle_b = visual_feats[18]
                #mle_brightness = visual_feats[20]
                #fps = visual_feats[22]
                #height = visual_feats[23]
                #width = visual_feats[24]
                #duration = visual_feats[25]
                print("trailer predicted value =", trailer_pediction)
                print("trailer_features:", trailer_features)
                resp += str(trailer_pediction)+","
            else:
                resp += "null,"
            if (metadata_pediction and trailer_pediction and request.POST['metadata_pred_bool'] == 'true' and request.POST['trailer_pred_bool'] == 'true'):
                weight = 0.92
                print("********************mixed trailer network***********************")
                print("meta prediction:", float(metadata_pediction.astype(np.float)))
                print("weighted meta prediction:", weight*float(metadata_pediction.astype(np.float)))
                print("trailer prediction:", float(trailer_pediction.astype(np.float)))
                print("weighted trailer prediction:", (1-weight)*float(trailer_pediction.astype(np.float)))
                resp += str(float(metadata_pediction.astype(np.float))*weight + (1-weight)*float(trailer_pediction.astype(np.float)))
            else:
                resp += str("null")
            if (request.POST['trailer_pred_bool'] == 'true'):
                resp +=","
                features_str = ""
                for x in trailer_features:
                    features_str += str(x)+","                    
                print("ressssss:", features_str)
                resp += features_str[:-1]
                
        except Exception as e:
            print("error:", e)
            resp = "null,null,null"
        print("response--------------------", resp)
        return HttpResponse(json.dumps(resp), content_type="application/json")
        
    
    if (request.POST['fct'] == "search_list"):
        context['query'] = request.POST['title']
        context['query_result'] = IMDbSearch(context['query'])
        list_movies = "";
        for res in context['query_result']:
            if ('url' in res.keys() and 'name' in res.keys() and 'image' in res.keys()):
                list_movies +=  "<div class='movie-item-style-2 movie-item-style-1'>"\
                "<div class='mv-img3'><img class='loadAsync preLoad' data-src='"+res['image']+"' alt=''></div>"\
                "<div class='hvr-inner'><a href='/movie?id="+res['url'][9:-1]+"'> Predict <i class='ion-android-arrow-dropright'></i> </a></div>"\
                "<div class='mv-item-infor'>"\
                 "<h6><a href='/movie?id="+res['url'][9:-1]+"' title='"+res['name']+"'>"+res['name']+"</a></h6>"\
                "</div>"\
                "</div>"
        return HttpResponse(json.dumps(list_movies), content_type="application/json")
    if (request.POST['fct'] == "search_movie"):
        print("****************search_movie****************")
        context['imdb_id'] = request.POST['id']
        real_value = None
        total_value = None
        for id in range(len(movies.name)):
            if movies.url[id] == context['imdb_id']:
                real_value = movies.revenue_week1[id]
                total_value = movies.revenue_total[id]
                break
        IMDb = imdb.IMDb()
        item = IMDb.get_movie(context['imdb_id'])
        movie = IMDbRetrieve(item.movieID)
        movie.update({'imdb_id': context['imdb_id']})
        movie.update({'year': item['year']})
        movie.update({'youtube': YoutubeRetrieve(movie['name'], item['year'])})
        movie.update({'outline': item.get('plot outline')})
        movie.update({'real_value': real_value})
        movie.update({'total_value': total_value})
        print("movie retrieved", movie)
        return HttpResponse(json.dumps(movie), content_type="application/json")
        
    if (request.POST['fct'] == "search_trailer"):
        print("zzzzzzzzzzzzzzzzz")
        context['youtube_id'] = request.POST['v']
        with open('predicted_movies.json') as json_file:
            data = json.load(json_file)
            movie = None
            for item_data in data:
                if 'youtube_id' in item_data.keys():
                    if context['youtube_id'] == item_data['youtube_id']:
                        movie = item_data
                        break
            if movie is None:
                real_value = None
                total_value = None
                for id in range(len(movies.name)):
                    if context['youtube_id'] in movies.youtube_trailer[id] :
                        real_value = movies.revenue_week1[id]
                        total_value = movies.revenue_total[id]
                predicted_value = None
                movie = {}
                movie.update({'youtube_id': context['youtube_id']})
                movie.update({'real_value': real_value})
                movie.update({'total_value': total_value})
                movie.update({'predicted_value': predicted_value})
                movie.update({'youtube_title': YoutubeRetrieveTitle(context['youtube_id'])})
                if (len(data) >= 100):
                    data.pop(0)
                data.append(movie)
                with open('predicted_movies.json', 'w') as outfile:
                    json.dump(data, outfile)
        return HttpResponse(json.dumps(movie), content_type="application/json")
    
#---------------------------------------------------------------------------
def movie(request):
    
    return render(request, "movie.html", context= context)
def trailer(request):
    
    return render(request, "trailer.html", context= context)
#---------------------------------------------------------------------------
def query(request):
    
    global context
    print(request.POST)
    """if request.POST['button'] == 'searchButn':
        #if request.method == 'POST':
        #if request.is_ajax():
        import json
        print("all:"); print(request.POST);
        print("search_field data: %s" % request.POST['search_field'])
        mydata = [{'foo': 1, 'baz': 2}]
        return HttpResponse(json.dumps(mydata), content_type="application/json")"""
    if request.POST['button'] == 'searchButn':
        return HttpResponse(json.dumps('callSearchPage'), content_type="application/json")

    if request.POST['button'] == 'speechButn':
        r = sr.Recognizer()
        speechtext = ""
        r.energy_threshold = 300
        with sr.Microphone() as source:
            print("say something")
            try:
                audio = r.listen(source , timeout=10)
            except sr.WaitTimeoutError:
                print("time error")
                return HttpResponse(json.dumps('-timeout-'), content_type="application/json")
            print("Time over")
        try:
            speechtext = r.recognize_google(audio)
        except:
            return HttpResponse(json.dumps('-recognition_failed-'), content_type="application/json")
            pass
        print(speechtext)
        return HttpResponse(json.dumps(speechtext), content_type="application/json")
        
    return render(request,"index.html")
    