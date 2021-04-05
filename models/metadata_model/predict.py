import json
import logging
import os
import urllib.request
import urllib.parse
import pickle
import sys

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import keras
from keras.models import load_model


LOGGER = logging.getLogger()
LOGGER.setLevel(20)


class MetadataPredictor:

    def __init__(self, models_dir=None, encoding_dir=None):
        if models_dir is None:
            self.models_dir = os.getcwd()
        else:
            self.models_dir = models_dir
        
        if encoding_dir is None:
            self.encoding_dir = os.getcwd()
        else:
            self.encoding_dir = encoding_dir
        keras.backend.clear_session()
        self.model = load_model(os.path.join(self.models_dir, 'metadata.model'))

    def predict(self, imdb_id):
        movie = self.IMDbRetrieve(imdb_id)
        (actors, creators, directors, genres, contentRating, year, duration) = self.get_movie_data(movie)

        # Prepare input data -> encode -> reduce dimensions
        with open(os.path.join(self.encoding_dir, 'encoding.pkl'), 'rb') as f:
            (
                all_actors,
                all_directors,
                all_creators,
                all_genres,
                all_contentRatings,
                min_year,
                max_duration,
                class_limits
            ) = pickle.load(f)

        # print('Encoding data ...')

        actors_coded = np.zeros(len(all_actors), )
        for x in actors:
            try:
                actors_coded += all_actors[x]
            except:  # unknown actors
                pass

        creators_coded = np.zeros(len(all_creators), )
        for x in creators:
            try:
                creators_coded += all_creators[x]
            except:  # unknown
                pass

        directors_coded = np.zeros(len(all_directors), )
        try:
            directors_coded += all_directors[directors]
        except:  # unknown
            pass

        genres_coded = np.zeros(len(all_genres), )
        for x in genres:
            try:
                genres_coded += all_genres[x]
            except:  # unknown
                pass

        contentRating_coded = np.zeros(len(all_contentRatings), )
        try:
            contentRating_coded += all_contentRatings[contentRating]
        except:  # unknown
            pass

        year_coded = (year - min_year) / (2019 - min_year)
        duration_coded = duration / max_duration

        # print(np.sum(actors_coded))
        # print(np.sum(creators_coded))
        # print(np.sum(directors_coded))
        # print(np.sum(genres_coded))
        # print(np.sum(contentRating_coded))
        # print(np.sum(year_coded))
        # print(np.sum(duration_coded))

        # Choose features to use
        # features = ['actors', 'creators', 'directors', 'genres', 'contentRating', 'year', 'duration']

        # Generate input data

        metadata_input = actors_coded
        metadata_input = np.concatenate((metadata_input, creators_coded))
        metadata_input = np.concatenate((metadata_input, directors_coded))
        metadata_input = np.concatenate((metadata_input, genres_coded))
        metadata_input = np.concatenate((metadata_input, contentRating_coded))
        metadata_input = np.append(metadata_input, year_coded)
        metadata_input = np.append(metadata_input, duration_coded)

        # Evaluate loaded model
        prediction = self.model.predict(np.array([metadata_input]))

        # Re-map class to revenue value
        index = np.argmax(prediction)
        revenue = np.ceil(class_limits[index] + (class_limits[index + 1] - class_limits[index]) / 2)

        # print(prediciton)
        # print(class_limits)

        # print(f'Predicted Revenue: {revenue}')
        return revenue

    @staticmethod
    def IMDbRetrieve(id):
        data = {}
        if id:
            if str(id)[:2] != 'tt':
                url = 'https://www.imdb.com/title/tt' + str(id)
            else:
                url = 'https://www.imdb.com/title/' + str(id)
            html = MetadataPredictor.urlopen(url)
            soup = BeautifulSoup(html, 'html.parser')
            load = json.loads(soup.find('script', type='application/ld+json').text)
            data.update(load)
        return data

    @staticmethod
    def get_movie_data(movie):
        actors = []
        for x in movie["actor"]:
            try:
                actors.append(x["name"])
            except:
                pass

        creators = []
        for x in movie["creator"]:
            try:
                creators.append(x["name"])
            except:
                pass

        try:
            director = movie["director"]["name"]
        except:
            director = ""

        try:
            genres = movie["genre"]
        except:
            genres = []

        try:
            contentRating = movie["contentRating"]
        except:
            contentRating = ""

        year = movie["datePublished"][0:4]
        duration = movie["duration"]

        if 'H' in duration and 'M' in duration:
            hours = int(duration[duration.find('PT') + 2:duration.find('H')])
            minutes = int(duration[duration.find('H') + 1:duration.find('M')])
        if 'H' in duration:
            hours = int(duration[duration.find('PT') + 2:duration.find('H')])
        elif 'M' in duration:
            minutes = int(duration[duration.find('PT') + 2:duration.find('M')])

        duration = (60 * hours) + minutes

        return (actors, creators, director, genres, contentRating, float(year), float(duration))

    @staticmethod
    def urlopen(url, mobile=False):

        if mobile:
            urlheader = {'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 5_0 like Mac OS X) AppleWebKit/534.46',
                         'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,/;q=0.8',
                         'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                         'Accept-Encoding': 'none',
                         'Accept-Language': 'en-US,en;q=0.8',
                         'Connection': 'keep-alive'}
        else:
            urlheader = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '
                                       'AppleWebKit/537.11 (KHTML, like Gecko) '
                                       'Chrome/23.0.1271.64 Safari/537.11',
                         'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,/;q=0.8',
                         'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                         'Accept-Encoding': 'none',
                         'Accept-Language': 'en-US,en;q=0.8',
                         'Connection': 'keep-alive'
                         }
        # header2 = 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'
        return urllib.request.urlopen(urllib.request.Request(url=url, data=None, headers=urlheader)).read().decode(
            'utf-8')


def predict(df_testset):
    predictor = MetadataPredictor('./trained_model/', './encoding/')
    predictions = list()

    for i, imdb_id in enumerate(df_testset['imdb_id'].values):
        # LOGGER.info(f"Processing {i+1}/{len(df_testset)}")
        print(f"Processing {i + 1}/{len(df_testset)}")

        prediction = predictor.predict(imdb_id)
        predictions.append(prediction)

    # LOGGER.info("Prediction finished.")
    print("Predictions finished.")

    return predictions
    
def predict_oneMovie(imdb_id, predict_dir = './'):
    predictor = MetadataPredictor(os.path.join(predict_dir, './trained_model/'), os.path.join(predict_dir, './encoding/'))

    print(f"Processing")
    prediction = predictor.predict(imdb_id)

    # LOGGER.info("Prediction finished.")
    print("Prediction finished.")

    return prediction

if __name__ == '__main__':
    argument = sys.argv[1]
    if '.csv' in argument:
        test_set = argument
        df_in = pd.read_csv(test_set)
        predictions = predict(df_in)
    
        df_out = df_in.copy()
        df_out['metadata_prediction'] = predictions
        df_out.to_csv('../metadata_predictions.csv', index=False)
        print("done")
    else:
        imdb_id = argument
        prediction = predict_oneMovie(imdb_id)
        prediction_value = prediction
        print("predicted value:", prediction)
    