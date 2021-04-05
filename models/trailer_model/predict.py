import csv
import json
import logging
import os
import pickle
import sys
import time
import urllib
import shutil
import glob
import cv2
import dlib
import imutils
import keras
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction as aF
from tensorflow.python.keras.models import load_model

LOGGER = logging.getLogger()
LOGGER.setLevel(20)


class AudioAnalyzer:

    def __init__(self, models_dir=None, predict_dir = './'):
        if models_dir is None:
            models_dir = os.getcwd()
        keras.backend.clear_session()
        self.predict_dir = predict_dir
        self.pca = self.spectrum_cnn = self.chroma_cnn = None
        self.load_models(models_dir)

        self.signal_feature_names = ['mfcc_1_mean', 'beat', 'beat_confidence'] + [f'score_{i}' for i in range(1, 6)]

        mask = [False] * 70
        mask[8] = mask[68] = mask[69] = True
        self.signal_feature_mask = np.array(mask)

        self.spectrum_feature_names = [f'spectrum_{i}' for i in range(1, 65)]

        self.audio_features = self.signal_feature_names + self.spectrum_feature_names

    def load_models(self, models_dir):
        """Load models for audio features handling."""

        with open(os.path.join(models_dir, 'feats_dimred.model'), 'rb') as fr:
            self.pca = pickle.load(fr)

        self.spectrum_cnn = load_model(os.path.join(models_dir,'spectrum_cnn_model.h5'))

        # ?? TODO
        # self.chroma_cnn = None

        return self.pca, self.spectrum_cnn  # , self.chroma_cnn

    def analyze_audio(self, trailer_file):
        """Get all audio-related features from given trailer for box office prediction."""

        audio_file = os.path.join(self.predict_dir, 'audio_tmp-signal.mp3')
        os.system(f'ffmpeg -loglevel warning -i "{trailer_file}" -f mp3 -vn "{audio_file}"')

        feats, _ = self.featurize_audio_signal(audio_file)
        if feats is not None:
            large_feats = feats[self.signal_feature_mask]
            other_feats = feats[~self.signal_feature_mask]
            other_feats = self.pca.transform(other_feats.reshape(1, -1))
            feats = np.append(large_feats, other_feats)
        else:
            feats = [np.nan] * len(self.signal_feature_names)

        spectrum = self.compute_mel_spectrogram(audio_file)
        spectrum = self.spectrum_cnn.predict(spectrum.reshape(1, 200, 500, 3))

        os.remove(audio_file)  # clean up
        return np.append(feats, spectrum)

    def analyze_audios(self, trailers, csv_file=None):
        if csv_file is None:
            csv_file = os.path.join(self.predict_dir, 'audio_analysis.csv')

        with open(csv_file, 'w') as fw:
            csv_writer = csv.writer(fw)

            # write header
            csv_writer.writerow(['movie_id'] + self.signal_feature_names + self.spectrum_feature_names)

            for trailer in trailers:
                movie_id = os.path.splitext(os.path.basename(trailer))[0]
                all_features = self.analyze_audio(trailer)
                csv_writer.writerow([movie_id] + list(all_features))

        return csv_file

    @staticmethod
    def featurize_audio_signal(audio_file, mid_term=(1, 1), short_term=(0.05, 0.05)):
        """Extract features from raw audio signal."""

        [fs, x] = audioBasicIO.readAudioFile(audio_file)
        if fs == -1:
            sys.stderr.write(f'Failed to extract features from {audio_file}')
            return None, None

        x = audioBasicIO.stereo2mono(x)

        [mt_features, st_features, mt_feature_names] = aF.mtFeatureExtraction(
            x, fs, round(fs * mid_term[0]), round(fs * mid_term[1]),
            round(fs * short_term[0]), round(fs * short_term[1])
        )

        [beat, beat_confidence] = aF.beatExtraction(st_features, short_term[1])

        mt_feature_names.extend(['beat', 'beat_confidence'])
        mt_features = np.transpose(mt_features).mean(axis=0)
        mt_features = np.append(mt_features, [beat, beat_confidence])

        return mt_features, mt_feature_names

    @staticmethod
    def compute_mel_spectrogram(audio_file):
        """Computes a mel spectrogram of the given audio file."""

        x, fs = librosa.load(audio_file)
        mel_spectrogram = librosa.feature.melspectrogram(x, fs)

        fig = plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            librosa.power_to_db(mel_spectrogram, ref=np.max),
            x_axis='time', y_axis='mel', fmax=8000
        )
        ax = plt.gca()
        ax.axis('off')

        fig.savefig('audio_tmp-spectrum.png', bbox_inches='tight', pad_inches=0)
        plt.close('all')

        img = cv2.imread('audio_tmp-spectrum.png')
        img = cv2.resize(img, (500, 200))
        os.remove('audio_tmp-spectrum.png')  # clean up
        return img


class VisualFeatureExtractor:

    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.emotionPredictionModel()
        self.detector = dlib.get_frontal_face_detector()
        
        self.visual_features = [
            "mean_Anger", "var_Anger", "mean_Disgust", "var_Disgust", "mean_Fear", "var_Fear",
            "mean_Happy", "var_Happy", "mean_Sad", "var_Sad", "mean_Surprise", "var_Surprise",
            "mean_Neutral", "var_Neutral", "R_mean", "R_var", "G_mean", "G_var", "B_mean", "B_var",
            "Brightness_mean", "Brightness_var", "video_fps", "width", "height", "duration"
        ]

    # ****** image processing ****** """
    def Flip(self, data):
        dataFlipped = data[..., ::-1].reshape(2304).tolist()
        return dataFlipped

    def Roated15Left(self, data):
        num_rows, num_cols = data.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), 20, 1)
        img_rotation = cv2.warpAffine(data, rotation_matrix, (num_cols, num_rows))
        return img_rotation.reshape(2304).tolist()

    def Roated15Right(self, data):
        num_rows, num_cols = data.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), -20, 1)
        img_rotation = cv2.warpAffine(data, rotation_matrix, (num_cols, num_rows))
        return img_rotation.reshape(2304).tolist()

    def shiftedUp20(self, data):
        translated = imutils.translate(data, 0, -5)
        translated2 = translated.reshape(2304).tolist()
        return translated2

    def shiftedDown20(self, data):
        translated = imutils.translate(data, 0, 5)
        translated2 = translated.reshape(2304).tolist()
        return translated2

    def shiftedLeft20(self, data):
        translated = imutils.translate(data, -5, 0)
        translated2 = translated.reshape(2304).tolist()
        return translated2

    def shiftedRight20(self, data):
        translated = imutils.translate(data, 5, 0)
        translated2 = translated.reshape(2304).tolist()
        return translated2

    def flatten_matrix(self, matrix):
        vector = matrix.flatten(1)
        vector = vector.reshape(1, len(vector))
        return vector

    def zca_whitening(self, inputs):
        sigma = np.dot(inputs, inputs.T) / inputs.shape[1]  # Correlation matrix
        U, S, V = np.linalg.svd(sigma)  # Singular Value Decomposition
        epsilon = 0.1  # Whitening constant, it prevents division by zero
        ZCAMatrix = np.dot(np.dot(U, np.diag(1.0 / np.sqrt(np.diag(S) + epsilon))), U.T)  # ZCA Whitening matrix
        return np.dot(ZCAMatrix, inputs)  # Data whitening

    def global_contrast_normalize(self, X, scale=1., subtract_mean=True, use_std=True, sqrt_bias=10, min_divisor=1e-8):
        assert X.ndim == 2, "X.ndim must be 2"
        scale = float(scale)
        assert scale >= min_divisor
        mean = X.mean(axis=1)
        if subtract_mean:
            X = X - mean[:, np.newaxis]  # Makes a copy.
        else:
            X = X.copy()

        if use_std:
            ddof = 1
            if X.shape[1] == 1:
                ddof = 0

            normalizers = np.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
        else:
            normalizers = np.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale

        # Don't normalize by anything too small.
        normalizers[normalizers < min_divisor] = 1.

        X /= normalizers[:, np.newaxis]  # Does not make a copy.
        return X

    def ZeroCenter(self, data):
        data = data - np.mean(data, axis=0)
        return data

    def normalize(self, arr):
        for i in range(3):
            minval = arr[..., i].min()
            maxval = arr[..., i].max()
            if minval != maxval:
                arr[..., i] -= minval
                arr[..., i] *= (255.0 / (maxval - minval))
        return arr

    def ConvertToArrayandReshape(self, List):
        numpyarray = np.asarray(List)
        numpyarray = numpyarray.reshape(1, 48, 48)
        numpyarray = numpyarray.reshape(1, 48, 48, 1)
        numpyarray = numpyarray.astype('float32')
        return numpyarray

    def imagePreprocessing(self, crop2):
        data2 = self.ZeroCenter(crop2)
        data3 = self.zca_whitening(self.flatten_matrix(data2)).reshape(48, 48)
        data4 = self.global_contrast_normalize(data3)
        data5 = np.rot90(data4, 3)
        return data5

    # calculate brightness
    def getBrightness(self, frame):
        import cv2
        # convert image to HSV format
        HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # calculate the brightness (mean of the V-Channel)
        brightness = np.mean(HSV[:, :, 2].flatten())
        return brightness

    # calculate sharpness
    # def getSharpness(self, frame):
    #     # install cpbd library and its dependies
    #
    #     # estimate sharpness
    #     sharpness = cpbd.compute(frame)
    #     return sharpness

    # ****** Emotion prediction (NN) ****** """
    def emotionPredictionModel(self):
        img_rows, img_cols = 48, 48
        # the CIFAR10 images are RGB
        img_channels = 1

        model = Sequential()
        model.add(Convolution2D(64, (5, 5), border_mode='valid',
                                input_shape=(img_rows, img_cols, 1)))
        model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None))
        model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2), data_format="channels_first"))
        model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

        model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), data_format="channels_first"))
        model.add(Convolution2D(64, (3, 3)))
        model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None))
        model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), data_format="channels_first"))
        model.add(Convolution2D(64, (3, 3)))
        model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None))
        model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), data_format="channels_first"))
        model.add(Convolution2D(128, (3, 3)))
        model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None))
        model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), data_format="channels_first"))
        model.add(Convolution2D(128, (3, 3)))
        model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None))

        model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), data_format="channels_first"))
        model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None))
        model.add(Dropout(0.2))
        model.add(Dense(1024))
        model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None))
        model.add(Dropout(0.2))

        model.add(Dense(7))

        model.add(Activation('softmax'))

        ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
        model.compile(loss='categorical_crossentropy',
                      optimizer=ada,
                      metrics=['accuracy'])
        filepath = os.path.join(self.models_dir, "Model.120-0.6343.hdf5")
        print(filepath)
        model.load_weights(filepath)
        self.model = model

    def emotionPrediction(self, data5):
        Train_x_Init = self.ConvertToArrayandReshape(data5)
        Train_x_Flip = self.ConvertToArrayandReshape(self.Flip(data5))
        Train_x_Rotleft = self.ConvertToArrayandReshape(self.Roated15Left(data5))
        Train_x_Rotright = self.ConvertToArrayandReshape(self.Roated15Right(data5))
        Train_x_ShiftedUp = self.ConvertToArrayandReshape(self.shiftedUp20(data5))
        Train_x_ShiftedDown = self.ConvertToArrayandReshape(self.shiftedDown20(data5))
        Train_x_ShiftedLeft = self.ConvertToArrayandReshape(self.shiftedLeft20(data5))
        Train_x_ShiftedRight = self.ConvertToArrayandReshape(self.shiftedRight20(data5))
        a = np.array([self.model.predict_proba(Train_x_Init, verbose=0)[0]])
        return a.mean(axis=0)

    # ****** Face detection ****** """
    def faceDetection(self, img):  # extract faces
        grayimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dets, scores, idx = self.detector.run(grayimg, 1)
        imgWithRect = img  # cv2.imread(imagepath)
        ListOfFaces = []
        FaceWriteList = []
        temp = 0
        for rectangle in dets:
            if (scores[temp] > 0.2):
                # print("face detected")
                left = rectangle.left()
                top = rectangle.top()
                right = rectangle.right()
                bottom = rectangle.bottom()
                if top < 0:
                    offset = 0 - top
                    top = 0
                    bottom = bottom + offset
                if left < 0:
                    offset = 0 - left
                    left = 0
                    right = right + offset
                cv2.rectangle(imgWithRect, (left, top), (right, bottom), (255, 255, 255), 2)
                crop = grayimg[top:bottom, left:right]
                colored_crop = img[top:bottom, left:right]
                crop2 = cv2.resize(crop, (48, 48))
                data = self.imagePreprocessing(crop2)
                ListOfFaces.append(data)
                FaceWriteList.append(colored_crop)
            temp += 1

        return ListOfFaces, FaceWriteList

    # ****** extraction ****** """
    def extract(self, file, resolution):

        # fps
        fps = 1
        # read the video from the file
        cap = cv2.VideoCapture(file)
        cap.set(cv2.CAP_PROP_FPS, 15)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # to add to csv
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        try:
            duration = n_frames / video_fps
        except:
            duration = 0.0
        LOGGER.debug(f'duration = {duration}')
        # initialization
        colors_R = []
        colors_G = []
        colors_B = []
        brightness = []
        sharpness = []
        Anger = [0.0]
        Disgust = [0.0]
        Fear = [0.0]
        Happy = [0.0]
        Sad = [0.0]
        Surprise = [0.0]
        Neutral = [0.0]

        ret = True
        frame_index = -1
        count = 0

        # pbar2 = tqdm_notebook(total=n_frames)
        # if duration <= 600:
        while ret:
            count = count + 1
            # next frame
            frame_index = frame_index + 1
            # read the frame
            ret, frame = cap.read()

            if video_fps > fps and frame_index % int(video_fps / fps) != 0:
                continue

            if ret:
                # extract colors
                colors_R.append(np.mean(frame[:, :, 0].flatten()))
                colors_G.append(np.mean(frame[:, :, 1].flatten()))
                colors_B.append(np.mean(frame[:, :, 2].flatten()))
                # extract brightness
                brightness.append(self.getBrightness(frame))
                # extract sharpness
                # sharpness.append(self.getSharpness(frame))
                # integrate the code of syrine to extract emotion
                if count % 10 == 0:

                    ListOfFaces, FaceWriteList = self.faceDetection(frame)
                    temp_list = []
                    for face, coors in zip(ListOfFaces, FaceWriteList):
                        result = self.emotionPrediction(face)
                        temp_list.append(result)
                    for y in temp_list:
                        Anger.append(y[0])
                        Disgust.append(y[1])
                        Fear.append(y[2])
                        Happy.append(y[3])
                        Sad.append(y[4])
                        Surprise.append(y[5])
                        Neutral.append(y[6])
                # pbar2.update(frame_index - pbar2.n)

        mean_Anger = np.mean(np.array(Anger))
        mean_Disgust = np.mean(np.array(Disgust))
        mean_Fear = np.mean(np.array(Fear))
        mean_Happy = np.mean(np.array(Happy))
        mean_Sad = np.mean(np.array(Sad))
        mean_Surprise = np.mean(np.array(Surprise))
        mean_Neutral = np.mean(np.array(Neutral))

        var_Anger = np.var(np.array(Anger))
        var_Disgust = np.var(np.array(Disgust))
        var_Fear = np.var(np.array(Fear))
        var_Happy = np.var(np.array(Happy))
        var_Sad = np.var(np.array(Sad))
        var_Surprise = np.var(np.array(Surprise))
        var_Neutral = np.var(np.array(Neutral))

        features = [mean_Anger, var_Anger, mean_Disgust, var_Disgust, mean_Fear, var_Fear, mean_Happy,
                          var_Happy, mean_Sad, var_Sad, mean_Surprise, var_Surprise, mean_Neutral, var_Neutral,
                          np.mean(np.array(colors_R)), np.var(np.array(colors_R)), np.mean(np.array(colors_G)),
                          np.var(np.array(colors_G)), np.mean(np.array(colors_B)), np.var(np.array(colors_B)),
                          np.mean(np.array(brightness)), np.mean(np.var(brightness)), video_fps, width, height,
                          duration]
        # df=df.append(pd.Series(list_to_append,index=["IMDB_ID","mean_Anger","var_Anger","mean_Disgust","var_Disgust","mean_Fear","var_Fear","mean_Happy","var_Happy","mean_Sad","var_Sad","mean_Surprise","var_Surprise","mean_Neutral","var_Neutral","R_mean","R_var","G_mean","G_var","B_mean","B_var","Brightness_mean","Brightness_var","Sharpness_mean","Sharpness_var"]),ignore_index=True)

        # else:
        #     list_to_append = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        #                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return np.array(features)

    # ****** iteration over all movies ****** """
    def run(self, movies, trailers_dir, resolution, output_path, output_file):

        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        features_name = ["IMDB_ID", "mean_Anger", "var_Anger", "mean_Disgust", "var_Disgust", "mean_Fear", "var_Fear",
                         "mean_Happy", "var_Happy", "mean_Sad", "var_Sad", "mean_Surprise", "var_Surprise",
                         "mean_Neutral", "var_Neutral", "R_mean", "R_var", "G_mean", "G_var", "B_mean", "B_var",
                         "Brightness_mean", "Brightness_var", "video_fps", "width", "height", "duration"]
        # df =pd.DataFrame(columns=features_name)
        '''with open('Akram.csv', 'w+') as fp:
            wr = csv.writer(fp)
            wr.writerow(features_name)'''

        # iterate over videos
        movies_to_iterate = movies.entries[4721:5000]  # start:end
        # pbar1 = tqdm_notebook(total=len(movies_to_iterate))
        for entry in movies.entries[4721:5000]:  # start:end
            # Store sample
            videoFile = os.path.join(trailers_dir, entry.movie_id + '.mp4')
            print('Retrieving movie ', entry.movie_id)
            if not os.path.isfile(videoFile):
                print('Donwload trailer')
                print(entry.youtube_trailer)
                movies.GetTrailer(entry.imdb_trailer, entry.youtube_trailer, trailers_dir, entry.movie_id)

            while not os.path.isfile(videoFile):
                # print('Trailer is not yet ready')
                time.sleep(1)
            print('Start extracting visual features')
            # video to frames and append to colors list
            list_to_append = [entry.movie_id, ] + self.extract(os.path.join(trailers_dir, entry.movie_id + '.mp4'),
                                                               resolution)
            # df=df.append(pd.Series(list_to_append,index=features_name),ignore_index=True)
            with open(os.path.join(output_path, output_file), 'a', newline='') as f:
                wr = csv.writer(f)
                wr.writerow(list_to_append)
            print('Extracting features done!')
            # delete video
            os.remove(os.path.join(trailers_dir, entry.movie_id + '.mp4'))
            # !rm -r output_path+entry.movie_id+'.mp4
            # !rm -r to_remove
            # pbar1.update(1)

        # df.to_csv(os.path.join(output_path, output_file), sep=',')
        print("Sucess, Completed")


def fetch_trailer(youtube_url, output_file):
    #os.system("youtube-dl https://www.youtube.com/watch?v=DW5jfjN-5RI");
    path = "./tmp"
    if os.path.exists(path):
        while os.path.isdir (path):
            shutil.rmtree (path, ignore_errors=True)
    os.makedirs(path)
    cmd = "youtube-dl --format bestvideo+bestaudio[ext=m4a]/bestvideo+bestaudio/best --merge-output-format mp4 "+youtube_url+ " -o ./tmp/trailer.mp4"
    os.system(cmd);
    
    for file in os.listdir(path):
        if file.endswith(".mp4"):
            if os.path.getsize(os.path.join(path, file)) >0:
                return os.path.join(path, file)
    return None


def query_azure(feature_names, features):
    data = {
        "Inputs": {
            "input1": {
                "ColumnNames": feature_names,
                "Values": features
            }
        },
        "GlobalParameters": {},
    }

    body = str.encode(json.dumps(data))

    url = 'https://ussouthcentral.services.azureml.net/workspaces/e09832186fd7451abe2a0ca30256cebb/services/14a37a5bfcc248aa8c93341e357bb2bd/execute?api-version=2.0&details=true'
    api_key = ''  # Replace this with the API key for the web service
    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read()

        print(result)
    except urllib.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())

        print(json.loads(error.read()))

    results = json.loads(result.decode())['Results']['output1']['value']['Values']
    return np.array(results).flatten()


def predict(df_testset):
    audio_analyzer = AudioAnalyzer('./trained_model')
    visual_analyzer = VisualFeatureExtractor('./trained_model')

    feature_names = visual_analyzer.visual_features + audio_analyzer.audio_features
    features = []

    tmp_video_file = 'tmp.mp4'
    for i, url in enumerate(df_testset['url'].values):
        # LOGGER.info(f"\nProcessing {i+1}/{len(df_testset)}")
        print(f"\nProcessing {i + 1}/{len(df_testset)}")

        # LOGGER.debug(f'Retrieving video from {url}')
        print(f'Retrieving video from {url}')
        fetch_trailer(url, tmp_video_file)

        # LOGGER.debug('Start extracting visual features')
        print('Start extracting visual features')
        visual_feats = visual_analyzer.extract(tmp_video_file, (640, 480))

        # LOGGER.debug('Start extracting audio features')
        print('Start extracting audio features')
        audio_feats = audio_analyzer.analyze_audio(tmp_video_file)

        features.append(list(np.append(visual_feats, audio_feats)))
        os.remove(tmp_video_file)

    # LOGGER.info("All features extracted.")
    print("All features extracted.")

    return query_azure(feature_names, features)

def predict_oneMovie(url, predict_dir = './'):
    print("urlurlurlurlurlurlurlurlurl", url)
    audio_analyzer = AudioAnalyzer(os.path.join(predict_dir, './trained_model/'))
    visual_analyzer = VisualFeatureExtractor(os.path.join(predict_dir, './trained_model/'))

    feature_names = visual_analyzer.visual_features + audio_analyzer.audio_features
    features = []

    tmp_video_file = os.path.join(predict_dir,'/tmp/trailer.mp4')
    
    # LOGGER.info(f"\nProcessing {i+1}/{len(df_testset)}")
    print(f"\nProcessing")

    # LOGGER.debug(f'Retrieving video from {url}')
    print(f'Retrieving video from {url}')
    tmp_video_file = fetch_trailer(url, tmp_video_file)
    # LOGGER.debug('Start extracting visual features')
    print('Start extracting visual features')
    visual_feats = visual_analyzer.extract(tmp_video_file, (640, 480))

    # LOGGER.debug('Start extracting audio features')
    print('Start extracting audio features')
    audio_feats = audio_analyzer.analyze_audio(tmp_video_file)

    features.append(list(np.append(visual_feats, audio_feats)))
    os.remove(tmp_video_file)

    # LOGGER.info("All features extracted.")
    print("All features extracted.")
    predictions = query_azure(feature_names, features)
    print("feature_names:", feature_names)
    return predictions[0], list(features[0])

if __name__ == '__main__':
    argument = sys.argv[1]
    print(argument)
    if '.csv' in argument:
        test_set = argument
        df_in = pd.read_csv(test_set)
        predictions = predict(df_in)
    
        df_out = df_in.copy()
        df_out['trailer_prediction'] = predictions
        df_out.to_csv('../trailer_predictions.csv', index=False)
    else:
        url = argument
        prediction = predict_oneMovie(url)
        print("Predicted value:", prediction)
