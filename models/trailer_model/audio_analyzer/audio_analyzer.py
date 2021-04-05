__doc__ = """Script for extracting various audio features from trailers.

All functionality is encapsulated in the class 'AudioAnalyzer'.
Main interface is through the method 'analyze_audios'.
Script can also be invoked from command line, usage: 'audio_analyzer <output_csv> <video1,video2,...>

script requires several pre-trained models, namely
    * 'feats_dimred.model'
    * 'spectrum_cnn_model.h5'
"""

import csv
import os
import pickle
import sys

import cv2
import librosa
import librosa.display
import numpy as np
import pandas as pd
from keras.models import load_model
from matplotlib import pyplot as plt
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction as aF


class AudioAnalyzer:

    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.getcwd()

        self.pca = self.spectrum_cnn = self.chroma_cnn = None
        self.load_models(models_dir)

        self.signal_feature_names = ['mfcc_1_mean', 'beat', 'beat_confidence'] + [f'score_{i}' for i in range(1, 6)]

        mask = [False] * 70
        mask[8] = mask[68] = mask[69] = True
        self.signal_feature_mask = np.array(mask)

        self.spectrum_feature_names = [f'spectrum_{i}' for i in range(1, 65)]

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

        audio_file = f'audio_tmp-signal.mp3'
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
            csv_file = os.path.join(os.getcwd(), 'audio_analysis.csv')

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
        ax.axis(False)

        fig.savefig('audio_tmp-spectrum.png', bbox_inches='tight', pad_inches=0)
        plt.close('all')

        img = cv2.imread('audio_tmp-spectrum.png')
        img = cv2.resize(img, (500, 200))
        os.remove('audio_tmp-spectrum.png')  # clean up
        return img

    @staticmethod
    def compute_chromagram(audio_file):
        # ?? TODO
        pass


def demo():
    """Sample usage."""
    videos = [f'demo/{f}' for f in os.listdir('demo')]

    aa = AudioAnalyzer()
    print(aa.analyze_audios(videos, 'demo/audio_analysis.csv'))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.stderr.write('Less than two arguments were passed. Running demo...\n')
        demo()
    else:
        output_csv = sys.argv[1]
        trailers = sys.argv[2].split(',')

        analyzer = AudioAnalyzer()
        analyzer.analyze_audios(trailers, output_csv)
