import csv
import json
import os
import urllib
import urllib.request

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction as aF
from pyAudioAnalysis.audioFeatureExtraction import mtFeatureExtraction, beatExtraction


class Utils:

    @staticmethod
    def urlopen(url, mobile=False):
        """
        Author: Oussema Dhaouadi

        :param url:
        :param mobile:
        :return:
        """
        if mobile:
            url_header = {
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 5_0 like Mac OS X) AppleWebKit/534.46',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                'Accept-Encoding': 'none',
                'Accept-Language': 'en-US,en;q=0.8',
                'Connection': 'keep-alive'
            }
        else:
            url_header = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '
                              'AppleWebKit/537.11 (KHTML, like Gecko) '
                              'Chrome/23.0.1271.64 Safari/537.11',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                'Accept-Encoding': 'none',
                'Accept-Language': 'en-US,en;q=0.8',
                'Connection': 'keep-alive'
            }

        return urllib.request.urlopen(
            urllib.request.Request(
                url=url,
                data=None,
                headers=url_header
            )
        ).read().decode('utf-8')


class AudioDataset:

    def __init__(self, audio_dataset):
        self.audio_dataset = os.path.abspath(audio_dataset)

        self.cmd_trailer_download = lambda yt_url, fname: f'''youtube-dl --quiet \
                            --format bestvideo+bestaudio[ext=m4a]/bestvideo+bestaudio/best \
                            --merge-output-format mp4 {yt_url} -o "{fname}.mp4"'''
        self.cmd_audio_extract = lambda fname: f'''ffmpeg -loglevel warning -i "{fname}.mp4" -f mp3 -vn "{fname}.mp3"'''

    @staticmethod
    def _sort_imdb_videos(imdb_video):
        quality = imdb_video['definition'][:-1]
        if quality.isnumeric():
            return int(quality)
        else:
            return - 0xffff

    def fetch_trailer(self, imdb_video_url, youtube_video_url, trailers_dir, filename):
        """Downloads a video specified by given URL.
        
        Author: Oussema Dhaouadi
        Adjusted by Jan Toth

        :param imdb_video_url:
        :param youtube_video_url:
        :param trailers_dir:
        :param filename:
        :return:
        """
        os.chdir(trailers_dir)
        if len(imdb_video_url):
            try:
                imdb_vid = imdb_video_url[imdb_video_url.index('/imdb/vi') + 6:]
                html = Utils.urlopen(imdb_video_url)
                script = BeautifulSoup(html, 'html.parser').find_all('script')[-3].text
                load = json.loads(script[script.index('push(') + len('push('):script.index(');')])

                videos = load['videos']['videoMetadata'][imdb_vid]['encodings']
                videos.sort(key=self._sort_imdb_videos, reverse=True)
                best_quality_video = videos[0]
                urllib.request.urlretrieve(best_quality_video['videoUrl'], filename + '.mp4')
            except:
                os.system(self.cmd_trailer_download(youtube_video_url, filename))
        else:
            os.system(self.cmd_trailer_download(youtube_video_url, filename))

    def fetch_audio(self, imdb_trailer, youtube_trailer, target_dir, filename, keep_video=False):
        """Downloads an audio from a video specified by given URL.

        :param imdb_trailer:
        :param youtube_trailer:
        :param target_dir:
        :param filename:
        :param keep_video:
        :return:
        """
        self.fetch_trailer(imdb_trailer, youtube_trailer, target_dir, filename)
        os.system(self.cmd_audio_extract(filename))

        if not keep_video:
            os.system('rm -f *.mp4')

    def build_dataset(self, target_dir, num_classes, elems_per_class=None, measurements=(True, True, True)):

        def compute_features(audio_file):
            if measurements[0]:
                features, _ = self.extract_features(audio_file)
                csv_writer.writerow([instance_id, class_label] + list(features))
                fw.flush()

            if measurements[1]:
                self.compute_chromagram(audio_file, chromagram_dir)

            if measurements[2]:
                self.compute_mel_spectrogram(audio_file, spectrogram_dir)

            os.system('rm -f *.mp3')

        def detect_previous_work():
            curr_class = -1
            done_instances = set()

            if os.path.exists('audio_features.csv'):
                df = pd.read_csv('audio_features.csv')
                curr_class = df.tail(1)['target_class'].values[0]
                done_instances = set(df['id'].values)

            return curr_class, done_instances

        cwd = os.getcwd()
        os.chdir(target_dir)
        target_dir = os.path.abspath('.')
        curr_class, done_instances = detect_previous_work()

        df = pd.read_csv(self.audio_dataset)
        sorted_df = df.sort_values('revenue_opening')
        data_split = np.array_split(sorted_df.index.values, num_classes)

        if elems_per_class is None or elems_per_class > len(data_split[-1]):
            elems_per_class = -1

        skipped = list()
        header = not os.path.exists('audio_features.csv')
        with open('audio_features.csv', 'a') as fw:
            csv_writer = csv.writer(fw)

            for class_label in range(num_classes):
                done = 0

                if class_label < curr_class:
                    continue

                class_data = data_split[class_label]
                class_data = self._sample_class(class_data, elems_per_class)
                os.chdir(target_dir)

                class_dir = str(class_label)
                if not os.path.exists(class_dir): os.mkdir(class_dir)
                os.chdir(class_dir)

                chromagram_dir = 'chromagrams'
                spectrogram_dir = 'spectrograms'
                if not os.path.exists(chromagram_dir): os.mkdir(chromagram_dir)
                if not os.path.exists(spectrogram_dir): os.mkdir(spectrogram_dir)

                for instance_id in class_data:
                    done += 1
                    if done % 10 == 0:
                        print(f"Class '{class_label}': Already processed {done - 1}/{len(class_data)}")

                    if instance_id in done_instances:
                        continue

                    name, revenue_opening, imdb_trailer, youtube_trailer = df.loc[instance_id, :]
                    fname = str(instance_id)

                    if imdb_trailer is np.nan:
                        imdb_trailer = ""

                    try:
                        self.fetch_audio(imdb_trailer, youtube_trailer, '.', fname)
                        audio_file = f'{fname}.mp3'
                        if not header:
                            _, features_names = self.extract_features(audio_file)
                            csv_writer.writerow(['id', 'target_class'] + features_names)
                            header = True

                        compute_features(audio_file)
                        plt.close('all')
                    except Exception as e:
                        print(f'Error while processing {instance_id}\n{e}')
                        skipped.append(instance_id)

        os.chdir(cwd)
        return skipped

    @staticmethod
    def _sample_class(instances, num_elements):
        if num_elements < 0:
            return instances
        else:
            return np.random.choice(instances, num_elements, replace=False)

    @staticmethod
    def extract_features(audio_file, mid_term=(1, 1), short_term=(0.05, 0.05), compute_beat=True):
        """Extracts features from an audio file.

        :param audio_file: path to the audio file
        :param mid_term: tuple: (mid-term window, mid-term step)
        :param short_term: tuple: (short-term window, short-term step)
        :param compute_beat: boolean if beat should also be computed
        :return: tuple: (numpy array with extracted features, list of extracted features' names)
        """
        # sampling frequency 'Fs' and the signal 'x'
        [fs, x] = audioBasicIO.readAudioFile(audio_file)
        # assert fs != -1
        x = audioBasicIO.stereo2mono(x)

        if compute_beat:
            [mt_features, st_features, mt_feature_names] = mtFeatureExtraction(
                x, fs, round(fs * mid_term[0]), round(fs * mid_term[1]),
                round(fs * short_term[0]), round(fs * short_term[1])
            )

            [beat, beat_confidence] = beatExtraction(st_features, short_term[1])

            mt_feature_names.extend(['beat', 'beat_confidence'])
            mt_features = np.transpose(mt_features).mean(axis=0)
            mt_features = np.append(mt_features, [beat, beat_confidence])
        else:
            [mt_features, _, mt_feature_names] = mtFeatureExtraction(
                x, fs, round(fs * mid_term[0]), round(fs * mid_term[1]),
                round(fs * short_term[0]), round(fs * short_term[1])
            )

            mt_features = np.transpose(mt_features).mean(axis=0)

        return mt_features, mt_feature_names

    @staticmethod
    def compute_chromagram(audio_file, target_dir=None, plot=False):
        """Computes chromagram of the given audio file.

        :param audio_file: path to the audio file
        :param target_dir: directory where the resulting PNG will be saved
        :param plot: Boolean flag if the saved plot should also be displayed
        :return: chromagram and plot's axes
        """
        # prepare target image name
        if target_dir is None:
            target = os.path.splitext(audio_file)[0]
        else:
            filename = os.path.split(audio_file)[1]
            target = os.path.join(target_dir, os.path.splitext(filename)[0])

        # sampling frequency 'Fs' and the signal 'x'
        fs, x = audioBasicIO.readAudioFile(audio_file)
        x = audioBasicIO.stereo2mono(x)

        chromagram, time_axis, freq_axis = aF.stChromagram(x, fs, round(fs * 0.040), round(fs * 0.040), plot)

        # prepare and save the desired image
        fig, ax = plt.subplots()
        ax.axis('off')
        chromagram_plot = chromagram.transpose()[::-1, :]

        ratio = int(chromagram_plot.shape[1] / (3 * chromagram_plot.shape[0]))
        if ratio < 1:
            ratio = 1
        chromagram_plot = np.repeat(chromagram_plot, ratio, axis=0)

        image_plot = plt.imshow(chromagram_plot)
        image_plot.set_cmap('jet')
        fig.savefig(f'{target}.png', bbox_inches='tight', pad_inches=0)

        return chromagram, time_axis, freq_axis

    @staticmethod
    def compute_mel_spectrogram(audio_file, target_dir=None, plot=False):
        """Computes a mel spectrogram of the given audio file.

        :param audio_file: path to the audio file
        :param target_dir: directory where the resulting PNG will be saved
        :param plot: Boolean flag if the saved plot should also be displayed
        :return: mel spectrogram and plot's axes
        """

        # prepare target image name
        if target_dir is None:
            target = os.path.splitext(audio_file)[0]
        else:
            filename = os.path.split(audio_file)[1]
            target = os.path.join(target_dir, os.path.splitext(filename)[0])

        # the signal 'x' and the sampling frequency 'Fs'
        x, fs = librosa.load(audio_file)
        mel_spectrogram = librosa.feature.melspectrogram(x, fs)

        fig = plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            librosa.power_to_db(mel_spectrogram, ref=np.max),
            x_axis='time', y_axis='mel', fmax=8000
        )
        ax = plt.gca()
        ax.axis(False)

        fig.savefig(f'{target}.png', bbox_inches='tight', pad_inches=0)

        if plot:
            ax.axis(True)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel spectrogram')
            plt.tight_layout()
            plt.show()


def main():
    audio_dataset = AudioDataset('data/trailer_dataset.csv')
    skipped = audio_dataset.build_dataset('data/audio_dataset', 2)

    print("Skipped:")
    print(skipped)


if __name__ == '__main__':
    np.random.seed(0)
    main()
