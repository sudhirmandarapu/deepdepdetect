import csv
import numpy as np
import os
from textblob import TextBlob
from operator import itemgetter


class Preprocessor:
    def __init__(self, path, transcripts):
        self.path = path
        self.transcripts = transcripts

    def get_all_transcript_features(self, config):
        x = []
        y = []
        expected_dict = self._build_expected_dict()
        liwc_features = self._build_liwc_feature_dict()
        lda_features = self._build_lda_features_dict()
        for transcript in self.transcripts:
            if transcript.id in expected_dict:
                x.append(self._get_features_for_transcript(transcript, liwc_features, lda_features, config))
                y.append(int(expected_dict[transcript.id]))

        print('\n\n***\n')
        print('Total number of examples:', len(y))
        print('Total number of non-depressed examples:', len(y) - sum(y))
        print('Total number of depressed examples:', sum(y))
        print('\n***\n\n')

        return np.array(x, dtype='float32'), np.array(y, dtype='int32')

    def compile_all_transcripts(self, dest_dir):
        for transcript in self.transcripts:
            f = open(dest_dir + '/' + transcript.id + '.txt', "w+")
            for row in transcript.rows:
                if row.speaker == 'Participant':
                    f.write(row.value+'\n')
            f.close()

    def _build_expected_dict(self):
        expected_dict = {}
        for file in [
            'dev_split_Depression_AVEC2017.csv',
            'train_split_Depression_AVEC2017.csv'
        ]:
            with open(self.path+'/'+file) as csv_file:
                reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
                for row in reader:
                    row = row[0].split(',')
                    if row[0] == 'Participant_ID':
                        continue
                    expected_dict[row[0]] = row[1] != '0'
        return expected_dict

    def _build_lda_features_dict(self):
        lda_dict = {}
        with open(self.path+'/lda_topics.csv') as csv_file:
            reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
            for row in reader:
                row = row[0].split(',')
                last_slash = row[1].rfind('/', 0, len(row[1]))
                lda_dict[row[1][last_slash+1:last_slash+4]] = np.array(row[2:], dtype=float)
        return lda_dict

    def _get_features_for_transcript(self, transcript, liwc_features, lda_features, config):
        features = np.array([], dtype=float)
        if "liwc" in config:
            if "liwc_indexes" in config:
                features = np.array(itemgetter(*config["liwc_indexes"])(liwc_features[transcript.id]))
            else:
                features = np.array(liwc_features[transcript.id])

        if "sentiment" in config:
            features = np.append(features, self._get_sentiment(transcript))

        if "lda" in config:
            features = np.append(features, lda_features[transcript.id])
        # avg_res = self._get_average_response_length(file)
        # features.append(avg_res)
        return features

    def _build_liwc_feature_dict(self):
        liwc_dict = {}
        with open(self.path+'/liwc_features.csv') as csv_file:
            reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
            for row in reader:
                row = row[0].split(',')
                if row[0] == 'Filename':
                    continue
                name, _ = os.path.splitext(row[0])
                features = [float(x) / 100 for x in row[3:]]  # ignore the first three items in the csv
                liwc_dict[name[:3]] = features
        return liwc_dict

    def print_liwc_features(self):
        with open(self.path+'/liwc_features.csv') as csv_file:
            reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
            for row in reader:
                feature_names = row[0].split(',')[3:]
                i = 0
                print('\n\n***\n')
                for name in feature_names:
                    print(i, name)
                    i += 1
                print('\n***\n\n')
                return feature_names

    @staticmethod
    def _get_sentiment(transcript):
        participant_responses = ""
        for row in transcript.rows:
            if row.speaker == 'Participant':
                participant_responses += row.value+'\n'
        t = TextBlob(participant_responses)
        return t.sentiment.polarity

    def _get_average_response_length(self, file):
        pass

    @staticmethod
    def _len_without_spaces(words):
        counter = 0
        for word in words:
            if word != '':
                counter += 1
        return counter
