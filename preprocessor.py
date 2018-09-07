import csv
import numpy as np
import os


class Preprocessor:
    def __init__(self, path):
        """
        Pre-processes the transcripts into to an array of features that can be inputted to an ML model.
        :param path: Location of all the transcripts.
        """
        self.path = path

    def get_all_transcript_features(self):
        """
        Creates the input features array and the expected outputs arrays based on transcripts.
        :return: Inputs (X) and expected outputs (Y).
        """
        x = []
        y = []
        expected_dict = self._build_expected_dict()
        liwc_features = self._build_liwc_feature_dict()
        for file in os.listdir(self.path):
            name, ext = os.path.splitext(file)
            if ext != '.csv' or name[4:] != 'TRANSCRIPT':
                continue
            x.append(self._get_features_for_transcript(file, liwc_features))
            y.append(int(expected_dict[name[:3]]))
        return np.array(x, dtype='float32'), np.array(y, dtype='int32')

    def compile_all_transcripts(self, dest_dir):
        for file in os.listdir(self.path):
            name, ext = os.path.splitext(file)
            if ext != '.csv' or name[4:] != 'TRANSCRIPT':
                continue
            self._compile_participant_text(file, dest_dir + '/' + name + '.txt')

    def _build_expected_dict(self):
        expected_dict = {}
        for file in [
            'dev_split_Depression_AVEC2017.csv',
            'test_split_Depression_AVEC2017.csv',
            'train_split_Depression_AVEC2017.csv'
        ]:
            with open(self.path+'/'+file) as csv_file:
                reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
                for row in reader:
                    row = row[0].split(',')
                    if row[0] == 'Participant_ID':
                        continue
                    expected_dict[row[0]] = row[1] == '1'
        return expected_dict

    def _get_features_for_transcript(self, file, liwc_features):
        """
        Compiles the features array for a particular transcript.
        :param file: Name of the transcript.
        :return: Features array.
        """
        name, _ = os.path.splitext(file)
        features = liwc_features[name]
        avg_res = self._get_average_response_length(file)
        features.append(avg_res)
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
                liwc_dict[name] = row[3:]
        return liwc_dict

    def _get_average_response_length(self, file):
        """
        Determines the average continuous response length for the participant in a transcript.
        :param file: Name of the transcript.
        :return: The average response length.
        """
        parts = 0
        total = 0
        with open(self.path+'/'+file) as csv_file:
            reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
            length = 0
            for row in reader:
                if len(row) < 1:
                    continue
                speaker, content = self._extract_transcript_row(row)
                if speaker == 'Participant':
                    number_of_words = self._len_without_spaces(content.split(' '))
                    length += number_of_words
                else:
                    if length > 0:
                        total += length
                        parts += 1
                        length = 0
        if length > 0:
            total += length
            parts += 1
        return total/parts

    def _compile_participant_text(self, file, dest):
        """
        Compiles all the participant responses from a transcript.
        :param file: Name of the transcript.
        :param dest: The destination file for the compiled text.
        """
        f = open(dest, "w+")
        with open(self.path+'/'+file) as csv_file:
            reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
            for row in reader:
                if len(row) < 1:
                    continue
                speaker, content = self._extract_transcript_row(row)
                if speaker == 'Participant':
                    f.write(content+'\n')
        f.close()

    @staticmethod
    def _len_without_spaces(words):
        counter = 0
        for word in words:
            if word != '':
                counter += 1
        return counter

    @staticmethod
    def _extract_transcript_row(row):
        first = row[0].split('\t')
        speaker = first[2]
        content = first[len(first)-1]+' '+' '.join(row[1:])
        return speaker, content
