import csv
import numpy as np
import os


def get_all_transcript_features(path):
    """
    Creates the input features array and the expected outputs arrays based on transcripts.
    :param path: Path to the transcripts.
    :return: Inputs (X).
    """
    x = []
    for file in os.listdir(path):
        name, ext = os.path.splitext(file)
        if ext != '.csv' or name[4:] != 'TRANSCRIPT':
            continue
        x.append(_get_features_for_transcript(path + '/' + file))
    return x


def compile_all_transcripts(src_dir, dest_dir):
    for file in os.listdir(src_dir):
        name, ext = os.path.splitext(file)
        if ext != '.csv' or name[4:] != 'TRANSCRIPT':
            continue
        _compile_participant_text(src_dir + '/' + file, dest_dir + '/' + name + '.txt')


def _get_features_for_transcript(path):
    """
    Compiles the features array for a particular transcript.
    :param path: Path to the transcript.
    :return: Features array.
    """
    features = []
    avg_res = _get_average_response_length(path)
    features.append(avg_res)
    return features


def _get_liwc_features(id):
    return


def _get_average_response_length(path):
    """
    Determines the average continuous response length for the participant in a transcript.
    :param path: Location of the transcript.
    :return: The average response length.
    """
    parts = 0
    total = 0
    with open(path) as csv_file:
        reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
        length = 0
        for row in reader:
            if len(row) < 1:
                continue
            speaker, content = _extract_transcript_row(row)
            if speaker == 'Participant':
                number_of_words = _len_without_spaces(content.split(' '))
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


def _compile_participant_text(path, dest):
    """
    Compiles all the participant responses from a transcript.
    :param path: Location of the transcript.
    :param dest: The destination file for the compiled text.
    """
    f = open(dest, "w+")
    with open(path) as csv_file:
        reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
        for row in reader:
            if len(row) < 1:
                continue
            speaker, content = _extract_transcript_row(row)
            if speaker == 'Participant':
                f.write(content+'\n')
    f.close()


def _len_without_spaces(words):
    counter = 0
    for word in words:
        if word != '':
            counter += 1
    return counter


def _extract_transcript_row(row):
    first = row[0].split('\t')
    speaker = first[2]
    content = first[len(first)-1]+' '+' '.join(row[1:])
    return speaker, content
