import csv
import os
import matplotlib.pyplot as plt
from preprocessor import Preprocessor
import numpy as np


def build_expected_dict():
    expected_dict = {}
    for file in [
        'dev_split_Depression_AVEC2017.csv',
        'train_split_Depression_AVEC2017.csv'
    ]:
        with open('./transcripts/'+file) as csv_file:
            reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
            for row in reader:
                row = row[0].split(',')
                if row[0] == 'Participant_ID':
                    continue
                expected_dict[row[0]] = row[1] == '1'
    return expected_dict


def get_feature_averages():
    d = [None]*80
    nd = [None]*80
    expected_dict = build_expected_dict()
    titles = None
    with open('./output/split_transcripts/liwc_features.csv') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
        for row in reader:
            split_row = row[0].split(',')
            if split_row[0] == 'Filename':
                titles = split_row[3:83]
                continue
            name, _ = os.path.splitext(row[0])
            t_id = name[:3]
            if t_id in expected_dict:
                for i in range(3, 83):
                    if expected_dict[name[:3]]:
                        if d[i-3] is None:
                            d[i-3] = [float(split_row[i])]
                        else:
                            d[i-3].append(float(split_row[i]))
                    else:
                        if nd[i-3] is None:
                            nd[i-3] = [float(split_row[i])]
                        else:
                            nd[i-3].append(float(split_row[i]))
            else:
                continue
    d_avg = [round(sum(x)/len(x), 4) for x in d]
    nd_avg = [round(sum(x)/len(x), 4) for x in nd]

    return np.array(d), np.array(nd), titles


def get_responses():
    expected_dict = build_expected_dict()
    p = Preprocessor('./transcripts')
    d = []
    nd = []
    for file in os.listdir('./transcripts'):
        name, ext = os.path.splitext(file)
        if ext != '.csv' or name[4:] != 'TRANSCRIPT':
            continue
        avg = p._get_average_response_length(file)
        t_id = name[:3]
        if t_id in expected_dict:
            if expected_dict[t_id]:
                d.append(avg)
            else:
                nd.append(avg)
    # for i in nd:
    #    print(i)
    return d, nd


def get_sentiments():
    expected_dict = build_expected_dict()
    p = Preprocessor('./transcripts')
    d = []
    nd = []
    for file in os.listdir('./output/compiled_transcripts'):
        name, ext = os.path.splitext(file)
        if ext != '.txt' or name[4:] != 'TRANSCRIPT':
            continue
        sentiment = p._get_sentiment(name)
        t_id = name[:3]
        if t_id in expected_dict:
            if expected_dict[t_id]:
                d.append(sentiment)
            else:
                nd.append(sentiment)
    for i in d:
        print(i)


d, nd, titles = get_feature_averages()
index = 33
while index < 50:
    fig, ax = plt.subplots()
    ax.boxplot([d[index], nd[index]])
    ax.set_title("Feature scores for "+titles[index]+" ("+str(index)+")")
    ax.xticks = ([1, 2], ['depressed', 'non depressed'])
    index += 1
plt.show()
