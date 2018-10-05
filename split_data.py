from preprocessor import Preprocessor
import os

path = './transcripts'
new_labels_file = ''
p = Preprocessor(path)


def split_even(input_dir, file, output):
    name, ext = os.path.splitext(file)
    f = open(input_dir+'/'+file, 'r')
    o_1 = open(output+'/'+name+'_1'+ext, "w+")
    o_2 = open(output+'/'+name+'_2'+ext, "w+")
    word_list = f.read().split()
    mid_length = len(word_list)//2
    o_1.write(" ".join(word_list[:mid_length]))
    o_2.write(" ".join(word_list[mid_length:]))


def split_transcripts(output, temp_dir, expected_dir):
    for file in os.listdir(temp_dir):
        name, ext = os.path.splitext(file)
        if ext != '.txt' or name[4:] != 'TRANSCRIPT' or not name[:3] in expected_dict:
            continue
        split_even(temp_dir, file, output)

expected_dict = p._build_expected_dict()
split_transcripts('./output/split_transcripts', './split_transcripts_temp', expected_dict)
