import os
import csv


class Transcript:
    def __init__(self, t_id):
        self.id = t_id
        self.rows = []
        self.features = None

    def add_row(self, row):
        assert(type(row) == TranscriptRow)
        self.rows.append(row)


class TranscriptRow:
    def __init__(self, speaker, value):
        self.speaker = speaker
        self.value = value


def get_transcripts_in_path(path):
    transcripts = []
    for file in os.listdir(path):
        name, ext = os.path.splitext(file)
        if ext != '.csv' or name[4:] != 'TRANSCRIPT':
            continue
        transcripts.append(get_transcript(path+'/'+file, name[:3]))
    return transcripts


def get_transcript(file, t_id):
    transcript = Transcript(t_id)
    with open(file) as csv_file:
        reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
        for row in reader:
            if len(row) < 2:
                continue
            speaker, content = _extract_transcript_row(row)
            transcript.add_row(TranscriptRow(speaker, content))
    return transcript


def _extract_transcript_row(row):
    first = row[0].split('\t')
    speaker = first[2]
    content = first[len(first)-1]+' '+' '.join(row[1:])
    return speaker, content
