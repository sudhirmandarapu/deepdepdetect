import preprocessor

'''
path = './transcripts/depressed_transcripts/319_TRANSCRIPT.csv'

x = preprocessor.get_all_transcript_features(
    './transcripts',
)

print(x)
'''

preprocessor.compile_all_transcripts('./transcripts', './liwc_output')
