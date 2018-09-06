from preprocessor import Preprocessor

preprocessor = Preprocessor('./transcripts')

x, y = preprocessor.get_all_transcript_features()

print(x)
print(y)

'''
preprocessor.compile_all_transcripts('./output/compiled_transcripts')
'''