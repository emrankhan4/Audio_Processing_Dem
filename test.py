import librosa
import IPython.display as ipd
path = 'release_in_the_wild/2.wav'
music_array, sample_rate = librosa.load(path)
print(type(sample_rate))

ipd.Audio(path)