# %matplotlib inline
import librosa
import librosa.display
import matplotlib.pyplot as plt
from userlib import *
audio_path = 'release_in_the_wild/2.wav'
music_array , sample_rate = librosa.load(audio_path, sr=44100)

def main():

    plot_waveform(music_array, sample_rate)
    plot_spectrogram(music_array, sample_rate)

    mfccs = feature_mfcc(music_array, sample_rate)
    chroma = feature_chroma(music_array, sample_rate)
    sc = feature_speactral_contrast(music_array, sample_rate)
    zcr = feature_zero_crossing_rate(music_array, sample_rate)
    sro = feature_spectral_roll_off(music_array, sample_rate)

    print(mfccs.shape, chroma.shape, sc.shape, zcr.shape, sro.shape)

if __name__ == '__main__':
    main()