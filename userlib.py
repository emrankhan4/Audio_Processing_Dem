import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def mkdir(path):
    try:
        os.mkdir(path)
    except:
        pass

def __save_to_csv(data,feature_name):
    file_name = 'audio_features.csv'
    d = np.mean(data, axis=1)
    if os.path.isfile(file_name):
        df = pd.read_csv(file_name)
    else:
        # Create a new empty DataFrame
        df = pd.DataFrame()
    df1 = pd.DataFrame()
    for i in range(data.shape[1]):
        df1[f'{feature_name}']=data[:,i]
    if feature_name in df.columns:
        df = df.drop(feature_name, axis=1)
    df = pd.concat([df, df1], axis=1)
    df.to_csv(file_name,index=False)



def plot_waveform(music_array, sample_rate=32000):
    plt.figure(figsize=(15, 4), facecolor=(.5, .5, .5))
    librosa.display.waveshow(music_array, sr=sample_rate, color='pink')
    try:
        os.mkdir('audio_plots')
    except:
        pass
    plt.savefig('audio_plots/waveform.png')
    plt.clf()
    
def plot_spectrogram(music_array, sample_rate=32000):
    spectrogram = librosa.stft(music_array)
    spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))
    fig,ax = plt.subplots(figsize=(15, 4), facecolor=(.5, .5, .5))
    img = librosa.display.specshow(spectrogram_db, sr=sample_rate, x_axis='time', y_axis='hz', ax=ax)   
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Spectrogram')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    mkdir('audio_plots')
    plt.savefig('audio_plots/spectrogram.png')
    plt.clf()
def feature_mfcc(music_array, sample_rate=32000):
    mfccs = librosa.feature.mfcc(y=music_array, sr=sample_rate, n_mfcc=13)
    plt.figure(figsize=(15, 4), facecolor=(.5, .5, .5))
    librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    # plt.show()
    mkdir('audio_plots')
    plt.savefig('audio_plots/mfcc.png')
    plt.clf()
    __save_to_csv(mfccs,'mfcc.csv')
    return mfccs

def feature_chroma(music_array, sample_rate=32000):
    chromagram = librosa.feature.chroma_stft(y=music_array, sr=sample_rate)
    plt.figure(figsize=(15, 4), facecolor=(.5, .5, .5))
    librosa.display.specshow(chromagram, sr=sample_rate, x_axis='time')
    plt.colorbar()
    plt.title('Chromagram')
    plt.tight_layout()
    # plt.show()
    mkdir('audio_plots')
    plt.savefig('audio_plots/chromagram.png')
    plt.clf()
    __save_to_csv(chromagram, 'chromagram.csv')
    return chromagram
def feature_speactral_contrast(music_array, sample_rate=32000):
    contrast = librosa.feature.spectral_contrast(y=music_array, sr=sample_rate)
    plt.figure(figsize=(15, 4), facecolor=(.5, .5, .5))
    librosa.display.specshow(contrast, sr=sample_rate, x_axis='time')
    plt.colorbar()
    plt.title('Spectral Contrast')
    plt.tight_layout()
    # plt.show()
    mkdir('audio_plots')
    plt.savefig('audio_plots/spectral_contrast.png')
    plt.clf()
    __save_to_csv(contrast,'spectral_contrast.csv')
    return contrast

def feature_zero_crossing_rate(music_array, sample_rate=32000):
   
    y, sr = music_array, sample_rate
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)
    fig, ax = plt.subplots(figsize=(10, 6))
    librosa.display.waveshow(y, sr=sr, alpha=0.4)
    plt.plot(zcr.T, color='r', label='Zero-Crossing Rate')
    plt.legend()
    ax.set_title('Zero-Crossing Rate')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    mkdir('audio_plots')
    plt.savefig('audio_plots/zero_crossing_rate.png')
    # plt.show()
    __save_to_csv(zcr, 'zero_crossing_rate.csv')
    plt.clf()
    return zcr

def feature_spectral_roll_off(music_array, sample_rate=32000):
    y, sr = music_array, sample_rate
    S, phase = librosa.magphase(librosa.stft(y))
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             y_axis='log', x_axis='time')
    plt.title('Spectrogram (db)')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    mkdir('audio_plots')
    plt.savefig('audio_plots/spectral_roll_off.png')
    # plt.show()
    # sro_df = pd.DataFrame(columns=['Frequency', 'Spectral Roll-Off'])
    __save_to_csv(S, 'spectral_roll_off.csv')
    plt.clf()
    return S