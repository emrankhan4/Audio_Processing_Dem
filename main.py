# %matplotlib inline
import librosa
import librosa.display
import matplotlib.pyplot as plt
audio_path = 'release_in_the_wild/2.wav'
music_array , sample_rate = librosa.load(audio_path, sr=44100)

#waveform
plt.figure(figsize=(15, 4), facecolor=(.5, .5, .5))
librosa.display.waveshow(music_array, sr=sample_rate, color='pink')
plt.savefig('waveform.png')


#spectrogram
spectrogram = librosa.stft(music_array)
print(spectrogram.shape)
print(type(spectrogram))

spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))
print(spectrogram_db.shape)
print(type(spectrogram_db))

fig,ax = plt.subplots(figsize=(15, 4), facecolor=(.5, .5, .5))
img = librosa.display.specshow(spectrogram_db, sr=sample_rate, x_axis='time', y_axis='hz', ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set_title('Spectrogram')
ax.set_xlabel('Time')
ax.set_ylabel('Frequency')  
plt.savefig('spectrogram.png')

#mel spectrogram
mfccs = librosa.feature.mfcc(y=music_array, sr=sample_rate, n_mfcc=13)
print(type(mfccs))