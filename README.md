# Audio_Processing_Dem
# Audio Feature Extraction

This repository contains Python scripts for extracting various audio features from audio files using the `librosa` library. The extracted features can be useful for tasks such as audio classification, music information retrieval, and speech recognition.

## Features

The following audio features can be extracted using the provided scripts:

- Mel-Frequency Cepstral Coefficients (MFCCs)
- Chroma Features
- Spectral Contrast
- Zero-Crossing Rate (ZCR)
- Spectrograms
- Waveforms

## Requirements

To run the scripts in this repository, you need to have the following dependencies installed:

- Python (version 3.6 or later)
- librosa
- pandas
- matplotlib

You can install the required dependencies using pip:



pip install librosa pandas matplotlib


## Usage

1. Clone this repository to your local machine:



git clone https://github.com/emrankhan4/Audio_Processing_Dem.git


2. Navigate to the repository directory:



cd Audio_Processing_Dem


3. Open the desired Python script main.py in a text editor or an integrated development environment (IDE).

4. Modify the script to specify the path to your audio file(s) and any other necessary configurations.

5. Run the script to extract the desired audio features.

6. The extracted features will be saved as a CSV file in the same directory as the script.

## Examples

Here are some examples of how to use the scripts in this repository:

- Extract MFCCs from an audio file:

```python
import librosa
import pandas as pd

# Load an audio file
audio_file = 'path/to/your/audio/file.wav'
y, sr = librosa.load(audio_file)

# Compute MFCCs,Plot and save as csv 
mfccs = librosa.feature.mfcc(y=y, sr=sr)

# Compute Chroma,Plot and save as csv 


# Load an audio file
audio_file = 'path/to/your/audio/file.wav'
y, sr = librosa.load(audio_file)

# Compute chroma features
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
```