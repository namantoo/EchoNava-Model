import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pygame

# Initialize pygame for audio playback
pygame.mixer.init()

# Define the path to the audio file
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'archive')
sample_file = os.path.join(DATA_DIR, 'Actor_01', '03-01-01-01-01-01-01.wav')

# Play the audio file
pygame.mixer.music.load(sample_file)
pygame.mixer.music.play()
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)

# Load the audio file using librosa
y, sr = librosa.load(sample_file)

# Extract features
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
zcr = librosa.feature.zero_crossing_rate(y)
rms_energy = librosa.feature.rms(y=y)

# Plot the features
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
librosa.display.specshow(contrast, x_axis='time')
plt.colorbar(label='Spectral Contrast')
plt.title('Spectral Contrast')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.semilogy(rms_energy.T, label='RMS Energy')
plt.ylabel('RMS Energy')
plt.xticks([])
plt.xlim([0, rms_energy.shape[-1]])
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(librosa.times_like(zcr), zcr[0], label='Zero Crossing Rate')
plt.title('Zero Crossing Rate')
plt.tight_layout()
plt.show()
