import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pygame.mixer

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'archive')
sample_file = os.path.join(DATA_DIR, 'Actor_01', '03-01-01-01-01-01-01.wav')

print(f"Playing audio file: {sample_file}")

# Play audio using pygame
pygame.mixer.init()
pygame.mixer.music.load(sample_file)
pygame.mixer.music.play()
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)

# Load the audio file
y, sr = librosa.load(sample_file)

# Plot the waveform using matplotlib directly
plt.figure(figsize=(10, 4))
plt.plot(y)
plt.title('Waveform')
plt.tight_layout()
plt.show()

# Plot the spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.tight_layout()
plt.show()