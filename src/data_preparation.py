import os
import pandas as pd

# Define a function to extract emotion labels
def extract_emotion_from_filename(filename):
    """Extract the emotion label from the filename."""
    emotion_code = int(filename.split("-")[2])
    emotion_labels = {
        1: "neutral",
        2: "calm",
        3: "happy",
        4: "sad",
        5: "angry",
        6: "fearful",
        7: "disgust",
        8: "surprised"
    }
    return emotion_labels[emotion_code]

# Define the data directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'archive')

data = []

# Loop through all files in the data directory
for subdir, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".wav"):
            filepath = os.path.join(subdir, file)
            emotion = extract_emotion_from_filename(file)
            data.append((filepath, emotion))

# Convert the data into a pandas dataframe
df = pd.DataFrame(data, columns=["filepath", "emotion"])

# Display the first few rows of the dataframe
print(df.head())

# Save the dataframe to a CSV file
df.to_csv('ravdess_data.csv', index=False)
