# Music Genre Classification

A machine learning project that classifies music into different genres using audio features and deep learning techniques.

## Project Overview

This project implements a music genre classifier that can categorize audio files into 10 different genres:
- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

## Dataset

The project uses the GTZAN dataset which includes:
- 1000 audio tracks
- 10 genres (100 tracks per genre)
- Each track is 30 seconds long
- 22050Hz Mono 16-bit audio files

Dataset structure:
```
Data/
    genres_original/
        blues/
        classical/
        country/
        disco/
        hiphop/
        jazz/
        metal/
        pop/
        reggae/
        rock/
```

## Features

- Audio feature extraction using Librosa
- Implementation of various classifiers:
  - K-Nearest Neighbors
  - Decision Trees
  - Support Vector Machines
  - AdaBoost
  - Logistic Regression
  - Neural Networks (using Keras)
- Feature analysis including:
  - Chroma spectrograms
  - Tempograms
  - Mel spectrograms

## Requirements

Install the required dependencies:

```sh
pip install -r requirements.txt
```

Main dependencies include:
- tensorflow
- sklearn
- librosa
- soundfile
- numpy
- pandas
- matplotlib

## Usage

1. Clone the repository
2. Install the requirements
3. Open and run the `music_classification.ipynb` notebook
4. The trained model is saved as `genre_classifier.keras`

## Model Architecture

The project implements multiple machine learning models, including a deep learning model built with Keras. The models are trained on audio features extracted from the music files, including:
- Chroma features
- Mel-frequency cepstral coefficients (MFCCs)
- Spectral features
- Rhythm features

## Results

The model achieves classification accuracy across different genres, with detailed performance metrics available in the notebook.

## Files Description

- `music_classification.ipynb`: Main Jupyter notebook containing all the code and analysis
- `genre_classifier.keras`: Saved trained model
- `model_history.pkl`: Training history of the model
- `requirements.txt`: List of Python dependencies
- `Data/`: Directory containing the dataset
