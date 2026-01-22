# Audio-Based Music Genre Classification

A system that classifies music genres by analyzing audio features extracted from music files. This repository contains Jupyter notebooks that walk through data preparation, feature extraction, modeling (classical ML and deep learning approaches), evaluation, and demonstration.

## Table of contents
- [Project overview](#project-overview)
- [Contents](#contents)
- [Key features](#key-features)
- [Dataset](#dataset)
- [Audio features used](#audio-features-used)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Notebooks & workflow](#notebooks--workflow)
- [How to reproduce results](#how-to-reproduce-results)
- [How to add your own audio files](#how-to-add-your-own-audio-files)
- [Results & evaluation](#results--evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project overview
This project demonstrates end-to-end genre classification from raw audio:
- extract audio features (e.g., MFCC, Chroma, Spectral Contrast),
- build feature sets,
- train and evaluate classifiers (traditional ML and optionally deep learning on spectrograms),
- analyze model performance and common failure modes.

It is organized as interactive Jupyter notebooks so you can reproduce experiments and iterate quickly.

## Contents
- One or more Jupyter notebooks (.ipynb) containing:
  - dataset exploration
  - feature extraction pipeline
  - model training and evaluation
  - visualization of results
- (Optional) helper scripts or saved models (if present in repo)
- README.md (this file)

## Key features
- Reproducible workflow in notebooks
- Typical audio features used for genre classification
- Examples for classical ML pipelines (e.g., Random Forest, SVM) and deep learning approaches (CNN on spectrograms), where provided
- Evaluation with confusion matrix and per-class metrics

## Dataset
This repo does not include large datasets. Common datasets used for music genre classification include:
- GTZAN (popular benchmark) — often used for prototyping
- Your own collection of labeled audio files

Make sure your dataset is organized and labeled consistently before running the notebooks.

## Audio features used
Typical features extracted in the notebooks:
- MFCCs (Mel-frequency cepstral coefficients)
- Chroma features
- Spectral contrast
- Tonnetz
- Zero-crossing rate
- Tempo (BPM) estimates

These are usually extracted using librosa and saved into a tabular format (CSV/Parquet) for model training.

## Requirements
Minimum recommended:
- Python 3.8+
- Jupyter Notebook or JupyterLab
- librosa
- numpy
- pandas
- scikit-learn
- matplotlib / seaborn
- (Optional) tensorflow or pytorch for deep learning notebooks

Example (pip):
pip install -r requirements.txt
If there is no requirements.txt in the repository, you can install the core libs manually:
pip install jupyterlab librosa numpy pandas scikit-learn matplotlib seaborn

Or with conda:
conda create -n genre-classify python=3.9
conda activate genre-classify
pip install jupyterlab librosa numpy pandas scikit-learn matplotlib seaborn

## Quick start
1. Clone the repo:
   git clone https://github.com/siddarammanur656/Audio-Based-Music-Genre-Classification.git
2. Create and activate your environment (see Requirements).
3. Start Jupyter:
   jupyter lab
4. Open the main notebook(s) and run cells in order. Notebooks contain explanatory text and code cells.

## Notebooks & workflow
Typical notebook workflow:
1. Explore dataset and labels.
2. Extract features from audio files and save as CSV.
3. Prepare train/val/test splits and scale features.
4. Train models (baseline classical models, optionally CNNs).
5. Evaluate with accuracy, F1, confusion matrix and per-class metrics.
6. Visualize results and inspect misclassifications.

## How to reproduce results
- Ensure the same dataset and preprocessing parameters (sample rate, window/hop size, number of MFCCs).
- Use provided random seeds in notebooks for deterministic splits where appropriate.
- If notebooks save intermediate feature files, reuse them to save time.

## How to add your own audio files
- Place audio files in a folder with subfolders per genre (recommended), e.g.:
  dataset/
    rock/
      track1.wav
      track2.wav
    jazz/
      ...
- Update the notebook paths to point to your dataset and run the feature extraction cells.

Example MFCC extraction snippet (from the notebook):
```python
import librosa
y, sr = librosa.load(path, sr=22050, mono=True)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_mean = mfcc.mean(axis=1)
```

## Results & evaluation
Notebooks include scripts to generate:
- accuracy and per-class precision/recall/F1
- confusion matrix visualizations
- feature importance (for tree-based models)
- sample predictions and audio examples of correct/incorrect classifications

## Contributing
Contributions are welcome. Good ways to contribute:
- Add reproducible notebooks demonstrating other models
- Add a requirements.txt or environment.yml
- Improve feature extraction pipelines and documentation
- Add unit tests or CI for preprocessing

Please open an issue or pull request to discuss changes.

## License
Check the repository for a LICENSE file. If there is none and you want a permissive license, consider adding an MIT license.

## Contact
Repository owner: siddarammanur656

If you'd like, I can:
- commit this README to the repository,
- generate a requirements.txt from the notebooks,
- or create a small CONTRIBUTING.md to guide contributors.
