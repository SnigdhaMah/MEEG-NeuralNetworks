# MEEG-NeuralNetworks
How do different architecture of neural networks affect eeg decoding ability?
How do different representations of EEG (spectrogram, waveform) affect decoding?

Data: https://drive.google.com/drive/folders/1Tabw5sjpFiwy88yP-C-LnunNFrrre9AR

- For each model:
  - For each number of subjects (the first 30, 20, 10):
    - Train on subjects, and test on those SAME SUBJECTS (different sessions) to understand baseline
    - See how the model fares on new subjects (and different sessions)

Transformer - Yuchen

Graph NN - Snigdha

CNN - Kellen


Trained each model on the first 18 trials for the training subjects(10/20/30).

Evaluate model on remaining 2 trials for training subjects and then for all trials of the other subjects.

For each model, calculate accuracy for valence, arousal, and exact valence/arousal match for the 2 two test cases (cross trial / cross subject)
