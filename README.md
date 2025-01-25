Code and data repository for the preprint titled ["Object representations reflect hierarchical scene structure and depend on high-level visual, semantic, and action information." (Aylin Kallmayer, Leila Zacharias, Luisa Jetter, Melissa Le-Hoa Võ)](https://doi.org/10.31234/osf.io/hs835)

## About

Epoched data can be downloaded [here](https://www.dropbox.com/scl/fi/001hjyw0r9lf6ezuynv5t/Archive.zip?rlkey=xhmini6w1xg30rjqvbh7u1xe5&st=lqpvfcra&dl=0)

Once downloaded and unzipped you have access to eeg epoch and ica files for each participant:

    .
    📦SceneGrammarEEG
     ┣ 📂sub-01
     ┃ ┣ 📂eeg
     ┃ ┃ ┣ 📜sub-01-ica.fif
     ┃ ┃ ┣ 📜sub-01_repaired-False-epo.fif
     ┃ ┃ ┗ 📜sub-01_repaired-True-epo.fif
     ┃ ┣ 📂cross_decoding
     ┃ ┣ 📂decoding
     ┣ 📂sub-02
     ┃ ┣ 📂eeg
     ┃ ┃ ┣ 📜sub-02-ica.fif
     ┃ ┃ ┣ 📜sub-02_repaired-False-epo.fif
     ┃ ┃ ┗ 📜sub-02_repaired-True-epo.fif
     ┗...

Each folder contains:
- eeg files:
    - ICA files (.fif): Independent Component Analysis files for artifact correction.
    - Epoch files (.fif): EEG epochs, both repaired and non-repaired, for further analysis.
- decoding results 
- cross-decoding results

The EEG preprocessing and decoding analyses were carried out using [mne Python](https://mne.tools/stable/index.html).
CORNet-S weights were obtained from [here](https://github.com/dicarlolab/CORnet).

## Usage
Download the repository.
Set up the python environment using conda:

    > conda create -n scenegrammar_eeg python=3.8 
    > conda activate scenegrammar_eeg

After activating your environment, install the package from the folder:

    pip install .

From the `/scripts` directory you can run decoding and rsa analysis scripts and "viz" jupyter notebooks for visualizing results. The `'run_preprocessing.py'` script is included but can only be run locally having access to raw EEG data. It is included in the repository for reference.

