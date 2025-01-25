# Config file, read in by all scripts to set global variables.
import numpy as np

subject_id = ["2pilot","subj_2","subj_3","subj_4","subj_5","subj_6","subj_7","subj_8","subj_9",
            "subj_10","subj_11","subj_12","subj_13","subj_14","subj_15","subj_16","subj_18","subj_17",
            "subj_19","subj_21","subj_22","subj_23","subj_24","subj_25","subj_26","subj_27","subj_28"]
subject_id_bad = ["subj_4","subj_21"]
subject_id_clean = [x for x in subject_id if x not in subject_id_bad]

# Raw EEG data is stored on external_SSD (not publicly available), so preprocessing can only be run locally.
external_SSD = "/Volumes/Extreme SSD/01_2023_EEG-Data/LeilaEEG/" 
unprocessed_dir = external_SSD + "raw/" 
raw_dir = external_SSD + "raw_fif/"

# Epoched data is stored in eeg_data_dir, which is publicly available.
eeg_data_dir = "data/"

tmin, tmax = -.1, 1      # epoch time window
eog = 'Fp2'              # EOG electrode

# We follow the resampling and decimating approach from the MNE-Python tutorial: https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html#resampling
current_sfreq = 1000     # current sampling rate 
desired_sfreq = 250      # desired sampling rate    
decim = np.round(current_sfreq / desired_sfreq).astype(int)
obtained_sfreq = current_sfreq / decim
lowpass_freq = obtained_sfreq / 3.0
l_freq, h_freq = None, lowpass_freq
baseline = (None, 0)   # baseline correction 

conditions = ["Toilette","Toilettenpapier","Waschbecken","Zahnbürste","Herd","Pfanne","Schneidebrett","Brotmesser"]