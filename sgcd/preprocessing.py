import mne
import os.path as op
from autoreject import AutoReject
import numpy as np
from pandas import read_excel

def create_event_id():
    """
    Creates an event ID mapping from the given Excel file.

    This function reads data from an Excel file named "triallist.xlsx" and processes it to create a mapping of event IDs.
    It limits the data to the first 80 rows, increments the "trigger_id" by 1, and selects specific columns for the mapping.
    The event ID mapping is created based on the concatenation of certain columns, and the result is printed and returned.

    Returns:
        dict: A dictionary where the keys are concatenated strings of specific columns and the values are the incremented "trigger_id".
    """
    conds = read_excel("triallist.xlsx")
    conds = conds[:80]
    conds["trigger_id"] += 1
    cols = ["scenename","phrase_id","object_name","fname","trigger_id"] 
    conds = conds[cols]
    conditions = []
    for c in conds.values:
        cond_tags = list(c[0:4])
        conditions.append('/'.join(map(str, cond_tags)))
    event_id = dict(zip(conditions, conds.trigger_id))
    print(f"created event_id...")
    return event_id
   
def load_raw_data(unprocessed_dir, raw_dir, subject_id, l_freq, h_freq, eog, overwrite=False):
    """
    Loads, processes, and saves raw EEG data for a given subject.

    This function reads raw EEG data from a BrainVision file, applies a standard 10-20 montage, filters the data,
    crops the data around the events, and saves the processed data to a FIF file.

    Args:
        unprocessed_dir (str): Directory containing the unprocessed EEG data files.
        raw_dir (str): Directory to save the processed raw EEG data files.
        subject_id (str): Identifier for the subject whose data is being processed.
        l_freq (float): Low cutoff frequency for the bandpass filter.
        h_freq (float): High cutoff frequency for the bandpass filter.
        eog (list): List of EOG channels.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.

    Returns:
        None
    """
    print(f'Loading raw data for {subject_id}...')

    in_file = op.join(unprocessed_dir,  f'{subject_id}.vhdr')
    out_file = op.join(raw_dir, f'{subject_id}-{l_freq}_{h_freq}-raw.fif')

    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    raw = mne.io.read_raw_brainvision(in_file, eog=[eog], preload=True)
    raw.drop_channels([ch for ch in raw.ch_names
                    if ch not in ten_twenty_montage.ch_names])
    raw.set_montage(ten_twenty_montage)

    # find events, filter, resample
    events = mne.events_from_annotations(raw)[0]
    raw.filter(l_freq, h_freq)
    
    # crop 10 seconds before the first event
    tmin = raw.times[events[0][0]] - 10
    if tmin < 0:  # if tmin is negative just set it at the beginning
        tmin = raw.times[0]
        # crop 10 seconds after the last event
    tmax = raw.times[events[-1][0]] + 10
    if tmax > raw.times[-1]:  # if tmax is bigger then the max time, just take that
        tmax = raw.times[-1]

    # crop it
    raw.crop(tmin=tmin, tmax=tmax)

    # save fif and save events file
    raw.save(out_file, overwrite=overwrite)
    print("raw data saved to disk")

def create_ica(raw_dir, sub_nr, subject_id_list, l_freq, h_freq, tmin, tmax, eog, eeg_data_dir):
    """
    Creates and saves Independent Component Analysis (ICA) for a subject's EEG data.

    This function reads a subject's raw EEG data, creates epochs, applies autorejection to remove
    artifacts, fits ICA, identifies components corresponding to EOG artifacts, excludes these components,
    and saves the ICA solution.

    Args:
        raw_dir (str): Directory where raw EEG data files are located.
        sub_nr (int): Subject number (index in the subject_id_list).
        subject_id_list (list): List of subject identifiers.
        l_freq (float): Low cutoff frequency for initial filtering.
        h_freq (Optional[float]): High cutoff frequency for initial filtering.
        tmin (float): Start time before event for epoching.
        tmax (float): End time after event for epoching.
        eog (str): Name of the EOG channel to use for artifact detection.
        eeg_data_dir (str): Directory where the ICA solution should be saved.

    Returns:
        None
    """
    # read raw file
    print(f'reading raw file for {subject_id_list[sub_nr]}...')
    raw = mne.io.read_raw_fif(raw_dir+f'{subject_id_list[sub_nr]}-{l_freq}_{h_freq}-raw.fif',preload=True)
    filt_raw = raw.copy().filter(l_freq=1., h_freq=None)
    events = mne.events_from_annotations(filt_raw)[0]

    print(f'creating epochs for {subject_id_list[sub_nr]}...')
    epochs = mne.Epochs(filt_raw, events, tmin=tmin, tmax=tmax, reject=None, preload=True, event_repeated='drop', baseline=None)
    
    # Autoreject (local) epochs to benefit ICA (fit on 20 epochs to save time)
    print(f'fitting autoreject for {subject_id_list[sub_nr]}...')
    auto_reject_pre_ica   = AutoReject(random_state = 100).fit(epochs[:20])
    epochs_ar, reject_log = auto_reject_pre_ica.transform(epochs, return_log = True)
                
    # Fit ICA on non-artifactual epochs 
    print(f'fitting ICA for {subject_id_list[sub_nr]}...')
    ica = mne.preprocessing.ICA(random_state = 100)
    ica.fit(epochs[~reject_log.bad_epochs])

    # exclude eog components
    ica.exclude = []
    
    # find which ICs match the EOG pattern
    eog_indices, eog_scores = ica.find_bads_eog(epochs, ch_name = eog)
    ica.exclude = eog_indices
    
    # save
    ica.save(eeg_data_dir+f'sub-{str(sub_nr+1).zfill(2)}/eeg/sub-{str(sub_nr+1).zfill(2)}-ica.fif', overwrite=True)
    print(f"ica created and stored to file for subject {subject_id_list[sub_nr]}...")

def make_epochs(raw_dir, eeg_data_dir, sub_nr, subject_id_list, l_freq, h_freq, tmin, tmax, decim, repair=False):
    """
    Creates and saves epochs for a subject's EEG data, optionally applying AutoReject.

    This function reads a subject's raw EEG data, reads the corresponding ICA solution,
    creates epochs, applies ICA to the epochs, decimates the epochs, optionally repairs
    the epochs using AutoReject, and saves the epochs to disk.

    Args:
        raw_dir (str): Directory where raw EEG data files are located.
        eeg_data_dir (str): Directory where processed EEG data files should be saved.
        sub_nr (int): Subject number (index in the subject_id_list).
        subject_id_list (list): List of subject identifiers.
        l_freq (float): Low cutoff frequency for filtering the raw data.
        h_freq (float): High cutoff frequency for filtering the raw data.
        tmin (float): Start time before event for epoching.
        tmax (float): End time after event for epoching.
        decim (int): Decimation factor for the epochs.
        repair (bool): Whether to apply AutoReject for epoch repair. Default is False.

    Returns:
        tuple: A tuple containing the number of epochs before and after repair (if applicable).
    """
    event_id = create_event_id()
    # read raw 
    raw = mne.io.read_raw_fif(raw_dir+f'{subject_id_list[sub_nr]}-{l_freq}_{h_freq}-raw.fif',preload=True)

    # read ica
    ica = mne.preprocessing.read_ica(eeg_data_dir+f'sub-{str(sub_nr+1).zfill(2)}/eeg/sub-{str(sub_nr+1).zfill(2)}-ica.fif')

    # epoch
    events = mne.events_from_annotations(raw)[0]
    epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, reject=None, preload=True, event_repeated='drop')
    epochs.event_id = event_id

    # apply ica
    epochs_ica = ica.apply(epochs.copy())
    epochs_ica.decimate(decim=decim)

    if repair:
        cv = 4
        ar = AutoReject(thresh_method='random_search', random_state=42, cv=cv, n_jobs=-1)
        ar.fit(epochs_ica)
        epochs_ar, reject_log = ar.transform(epochs_ica, return_log=True)
        epochs_ar.save(eeg_data_dir+f'sub-{str(sub_nr+1).zfill(2)}/eeg/sub-{str(sub_nr+1).zfill(2)}_repaired-True-epo.fif', overwrite=True)
    # save
    epochs_ica.save(eeg_data_dir+f'sub-{str(sub_nr+1).zfill(2)}/eeg/sub-{str(sub_nr+1).zfill(2)}_repaired-False-epo.fif', overwrite=True)
    
    if repair == True:
        return len(epochs_ica), len(epochs_ar)
    else: 
        return len(epochs_ica)

