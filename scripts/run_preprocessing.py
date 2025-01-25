import sgcd.preprocessing as pp
import os
import sgcd.config as config

# NOTE: These steps can only be run locally, having access to raw EEG data. It is included in the repository for reference.

# preprocess all subjects. Steps include: loading raw data, creating ICA to remove eog related components automatically,
# creating epochs, and repairing epochs using autoreject.
if __name__ == "__main__":
    os.makedirs(config.raw_dir, exist_ok=True)
    os.makedirs(config.eeg_data_dir, exist_ok=True)

    for sub, _ in enumerate(config.subject_id):
        pp.load_raw_data(config.unprocessed_dir, config.raw_dir, sub, config.l_freq, config.h_freq, config.eog, overwrite=True)
        pp.create_ica(config.raw_dir, sub, config.subject_id, config.l_freq, config.h_freq, config.tmin, config.tmax, config.eog, config.eeg_data_dir)
        pp.make_epochs(config.raw_dir, config.eeg_data_dir, sub, config.subject_id, config.l_freq, config.h_freq, config.tmin, config.tmax, config.decim, repair=True)