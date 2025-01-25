import os
import mne
import numpy as np
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA

def load_eeg_data(sub, eeg_data_dir):
    """
    Load epoched EEG data for a specific subject.

    Parameters:
    - sub (int): Subject identifier.
    - eeg_data_dir (str): Directory where epoched EEG data is stored.
    return: mne.Epochs object
    """
    epochs = mne.read_epochs(eeg_data_dir+f'sub-{str(sub+1).zfill(2)}/eeg/sub-{str(sub+1).zfill(2)}_repaired-True-epo.fif', verbose="warning")
    return epochs

def run_decoding_subject(sub, epochs, conditions, estimator, objects='all', cond='object', pca=True):
    """
    Run decoding analysis on EEG data for a specific subject.

    Parameters:
    - sub (int): Subject identifier.
    - conditions (list): List of conditions to include in the analysis.
    - estimator (str): Estimator to use for decoding. Options: "LDA" or "SVM".
    - objects (str, optional): Object condition to consider. Default is 'all'.
    - cond (str, optional): Condition type. Options: "object", "phrase", "scene", "avl". Default is 'object'.
    - pca (bool, optional): Whether to apply PCA. Default is True.
    """
    decoding_dir = f'data/sub-{str(sub+1).zfill(2)}/decoding/{estimator}/{cond}/{objects}/pca_{pca}/'
    if not os.path.exists(decoding_dir):
        os.makedirs(decoding_dir, exist_ok=True)

    if estimator == "LDA":
        est = LinearDiscriminantAnalysis()
    elif estimator == "SVM":
        est = LinearSVC(max_iter=1000, dual='auto')
        
    # prepare data
    epochs = mne.concatenate_epochs([epochs[x] for x in conditions])
    epochs_pseudo = np.array([epochs[x].get_data(copy=True).mean(axis=0) for x in list(epochs.event_id.keys())])

    X = epochs_pseudo
    if cond=="object":
        y = np.repeat(range(len(conditions)), 10)
        n_splits = 10
    elif cond=="phrase":
        y = np.repeat([0,1,2,3], 20)
        n_splits = 20
    elif cond=="scene":
        y = np.repeat([0,1], 40) if objects=="all" else np.repeat([0,1], 20)
        n_splits = 40 if objects=="all" else 20
    elif cond=="avl":
        y = np.repeat([0,1], 40)
        n_splits = 40

    # define pipeline
    cv = StratifiedKFold(n_splits=n_splits)

    if pca:
        clf = make_pipeline(StandardScaler(), PCA(n_components=.99, svd_solver='full'), est)
    else:
        clf = make_pipeline(StandardScaler(), est)

    time_decod = SlidingEstimator(clf, verbose=True)

    # accuracy scores per validation split
    scores = cross_val_multiscore(time_decod, X, y, cv=cv, n_jobs=6, verbose="warning")

    # mean across validation splits
    mean_scores = np.mean(scores, axis=0)
    
    np.save(f'{decoding_dir}{str(sub+1).zfill(2)}-{cond}_{objects}-Decoding-Acc-pseudo-loeo-PCA_{pca}-{estimator}.npy', mean_scores)

def run_cross_decoding_subject(sub, epochs, conditions_train, conditions_test, estimator, direction_flag, pca=True, cond="phrase"):
    """
    Run cross-decoding analysis on EEG data.

    Parameters:
    - sub (int): Subject identifier.
    - conditions_train (list): List of conditions to use for training.
    - conditions_test (list): List of conditions to use for testing.
    - estimator (str): Estimator to use for classification. Options: "LDA", "SVM".
    - direction_flag (str): Direction flag.
    - pca (bool): Whether to apply PCA (Principal Component Analysis) before classification. Default is True.
    - cond (str): Condition type. Options: "phrase", "scene". Default is "phrase".
    """
    decoding_dir = f'data/sub-{str(sub+1).zfill(2)}/cross_decoding/{direction_flag}/{cond}/{estimator}/pca_{pca}/'
    if not os.path.exists(decoding_dir):
        os.makedirs(decoding_dir, exist_ok=True)

    if estimator == "LDA":
        est = LinearDiscriminantAnalysis()
    elif estimator == "SVM":
        est = LinearSVC(max_iter=1000, dual='auto')
    # prepare data
    epochs_train = mne.concatenate_epochs([epochs[x] for x in conditions_train])
    epochs_test = mne.concatenate_epochs([epochs[x] for x in conditions_test])

    epochs_train_pseudo = np.array([epochs_train[x].get_data(copy=True).mean(axis=0) for x in epochs_train.event_id.keys()])
    epochs_test_pseudo = np.array([epochs_test[x].get_data(copy=True).mean(axis=0) for x in epochs_test.event_id.keys()])
    
    X_train = epochs_train_pseudo
    y_train = np.repeat(range(len(conditions_train)), 10) if cond=="phrase" else np.repeat([0,1], 20)

    X_test = epochs_test_pseudo
    y_test = np.repeat(range(len(conditions_test)), 10) if cond=="phrase" else np.repeat([0,1], 20)

    # define pipeline   
    if pca:
        clf = make_pipeline(StandardScaler(), PCA(n_components=.99, svd_solver='full'), est)
    else:
        clf = make_pipeline(StandardScaler(), est)   
    
    time_decod = SlidingEstimator(clf, n_jobs=6, verbose="warning")
    
    time_decod.fit(X_train, y_train)
    scores = time_decod.score(X_test, y_test)
    y_pred = time_decod.predict(X_test)

    # confusion matrix converted to distance matrix
    # for each exemplar start with 1 (maximum distance), convert to 0 if predicted by classifier
    Cmatrix = np.ones((40,40,len(epochs.times)))
    for time1 in range(y_pred.shape[1]):
        y_p = y_pred[:,time1]
        for i in range(40):
            if cond=="phrase":
                Cmatrix[i,y_p[i]*10:(y_p[i]*10)+10,time1] = 0
            else:
                Cmatrix[i,y_p[i]*20:(y_p[i]*20)+20,time1] = 0

    np.save(f'{decoding_dir}{str(sub+1).zfill(2)}-{direction_flag}-Decoding-Acc-pseudo-diagtimes-PCA_{pca}-{estimator}.npy', scores)
    np.save(f'{decoding_dir}{str(sub+1).zfill(2)}-{direction_flag}-confusion-exemplar-pseudo-diagtimes-PCA_{pca}-{estimator}.npy', Cmatrix)