import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage import metrics
import sklearn
from pandas import read_excel
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mne.stats import permutation_cluster_1samp_test
import scipy
from scipy import linalg
import mne
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.ndimage
from scipy.ndimage import gaussian_filter
import pingouin 

def create_ssim_rdm(images_list, conditions_anchor, conditions_local):
    """
    Create structural similarity RDM for the given images list.
    Parameters:
    - images_list (list): List of image names.
    - conditions_anchor (list): List of anchor object indices.
    - conditions_local (list): List of local object indices.
    Returns:
    - ssim_rdm (np.array): Structural similarity RDM all images.
    - sub_rdm_al (np.array): Structural similarity subRDM for anchor-local pairing.
    - sub_rdm_la (np.array): Structural similarity subRDM for local-anchor pairing.
    """
    ssim_rdm = np.zeros((80,80))

    for n, i in enumerate(images_list):
        for m, j in enumerate(images_list):
            img1 = cv2.imread(f'IMAGES_CORnet/{i}')
            img2 = cv2.imread(f'IMAGES_CORnet/{j}')
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation = cv2.INTER_AREA)
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            ssim_score = metrics.structural_similarity(img1_gray, img2_gray, full=True)
            ssim_rdm[n,m] = 1-ssim_score[0]
    sub_rdm_al = ssim_rdm[np.array(conditions_local)[:,None], conditions_anchor]
    sub_rdm_la = ssim_rdm[np.array(conditions_anchor)[:,None], conditions_local]
    return ssim_rdm, sub_rdm_al, sub_rdm_la

def create_gpt_rdm(feature_path, conditions_anchor, conditions_local):
    """
    Create GPT-2 RDM for the object category labels.
    Parameters:
    - feature_path (str): Path to the GPT-2 features.
    - conditions_anchor (list): List of anchor object indices.
    - conditions_local (list): List of local object indices.
    Returns:
    - rdm (np.array): GPT-2 RDM for all objects.
    - sub_rdm_al (np.array): GPT-2 subRDM for anchor-local pairings.
    - sub_rdm_la (np.array): GPT-2 subRDM for local-anchor pairings.
    """
    gpt2 = np.load(feature_path)

    rdm = sklearn.metrics.pairwise.euclidean_distances(gpt2)
    # normalize
    rdm = (rdm-np.min(rdm))/(np.max(rdm)-np.min(rdm))

    rdm = broadcast_tile(rdm, 10, 10)

    sub_rdm_al = rdm[np.array(conditions_local)[:,None], conditions_anchor]
    sub_rdm_la = rdm[np.array(conditions_anchor)[:,None], conditions_local]
    return rdm, sub_rdm_al, sub_rdm_la

def create_actions_rdm(feature_path, conditions_anchor, conditions_local):
    """
    Create RDM for the action features.
    Parameters:
    - feature_path (str): Path to the action features.
    - conditions_anchor (list): List of anchor object indices.
    - conditions_local (list): List of local object indices.
    Returns:
    - actions_rdm (np.array): RDM for all actions.
    - actions_rdm_al (np.array): SubRDM for anchor-local pairings.
    - actions_rdm_la (np.array): SubRDM for local-anchor pairings.
    """
    actions = pd.read_csv(feature_path)
    actions = np.array(actions.drop(actions.columns[0],axis=1))
    actions_rdm = 1-np.corrcoef(actions)
    actions_rdm = broadcast_tile(actions_rdm, 10, 10)

    actions_rdm_al = actions_rdm[np.array(conditions_local)[:,None], conditions_anchor]
    actions_rdm_la = actions_rdm[np.array(conditions_anchor)[:,None], conditions_local]

    return actions_rdm, actions_rdm_al, actions_rdm_la

def load_cor_rdms(paths, conditions_anchor, conditions_local):
    """
    Load CORnet RDMs.
    Parameters:
    - paths (list): List of paths to the CORnet RDMs.
    - conditions_anchor (list): List of anchor objects.
    - conditions_local (list): List of local objects.
    Returns:
    - v1_rdm (np.array): RDM for CORnet v1.
    - v2_rdm (np.array): RDM for CORnet v2.
    - v4_rdm (np.array): RDM for CORnet v4.
    - it_rdm (np.array): RDM for CORnet IT.
    - layers_al (dict): Dictionary of subRDMs for anchor-local pairings.
    - layers_la (dict): Dictionary of subRDMs for local-anchor pairings.
    """
    v1 = np.load(paths[0])
    v2 = np.load(paths[1])
    v4 = np.load(paths[2])
    it = np.load(paths[3])

    v1_rdm = 1 - np.corrcoef(v1)
    v2_rdm = 1 - np.corrcoef(v2)
    v4_rdm = 1 - np.corrcoef(v4)
    it_rdm = 1 - np.corrcoef(it)

    layers_al = dict()
    layers_la = dict()
    for n_layer, layer in enumerate([v1_rdm,v2_rdm,v4_rdm,it_rdm]):
        vision_rdm = layer
        sub_rdm_al = vision_rdm[np.array(conditions_local)[:,None], conditions_anchor]
        sub_rdm_la = vision_rdm[np.array(conditions_anchor)[:,None], conditions_local]
        layers_al[n_layer] = sub_rdm_al.flatten()
        layers_la[n_layer] = sub_rdm_la.flatten()

    return v1_rdm, v2_rdm, v4_rdm, it_rdm, layers_al, layers_la

def load_cluster_times(pca=True,estimator="SVM",prefix="",cluster_type="base"):
    if cluster_type=="base":
        cluster = np.load(f"{prefix}results/good_clusters-al-diagtimes-PCA_{pca}-{estimator}-1.npy")
    elif cluster_type=="diff_within":
        cluster = np.load(f"{prefix}results/good_clusters-al_diff_within-diagtimes-PCA_{pca}-{estimator}-{2}.npy")
    sig_times = cluster[0]
    return sig_times

def create_phrase_rdm(feature_path):
    """
    Create RDM for phrase co-occurance.
    Parameters:
    - feature_path (str): Path to the phrase co-occurance matrix.
    Returns:
    - co_occ_mat_al (np.array): Co-occurance matrix for anchor-local pairings.
    - co_occ_mat_la (np.array): Co-occurance matrix for local-anchor pairings.
    """
    co_occ_mat_al = read_excel(feature_path, header=None)
    co_occ_mat_al = 1-co_occ_mat_al
    co_occ_mat_la = co_occ_mat_al.T

    return np.array(co_occ_mat_al), np.array(co_occ_mat_la)

def compute_vif(preds, names):
    """
    Compute and print Variance Inflation Factor for the given predictors.
    Parameters:
    - preds (list): List of predictors.
    - names (list): List of predictor names.
    """
    df = pd.DataFrame({
        names[0]: preds[0],
        names[1]: preds[1],
        names[2]: preds[2],
        names[3]: preds[3],
        names[4]: preds[4],
        names[5]: preds[5],
        names[6]: preds[6],
        names[7]: preds[7]
    })


    # the independent variables set
    x = df[names]

    # VIF dataframe 
    vif_data = pd.DataFrame()
    vif_data["feature"] = x.columns

    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(x.values, i)
                            for i in range(len(x.columns))]

    print(vif_data)

def whiten_preds(preds):
    """
    Whiten the predictors.
    Parameters:
    - preds (list): List of predictors.
    Returns:
    - x (np.array): Stacked predictors.
    - w (np.array): Whitened predictors.
    """
    # one feature per row
    x = np.stack(preds)
    w = whiten(x)

    for i in [x,w]:
        fig, ax = plt.subplots(figsize = (3,3))
        ax.imshow(np.cov(i))
        plt.show(block=False)
    return x, w

def plot_rdm(rdm):
    """
    Plot the given RDM.
    Parameters:
    - rdm (np.array): RDM to be plotted.
    """
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position="right", size="2%", pad=0.1)
    im=ax.matshow(rdm,vmin=0,vmax=1)
    plt.colorbar(im,cax=cax)
    plt.show(block=False)

def prepare_rsa():
    """
    Prepare data paths for RDM creation.
    Returns:
    - images_list_ssim (list): List of image names.
    - feature_path_gpt (str): Path to the GPT-2 features.
    - feature_path_actions (str): Path to the action embeddings.
    - feature_paths_cornet (tuple): Paths to the CORnet features.
    - feature_path_phrase (str): Path to the phrase co-occurance matrix.
    """
    # prepare images for SSIM
    images_list_ssim = []
    for i in os.listdir("IMAGES_CORnet"):
        if i[-3:] == "png" or i[-3:] == "jpg":
            images_list_ssim.append(i)
    images_list_ssim.sort()
    assert len(images_list_ssim) == 80, "Number of images is not 80."

    # prepare GPT2 embeddings
    language = "german"
    h = 12 # chose layer
    feature_path_gpt = f"GPT2/embeddings_{language}_layer_{h}.npy"

    # prepare action embeddings
    feature_path_actions = "action_embeddings_EEG.csv"

    # prepare CORnet features
    path_v1 = "CORnet/FEATURES/CORnet-S_V1_output_feats.npy"
    path_v2 = "CORnet/FEATURES/CORnet-S_V2_output_feats.npy"
    path_v4 = "CORnet/FEATURES/CORnet-S_V4_output_feats.npy"
    path_it = "CORnet/FEATURES/CORnet-S_IT_output_feats.npy"
    feature_paths_cornet = (path_v1,path_v2,path_v4,path_it)

    # prepare phrase co-occurrence
    feature_path_phrase = "co-occ_phrase.xlsx"

    return images_list_ssim, feature_path_gpt, feature_path_actions, feature_paths_cornet, feature_path_phrase

def create_rdms(conditions_anchor=range(40), conditions_local=range(40,80)):
    """
    Create RDMs for the predictors for GLM.
    Parameters:
    - conditions_anchor (list): List of anchor object indices.
    - conditions_local (list): List of local object indices.
    Returns:
    - preds_al (list): List of predictors for anchor-local pairings.
    - preds_la (list): List of predictors for local-anchor pairings.
    """
    images_list_ssim, feature_path_gpt, feature_path_actions, feature_paths_cornet, feature_path_phrase = prepare_rsa()
    
    #### SSIM RDMs
    # create and plot SSIM rdms
    ssim_rdm, sub_rdm_al, sub_rdm_la = create_ssim_rdm(images_list_ssim, 
                                                           conditions_anchor, 
                                                           conditions_local)
    assert ssim_rdm.shape == (80,80), "SSIM RDM has wrong shape, should be (80,80)."
    plot_rdm(ssim_rdm)
    # flatten rdms
    ssim_al_vec = sub_rdm_al.flatten()
    ssim_la_vec = sub_rdm_la.flatten()

    #### ChatGPT2 RDMs
    _, sub_rdm_al, sub_rdm_la = create_gpt_rdm(feature_path_gpt, 
                                                   conditions_anchor, 
                                                   conditions_local)
    plot_rdm(_)
    # flatten rdms
    gpt_al_vec = sub_rdm_al.flatten()
    gpt_la_vec= sub_rdm_la.flatten()

    #### Action RDMs
    _, actions_rdm_al, actions_rdm_la = create_actions_rdm(feature_path_actions, 
                                                               conditions_anchor, 
                                                               conditions_local)
    plot_rdm(_)
    # flatten rdms
    actions_al_vec = actions_rdm_al.flatten()
    actions_la_vec = actions_rdm_la.flatten()

    #### CORnet RDMs
    v1_rdm, v2_rdm, v4_rdm, it_rdm, layers_al, layers_la = load_cor_rdms(feature_paths_cornet, 
                                                                             conditions_anchor, 
                                                                             conditions_local)
    for l in [v1_rdm, v2_rdm, v4_rdm, it_rdm]:
        plot_rdm(l)

    #### Phrase RDMs
    co_occ_mat_al, co_occ_mat_la = create_phrase_rdm(feature_path_phrase)
    co_occ_al_vec = np.array(co_occ_mat_al).flatten()
    co_occ_la_vec = np.array(co_occ_mat_la).flatten()
    plot_rdm(co_occ_mat_al)

    names = ["ssim", "V1", "V2", "V4", "IT", "gpt", "phrase", "action"]
    preds_al = [ssim_al_vec, layers_al[0], layers_al[1], layers_al[2], layers_al[3], gpt_al_vec, co_occ_al_vec, actions_al_vec]
    preds_la = [ssim_la_vec, layers_la[0], layers_la[1], layers_la[2], layers_la[3], gpt_la_vec, co_occ_la_vec, actions_la_vec]

    return preds_al, preds_la, names

def run_semipartial_rsa(neural_rdms,preds,names,n_s=25, average=True, cluster_type="base"):
    times = load_cluster_times(cluster_type=cluster_type)
    num_timepoints = neural_rdms[0].shape[2]  # Number of timepoints
    num_preds = len(preds)
    
    if average:
        results_semi = np.zeros((n_s, len(preds[4:])))
        results_corr = np.zeros((n_s, num_preds))
        results_semi_all = np.zeros((n_s, len(preds[4:])))
    else:
        results_semi = np.zeros((n_s, len(preds[4:]), num_timepoints))
        results_corr = np.zeros((n_s, num_preds, num_timepoints))
        results_semi_all = np.zeros((n_s, len(preds[4:]), num_timepoints))

    transposed_vectors = list(map(list, zip(*preds)))
    df = pd.DataFrame(transposed_vectors, columns=names)

    for n in range(n_s):
        print(f'Processing subject {n+1}')
        neural_rdm = neural_rdms[n]

        if average:
            # Compute average across the selected time points
            neural_rdm_avg_time = np.mean(neural_rdm[:, :, times], axis=2)
            plt.imshow(neural_rdm_avg_time)
            plt.show(block=False)
            neural_vec = neural_rdm_avg_time.flatten()
            df["neural"] = neural_vec
            
            for i, name in enumerate(names):
                res_corr = pingouin.corr(df[name], df["neural"], method="pearson")   
                results_corr[n, i] = res_corr["r"]
                
                if name in ["IT", "gpt", "phrase", "action"]:
                    res_semi = pingouin.partial_corr(data=df, x=name, y="neural", covar=names[:4], method="pearson")
                    res_semi_all = pingouin.partial_corr(data=df, x=name, y="neural", covar=[n for n in names if n != name], method="pearson")
                    results_semi[n, i-4] = res_semi["r"]
                    results_semi_all[n, i-4] = res_semi_all["r"]
        else:
            # Loop over all timepoints
            for t in range(num_timepoints):
                neural_vec = neural_rdm[:, :, t].flatten()
                df["neural"] = neural_vec
                
                for i, name in enumerate(names):
                    res_corr = pingouin.corr(df[name], df["neural"], method="pearson")
                    results_corr[n, i, t] = res_corr["r"]
                    
                    if name in ["IT", "gpt", "phrase", "action"]:
                        res_semi = pingouin.partial_corr(data=df, x=name, y="neural", covar=names[:4], method="pearson")
                        res_semi_all = pingouin.partial_corr(data=df, x=name, y="neural", covar=[n for n in names if n != name], method="pearson")
                        results_semi[n, i-4, t] = res_semi["r"]
                        results_semi_all[n, i-4, t] = res_semi_all["r"]
    
    return results_semi, results_corr, results_semi_all


def correlation_x_w(x_list,w_list,plotname):
    """
    Plot the correlation between the original and whitened predictors.
    Parameters:
    - x_list (list): List of original predictors.
    - w_list (list): List of whitened predictors.
    - plotname (str): Name of the plot.
    """
    corrs = []
    for x in range(len(x_list)):
        orig = x_list[x]
        whit = w_list[x]
        corrs.append(np.corrcoef(orig, whit)[0][1])

    fig, ax = plt.subplots(figsize = (3,6))
    bplot = ax.boxplot(corrs, notch=False, patch_artist=True, showfliers=False)
    # fill with colors
    for median in bplot['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    for patch, color in zip(bplot['boxes'], ["lightblue"]):
        patch.set_facecolor(color)

    ax.set_ylabel("Correlation between original and whitened predictors", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.plot(np.repeat(1,8),corrs,'r.',alpha=.5,color="black",)

    plt.yticks(ticks=np.linspace(0,1,6))
    plt.subplots_adjust(left=0.21)
    plt.subplots_adjust(bottom=0.21)
    plt.savefig(f"results/plots/correlation_whitening_{plotname}.png", dpi=300)
    plt.show(block=False)
    print(corrs)

def load_neural_rdms(subject_id_clean, cond="phrase",pca=True, estimator="SVM", prefix=''):
    """
    Load neural RDMs.
    Parameters:
    - subject_id_clean (list): List of subject IDs.
    - cond (str): Condition for the neural RDMs.
    - pca (bool): Whether to use PCA.
    - estimator (str): Estimator for the neural RDMs.
    Returns:
    - neural_rdms_al (dict): Dictionary of neural RDMs for anchor-local pairings.
    - neural_rdms_la (dict): Dictionary of neural RDMs for local-anchor pairings.
    """
    neural_rdms_al = dict()
    neural_rdms_la = dict()
    for sub, subname in enumerate(subject_id_clean):
        direction_flag = "CrossAnchorLocal"
        decoding_dir = f'{prefix}data/sub-{str(sub+1).zfill(2)}/cross_decoding/{direction_flag}/{cond}/{estimator}/pca_{pca}'
        neural_rdm_1 = np.load(f'{decoding_dir}/{str(sub+1).zfill(2)}-{direction_flag}-confusion-exemplar-pseudo-diagtimes-PCA_{pca}-{estimator}.npy')
        
        direction_flag="CrossLocalAnchor"
        decoding_dir = f'{prefix}data/sub-{str(sub+1).zfill(2)}/cross_decoding/{direction_flag}/{cond}/{estimator}/pca_{pca}'
        neural_rdm_2 = np.load(f'{decoding_dir}/{str(sub+1).zfill(2)}-{direction_flag}-confusion-exemplar-pseudo-diagtimes-PCA_{pca}-{estimator}.npy')
        
        neural_rdms_al[sub] = neural_rdm_1
        neural_rdms_la[sub] = neural_rdm_2
    return neural_rdms_al, neural_rdms_la
        
def get_times(eeg_data_dir):
    """
    Get the time points for the EEG data.
    Parameters:
    - eeg_data_dir (str): Directory containing the EEG data.
    Returns:
    - times (int): Number of time points.
    """
    epochs = mne.read_epochs(eeg_data_dir+f'sub-{str(1).zfill(2)}/eeg/sub-{str(1).zfill(2)}_repaired-True-epo.fif', verbose="warning")
    times = epochs.times
    return times

def run_glm_rsa(neural_rdms,epochs_times,w,n_s=25):
    """
    Run RSA analysis. Predict neural RDMs using the whitened predictors in a logistic regression model.
    Parameters:
    - neural_rdms (np.array): Neural RDMs.
    - epochs_times (int): Number of time points.
    - w (np.array): Whitened predictors.
    - n_s (int): Number of subjects to include in the analysis.
    Returns:
    - results (np.array): Results of the RSA analysis. Beta values for each predictor.
    """
    results = np.zeros((n_s, len(w), len(epochs_times)))

    for n in range(n_s):
        neural_rdm = neural_rdms[n]

        for time in range(len(epochs_times)):
            neural_rdm_t = neural_rdm[:,:,time]

            neural_vec_t = neural_rdm_t.flatten()

            logreg = LogisticRegression(solver='liblinear', random_state=0)
            res = logreg.fit(np.vstack((w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7])).T, neural_vec_t)

            results[n,:,time] = res.coef_[0]
    return results

def create_cols(n_curves, custom_colors):
    """
    Create colors for the given number of curves.
    Parameters:
    - n_curves (int): Number of curves.
    - custom_colors (list): List of custom colors.
    Returns:
    - color_vals (list): List of colors.
    """
    values = range(n_curves)
    cividis = plt.get_cmap('plasma') 
    c_norm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cividis)

    color_vals = [scalar_map.to_rgba(values[x]) for x in range(n_curves-1)]
    color_vals.insert(0,custom_colors[0])
    color_vals.append(custom_colors[1])
    color_vals.append(custom_colors[2])
    color_vals.append(custom_colors[3])

    return color_vals

def create_cols_all(n_curves,alpha=1):
    """
    Create colors for the given number of curves.
    Parameters:
    - n_curves (int): Number of curves.
    - alpha (float): Alpha value for the colors.
    Returns:
    - color_vals (list): List of colors.
    """

    values = range(n_curves)
    cividis = plt.get_cmap('plasma') 
    c_norm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cividis)

    color_vals = [scalar_map.to_rgba(values[x],alpha=alpha) for x in range(n_curves)]

    return color_vals

def compute_p_values(diff, tail=1, return_clusters=False):
    """
    Compute p-values for the given difference.
    Parameters:
    - diff (np.array): Difference scores (compared to 0).
    - tail (int): Tail of the test.
    - return_clusters (bool): Whether to return the cluster indices.
    Returns:
    - pvals_plot (np.array): P-values for the difference scores.
    """
    _, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(diff, n_permutations=10000, out_type="indices",tail=tail)

    good_clusters_idx = np.where(cluster_pv < 0.05)[0]
    good_clusters = [clusters[idx] for idx in good_clusters_idx]

    pvals_plot = np.full((diff.shape[1]), None)
    for c in good_clusters:
        pvals_plot[c] = -2
    if return_clusters:
        return pvals_plot, good_clusters
    else:
        return pvals_plot

def smooth_for_plot(res):
    """
    Smooth the results for plotting.
    Parameters:
    - res (np.array): Results to be smoothed.
    Returns:
    - ga_scores (np.array): Smoothed mean values.
    - sigma (np.array): Smoothed sigma values.
    """    
    ga_scores = np.mean(res, axis=0)
    sigma = scipy.stats.sem(res, axis=0)

    # smoothing
    ga_scores = scipy.ndimage.gaussian_filter(ga_scores,2)
    sigma = scipy.ndimage.gaussian_filter(sigma,2)
    return ga_scores, sigma

def add_plot_configs(ax, pred_type="al_orig"):
    """
    Add plot configurations.
    Parameters:
    - pred_type (str): Type of predictors.
    """
    if pred_type=="al_orig" or pred_type=="la_orig":
        con = "original"
    else:
        con = "whitened"
    ax.axhline(0, color='k', linestyle='--', label='chance', linewidth=.8)
    ax.axvline(.0, color='k', linestyle='-', linewidth=.8)
    ax.axvline(.5, color='k', linestyle='-', linewidth=.8)
    ax.set_xlabel("time (s)", fontsize=20)
    ax.set_ylabel("beta", fontsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(bottom=0.15)
    plt.yticks(ticks=np.linspace(-0.5,0.5,3)) if con=="whitened" else plt.yticks(ticks=np.linspace(-3,3,7))
    plt.xticks(ticks=np.linspace(-0.1,1,12))

def add_boxplot_configs(ax,pred_type="al_orig",corr_type="glm"):
    """
    Add boxplot configurations.
    Parameters:
    - pred_type (str): Type of predictors.
    """
    if pred_type=="al_orig" or pred_type=="la_orig":
        con = "original"
    else:
        con = "whitened"
    ax.axhline(0, color='k', linestyle='--', label='chance', linewidth=.8)
    ax.set_ylabel("beta",fontsize=20) if corr_type=="glm" else ax.set_ylabel("r (pearson correlation)",fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(bottom=0.15, left=0.3)
    plt.yticks(ticks=np.linspace(-0.8,0.6,8)) if con=="whitened" else plt.yticks(ticks=np.linspace(-1,1,5))
    plt.ylim(top=0.6,bottom=-0.8) if con=="whitened" else plt.ylim(top=1,bottom=-1)
    ax.tick_params(axis='both', which='major', labelsize=12)

def broadcast_tile(a, h, w):
    """
    Broadcast and tile the given array.
    Parameters:
    - a (np.array): Array to be tiled.
    - h (int): Height of the tile.
    - w (int): Width of the tile.
    Returns:
    - np.array: Tiled array.
    """
    x, y = a.shape
    m, n = x * h, y * w
    return np.broadcast_to(
        a.reshape(x, 1, y, 1), (x, h, y, w)
    ).reshape(m, n)

def whiten(x):
    """
    Whiten the given data using ZCA (https://arxiv.org/abs/1512.00809).
    Parameters:
    - x (np.array): Data to be whitened.
    Returns:
    - wzca (np.array): Whitened data.
    """
    # Center data by subtracting mean for each feature
    xc = x.T - np.mean(x.T, axis=0)
    xc = xc.T

    # Calculate covariance matrix
    xcov = np.cov(xc, rowvar=True, bias=True)

    # Calculate Eigenvalues and Eigenvectors
    w, v = linalg.eig(xcov)

    # Calculate inverse square root of Eigenvalues
    # Optional: Add '.1e5' to avoid division errors if needed
    # Create a diagonal matrix
    diagw = np.diag(1/(w**0.5)) # or np.diag(1/((w+.1e-5)**0.5))
    diagw = diagw.real.round(4) #convert to real and round off

    # Whitening transform using ZCA (Zero Component Analysis)
    wzca = np.dot(np.dot(np.dot(v, diagw), v.T), xc)
    return wzca

