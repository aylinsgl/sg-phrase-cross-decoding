import numpy as np
import sgcd.config as config
from sgcd import rsa

CLUSTER_TYPE = "base"
AVG = True

def glm_rsa():
    # create RDMs for the predictors
    preds_al, preds_la, names = rsa.create_rdms()
    
    # compute and print Variance Inflation Factors
    rsa.compute_vif(preds_al,
                    names = names)
    rsa.compute_vif(preds_la,
                    names = names)
    
    # do whitening transformation on the predictors, keep original predictors for later
    x_al, w_al = rsa.whiten_preds(preds_al)
    x_la, w_la = rsa.whiten_preds(preds_la)
    
    # plot correlation between original and whitened predictors
    rsa.correlation_x_w(x_al,w_al,"al")
    rsa.correlation_x_w(x_la,w_la,"la")
    
    # compute RSA
    neural_rdms_al, neural_rdms_la = rsa.load_neural_rdms(subject_id_clean=config.subject_id_clean)
    epochs_times = rsa.get_times(eeg_data_dir=config.eeg_data_dir)
    results_al_w = rsa.run_glm_rsa(neural_rdms_al,epochs_times,w_al,n_s=25)
    results_la_w = rsa.run_glm_rsa(neural_rdms_la,epochs_times,w_la,n_s=25)

    np.save("results/rsa/rsa_results_al_whitened.npy",results_al_w)
    np.save("results/rsa/rsa_results_la_whitened.npy",results_la_w)

    results_al = rsa.run_glm_rsa(neural_rdms_al,epochs_times,x_al,n_s=25)
    results_la = rsa.run_glm_rsa(neural_rdms_la,epochs_times,x_la,n_s=25)

    np.save("results/rsa/rsa_results_al_original.npy",results_al)
    np.save("results/rsa/rsa_results_la_original.npy",results_la)

def semi_partial_rsa(cluster_type, avg):
    preds_al, preds_la, names = rsa.create_rdms()
    del(preds_al[-2])
    del(preds_la[-2])
    del(names[-2])
    print(names)
    neural_rdms_al, neural_rdms_la = rsa.load_neural_rdms(subject_id_clean=config.subject_id_clean)
    results_semi_al, results_corr_al, results_semi_all_al = rsa.run_semipartial_rsa(neural_rdms_al,preds_al,names,len(config.subject_id_clean), average=avg, cluster_type=cluster_type)
    results_semi_la, results_corr_la, results_semi_all_la = rsa.run_semipartial_rsa(neural_rdms_la,preds_la,names,len(config.subject_id_clean), average=avg, cluster_type=cluster_type)
    
    np.save(f"results/rsa/semipartial_al_{cluster_type}_avg{avg}.npy",results_semi_al)
    np.save(f"results/rsa/semipartial_la_{cluster_type}_avg{avg}.npy",results_semi_la)
    np.save(f"results/rsa/correlation_al_{cluster_type}_avg{avg}.npy",results_corr_al)
    np.save(f"results/rsa/correlation_la_{cluster_type}_avg{avg}.npy",results_corr_la)
    np.save(f"results/rsa/semipartial_all_al_{cluster_type}_avg{avg}.npy",results_semi_all_al)
    np.save(f"results/rsa/semipartial_all_la_{cluster_type}_avg{avg}.npy",results_semi_all_la)
    np.save("results/rsa/preds_al_corr.npy",np.corrcoef(np.stack(preds_al)))
    np.save("results/rsa/preds_la_corr.npy",np.corrcoef(np.stack(preds_la)))

def run_rsa(rsa_type):
    if rsa_type=="glm":
        glm_rsa()
    elif rsa_type=="semi_partial":
        semi_partial_rsa(CLUSTER_TYPE, AVG)

run_rsa('semi_partial')