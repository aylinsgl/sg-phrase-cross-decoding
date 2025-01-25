import sgcd.decoding as dc
import os
import config
from tqdm import tqdm

if __name__ == "__main__":
    print(config.conditions)
    for sub, _ in tqdm(enumerate(config.subject_id_clean), total=len(config.subject_id_clean), desc="decoding subjects"):
        epochs = dc.load_eeg_data(sub, config.eeg_data_dir)
        conditions_anchors = ["Toilette","Waschbecken","Herd","Schneidebrett"] #toilet, sink, stove, cuttingboard
        conditions_local = ["Toilettenpapier","Zahnbürste","Pfanne","Brotmesser"]
        conditions_avl = conditions_anchors+conditions_local
        dc.run_decoding_subject(sub, epochs, conditions_avl, 'SVM', objects='all', cond='avl', pca=True)