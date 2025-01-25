import sgcd.decoding as dc
import os
import sgcd.config as config
from tqdm import tqdm

# Run all decoding and cross-decoding analyses for all subjects in all configurations and store to file
if __name__ == "__main__":
    print(config.conditions)
    for sub, _ in tqdm(enumerate(config.subject_id_clean), total=len(config.subject_id_clean), desc="decoding subjects"):
        epochs = dc.load_eeg_data(sub, config.eeg_data_dir)
        
        dc.run_decoding_subject(sub, epochs, config.conditions, 'SVM', objects='all', cond='object', pca=True)
        conditions_anchors = ["Toilette","Waschbecken","Herd","Schneidebrett"]
        dc.run_decoding_subject(sub, epochs, conditions_anchors, 'SVM', objects='anchor', cond='object', pca=True)
        conditions_local = ["Toilettenpapier","Zahnbürste","Pfanne","Brotmesser"] 
        
        dc.run_decoding_subject(sub, epochs, conditions_local, 'SVM', objects='local', cond='object', pca=True)
        dc.run_decoding_subject(sub, epochs, config.conditions, 'SVM', objects='all', cond='phrase', pca=True)
        dc.run_decoding_subject(sub, epochs, config.conditions, 'SVM', objects='all', cond='scene', pca=True)
        dc.run_decoding_subject(sub, epochs, conditions_anchors, 'SVM', objects='anchor', cond='scene', pca=True)
        dc.run_decoding_subject(sub, epochs, conditions_local, 'SVM', objects='local', cond='scene', pca=True)
        conditions_anchors = ["Toilette","Waschbecken","Herd","Schneidebrett"] #toilet, sink, stove, cuttingboard
        conditions_local = ["Toilettenpapier","Zahnbürste","Pfanne","Brotmesser"]
        conditions_avl = conditions_anchors+conditions_local
        dc.run_decoding_subject(sub, epochs, conditions_avl, 'SVM', objects='all', cond='avl', pca=True)
        
        conditions_anchors = ["Toilette","Waschbecken","Herd","Schneidebrett"] #toilet, sink, stove, cuttingboard
        conditions_local = ["Toilettenpapier","Zahnbürste","Pfanne","Brotmesser"] #toiletpaper, toothbrush, pan, breadknife
        dc.run_cross_decoding_subject(sub, epochs, conditions_anchors, conditions_local, 'SVM', "CrossAnchorLocal", pca=True, cond="phrase")
        dc.run_cross_decoding_subject(sub, epochs, conditions_local, conditions_anchors, 'SVM', "CrossLocalAnchor", pca=True, cond="phrase")
        dc.run_cross_decoding_subject(sub, epochs, conditions_anchors, conditions_local, 'SVM', "CrossAnchorLocal_Scene", pca=True, cond="scene")
        dc.run_cross_decoding_subject(sub, epochs, conditions_local, conditions_anchors, 'SVM', "CrossLocalAnchor_Scene", pca=True, cond="scene")

        conditions_anchors = ["Toilette","Waschbecken","Herd","Schneidebrett"] #toilet, sink, stove, cuttingboard
        conditions_local = ["Zahnbürste","Toilettenpapier","Brotmesser","Pfanne"] #toothbrush, toiletpaper, knife, pan
        dc.run_cross_decoding_subject(sub, epochs, conditions_anchors, conditions_local, 'SVM', "CrossAnchorLocalDiffPhraseSameScene", pca=True, cond="phrase")
        dc.run_cross_decoding_subject(sub, epochs, conditions_local, conditions_anchors, 'SVM', "CrossLocalAnchorDiffPhraseSameScene", pca=True, cond="phrase")

        conditions_anchors = ["Toilette","Waschbecken","Herd","Schneidebrett"] #toilet, sink, stove, cuttingboard
        conditions_local = ["Brotmesser","Pfanne","Zahnbürste","Toilettenpapier"] #toothbrush, toiletpaper, knife, pan
        dc.run_cross_decoding_subject(sub, epochs, conditions_anchors, conditions_local, 'SVM', "CrossAnchorLocalDiffPhraseDiffScene", pca=True, cond="phrase")
        dc.run_cross_decoding_subject(sub, epochs, conditions_local, conditions_anchors, 'SVM', "CrossLocalAnchorDiffPhraseDiffScene", pca=True, cond="phrase")

        conditions_anchors = ["Toilette","Waschbecken","Herd","Schneidebrett"] #toilet, sink, stove, cuttingboard
        conditions_local = ["Pfanne","Brotmesser","Toilettenpapier","Zahnbürste"] #toothbrush, toiletpaper, knife, pan
        dc.run_cross_decoding_subject(sub, epochs, conditions_anchors, conditions_local, 'SVM', "CrossAnchorLocalDiffPhraseDiffScene2", pca=True, cond="phrase")
        dc.run_cross_decoding_subject(sub, epochs, conditions_local, conditions_anchors, 'SVM', "CrossLocalAnchorDiffPhraseDiffScene2", pca=True, cond="phrase")