# import os
# import time
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf

# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import categorical_crossentropy
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
# from sklearn.metrics import cohen_kappa_score
# import pandas as pd
# import models 
# from preprocess import get_data
# # from keras.utils.vis_utils import plot_model
# from eeg_fsgm import ica_fsgm_tta

# #%%
# def draw_learning_curves(history):
#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.title('Model accuracy')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'val'], loc='upper left')
#     plt.show()
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'val'], loc='upper left')
#     plt.show()
#     plt.close()

# def draw_confusion_matrix(cf_matrix, sub, results_path, classes_labels):
#     # Generate confusion matrix plot
#     display_labels = classes_labels
#     disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, 
#                                 display_labels=display_labels)
#     disp.plot()
#     disp.ax_.set_xticklabels(display_labels, rotation=12)
#     plt.title('Confusion Matrix of Subject: ' + sub )
#     plt.savefig(results_path + '/subject_' + sub + '.png')
#     plt.show()

# def draw_performance_barChart(num_sub, metric, label):
#     fig, ax = plt.subplots()
#     x = list(range(1, num_sub+1))
#     ax.bar(x, metric, 0.5, label=label)
#     ax.set_ylabel(label)
#     ax.set_xlabel("Subject")
#     ax.set_xticks(x)
#     ax.set_title('Model '+ label + ' per subject')
#     ax.set_ylim([0,1])
    
    



# def test(model, dataset_conf, results_path, n_imfs, num_arti_FRMs, run_id, allRuns=False):
#     log_write = open(results_path + f"/log_num_arti_FRMs_{num_arti_FRMs}_imfs_{n_imfs}.txt", "a")
#     best_models = open(results_path + "/best models.txt", "r")
    
#     dataset = dataset_conf.get('name')
#     n_classes = dataset_conf.get('n_classes')
#     n_sub = dataset_conf.get('n_sub')
#     data_path = dataset_conf.get('data_path')
#     isStandard = dataset_conf.get('isStandard')
#     LOSO = dataset_conf.get('LOSO')
#     classes_labels = dataset_conf.get('cl_labels')
    
#     acc_bestRun = np.zeros(n_sub)
#     kappa_bestRun = np.zeros(n_sub)
#     cf_matrix = np.zeros([n_sub, n_classes, n_classes])
#     csv_data = []

#     for sub in range(n_sub):
#         _, _, _, X_test, _, y_test_onehot = get_data(data_path, sub, dataset, LOSO, isStandard)
#         X_test, y_test_onehot = ica_fsgm_tta(X_test, y_test_onehot, dataset_conf, n_imfs, num_arti_FRMs, random_seed=run_id)
        
#         filepath = best_models.readline()
#         model.load_weights(results_path + filepath[:-1])
#         y_pred = model.predict(X_test).argmax(axis=-1)
#         labels = y_test_onehot.argmax(axis=-1)
#         acc_bestRun[sub] = accuracy_score(labels, y_pred)
#         kappa_bestRun[sub] = cohen_kappa_score(labels, y_pred)
#         cf_matrix[sub, :, :] = confusion_matrix(labels, y_pred, normalize='true')
#         # draw_confusion_matrix(cf_matrix[sub, :, :], f"{sub+1}_run_{run_id}", results_path, classes_labels)
        
#         subject_data = {
#             'Subject': sub + 1,
#             'Best_Run': filepath[filepath.find('run-')+4:filepath.find('/sub')],
#             'Accuracy': acc_bestRun[sub],
#             'Kappa': kappa_bestRun[sub],
#             'Num_Arti_FRMs': num_arti_FRMs,
#             'Run_ID': run_id
#         }
#         subject_data['Confusion_Matrix'] = cf_matrix[sub, :, :].flatten().tolist()
#         csv_data.append(subject_data)
        
#         info = f"Run {run_id}, Subject: {sub+1}   best_run: {subject_data['Best_Run']}   acc: {acc_bestRun[sub]:.4f}   kappa: {kappa_bestRun[sub]:.4f}"
#         print(info)
#         log_write.write('\n' + info)
    
#     info = f"\nRun {run_id}, Average of {n_sub} subjects - best runs (num_arti_FRMs={num_arti_FRMs}):\nAccuracy = {np.average(acc_bestRun):.4f}   Kappa = {np.average(kappa_bestRun):.4f}\n"
#     print(info)
#     log_write.write(info)
    
#     # draw_performance_barChart(n_sub, acc_bestRun, f'Accuracy_run_{run_id}')
#     # draw_performance_barChart(n_sub, kappa_bestRun, f'K-score_run_{run_id}')
#     # draw_confusion_matrix(cf_matrix.mean(0), f'All_run_{run_id}', results_path, classes_labels)
    
#     # Append to a single CSV
#     csv_df = pd.DataFrame(csv_data)
#     # csv_path = results_path + f"/test_results_num_arti_FRMs_{num_arti_FRMs}.csv"
#     csv_path = results_path + f"/test_results_num_arti_FRMs_{num_arti_FRMs}_imfs_{n_imfs}.csv"

#     # Append mode; write header only if file doesn't exist
#     csv_df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
#     print(f"Test results for run {run_id} appended to {csv_path}")
    
#     log_write.close()
#     best_models.close()
    
#     return csv_data
    
    
# #%%
# def getModel(model_name, dataset_conf):
    
#     n_classes = dataset_conf.get('n_classes')
#     n_channels = dataset_conf.get('n_channels')
#     in_samples = dataset_conf.get('in_samples')

#     # Select the model
#     if(model_name == 'ATCNet'):
#         # Train using the proposed ATCNet model: https://doi.org/10.1109/TII.2022.3197419
#         model = models.ATCNet_( 
#             # Dataset parameters
#             n_classes = n_classes, 
#             in_chans = n_channels, 
#             in_samples = in_samples, 
#             # Sliding window (SW) parameter
#             n_windows = 5, 
#             # Attention (AT) block parameter
#             attention = 'mha', # Options: None, 'mha','mhla', 'cbam', 'se'
#             # Convolutional (CV) block parameters
#             eegn_F1 = 16,
#             eegn_D = 2, 
#             eegn_kernelSize = 64,
#             eegn_poolSize = 7,
#             eegn_dropout = 0.3,
#             # Temporal convolutional (TC) block parameters
#             tcn_depth = 2, 
#             tcn_kernelSize = 4,
#             tcn_filters = 32,
#             tcn_dropout = 0.3, 
#             tcn_activation='elu'
#             )
#     elif(model_name == 'TCNet_Fusion'):
#         # Train using TCNet_Fusion: https://doi.org/10.1016/j.bspc.2021.102826
#         model = models.TCNet_Fusion(n_classes = n_classes, Chans=n_channels, Samples=in_samples)      
#     elif(model_name == 'EEGTCNet'):
#         # Train using EEGTCNet: https://arxiv.org/abs/2006.00622
#         model = models.EEGTCNet(n_classes = n_classes, Chans=n_channels, Samples=in_samples)          
#     elif(model_name == 'EEGNet'):
#         # Train using EEGNet: https://arxiv.org/abs/1611.08024
#         model = models.EEGNet_classifier(n_classes = n_classes, Chans=n_channels, Samples=in_samples) 
#     elif(model_name == 'EEGNeX'):
#         # Train using EEGNeX: https://arxiv.org/abs/2207.12369
#         model = models.EEGNeX_8_32(n_timesteps = in_samples , n_features = n_channels, n_outputs = n_classes)
#     elif(model_name == 'DeepConvNet'):
#         # Train using DeepConvNet: https://doi.org/10.1002/hbm.23730
#         model = models.DeepConvNet(nb_classes = n_classes , Chans = n_channels, Samples = in_samples)
#     elif(model_name == 'ShallowConvNet'):
#         # Train using ShallowConvNet: https://doi.org/10.1002/hbm.23730
#         model = models.ShallowConvNet(nb_classes = n_classes , Chans = n_channels, Samples = in_samples)
#     elif(model_name == 'MBEEG_SENet'):
#         # Train using MBEEG_SENet: https://www.mdpi.com/2075-4418/12/4/995
#         model = models.MBEEG_SENet(nb_classes = n_classes , Chans = n_channels, Samples = in_samples)

#     else:
#         raise Exception("'{}' model is not supported yet!".format(model_name))

#     return model

# def calculate_uncertainty(results_path, num_arti_FRMs, n_sub,n_imfs):
#     csv_path = results_path + f"/test_results_num_arti_FRMs_{num_arti_FRMs}_imfs_{n_imfs}.csv"
#     if not os.path.exists(csv_path):
#         print(f"Error: {csv_path} not found")
#         return
    
#     combined_df = pd.read_csv(csv_path)
    
#     uncertainty_data = []
#     for sub in range(1, n_sub + 1):
#         sub_data = combined_df[combined_df['Subject'] == sub]
#         if not sub_data.empty:
#             acc_variance = sub_data['Accuracy'].var() if len(sub_data) > 1 else 0
#             kappa_variance = sub_data['Kappa'].var() if len(sub_data) > 1 else 0
#             uncertainty_data.append({
#                 'Subject': sub,
#                 'Accuracy_Variance': acc_variance,
#                 'Kappa_Variance': kappa_variance,
#                 'Mean_Accuracy': sub_data['Accuracy'].mean(),
#                 'Mean_Kappa': sub_data['Kappa'].mean(),
#                 'Num_Arti_FRMs': num_arti_FRMs
#             })
    
#     uncertainty_df = pd.DataFrame(uncertainty_data)
#     uncertainty_csv_path = results_path + f"/uncertanity_num_arti_FRMs_{num_arti_FRMs}_imfs_{n_imfs}.csv"

#     uncertainty_df.to_csv(uncertainty_csv_path, index=False)
#     print(f"Uncertainty results for num_arti_FRMs={num_arti_FRMs} saved to {uncertainty_csv_path}")


# #%%
# def run(num_arti_FRMs,num_runs,n_imfs):
#     # Define dataset parameters
#     dataset = 'BCI2a' # Options: 'BCI2a','HGD', 'CS2R'
    
#     if dataset == 'BCI2a': 
#         in_samples = 1125
#         n_channels = 22
#         n_sub = 9
#         n_classes = 4
#         classes_labels = ['Left hand', 'Right hand','Foot','Tongue']
#         data_path = os.path.expanduser('~') + '/Tushar/EEG-ATCNet(TTA)/BCI/'
#     elif dataset == 'HGD': 
#         in_samples = 1125
#         n_channels = 44
#         n_sub = 14
#         n_classes = 4
#         classes_labels = ['Right Hand', 'Left Hand','Rest','Feet']     
#         data_path = os.path.expanduser('~') + '/mne_data/MNE-schirrmeister2017-data/robintibor/high-gamma-dataset/raw/master/data/'
#     elif dataset == 'CS2R': 
#         in_samples = 1125
#         # in_samples = 576
#         n_channels = 32
#         n_sub = 18
#         n_classes = 3
#         # classes_labels = ['Fingers', 'Wrist','Elbow','Rest']     
#         classes_labels = ['Fingers', 'Wrist','Elbow']     
#         # classes_labels = ['Fingers', 'Elbow']     
#         data_path = os.path.expanduser('~') + '/CS2R MI EEG dataset/all/EDF - Cleaned - phase one (remove extra runs)/two sessions/'
#     else:
#         raise Exception("'{}' dataset is not supported yet!".format(dataset))
        
#     # Create a folder to store the results of the experiment
#     results_path = os.getcwd() + "/results"
#     if not  os.path.exists(results_path):
#       os.makedirs(results_path)   # Create a new directory if it does not exist 
      
#     # Set dataset paramters 
#     dataset_conf = { 'name': dataset, 'n_classes': n_classes, 'cl_labels': classes_labels,
#                     'n_sub': n_sub, 'n_channels': n_channels, 'in_samples': in_samples,
#                     'data_path': data_path, 'isStandard': True, 'LOSO': False}
#     # Set training hyperparamters
#     train_conf = { 'batch_size': 32, 'epochs': 1000, 'patience': 300, 'lr': 0.001,
#                   'LearnCurves': True, 'n_train': 10, 'model':'ATCNet'}
           
#     # Train the model
#     # train(dataset_conf, train_conf, results_path)

#     # Clear existing CSV to avoid appending to old data
#     csv_path = os.path.join(results_path, f"test_results_num_arti_FRMs_{num_arti_FRMs}_imfs_{n_imfs}.csv")
#     print(f"CSV path: {csv_path}")
#     if os.path.exists(csv_path):
#         os.remove(csv_path)
#     # Evaluate the model based on the weights saved in the '/results' folder
#     model = getModel(train_conf.get('model'), dataset_conf)
#     for run_id in range(1, num_runs + 1):
#         try:
#             print(f"\nRunning test run {run_id} with num_arti_FRMs = {num_arti_FRMs}, n_imfs = {n_imfs}")
#             test(model, dataset_conf, results_path, n_imfs, num_arti_FRMs, run_id, allRuns=False)
#         except Exception as e:
#             print(f"Error in run {run_id}: {str(e)}")
#             raise
    
#     try:
#         calculate_uncertainty(results_path, num_arti_FRMs, n_sub,n_imfs)
#     except Exception as e:
#         print(f"Error in calculate_uncertainty: {str(e)}")
#         raise
# #%%
# # if __name__ == "__main__":
# #     run(num_arti_FRMs = 2,num_runs =2,n_imfs =2)
    
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Run EEG-ATCNet test with specified parameters")
#     parser.add_argument('--num_arti_FRMs', type=int, default=1, help='Number of artificial frames for TTA')
#     parser.add_argument('--num_runs', type=int, default=10, help='Number of test runs')
#     parser.add_argument('--imfs', type=int, default=1, help='Number of IMFs for EMD in TTA')
#     args = parser.parse_args()
#     run(num_arti_FRMs=args.num_arti_FRMs, num_runs=args.num_runs, n_imfs=args.imfs)

# # python main_TTA.py --num_arti_FRMs 1 --num_runs 2 --imfs 1
# # CUDA_VISIBLE_DEVICES=0 python main_TTA.py --num_arti_FRMs 1 --num_runs 2 --imfs 2
# # CUDA_VISIBLE_DEVICES=1 python main_TTA.py --num_arti_FRMs 5 --num_runs 2 --imfs 2
                
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import models 
from preprocess import get_data
from eeg_fsgm_new import ica_fsgm_tta
import logging
import sys
import traceback

# Custom class to redirect print statements to logger
class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())
        sys.__stdout__.write(buf)

    def flush(self):
        sys.__stdout__.flush()

def test(model, dataset_conf, results_path, output_path, epsilon, n_augs_per_class, run_id):
    log_write = open(os.path.join(output_path, f"log_epsilon_{epsilon}_aug_{sum(n_augs_per_class)}.txt"), "a")
    best_models = open(os.path.join(results_path, "best models.txt"), "r")
    
    dataset = dataset_conf.get('name')
    n_classes = dataset_conf.get('n_classes')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')
    classes_labels = dataset_conf.get('cl_labels')
    
    acc_bestRun = np.zeros(n_sub)
    acc_tta_bestRun = np.zeros(n_sub)
    kappa_bestRun = np.zeros(n_sub)
    cf_matrix = np.zeros([n_sub, n_classes, n_classes])
    csv_data = []

    for sub in range(n_sub):
        print(f"Evaluating subject {sub+1}...")
        try:
            _, _, _, X_test, _, y_test_onehot = get_data(data_path, sub, dataset, LOSO, isStandard)
            expected_shape = (X_test.shape[0], 1, dataset_conf['n_channels'], dataset_conf['in_samples'])
            if X_test.shape != expected_shape:
                raise ValueError(f"X_test shape {X_test.shape} does not match expected {expected_shape}")
            X_test_aug, y_test_onehot_aug, class_variance = ica_fsgm_tta(
                model, X_test, y_test_onehot, dataset_conf, 
                n_augs_per_class=n_augs_per_class, epsilon=epsilon
            )
            N_original = X_test.shape[0]
            N_aug = sum(n_augs_per_class)
            expected_aug_shape = (N_original + N_aug, 1, dataset_conf['n_channels'], dataset_conf['in_samples'])
            if X_test_aug.shape != expected_aug_shape:
                raise ValueError(f"X_test_aug shape {X_test_aug.shape} does not match expected {expected_aug_shape}")
            if y_test_onehot_aug.shape != (N_original + N_aug, n_classes):
                raise ValueError(f"y_test_onehot_aug shape {y_test_onehot_aug.shape} does not match expected {(N_original + N_aug, n_classes)}")
            
            labels = y_test_onehot.argmax(axis=-1)
            labels_aug = y_test_onehot_aug.argmax(axis=-1)
            
            filepath = best_models.readline().strip()
            if not filepath:
                raise ValueError(f"No model path found for subject {sub+1} in best models.txt")
            model_path = os.path.join(results_path, filepath.lstrip('/'))
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file {model_path} does not exist")
            model.load_weights(model_path)
            # Baseline
            logits = model.predict(X_test, verbose=0)
            probs_baseline = tf.nn.softmax(logits, axis=-1).numpy()
            predictions = np.argmax(probs_baseline, axis=-1)
            acc_orig = accuracy_score(labels, predictions)
            
            # TTA
            logits_aug = model.predict(X_test_aug, verbose=0)
            probs_aug = tf.nn.softmax(logits_aug, axis=-1).numpy()
            predictions_aug = np.argmax(probs_aug, axis=-1)
            
            # TTA with improved averaging of top-k similar samples
            valid_indices = []
            predictions_tta = np.zeros(N_original, dtype=np.int32)
            present_classes = set()
            top_k = 3  # Number of top similar samples to average
            for i in range(N_original):
                trial_indices = [j for j in range(N_original, N_original + N_aug) if labels_aug[j] == labels[i]]
                if len(trial_indices) >= 1:
                    valid_indices.append(i)
                    present_classes.add(labels[i])
                    # Compute cosine similarity between original and augmented sample probabilities
                    orig_probs = probs_baseline[i]
                    aug_probs = probs_aug[trial_indices]
                    similarities = [
                        np.dot(orig_probs, aug_probs[k]) / (np.linalg.norm(orig_probs) * np.linalg.norm(aug_probs[k]) + 1e-10)
                        for k in range(len(trial_indices))
                    ]
                    # Average the softmax probabilities of the top-k most similar samples
                    top_indices = np.argsort(similarities)[-min(top_k, len(trial_indices)):]  # Top-k indices
                    top_probs = aug_probs[top_indices]
                    avg_probs = np.mean(top_probs, axis=0)
                    predictions_tta[i] = np.argmax(avg_probs)
                else:
                    predictions_tta[i] = predictions[i]
                    valid_indices.append(i)
                    present_classes.add(labels[i])
            if len(present_classes) < n_classes:
                print(f"Warning: Only {len(present_classes)} classes ({sorted(present_classes)}) present in TTA for subject {sub+1}, epsilon {epsilon}, aug {n_augs_per_class}")
            if not valid_indices:
                print(f"Warning: No valid indices for TTA in subject {sub+1}, epsilon {epsilon}, aug {n_augs_per_class}. Setting acc_tta and kappa_tta to 0.")
                acc_tta = 0.0
                kappa_tta = 0.0
                cf_matrix[sub, :, :] = np.zeros((n_classes, n_classes))
            else:
                print(f"Subject {sub+1} - True labels: {labels[valid_indices][:10]}, TTA predictions: {predictions_tta[valid_indices][:10]}")
                acc_tta = accuracy_score(labels[valid_indices], predictions_tta[valid_indices])
                kappa_tta = cohen_kappa_score(labels[valid_indices], predictions_tta[valid_indices])
                cf_matrix[sub, :, :] = confusion_matrix(
                    labels[valid_indices], predictions_tta[valid_indices], 
                    labels=list(range(n_classes)), normalize='true'
                )
            
            acc_bestRun[sub] = acc_orig
            acc_tta_bestRun[sub] = acc_tta
            kappa_bestRun[sub] = kappa_tta
            
            subject_data = {
                'Subject': sub + 1,
                'Best_Run': filepath[filepath.find('run-')+4:filepath.find('/sub')],
                'Accuracy_Orig': acc_orig,
                'Accuracy_TTA': acc_tta,
                'Kappa_TTA': kappa_tta,
                'Epsilon': epsilon,
                'Aug_Count': str(n_augs_per_class),
                'Run_ID': run_id,
                'Class_Variance': str(class_variance)
            }
            subject_data['Confusion_Matrix'] = cf_matrix[sub, :, :].flatten().tolist()
            csv_data.append(subject_data)
            
            info = f"Run {run_id}, Epsilon {epsilon}, Aug {n_augs_per_class}, Subject: {sub+1}   best_run: {subject_data['Best_Run']}   acc_orig: {acc_orig:.4f}   acc_tta: {acc_tta:.4f}   kappa: {kappa_tta:.4f}"
            print(info)
            log_write.write('\n' + info)
        except Exception as e:
            error_msg = f"Error processing subject {sub+1}, epsilon {epsilon}, aug {n_augs_per_class}: {str(e)}"
            print(error_msg)
            log_write.write('\n' + error_msg)
            continue
    
    log_write.close()
    best_models.close()
    
    return csv_data, acc_bestRun, acc_tta_bestRun, kappa_bestRun

def calculate_uncertainty(csv_path, epsilon, n_augs_per_class, n_sub):
    try:
        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} not found")
            return
        
        combined_df = pd.read_csv(csv_path)
        
        uncertainty_data = []
        for sub in range(1, n_sub + 1):
            sub_data = combined_df[combined_df['Subject'] == sub]
            if not sub_data.empty:
                acc_variance = sub_data['Accuracy_TTA'].var() if len(sub_data) > 1 else 0
                kappa_variance = sub_data['Kappa_TTA'].var() if len(sub_data) > 1 else 0
                uncertainty_data.append({
                    'Subject': sub,
                    'Epsilon': epsilon,
                    'Aug_Count': str(n_augs_per_class),
                    'Accuracy_TTA_Variance': acc_variance,
                    'Kappa_TTA_Variance': kappa_variance,
                    'Mean_Accuracy_TTA': sub_data['Accuracy_TTA'].mean(),
                    'Mean_Kappa_TTA': sub_data['Kappa_TTA'].mean()
                })
        
        uncertainty_df = pd.DataFrame(uncertainty_data)
        uncertainty_csv_path = os.path.join('/home/teaching/Nikhil/ATCNet-EMD-TTA/results_plot/CSV', f"uncertainty_epsilon_{epsilon}_aug_{sum(n_augs_per_class)}.csv")
        uncertainty_df.to_csv(uncertainty_csv_path, index=False)
        print(f"Uncertainty results for epsilon={epsilon}, aug={sum(n_augs_per_class)} saved to {uncertainty_csv_path}")
    except Exception as e:
        print(f"Error in calculate_uncertainty for epsilon={epsilon}, aug={sum(n_augs_per_class)}: {str(e)}")

def getModel(model_name, dataset_conf):
    n_classes = dataset_conf.get('n_classes')
    n_channels = dataset_conf.get('n_channels')
    in_samples = dataset_conf.get('in_samples')
    if model_name == 'ATCNet':
        model = models.ATCNet_( 
            n_classes=n_classes, 
            in_chans=n_channels, 
            in_samples=in_samples, 
            n_windows=5, 
            attention='mha',
            eegn_F1=16,
            eegn_D=2, 
            eegn_kernelSize=64,
            eegn_poolSize=7,
            eegn_dropout=0.3,
            tcn_depth=2, 
            tcn_kernelSize=4,
            tcn_filters=32,
            tcn_dropout=0.3, 
            tcn_activation='elu'
        )
    else:
        raise Exception(f"'{model_name}' model is not supported yet!")
    return model

def run(epsilon_list, aug_counts_list, num_runs):
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/home/teaching/Nikhil/ATCNet-EMD-TTA/eeg_results/run_progress.log', mode='a'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    sys.stdout = StreamToLogger(logger, logging.INFO)

    dataset = 'BCI2a'
    if dataset == 'BCI2a':
        in_samples = 1125
        n_channels = 22
        n_sub = 9
        n_classes = 4
        classes_labels = ['Left hand', 'Right hand', 'Foot', 'Tongue']
        data_path = '/home/teaching/Nikhil/ATCNet-EMD-TTA/BCI/'
    else:
        raise Exception(f"'{dataset}' dataset is not supported yet!")
        
    results_path = '/home/teaching/Nikhil/ATCNet-EMD-TTA/results'
    output_path = '/home/teaching/Nikhil/ATCNet-EMD-TTA/eeg_results'
    csv_path = '/home/teaching/Nikhil/CSV'
    for path in [results_path, output_path, csv_path]:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
      
    dataset_conf = {
        'name': dataset,
        'n_classes': n_classes,
        'cl_labels': classes_labels,
        'n_sub': n_sub,
        'n_channels': n_channels,
        'in_samples': in_samples,
        'data_path': data_path,
        'isStandard': True,
        'LOSO': False,
        'sampling_rate': 250,
        'max_ica_components': 22,
        'noise_scale': 0.5,
        'key_channels': {0: [7], 1: [10], 2: [9], 3: [15, 16, 17]}
    }
    train_conf = {
        'batch_size': 32,
        'epochs': 1000,
        'patience': 300,
        'lr': 0.001,
        'LearnCurves': True,
        'n_train': 10,
        'model': 'ATCNet'
    }
    
    model = getModel(train_conf.get('model'), dataset_conf)
    
    start_time = time.time()
    print(f"Started full run at {time.strftime('%H:%M:%S', time.localtime())}")
    
    for epsilon in epsilon_list:
        for aug_counts in aug_counts_list:
            csv_file = os.path.join(csv_path, f"test_results_epsilon_{epsilon}_aug_{sum(aug_counts)}.csv")
            if os.path.exists(csv_file):
                os.remove(csv_file)
            
            # Arrays to store results across all runs for each subject
            all_acc_orig = np.zeros((n_sub, num_runs))
            all_acc_tta = np.zeros((n_sub, num_runs))
            all_kappa_tta = np.zeros((n_sub, num_runs))
            
            for run_id in range(1, num_runs + 1):
                print(f"\nRunning test run {run_id} with epsilon = {epsilon}, aug_counts = {aug_counts}")
                try:
                    csv_data, acc_orig, acc_tta, kappa_tta = test(
                        model, dataset_conf, results_path, output_path, epsilon, aug_counts, run_id
                    )
                    all_acc_orig[:, run_id-1] = acc_orig
                    all_acc_tta[:, run_id-1] = acc_tta
                    all_kappa_tta[:, run_id-1] = kappa_tta
                    
                    # Save per-run results to CSV
                    csv_df = pd.DataFrame(csv_data)
                    csv_df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
                    print(f"Test results for run {run_id}, epsilon {epsilon}, aug {sum(aug_counts)} appended to {csv_file}")
                except Exception as e:
                    error_msg = f"Error in run {run_id}, epsilon {epsilon}, aug {aug_counts}: {str(e)}\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    continue
            
            # Calculate averages across runs for each subject
            log_write = open(os.path.join(output_path, f"log_epsilon_{epsilon}_aug_{sum(aug_counts)}.txt"), "a")
            avg_csv_data = []
            for sub in range(n_sub):
                avg_acc_orig = np.mean(all_acc_orig[sub])
                avg_acc_tta = np.mean(all_acc_tta[sub])
                avg_kappa_tta = np.mean(all_kappa_tta[sub])
                info = f"\nSubject {sub+1}, Epsilon {epsilon}, Aug {aug_counts}, Average over {num_runs} runs:\nAccuracy_Orig = {avg_acc_orig:.4f}   Accuracy_TTA = {avg_acc_tta:.4f}   Kappa_TTA = {avg_kappa_tta:.4f}"
                print(info)
                log_write.write(info)
                
                avg_subject_data = {
                    'Subject': sub + 1,
                    'Best_Run': 'Average',
                    'Accuracy_Orig': avg_acc_orig,
                    'Accuracy_TTA': avg_acc_tta,
                    'Kappa_TTA': avg_kappa_tta,
                    'Epsilon': epsilon,
                    'Aug_Count': str(aug_counts),
                    'Run_ID': 'Average',
                    'Class_Variance': ''
                }
                avg_subject_data['Confusion_Matrix'] = [0] * (n_classes * n_classes)
                avg_csv_data.append(avg_subject_data)
            
            # Calculate overall average across all subjects
            overall_avg_acc_orig = np.mean(all_acc_orig)
            overall_avg_acc_tta = np.mean(all_acc_tta)
            overall_avg_kappa_tta = np.mean(all_kappa_tta)
            overall_info = f"\nOverall Average for Epsilon {epsilon}, Aug {aug_counts} over {num_runs} runs and {n_sub} subjects:\nAccuracy_Orig = {overall_avg_acc_orig:.4f}   Accuracy_TTA = {overall_avg_acc_tta:.4f}   Kappa_TTA = {overall_avg_kappa_tta:.4f}\n"
            print(info)
            log_write.write(overall_info)
            
            avg_overall_data = {
                'Subject': 'Overall Average',
                'Best_Run': '',
                'Accuracy_Orig': overall_avg_acc_orig,
                'Accuracy_TTA': overall_avg_acc_tta,
                'Kappa_TTA': overall_avg_kappa_tta,
                'Epsilon': epsilon,
                'Aug_Count': str(aug_counts),
                'Run_ID': 'Overall Average',
                'Class_Variance': ''
            }
            avg_overall_data['Confusion_Matrix'] = [0] * (n_classes * n_classes)
            avg_csv_data.append(avg_overall_data)
            
            # Append averages to CSV
            avg_csv_df = pd.DataFrame(avg_csv_data)
            avg_csv_df.to_csv(csv_file, mode='a', header=False, index=False)
            
            log_write.close()
            
            try:
                calculate_uncertainty(csv_file, epsilon, aug_counts, n_sub)
                print(f"Completed uncertainty calculation for epsilon {epsilon}, aug {aug_counts}")
            except Exception as e:
                error_msg = f"Error in calculate_uncertainty for epsilon {epsilon}, aug {aug_counts}: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                continue
    
    print(f"Completed full run in {(time.time() - start_time)/60:.2f} minutes at {time.strftime('%H:%M:%S', time.localtime())}")
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run EEG-ATCNet TTA with ICA+FGSM")
    parser.add_argument('--epsilon', type=float, default=0.005, help='Perturbation strength for FGSM')
    parser.add_argument('--num_aug', type=int, default=10, help='Number of augmentations per class (uniform for all classes)')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of test runs')
    args = parser.parse_args()
    
    epsilon_list = [0.025, 0.05, 0.1]
    aug_counts_list = [[10, 10, 10, 10], [20, 20, 20, 20], [30, 30, 30, 30]]
    if args.epsilon != 0.005:
        epsilon_list = [args.epsilon]
    if args.num_aug != 10:
        aug_counts_list = [[args.num_aug] * 4]
    
    run(epsilon_list, aug_counts_list, args.num_runs)
