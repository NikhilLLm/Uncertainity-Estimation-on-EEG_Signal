import os
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import models 
from preprocess import get_data
from eeg_fsgm_new import ica_fsgm_tta

# Test Function with ICA+FGSM TTA
def test(model, dataset_conf, results_path, output_path, epsilon, n_augs_per_class, run_id, allRuns=False):
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
    kappa_bestRun = np.zeros(n_sub)
    cf_matrix = np.zeros([n_sub, n_classes, n_classes])
    csv_data = []
    # Lists to store t-SNE data for aggregation across runs
    all_tsne_data_per_subject = [[] for _ in range(n_sub)]  # Per subject, across runs
    all_tsne_labels_per_subject = [[] for _ in range(n_sub)]
    all_subject_ids_per_subject = [[] for _ in range(n_sub)]

    for sub in range(n_sub):
        try:
            _, _, _, X_test, _, y_test_onehot = get_data(data_path, sub, dataset, LOSO, isStandard)
            N = X_test.shape[0]
            # Clear GPU memory before each subject
            tf.keras.backend.clear_session()
            X_test, y_test_onehot, class_variance, entropy, X_embedded, labels_aug = ica_fsgm_tta(
                predict_fn=lambda X: model(X, training=False),
                X_test=X_test,
                y_test_onehot=y_test_onehot,
                dataset_conf=dataset_conf,
                n_augs_per_class=n_augs_per_class,
                epsilon=epsilon,
                clip_min=-5.0,
                clip_max=5.0,
                perturb_all_components=True,
                apply_bandpass=True,
                save_tsne_path=None,  # We'll handle plotting after aggregating runs
                subject_id=sub+1,
                verbose=1
            )
            
            filepath = best_models.readline().strip()
            model.load_weights(os.path.join(results_path, filepath.lstrip('/')))
            batch_size = 16  # Reduced batch size to mitigate GPU memory issues
            n_samples = X_test.shape[0]
            y_pred = []
            for i in range(0, n_samples, batch_size):
                batch_X = X_test[i:i+batch_size]
                batch_pred = model.predict(batch_X, verbose=0).argmax(axis=-1)
                y_pred.append(batch_pred)
            y_pred = np.concatenate(y_pred, axis=0)
            labels = y_test_onehot.argmax(axis=-1)
            
            orig_indices = np.arange(N)
            aug_indices = np.arange(N, n_samples)
            orig_acc = accuracy_score(labels[orig_indices], y_pred[orig_indices])
            aug_acc = accuracy_score(labels[aug_indices], y_pred[aug_indices]) if aug_indices.size > 0 else 0.0
            total_acc = accuracy_score(labels, y_pred)
            kappa_bestRun[sub] = cohen_kappa_score(labels, y_pred)
            cf_matrix[sub, :, :] = confusion_matrix(labels, y_pred, normalize='true')
            
            softmax_variance = np.mean(list(class_variance.values()))
            
            subject_data = {
                'Subject': sub + 1,
                'Best_Run': filepath[filepath.find('run-')+4:filepath.find('/sub')],
                'Accuracy_Orig': orig_acc,
                'Accuracy_TTA': total_acc,
                'Augmented_Accuracy': aug_acc,
                'Kappa': kappa_bestRun[sub],
                'Epsilon': epsilon,
                'Aug_Count': str(n_augs_per_class),
                'Run_ID': run_id,
                'Softmax_Variance': softmax_variance,
                'Entropy': entropy
            }
            subject_data['Confusion_Matrix'] = cf_matrix[sub, :, :].flatten().tolist()
            csv_data.append(subject_data)
            
            info = (f"Run {run_id}, Epsilon {epsilon}, Aug {n_augs_per_class}, Subject: {sub+1}   "
                    f"best_run: {subject_data['Best_Run']}   acc_orig: {orig_acc:.4f}   acc_tta: {total_acc:.4f}   "
                    f"kappa: {kappa_bestRun[sub]:.4f}   softmax_variance: {softmax_variance:.4f}   entropy: {entropy:.4f}")
            print(info)
            log_write.write('\n' + info)
            
            # Collect t-SNE data only if X_embedded is not None
            if X_embedded is not None:
                all_tsne_data_per_subject[sub].append(X_embedded)
                all_tsne_labels_per_subject[sub].append(labels_aug)
                all_subject_ids_per_subject[sub].extend([sub+1] * len(labels_aug))
            else:
                print(f"Skipping t-SNE data collection for subject {sub+1} due to computation failure")
            
        except Exception as e:
            print(f"Error processing subject {sub+1}, epsilon {epsilon}, aug {n_augs_per_class}: {str(e)}")
            log_write.write(f"\nError processing subject {sub+1}, epsilon {epsilon}, aug {n_augs_per_class}: {str(e)}")
            continue
    
    info = (f"\nRun {run_id}, Average of {n_sub} subjects - best runs (epsilon={epsilon}, aug={sum(n_augs_per_class)}):\n"
            f"Accuracy_Orig = {np.mean([d['Accuracy_Orig'] for d in csv_data]):.4f}   "
            f"Accuracy_TTA = {np.mean([d['Accuracy_TTA'] for d in csv_data]):.4f}   "
            f"Kappa_TTA = {np.average(kappa_bestRun):.4f}\n")
    print(info)
    log_write.write(info)
    
    csv_path = os.path.join(output_path, f"test_results_epsilon_{epsilon}_aug_{sum(n_augs_per_class)}.csv")
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
    print(f"Test results for run {run_id} appended to {csv_path}")
    
    log_write.close()
    best_models.close()
    
    return csv_data, all_tsne_data_per_subject, all_tsne_labels_per_subject, all_subject_ids_per_subject

# Function to Plot Aggregated t-SNE
def plot_aggregated_tsne(tsne_data, tsne_labels, subject_ids, output_path, epsilon, n_augs_per_class, n_classes, title_prefix, filename_prefix):
    if not tsne_data:
        print(f"No t-SNE data to plot for {title_prefix}")
        return
    
    tsne_data = np.vstack(tsne_data)
    tsne_labels = np.concatenate(tsne_labels)
    subject_ids = np.array(subject_ids)
    
    class_names = ['Left hand', 'Right hand', 'Foot', 'Tongue']
    colors = {
        'original': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
        'augmented': ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896']
    }
    markers = {'original': 'o', 'augmented': '^'}
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    unique_subjects = np.unique(subject_ids)
    N = len(tsne_labels) // len(unique_subjects) // 2  # Approximate number of original samples per subject
    
    for class_idx in range(n_classes):
        ax = axes[class_idx]
        for sub in unique_subjects:
            sub_mask = subject_ids == sub
            orig_mask = (tsne_labels == class_idx) & (np.arange(len(tsne_labels)) % (2 * N) < N) & sub_mask
            aug_mask = (tsne_labels == class_idx) & (np.arange(len(tsne_labels)) % (2 * N) >= N) & sub_mask
            
            if orig_mask.sum() > 0:
                ax.scatter(tsne_data[orig_mask, 0], tsne_data[orig_mask, 1],
                           c=colors['original'][class_idx], marker=markers['original'],
                           label=f'Sub {sub} Orig' if class_idx == 0 and sub == unique_subjects[0] else None,
                           alpha=0.6, s=50)
            if aug_mask.sum() > 0:
                ax.scatter(tsne_data[aug_mask, 0], tsne_data[aug_mask, 1],
                           c=colors['augmented'][class_idx], marker=markers['augmented'],
                           label=f'Sub {sub} Aug' if class_idx == 0 and sub == unique_subjects[0] else None,
                           alpha=0.6, s=50)
        
        ax.set_title(f'Class {class_idx}: {class_names[class_idx]}')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        if class_idx == 0:
            ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle(f'{title_prefix} (Epsilon {epsilon}, Aug {n_augs_per_class})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(output_path, f'tsne_plots/epsilon_{epsilon}_aug_{sum(n_augs_per_class)}')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{filename_prefix}.png'), dpi=300)
    plt.close()

# Uncertainty Calculation and Per-Subject Averages
def calculate_uncertainty_and_averages(results_path, output_path, epsilon, n_augs_per_class, n_sub, num_runs,
                                       all_tsne_data_per_subject, all_tsne_labels_per_subject, all_subject_ids_per_subject):
    csv_path = os.path.join(output_path, f"test_results_epsilon_{epsilon}_aug_{sum(n_augs_per_class)}.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return
    
    combined_df = pd.read_csv(csv_path)
    
    # Per-subject averages
    for sub in range(1, n_sub + 1):
        sub_data = combined_df[combined_df['Subject'] == sub]
        if not sub_data.empty:
            acc_orig_mean = sub_data['Accuracy_Orig'].mean()
            acc_tta_mean = sub_data['Accuracy_TTA'].mean()
            kappa_mean = sub_data['Kappa'].mean()
            print(f"Subject {sub}, Epsilon {epsilon}, Aug {n_augs_per_class}, Average over {len(sub_data)} runs:")
            print(f"Accuracy_Orig = {acc_orig_mean:.4f}   Accuracy_TTA = {acc_tta_mean:.4f}   Kappa_TTA = {kappa_mean:.4f}")
            
            # Plot aggregated t-SNE for this subject
            tsne_data = all_tsne_data_per_subject[sub-1]
            tsne_labels = all_tsne_labels_per_subject[sub-1]
            subject_ids = all_subject_ids_per_subject[sub-1]
            plot_aggregated_tsne(
                tsne_data, tsne_labels, subject_ids, output_path, epsilon, n_augs_per_class, 4,
                title_prefix=f't-SNE Visualization of EEG Trials by Class (Subject {sub})',
                filename_prefix=f'tsne_subject_{sub}'
            )
    
    # Overall average
    acc_orig_overall = combined_df['Accuracy_Orig'].mean()
    acc_tta_overall = combined_df['Accuracy_TTA'].mean()
    kappa_overall = combined_df['Kappa'].mean()
    print(f"\nOverall Average for Epsilon {epsilon}, Aug {n_augs_per_class} over {num_runs} runs and {n_sub} subjects:")
    print(f"Accuracy_Orig = {acc_orig_overall:.4f}   Accuracy_TTA = {acc_tta_overall:.4f}   Kappa_TTA = {kappa_overall:.4f}")

    # Plot aggregated overall t-SNE
    all_tsne_data = [item for sublist in all_tsne_data_per_subject for item in sublist]
    all_tsne_labels = [item for sublist in all_tsne_labels_per_subject for item in sublist]
    all_subject_ids = [item for sublist in all_subject_ids_per_subject for item in sublist]
    plot_aggregated_tsne(
        all_tsne_data, all_tsne_labels, all_subject_ids, output_path, epsilon, n_augs_per_class, 4,
        title_prefix=f'Overall t-SNE Visualization',
        filename_prefix='tsne_overall'
    )

    # Uncertainty calculation
    uncertainty_data = []
    for sub in range(1, n_sub + 1):
        sub_data = combined_df[combined_df['Subject'] == sub]
        if not sub_data.empty:
            acc_variance = sub_data['Accuracy_TTA'].var() if len(sub_data) > 1 else 0
            kappa_variance = sub_data['Kappa'].var() if len(sub_data) > 1 else 0
            uncertainty_data.append({
                'Subject': sub,
                'Accuracy_Variance': acc_variance,
                'Kappa_Variance': kappa_variance,
                'Mean_Accuracy_Orig': sub_data['Accuracy_Orig'].mean(),
                'Mean_Accuracy_TTA': sub_data['Accuracy_TTA'].mean(),
                'Mean_Kappa': sub_data['Kappa'].mean(),
                'Epsilon': epsilon,
                'Aug_Count': str(n_augs_per_class)
            })
    
    uncertainty_df = pd.DataFrame(uncertainty_data)
    uncertainty_csv_path = os.path.join(output_path, f"uncertainty_epsilon_{epsilon}_aug_{sum(n_augs_per_class)}.csv")
    uncertainty_df.to_csv(uncertainty_csv_path, index=False)
    print(f"Uncertainty results for epsilon={epsilon}, aug={sum(n_augs_per_class)} saved to {uncertainty_csv_path}")

# Model Definition
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
    elif model_name == 'TCNet_Fusion':
        model = models.TCNet_Fusion(n_classes=n_classes, Chans=n_channels, Samples=in_samples)      
    elif model_name == 'EEGTCNet':
        model = models.EEGTCNet(n_classes=n_classes, Chans=n_channels, Samples=in_samples)          
    elif model_name == 'EEGNet':
        model = models.EEGNet_classifier(n_classes=n_classes, Chans=n_channels, Samples=in_samples) 
    elif model_name == 'EEGNeX':
        model = models.EEGNeX_8_32(n_timesteps=in_samples, n_features=n_channels, n_outputs=n_classes)
    elif model_name == 'DeepConvNet':
        model = models.DeepConvNet(nb_classes=n_classes, Chans=n_channels, Samples=in_samples)
    elif model_name == 'ShallowConvNet':
        model = models.ShallowConvNet(nb_classes=n_classes, Chans=n_channels, Samples=in_samples)
    elif model_name == 'MBEEG_SENet':
        model = models.MBEEG_SENet(nb_classes=n_classes, Chans=n_channels, Samples=in_samples)
    else:
        raise Exception(f"'{model_name}' model is not supported yet!")

    return model

# Main Run Function
def run(epsilons, num_augs, num_runs):
    dataset = 'BCI2a'
    in_samples = 1125
    n_channels = 22
    n_sub = 9
    n_classes = 4
    classes_labels = ['Left hand', 'Right hand', 'Foot', 'Tongue']
    data_path = '/home/teaching/Nikhil/ATCNet-EMD-TTA/BCI/'
    results_path = '/home/teaching/Nikhil/ATCNet-EMD-TTA/results'
    output_path = '/home/teaching/Nikhil/ATCNet-EMD-TTA/eeg_results'
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
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
        'max_ica_components': 10,
        'key_channels': {0: [7], 1: [10], 2: [9], 3: [15, 16, 17]},
        'class_freq_bands': {
            0: (7, 13),
            1: (7, 12),
            2: (12, 32),
            3: (12, 32)
        }
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
    
    for epsilon in epsilons:
        for num_aug in num_augs:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            n_augs_per_class = [num_aug] * n_classes
            csv_path = os.path.join(output_path, f"test_results_epsilon_{epsilon}_aug_{sum(n_augs_per_class)}_{timestamp}.csv")
            print(f"CSV path: {csv_path}")
            if os.path.exists(csv_path):
                os.remove(csv_path)
            
            # Aggregate t-SNE data across all runs
            all_tsne_data_per_subject = [[] for _ in range(n_sub)]
            all_tsne_labels_per_subject = [[] for _ in range(n_sub)]
            all_subject_ids_per_subject = [[] for _ in range(n_sub)]
            
            for run_id in range(1, num_runs + 1):
                print(f"\nRunning test run {run_id} with epsilon = {epsilon}, num_aug = {num_aug}")
                csv_data, tsne_data, tsne_labels, subject_ids = test(
                    model, dataset_conf, results_path, output_path, epsilon, n_augs_per_class, run_id, allRuns=False
                )
                # Aggregate t-SNE data
                for sub in range(n_sub):
                    all_tsne_data_per_subject[sub].extend(tsne_data[sub])
                    all_tsne_labels_per_subject[sub].extend(tsne_labels[sub])
                    all_subject_ids_per_subject[sub].extend(subject_ids[sub])
            
            calculate_uncertainty_and_averages(
                results_path, output_path, epsilon, n_augs_per_class, n_sub, num_runs,
                all_tsne_data_per_subject, all_tsne_labels_per_subject, all_subject_ids_per_subject
            )

# Entry Point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run EEG-ATCNet test with ICA+FGSM TTA")
    parser.add_argument('--num_runs', type=int, default=10, help='Number of test runs')
    args = parser.parse_args()
    epsilons = [0.01, 0.1, 0.2, 0.3, 0.4]
    num_augs = [10, 20, 30, 40, 50]
    run(epsilons=epsilons, num_augs=num_augs, num_runs=args.num_runs)
