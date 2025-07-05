import numpy as np
import tensorflow as tf
from scipy.signal import welch, butter, filtfilt
from sklearn.decomposition import FastICA, PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

def ica_fsgm_tta(predict_fn, X_test, y_test_onehot, dataset_conf, clip_min=-3.0, clip_max=3.0, 
                 n_augs_per_class=10, epsilon=0.025, perturb_all_components=True, 
                 apply_bandpass=True, save_tsne_path=None, subject_id=None, verbose=1):
    if verbose > 0:
        print("Applying ICA+FGSM-based TTA...")
    N, _, C, T = X_test.shape
    n_classes = dataset_conf['n_classes']
    fs = dataset_conf['sampling_rate']

    required_keys = ['n_classes', 'sampling_rate', 'key_channels', 'class_freq_bands']
    for key in required_keys:
        if key not in dataset_conf:
            raise ValueError(f"dataset_conf must contain '{key}'")

    if X_test.shape[0] != y_test_onehot.shape[0]:
        raise ValueError(f"X_test.shape[0]={X_test.shape[0]} does not match y_test_onehot.shape[0]={y_test_onehot.shape[0]}")

    if isinstance(n_augs_per_class, int):
        n_augs_per_class = [n_augs_per_class] * n_classes
    if len(n_augs_per_class) != n_classes:
        raise ValueError(f"n_augs_per_class length {len(n_augs_per_class)} does not match n_classes {n_classes}")
    total_aug_samples = sum(n_augs_per_class)
    total_samples = N + total_aug_samples

    X_test_combined = np.zeros((total_samples, 1, C, T), dtype=np.float32)
    y_test_combined = np.zeros((total_samples, n_classes), dtype=np.float32)

    X_test_copy = X_test.copy()
    y_test_onehot_copy = y_test_onehot.copy()

    key_channels = dataset_conf['key_channels']
    class_freq_bands = dataset_conf['class_freq_bands']
    labels = np.argmax(y_test_onehot_copy, axis=1)

    n_components = min(dataset_conf.get('max_ica_components', C), C)
    ica = FastICA(n_components=n_components, random_state=42, whiten='unit-variance')
    X_flat = X_test_copy[:, 0, :, :].transpose(0, 2, 1).reshape(N * T, C)
    S = ica.fit_transform(X_flat)
    A = ica.mixing_
    S = S.reshape(N, T, n_components)
    if verbose > 1:
        print(f"ICA: n_components={n_components}, S.shape={S.shape}, A.shape={A.shape}")

    component_stds = np.std(S, axis=(0, 1))
    if verbose > 1:
        print(f"Component standard deviations: {component_stds}")

    aug_trial_indices = []
    ic_indices = {}
    for class_idx in range(n_classes):
        class_trials = np.where(labels == class_idx)[0]
        if verbose > 1:
            print(f"Class {class_idx}: class_trials={class_trials}")
        if class_trials.size == 0:
            if verbose > 0:
                print(f"Warning: No trials for class {class_idx}. Skipping augmentation for this class.")
            continue

        channel_idx = key_channels[class_idx][0]
        low, high = class_freq_bands[class_idx]
        trial_powers = []
        for trial in class_trials:
            signal = X_test_copy[trial, 0, channel_idx, :]
            f, Pxx = welch(signal, fs=fs, nperseg=min(256, T))
            power = np.sum(Pxx[(f >= low) & (f <= high)]) / np.sum(Pxx) if np.sum(Pxx) > 0 else 0
            trial_powers.append(power)
        if len(trial_powers) < n_augs_per_class[class_idx]:
            if verbose > 0:
                print(f"Warning: Not enough trials ({len(trial_powers)}) for class {class_idx} to meet n_augs_per_class={n_augs_per_class[class_idx]}. Adjusting.")
            selected = np.random.choice(class_trials, size=n_augs_per_class[class_idx], replace=True)
        else:
            top_trials = class_trials[np.argsort(trial_powers)[-n_augs_per_class[class_idx]:]]
            selected = np.random.choice(top_trials, size=n_augs_per_class[class_idx], replace=True)
        aug_trial_indices.extend(selected)

        ic_powers = []
        for ic in range(n_components):
            ic_signal = S[class_trials, :, ic].flatten()
            f, Pxx = welch(ic_signal, fs=fs, nperseg=min(256, T))
            power = np.sum(Pxx[(f >= low) & (f <= high)]) / np.sum(Pxx) if np.sum(Pxx) > 0 else 0
            if class_idx == n_classes - 1 and len(key_channels[class_idx]) > 1:
                channel_weights = [np.abs(A[ch, ic]) for ch in key_channels[class_idx]]
                topo_weight = np.mean(channel_weights) / np.abs(A[:, ic]).max() if np.abs(A[:, ic]).max() > 0 else 0
            else:
                topo_weight = np.abs(A[channel_idx, ic]) / np.abs(A[:, ic]).max() if np.abs(A[:, ic]).max() > 0 else 0
            ic_powers.append(power * topo_weight)
        top_ic_indices = np.argsort(ic_powers)[-2:] if len(ic_powers) >= 2 else [np.argmax(ic_powers)]
        ic_indices[class_idx] = top_ic_indices
        if verbose > 1:
            for ic_idx in top_ic_indices:
                topo_weights = {ch: A[ch, ic_idx] for ch in [7, 10, 9] if ch < C}
                print(f"Class {class_idx} selected IC {ic_idx} with power {ic_powers[ic_idx]:.4f}, topo weights (C3/C4/Cz): {topo_weights}")

    aug_trial_indices = sorted(aug_trial_indices)
    max_index = N - 1
    invalid_indices = [idx for idx in aug_trial_indices if idx > max_index]
    if invalid_indices:
        raise ValueError(f"aug_trial_indices contains invalid indices {invalid_indices} exceeding max index {max_index} for N={N}")
    if verbose > 1:
        print(f"Selected {len(aug_trial_indices)} trials for augmentation: {aug_trial_indices}")

    X_test_combined[:N] = X_test_copy
    y_test_combined[:N] = y_test_onehot_copy

    aug_idx = N
    for i in aug_trial_indices:
        class_idx = labels[i]
        s = S[i:i+1]
        s = s * np.random.uniform(0.8, 1.2)
        y = y_test_onehot_copy[i:i+1]
        weight = 1.0

        s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(s_tensor)
            x_reconstructed = tf.matmul(s_tensor, tf.convert_to_tensor(A.T, dtype=tf.float32))
            x_reconstructed = tf.transpose(x_reconstructed, [0, 2, 1])[:, tf.newaxis, :, :]
            logits = predict_fn(x_reconstructed)
            y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
            loss = tf.keras.losses.categorical_crossentropy(y_tensor, logits, from_logits=False, label_smoothing=0.1)
        grad = tape.gradient(loss, s_tensor)
        grad_np = grad.numpy() if grad is not None else np.zeros_like(s)
        grad_sign = np.sign(grad_np)
        if np.any(np.isnan(grad_np)):
            if verbose > 0:
                print(f"Skipping trial {i}, class {class_idx}: Gradient contains NaN")
            continue
        grad_norm = np.linalg.norm(grad_np.flatten(), ord=2)

        epsilon_var = epsilon * np.random.uniform(0.5, 1.5)
        perturbation = np.zeros_like(s)
        components_to_perturb = ic_indices[class_idx] if perturb_all_components else [np.random.choice(ic_indices[class_idx])]
        for ic_to_perturb in components_to_perturb:
            ic_epsilon = epsilon_var * component_stds[ic_to_perturb]
            ic_perturbation = ic_epsilon * grad_sign[:, :, ic_to_perturb] * weight
            if apply_bandpass:
                low, high = class_freq_bands[class_idx]
                b, a = butter(2, [low/(fs/2), high/(fs/2)], btype='band')
                ic_perturbation = filtfilt(b, a, ic_perturbation)
            perturbation[:, :, ic_to_perturb] = ic_perturbation

        s_adv = s + perturbation
        x_adv = np.matmul(s_adv, A.T)
        x_adv = x_adv.transpose(0, 2, 1)[:, np.newaxis, :, :]
        x_adv = np.clip(x_adv, clip_min, clip_max)
        X_test_combined[aug_idx] = x_adv
        y_test_combined[aug_idx] = y
        if verbose > 1:
            pre_filter_std = np.std(epsilon_var * grad_sign * weight)
            post_filter_std = np.std(perturbation)
            print(f"Trial {i}, Class {class_idx}: perturbed ICs {components_to_perturb}, epsilon={epsilon_var:.4f}, grad_norm={grad_norm:.2f}, pre_filter_std={pre_filter_std:.2f}, post_filter_std={post_filter_std:.2f}")
        aug_idx += 1

    total_samples = aug_idx
    X_test_combined = X_test_combined[:total_samples]
    y_test_combined = y_test_combined[:total_samples]

    expected_shape = (total_samples, 1, C, T)
    if X_test_combined.shape != expected_shape:
        raise ValueError(f"X_test_combined shape {X_test_combined.shape} does not match expected {expected_shape}")
    if y_test_combined.shape != (total_samples, n_classes):
        raise ValueError(f"y_test_combined shape {y_test_combined.shape} does not match expected {(total_samples, n_classes)}")

    probs_aug = predict_fn(X_test_combined)
    probs_aug = probs_aug.numpy() if isinstance(probs_aug, tf.Tensor) else probs_aug
    preds_aug = np.argmax(probs_aug, axis=-1)
    labels_aug = np.argmax(y_test_combined, axis=1)
    aug_preds = preds_aug[N:total_samples]
    pred_dist = {i: np.sum(aug_preds == i) for i in range(n_classes)}
    if verbose > 0:
        print(f"Augmented prediction distribution: {pred_dist}")
        print(f"True aug labels (first 10): {labels_aug[N:total_samples][:10]}, Predicted aug labels (first 10): {aug_preds[:10]}")

    class_variance = {}
    entropy = -np.sum(probs_aug * np.log(probs_aug + 1e-10), axis=-1).mean()  # Entropy calculation
    for class_idx in range(n_classes):
        aug_indices = np.where(labels_aug[N:total_samples] == class_idx)[0] + N
        if verbose > 1:
            print(f"Class {class_idx}: aug_indices={aug_indices}")
        if aug_indices.size > 0:
            class_probs = probs_aug[aug_indices]
            class_variance[class_idx] = np.var(class_probs, axis=1).mean()
        else:
            class_variance[class_idx] = 0.0
            if verbose > 0:
                print(f"Warning: No augmented trials for class {class_idx}")
    if verbose > 0:
        print(f"Class variances: {class_variance}")

    # Initialize t-SNE variables as None in case computation fails
    X_embedded = None
    try:
        # Always compute t-SNE embeddings (even if save_tsne_path is None) since main.py needs X_embedded
        print(f"Subject {subject_id}: Total samples for t-SNE = {total_samples}")

        # Skip t-SNE if total_samples is too small
        if total_samples < 5:
            raise ValueError(f"Too few samples ({total_samples}) for t-SNE computation (minimum 5 required)")

        X_flat = X_test_combined[:, 0, :, :].reshape(total_samples, C * T)

        # Check for NaNs or infinities in X_test_combined
        has_nans = np.any(np.isnan(X_test_combined))
        has_infs = np.any(np.isinf(X_test_combined))
        if has_nans or has_infs:
            print(f"Subject {subject_id}: X_test_combined contains NaNs: {has_nans}, Infinities: {has_infs}. Replacing with 0 for t-SNE.")
            X_test_combined = np.nan_to_num(X_test_combined, nan=0.0, posinf=0.0, neginf=0.0)
            X_flat = X_test_combined[:, 0, :, :].reshape(total_samples, C * T)

        # Apply PCA to reduce dimensionality before t-SNE
        n_pca_components = min(50, total_samples, C * T)  # Reduce to 50 components or fewer
        print(f"Subject {subject_id}: Applying PCA to reduce dimensions to {n_pca_components}")
        pca = PCA(n_components=n_pca_components, random_state=42)
        X_reduced = pca.fit_transform(X_flat)
        explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
        print(f"Subject {subject_id}: PCA explained variance ratio = {explained_variance_ratio:.4f}")

        # Adjust perplexity dynamically
        max_perplexity = min(30, total_samples - 1)
        perplexity = min(max_perplexity, max(5, total_samples // 3))
        print(f"Subject {subject_id}: Using perplexity = {perplexity} for t-SNE")

        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_embedded = tsne.fit_transform(X_reduced)

        # If save_tsne_path is provided, generate the plot (not used in your case since main.py handles plotting)
        if save_tsne_path and subject_id is not None:
            os.makedirs(save_tsne_path, exist_ok=True)
            
            class_names = ['Left hand', 'Right hand', 'Foot', 'Tongue']
            colors = {
                'original': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                'augmented': ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896']
            }
            markers = {'original': 'o', 'augmented': '^'}
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
            axes = axes.flatten()
            
            orig_indices = np.arange(N)
            aug_indices = np.arange(N, total_samples)
            orig_labels = labels_aug[orig_indices]
            aug_labels = labels_aug[aug_indices]
            
            for class_idx in range(n_classes):
                ax = axes[class_idx]
                orig_mask = orig_labels == class_idx
                ax.scatter(X_embedded[orig_indices[orig_mask], 0], 
                           X_embedded[orig_indices[orig_mask], 1],
                           c=colors['original'][class_idx], 
                           marker=markers['original'], 
                           label='Original', 
                           alpha=0.6, 
                           s=50)
                
                aug_mask = aug_labels == class_idx
                ax.scatter(X_embedded[aug_indices[aug_mask], 0], 
                           X_embedded[aug_indices[aug_mask], 1],
                           c=colors['augmented'][class_idx], 
                           marker=markers['augmented'], 
                           label='Augmented', 
                           alpha=0.6, 
                           s=50)
                
                ax.set_title(f'Class {class_idx}: {class_names[class_idx]}')
                ax.set_xlabel('t-SNE 1')
                ax.set_ylabel('t-SNE 2')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.suptitle(f't-SNE Visualization of EEG Trials by Class (Subject {subject_id})', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(save_tsne_path, f'tsne_subject_{subject_id}.png'), dpi=300)
            plt.close()
    except Exception as e:
        print(f"Error computing t-SNE for subject {subject_id}: {str(e)}")
        X_embedded = None

    return X_test_combined, y_test_combined, class_variance, entropy, X_embedded, labels_aug
