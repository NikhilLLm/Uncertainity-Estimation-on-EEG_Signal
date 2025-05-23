from create_artificial_frame import create_artifical_frame
import numpy as np
from PyEMD import EMD
import tensorflow as tf

def emd_tta(X_test, y_test_onehot, dataset_conf,n_imfs = 3,num_arti_FRMs = 10,random_seed = None):
    print(X_test.shape)
    # print(y_test_onehot)

    augmented_trials = []
        # Extract class labels and unique classes
    class_labels = np.argmax(y_test_onehot, axis=1)
    unique_classes = np.unique(class_labels)
    class_imfs = {cls: [] for cls in unique_classes}

        # Step 1: Decompose trials and store IMFs per class
    for trial_idx, trial in enumerate(X_test):
        trial_data = trial[0]  # Shape [C, T]
        trial_class = class_labels[trial_idx]
            
        imfs_per_channel = []
        for ch in trial_data:
            emd = EMD()
            full_imfs = emd(ch)
            imfs = full_imfs[:n_imfs] if full_imfs.shape[0] >= n_imfs else np.pad(full_imfs, ((0,n_imfs-full_imfs.shape[0]),(0,0)))
            imfs_per_channel.append(imfs)
            
        class_imfs[trial_class].append(np.array(imfs_per_channel))  # [C, n_imfs, T]

        # Step 2: Generate artificial frames per class
      # Adjust as needed
    artificial_trials, artificial_labels = [], []

    for cls in unique_classes:
        if not class_imfs[cls]:
            continue
            
            # Prepare IMF data structure [T, n_imfs, num_frames, C]
        trials_imfs = np.transpose(np.array(class_imfs[cls]), (0, 3, 2, 1))  # [num_frames, T, n_imfs, C]
        set_real_IMFs = np.transpose(trials_imfs, (1, 2, 0, 3))  # [T, n_imfs, num_frames, C]
            
            # Generate artificial frames
        arti_FRMs = create_artifical_frame(
            set_real_FRMs_idx=list(range(len(class_imfs[cls]))),
            num_arti_FRMs=num_arti_FRMs,
            set_real_IMFs=set_real_IMFs,
            set_real_IMFs_idx=list(range(n_imfs)),
            random_seed=random_seed
        )
            
            # Reshape and add to artificial data
        arti_FRMs = arti_FRMs.transpose(0, 2, 1)[:, np.newaxis, :, :]  # [num_arti_FRMs, 1, C, T]
        artificial_trials.extend(arti_FRMs)
        artificial_labels.extend([cls] * num_arti_FRMs)

        # Combine original and artificial data
    X_test_combined = np.concatenate([X_test, np.array(artificial_trials)], axis=0)
    y_test_combined = np.concatenate([
        y_test_onehot,
        tf.keras.utils.to_categorical(artificial_labels, num_classes=dataset_conf['n_classes'])
        ], axis=0)

    print("Augmented data shape:", X_test_combined.shape)
    print("Augmented labels shape:", y_test_combined.shape)
    #     # Process combined data through EMD augmentation
    # for trial in X_test_combined:
    #         trial_data = trial[0]  # [C, T]
    #         imfs_per_channel = []
            
    #         for ch in trial_data:
    #             emd = EMD()
    #             full_imfs = emd(ch)
    #             imfs = full_imfs[:n_imfs] if full_imfs.shape[0] >= n_imfs else np.pad(full_imfs, ((0,n_imfs-full_imfs.shape[0]),(0,0)))
    #             imfs_per_channel.append(imfs)
            
    #         # Flatten IMF structure
    #         imfs_flat = np.array(imfs_per_channel).reshape(-1, trial_data.shape[1])
    #         augmented_trials.append(imfs_flat)

    # X_test_augmented = np.array(augmented_trials)[:, np.newaxis, :, :]
    # y_test_augmented = y_test_combined

    # print("Augmented data shape:", X_test_augmented.shape)
    # print("Augmented labels shape:", y_test_augmented.shape)
    # return X_test_augmented, y_test_augmented
    return X_test_combined, y_test_combined

if __name__ == "__main__":
    # Example data
    X_test = np.random.rand(5, 1, 64, 128)  # 5 trials, 1 channel, 64 samples, 128 time points
    y_test_onehot = np.random.randint(0, 2, (5, 2))  # 5 trials, 2 classes (one-hot encoded)
    dataset_conf = {'n_classes': 2}

    # Call the tta function
    X_test_augmented, y_test_augmented = emd_tta(X_test, y_test_onehot, dataset_conf,n_imfs=4, num_arti_FRMs=50)


        # # Example: pick top 3 IMFs per channel
        # n_imfs = 2
        # augmented_trials = []
        # for trial in X_test:  # shape: [1, C, T]
        #     trial = trial[0]  # shape: [C, T]
        #     imfs_per_channel = []
        #     for ch in trial:
        #         emd = EMD()
        #         full_imfs = emd(ch)
        #         if full_imfs.shape[0] >= n_imfs:
        #             imfs = full_imfs[:n_imfs]
        #         else:
        #             padded = np.zeros((n_imfs, full_imfs.shape[1]))
        #             padded[:full_imfs.shape[0]] = full_imfs
        #             imfs = padded
        #         imfs_per_channel.append(imfs)  # [n_imfs, T]
        #     imfs_per_channel = np.array(imfs_per_channel)  # [C, n_imfs, T]
        #     imfs_per_channel = imfs_per_channel.reshape(-1, imfs_per_channel.shape[-1])  # [C * n_imfs, T]
        #     augmented_trials.append(imfs_per_channel)

        # X_test = np.array(augmented_trials)  # [N, C * n_imfs, T]
        # X_test = X_test[:, np.newaxis, :, :]  # add channel dim: [N, 1, C * n_imfs, T]
        # print(X_test.shape)
        # print(y_test_onehot.shape)
        # Iteration over runs (seeds)
        # Inside the test function after loading X_test and y_test_onehot

    # Apply EMD-based augmentation and generate artificial frames
    