import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# Define paths
input_dir = "/home/teaching/Nikhil/ATCNet-EMD-TTA/eeg_results/"
output_dir = "/home/teaching/Nikhil/ATCNet-EMD-TTA/final_plot/"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the expected epsilon and aug_count values
epsilons = [0.01, 0.025, 0.05, 0.075, 0.1]
aug_counts = [20, 40, 60, 80, 100]

# Colors for each aug_count (matching the Chart.js colors)
colors = {
    20: '#1f77b4',   # Blue
    40: '#ff7f0e',   # Orange
    60: '#2ca02c',   # Green
    80: '#d62728',   # Red
    100: '#9467bd'   # Purple
}

# Load all uncertainty CSV files
csv_pattern = os.path.join(input_dir, "uncertainty_epsilon_*.csv")
csv_files = glob.glob(csv_pattern)

if not csv_files:
    raise FileNotFoundError(f"No uncertainty CSV files found in {input_dir}")

# Combine all CSV files into a single DataFrame
all_data = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    all_data.append(df)
data = pd.concat(all_data, ignore_index=True)

# Ensure numeric columns are properly typed
data['Epsilon'] = data['Epsilon'].astype(float)
data['Aug_Count'] = data['Aug_Count'].astype(int)
data['Mean_Accuracy_TTA'] = data['Mean_Accuracy_TTA'].astype(float)
data['Mean_Entropy'] = data['Mean_Entropy'].astype(float)

# Function to plot Accuracy or Uncertainty vs. Epsilon
def plot_metric_vs_epsilon(df, subject, metric, y_label, filename, y_min, y_max):
    plt.figure(figsize=(8, 6))
    
    # Plot a line for each Aug_Count
    for aug_count in aug_counts:
        subset = df[df['Aug_Count'] == aug_count]
        # Sort by Epsilon to ensure the line is drawn in order
        subset = subset.sort_values('Epsilon')
        plt.plot(subset['Epsilon'], subset[metric], 
                 label=f'Aug Count: {aug_count}', 
                 color=colors[aug_count], 
                 marker='o',  # Add markers to data points
                 linewidth=2, 
                 markersize=6)
    
    # Customize the plot
    plt.xlabel('Epsilon', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(f'{subject} {y_label} vs. Epsilon', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Augmentation Count', loc='best')
    plt.xticks(epsilons)
    plt.ylim(y_min, y_max)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Plot for Overall data
overall_data = data[data['Subject'] == 'Overall']
if overall_data.empty:
    print("Warning: No 'Overall' data found in the CSVs. Skipping overall plots.")
else:
    # Overall Accuracy vs. Epsilon
    plot_metric_vs_epsilon(
        overall_data,
        'Overall',
        'Mean_Accuracy_TTA',
        'Mean Accuracy (TTA)',
        'accuracy_vs_epsilon_overall.png',
        y_min=0.75,
        y_max=0.95
    )
    
    # Overall Uncertainty (Entropy) vs. Epsilon
    plot_metric_vs_epsilon(
        overall_data,
        'Overall',
        'Mean_Entropy',
        'Mean Entropy',
        'uncertainty_vs_epsilon_overall.png',
        y_min=1.3,
        y_max=1.5
    )

# Plot for each subject (1 to 9)
for subject in range(1, 10):
    subject_data = data[data['Subject'] == str(subject)]
    if subject_data.empty:
        print(f"Warning: No data found for Subject {subject}. Skipping plots for this subject.")
        continue
    
    # Subject-specific Accuracy vs. Epsilon
    plot_metric_vs_epsilon(
        subject_data,
        f'Subject {subject}',
        'Mean_Accuracy_TTA',
        'Mean Accuracy (TTA)',
        f'accuracy_vs_epsilon_subject_{subject}.png',
        y_min=0.75,
        y_max=0.95
    )
    
    # Subject-specific Uncertainty (Entropy) vs. Epsilon
    plot_metric_vs_epsilon(
        subject_data,
        f'Subject {subject}',
        'Mean_Entropy',
        'Mean Entropy',
        f'uncertainty_vs_epsilon_subject_{subject}.png',
        y_min=1.3,
        y_max=1.5
    )

print(f"Plots have been saved to {output_dir}")
