import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import ast
import re
import matplotlib

# Use a clean, minimalistic style
matplotlib.rcParams.update({
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 12,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

# Define paths
input_dir = "/home/teaching/Nikhil/ATCNet-EMD-TTA/eeg_results/"
output_dir = "/home/teaching/Nikhil/ATCNet-EMD-TTA/final_plot/"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load all per-run CSV files (test_results_*.csv, exclude .txt)
csv_pattern = os.path.join(input_dir, "test_results_*.csv")
csv_files = [f for f in glob.glob(csv_pattern) if f.endswith('.csv')]

if not csv_files:
    raise FileNotFoundError(f"No test_results CSV files found in {input_dir}")

# Combine all CSV files into a single DataFrame
all_data = []
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        all_data.append(df)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        continue

if not all_data:
    raise ValueError("No valid CSV files could be read.")

all_data = pd.concat(all_data, ignore_index=True)

# Add Subject column (Best_Run corresponds to Subject)
all_data['Subject'] = all_data['Best_Run'].astype(str)

# Debug: Print unique Best_Run values and sample data
print("Unique Best_Run values before filtering:", sorted(all_data['Best_Run'].unique()))
print("Sample of all_data before filtering:")
print(all_data[['Best_Run', 'Epsilon', 'Aug_Count', 'Accuracy_TTA', 'Entropy']].head())

# Exclude Subject 10 (non-existent)
all_data = all_data[all_data['Subject'].astype(int).between(1, 9)]
print("Unique Subjects after excluding Subject 10:", sorted(all_data['Subject'].unique()))

# Parse Aug_Count from string list to total integer
def parse_aug_count(aug_count_str):
    try:
        cleaned_str = re.sub(r'[^0-9,\[\]\s]', '', aug_count_str)
        aug_list = ast.literal_eval(cleaned_str)
        return sum(aug_list)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing Aug_Count '{aug_count_str}': {e}")
        return None

all_data['Aug_Count'] = all_data['Aug_Count'].apply(parse_aug_count)

# Debug: Print data for Subjects 5, 7, 8 before any filtering
for subject in ['5', '7', '8']:
    subject_data = all_data[all_data['Subject'] == subject]
    if not subject_data.empty:
        print(f"Data found for Subject {subject} before filtering:")
        print(subject_data[['Best_Run', 'Epsilon', 'Aug_Count', 'Accuracy_TTA', 'Entropy']].head())
    else:
        print(f"No data found for Subject {subject} before filtering.")

# Ensure numeric columns are properly typed
all_data['Epsilon'] = pd.to_numeric(all_data['Epsilon'], errors='coerce')
all_data['Accuracy_TTA'] = pd.to_numeric(all_data['Accuracy_TTA'], errors='coerce')
all_data['Entropy'] = pd.to_numeric(all_data['Entropy'], errors='coerce')

# Drop rows with NaN in critical columns (instead of filling)
required_cols = ['Epsilon', 'Accuracy_TTA', 'Entropy', 'Aug_Count']
all_data = all_data.dropna(subset=required_cols)

# Debug: Check data after dropping NaNs
for subject in ['5', '7', '8']:
    subject_data = all_data[all_data['Subject'] == subject]
    if not subject_data.empty:
        print(f"Data found for Subject {subject} after dropping NaNs:")
        print(subject_data[['Best_Run', 'Epsilon', 'Aug_Count', 'Accuracy_TTA', 'Entropy']].head())
    else:
        print(f"No data found for Subject {subject} after dropping NaNs.")

# Prepare averaged data
avg_data = all_data.groupby(['Aug_Count', 'Epsilon']).agg({
    'Accuracy_TTA': 'mean',
    'Entropy': 'mean'
}).reset_index()

avg_data.rename(columns={
    'Accuracy_TTA': 'Mean_Accuracy_TTA',
    'Entropy': 'Mean_Entropy'
}, inplace=True)

# Print columns and unique values for debugging
print("Columns in aggregated DataFrame:", avg_data.columns.tolist())
print("Unique Aug_Count values:", sorted(avg_data['Aug_Count'].unique()))
print("Unique Epsilon values:", sorted(avg_data['Epsilon'].unique()))

# Function for averaged plots
def plot_avg(data, x_col, y_col, line_col, title, filename):
    plt.figure(figsize=(10, 6))
    for value in sorted(data[line_col].unique()):
        subset = data[data[line_col] == value]
        subset = subset.sort_values(x_col)
        plt.plot(subset[x_col], subset[y_col], label=f'{line_col}={value}', marker='o', linewidth=2, markersize=8)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    if 'Accuracy' in y_col:
        plt.ylim(0.6, 1.0)
    else:
        plt.ylim(0.3, 1.0)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Generate averaged plots with clear names
plot_avg(avg_data, 'Aug_Count', 'Mean_Accuracy_TTA', 'Epsilon',
         'Overall Accuracy vs Augmentation Count',
         'Overall_Accuracy_vs_AugCount_Epsilon_Lines.png')
plot_avg(avg_data, 'Epsilon', 'Mean_Accuracy_TTA', 'Aug_Count',
         'Overall Accuracy vs Epsilon',
         'Overall_Accuracy_vs_Epsilon_AugCount_Lines.png')
plot_avg(avg_data, 'Aug_Count', 'Mean_Entropy', 'Epsilon',
         'Overall Uncertainty vs Augmentation Count',
         'Overall_Entropy_vs_AugCount_Epsilon_Lines.png')
plot_avg(avg_data, 'Epsilon', 'Mean_Entropy', 'Aug_Count',
         'Overall Uncertainty vs Epsilon',
         'Overall_Entropy_vs_Epsilon_AugCount_Lines.png')

# Function for subject-wise plots
def plot_subject(subject_data, subject, x_col, y_col, line_col, title, filename):
    plt.figure(figsize=(10, 6))
    for value in sorted(subject_data[line_col].unique()):
        subset = subject_data[subject_data[line_col] == value]
        subset = subset.sort_values(x_col)
        plt.plot(subset[x_col], subset[y_col], label=f'{line_col}={value}', marker='o', linewidth=2, markersize=8)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    if 'Accuracy' in y_col:
        plt.ylim(0.6, 1.0)
    else:
        plt.ylim(0.3, 1.0)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Generate subject-wise plots with clear names
for subject in sorted(all_data['Subject'].unique()):
    subject_data = all_data[all_data['Subject'] == subject]
    subject_data = subject_data.groupby(['Subject', 'Aug_Count', 'Epsilon']).agg({
        'Accuracy_TTA': 'mean',
        'Entropy': 'mean'
    }).reset_index()
    subject_data.rename(columns={
        'Accuracy_TTA': 'Mean_Accuracy_TTA',
        'Entropy': 'Mean_Entropy'
    }, inplace=True)
    if subject_data.empty:
        print(f"Warning: No data found for Subject {subject}. Skipping plots for this subject.")
        continue
    
    plot_subject(subject_data, subject, 'Aug_Count', 'Mean_Accuracy_TTA', 'Epsilon',
                 f'Subject {subject} Accuracy vs Augmentation Count',
                 f'Subject_{subject}_Accuracy_vs_AugCount_Epsilon_Lines.png')
    plot_subject(subject_data, subject, 'Epsilon', 'Mean_Accuracy_TTA', 'Aug_Count',
                 f'Subject {subject} Accuracy vs Epsilon',
                 f'Subject_{subject}_Accuracy_vs_Epsilon_AugCount_Lines.png')
    plot_subject(subject_data, subject, 'Aug_Count', 'Mean_Entropy', 'Epsilon',
                 f'Subject {subject} Uncertainty vs Augmentation Count',
                 f'Subject_{subject}_Entropy_vs_AugCount_Epsilon_Lines.png')
    plot_subject(subject_data, subject, 'Epsilon', 'Mean_Entropy', 'Aug_Count',
                 f'Subject {subject} Uncertainty vs Epsilon',
                 f'Subject_{subject}_Entropy_vs_Epsilon_AugCount_Lines.png')

# Specific Plot: Average Accuracy vs Aug_Count for Epsilon=0.01
def plot_specific_epsilon(avg_data, epsilon=0.01):
    plt.figure(figsize=(10, 6))
    subset = avg_data[avg_data['Epsilon'] == epsilon]
    subset = subset.sort_values('Aug_Count')
    plt.plot(subset['Aug_Count'], subset['Mean_Accuracy_TTA'], label=f'Epsilon={epsilon}', marker='o', linewidth=2, markersize=8)
    plt.xlabel('Aug_Count')
    plt.ylabel('Average Accuracy (TTA)')
    plt.title(f'Overall Accuracy vs Augmentation Count for Epsilon={epsilon}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.ylim(0.6, 1.0)
    plt.savefig(os.path.join(output_dir, 'Overall_Accuracy_vs_AugCount_Epsilon_0.01.png'), dpi=300, bbox_inches='tight')
    plt.close()

plot_specific_epsilon(avg_data, epsilon=0.01)

print(f"All plots have been saved to {output_dir}")
