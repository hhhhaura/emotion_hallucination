import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

# 1. Define file paths (Update these to match your actual local filenames)
FILE_NONE = 'squad_v2.jsonl'
FILE_FEAR1 = 'squad_v2_fear_1.jsonl'
FILE_FEAR2 = 'squad_v2_fear_2.jsonl'
FILE_FEAR3 = 'squad_v2_fear_3.jsonl'

# 2. Data Loading Function
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            
            # Support both 'metrics' dict and the 'samples' array format
            if 'metrics' in obj:
                metrics = obj.get('metrics', {})
                if 'f1' in metrics:
                    score = metrics['f1']
                elif 'accuracy' in metrics:
                    score = metrics['accuracy']
                elif 'em' in metrics:
                    score = metrics['em']
                else:
                    score = 0.0
            elif 'samples' in obj:
                # Fallback format if metrics isn't explicitly defined
                score = np.mean([float(sample[1]) for sample in obj['samples']])
            else:
                score = 0.0
                
            # Normalize score to 0.0 - 1.0 if it is represented as 0 - 100
            if score > 1.0:
                score = score / 100.0
                
            data.append({
                'example_id': obj['example_id'],
                'score': score
            })
    return pd.DataFrame(data)

# 3. Load Datasets
print("Loading data...")
df_none = load_jsonl(FILE_NONE).rename(columns={'score': 'before'})
df_fear1 = load_jsonl(FILE_FEAR1).rename(columns={'score': 'fear_1'})
df_fear2 = load_jsonl(FILE_FEAR2).rename(columns={'score': 'fear_2'})
df_fear3 = load_jsonl(FILE_FEAR3).rename(columns={'score': 'fear_3'})

# 4. Merge all into one DataFrame based on example_id
df = df_none.merge(df_fear1, on='example_id')
df = df.merge(df_fear2, on='example_id')
df = df.merge(df_fear3, on='example_id')

# 5. Calculate 'after' score
df['after'] = df[['fear_1', 'fear_2', 'fear_3']].mean(axis=1)

print(f"Data merged successfully. Total examples processed: {len(df)}")
print("Generating plots...")

# Plotting styles
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# Plot 1: Scatter Plot (Before vs After)
# ==========================================
plt.figure(figsize=(8, 6))
plt.scatter(df['before'], df['after'], alpha=0.3, s=10, label='Data points')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2, label='x = y (No change)')
plt.title('Scatter Plot: Before vs After Score (SQuAD v2)')
plt.xlabel('Before Score')
plt.ylabel('After Score (Mean of Fear 1-3)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.savefig('squad_v2_scatter.png', dpi=300)
plt.close()

# ==========================================
# Plot 2: Histogram with 4 bars
# ==========================================
bins = np.arange(0, 1.1, 0.1)
labels = [f'{i:.1f}-{i+0.1:.1f}' for i in np.arange(0, 1.0, 0.1)]

def get_counts(series):
    counts, _ = np.histogram(series, bins=bins)
    return counts

counts_before = get_counts(df['before'])
counts_f1 = get_counts(df['fear_1'])
counts_f2 = get_counts(df['fear_2'])
counts_f3 = get_counts(df['fear_3'])

x = np.arange(len(labels))
width = 0.2

plt.figure(figsize=(12, 6))
plt.bar(x - 1.5*width, counts_before, width, label='Before')
plt.bar(x - 0.5*width, counts_f1, width, label='Fear 1')
plt.bar(x + 0.5*width, counts_f2, width, label='Fear 2')
plt.bar(x + 1.5*width, counts_f3, width, label='Fear 3')
plt.xticks(x, labels, rotation=45)
plt.xlabel('Score Range')
plt.ylabel('Count')
plt.title('Score Histogram (Before and 3 Fear variants - SQuAD v2)')
plt.legend()
plt.tight_layout()
plt.savefig('squad_v2_histogram.png', dpi=300)
plt.close()

# ==========================================
# Plot 3: Mean and Delta for each before accuracy bin
# ==========================================
df['bin'] = pd.cut(df['before'], bins=bins, right=False, include_lowest=True, labels=labels)
# 確保滿分 1.0 會被歸類到最後一個 bin
df.loc[df['before'] == 1.0, 'bin'] = labels[-1]

grouped = df.groupby('bin', observed=False)
means = grouped['after'].mean().fillna(0)
deltas = (means - grouped['before'].mean().fillna(0)).fillna(0)

plt.figure(figsize=(10, 6))
x_bin = np.arange(len(labels))
plt.bar(x_bin - 0.2, means, 0.4, label='Mean After Score')
plt.bar(x_bin + 0.2, deltas, 0.4, label='Delta (After - Before)')
plt.xticks(x_bin, labels, rotation=45)
plt.axhline(0, color='black', linewidth=1)
plt.xlabel('Before Score Range')
plt.ylabel('Value')
plt.title('Mean After Score and Delta by Before Score Range (SQuAD v2)')
plt.legend()
plt.tight_layout()
plt.savefig('squad_v2_mean_delta.png', dpi=300)
plt.close()

print("Plots generated successfully.")
print(f"SQuAD v2 整體平均 Before: {df['before'].mean():.4f}")
print(f"SQuAD v2 整體平均 After:  {df['after'].mean():.4f}")
print(f"SQuAD v2 整體平均 Delta:  {df['after'].mean() - df['before'].mean():.4f}")