#!/usr/bin/env python3
# Script for neural data analysis

# Author: Virginia Casasnovas
# Date: 2025-07-10

#%% --- SECTION: Imports ---
import numpy as np
import pandas as pd
import os, glob
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
from neural_functions import *

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm

#%% --- SECTION: Inits ---
data_root = '/Users/virginia/Desktop/'

meta = {
    'project' : '',
    'task' : 'BMI',
    'taskAlias' : 'pool_py',
    'subject': 'yod'
        }
meta['superfolder_pool'] = os.path.join(data_root, meta['task'], meta['taskAlias'], 'pool')
meta['superfolder'] = os.path.join(data_root, meta['task'], meta['taskAlias'], meta['subject'])
mky_lab = {'yod': 'MY', 'zep': 'MZ'}

# Saving figures
meta['figfolder'] = os.path.join(data_root, meta['task'], meta['taskAlias'], 'Figs')
save_figs = 0
fmt = 'pdf'

spike_files = sorted(glob.glob(os.path.join(meta['superfolder'], "*_spikes_s.csv")))
task_files = sorted(glob.glob(os.path.join(meta['superfolder'], "*_BCI_Task.txt")))

my_pal = {"CRS": [117/255, 107/255, 177/255], "CLD": [188/255, 189/255, 220/255]}
my_pal2 = {"CRS": [190/255, 48/255, 54/255], "CLD": [44/255, 117/255, 179/255]}
unc_lab = {False: 'CRS', True: 'CLD'}

#%% Compute or load DataFrame
all_mky = True # Load Dataframe from both monkeys
overwrite = False # Overwrite DataFrame
file_path = os.path.join(meta['superfolder'], 'df_clf_M'+meta['subject'][0].upper()+'.pkl')

if not os.path.exists(file_path) or overwrite:

    # Load and transform data into DataFrame
    df_list = []

    for file_idx, (spike_file, task_file) in enumerate(tqdm(zip(spike_files, task_files), total=len(spike_files))):
        spike_data = pd.read_csv(spike_file).to_numpy()
        task_data = task_selection(task_file, file_idx)
        spk, dens, target, uncert = extract_trial_data(spike_data, task_data, t_start_add=-0.2, t_end_add=0.6)
        df_file = pd.DataFrame({
            'monkey': mky_lab[task_data.monkey.values[0]],
            'file_index': file_idx,
            'uncertainty': uncert,
            'target_direction': target,
            'density': dens})
        
        neuron_filter = df_file.groupby(['target_direction']).agg(
            density=('density', lambda group: np.stack(group).mean(axis=0).mean(axis=1))).reset_index()
        neuron_filter = np.max(np.stack(neuron_filter['density'].values), axis=0) > 2
        df_file['density'] = df_file['density'].apply(lambda x: x[neuron_filter, :])

        df_list.append(df_file)

    df = pd.concat(df_list, ignore_index=True)

    # Save DataFrame
    with open(os.path.join(meta['superfolder'], 'df_clf_M'+meta['subject'][0].upper()+'.pkl'), 'wb') as file: 
        pickle.dump(df, file) 
else:
    if all_mky:
       # Load DataFrames from both monkeys 
        with open(os.path.join(data_root, meta['task'], meta['taskAlias'], 'yod', 'df_clf_MY.pkl'), 'rb') as file:
            df_y = pickle.load(file)
            
        with open(os.path.join(data_root, meta['task'], meta['taskAlias'], 'zep', 'df_clf_MZ.pkl'), 'rb') as file:
                df_z = pickle.load(file)

        df = pd.concat([df_y, df_z], ignore_index=True)

        monkey = ['MY', 'MZ']
    else:
        # Load DataFrame from one monkey
        with open(os.path.join(meta['superfolder'], 'df_clf_M'+meta['subject'][0].upper()+'.pkl'), 'wb') as file: 
            df = pickle.load(file) 

#%% Number of trials per condition
df_counts = df.groupby(['monkey', 'uncertainty', 'target_direction'])['density'].count()
print(df_counts)

#%% PLOT: single-neuron response, trial average
sel_mky = 'MY'
sel_file = 0
sel = (df['monkey'] == sel_mky) & (df['file_index'] == sel_file)

sn_trial_average = df.loc[sel].groupby(['target_direction', 'uncertainty']).agg(
    density=('density', lambda group: np.stack(group).mean(axis=0))).reset_index()

neu_plt = 1
time = np.arange(-200, 605, 5)
tick_positions = [-200, 0, 200, 400, 600]
tick_labels = ['-200', '0', '200', '400', '600']

fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True, sharey=True)
for i, u in enumerate(sn_trial_average['uncertainty'].unique()):
    for t in sn_trial_average['target_direction'].unique():
        sel = (sn_trial_average.target_direction == t) & (sn_trial_average.uncertainty == u)
        data_plt = sn_trial_average.loc[sel, 'density'].values[0]
        data_plt = data_plt[neu_plt, :]
        ax[i].plot(time, data_plt, linewidth=2)
    ax[i].spines[['right', 'top']].set_visible(False)
    ax[i].spines[['left', 'bottom']].set_linewidth(1)
    ax[i].tick_params(axis='both', labelsize=13, width=1, length=3)
    ax[i].set_xticks(tick_positions)
    ax[i].set_xticklabels(tick_labels)
    ax[i].margins(0.05, 0.05)
    ax[i].set_title(unc_lab[u], fontsize=14, fontweight='bold')
    ax[i].set_xlabel('Time from go cue [ms]', fontsize=14)
    if i == 0:
        ax[i].set_ylabel('Firing rate [Hz]', fontsize=14)

#%% PLOT: Neural population response, trial average 
sel_mky = 'MY'
sel_file = 0
sel = (df['monkey'] == sel_mky) & (df['file_index'] == sel_file)

pp_trial_average = df.loc[sel].groupby(['target_direction', 'uncertainty']).agg(
    density=('density', lambda group: np.stack(group).mean(axis=0).mean(axis=0))).reset_index()

fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True, sharey=True)
for i, u in enumerate(pp_trial_average['uncertainty'].unique()):
    for t in pp_trial_average['target_direction'].unique():
        sel = (pp_trial_average.target_direction == t) & (pp_trial_average.uncertainty == u)
        data_plt = pp_trial_average.loc[sel, 'density'].values[0]
        ax[i].plot(time, data_plt, linewidth=2)
    ax[i].spines[['right', 'top']].set_visible(False)
    ax[i].spines[['left', 'bottom']].set_linewidth(1)
    ax[i].tick_params(axis='both', labelsize=13, width=1, length=3)
    ax[i].set_xticks(tick_positions)
    ax[i].set_xticklabels(tick_labels)
    ax[i].margins(0.05, 0.05)
    ax[i].set_title(unc_lab[u], fontsize=14, fontweight='bold')
    ax[i].set_xlabel('Time from go cue [ms]', fontsize=14)
    if i == 0:
        ax[i].set_ylabel('Firing rate [Hz]', fontsize=14)

#%% Within-condition classification
clf = make_pipeline(StandardScaler(), svm.SVC(kernel='linear', random_state=42))

all_scores_list = []
all_clf_list = []

for m in monkey:
    print(f"Monkey: {m}")
    df_mky = df.loc[df['monkey'] == m]

    for file_idx in df_mky['file_index'].unique():
        print(f"File index: {file_idx}")
        df_file = df_mky.loc[df_mky['file_index'] == file_idx]

        for unc in [False, True]:
            mean_scores, all_scores = classify_neural_window(
                df_file.loc[df_file['uncertainty'] == unc], 
                'density', clf)
    
            n_folds, n_windows = all_scores.shape
            for fold in range(n_folds):
                for t, time in enumerate(np.arange(-150, 550, 50)):
                    all_scores_list.append({
                        'monkey': m,
                        'file_index': file_idx,
                        'uncertainty': unc_lab[unc],
                        'fold': fold,
                        'time': time,
                        'score': all_scores[fold, t]*100})
        
scores_df = pd.DataFrame(all_scores_list)

#%% PLOT: Within-condition classifier performance
scores_df_mean = scores_df.groupby(by=['monkey', 'file_index', 'uncertainty', 'time'], as_index=False)[['score']].mean()

tick_positions = [-200, 0, 200, 400, 600]
tick_labels = ['-200', '0', '200', '400', '600']

# For each monkey 
fig, ax = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True, sharey=True)
for i, m in enumerate(scores_df_mean['monkey'].unique()):
    data_plt = scores_df_mean.loc[scores_df_mean['monkey'] == m]

    ax[i].axvline(color=[0.3, 0.3, 0.3], linewidth=1) 
    sns.lineplot(data=data_plt, x='time', y='score', hue='uncertainty', 
                 hue_order=('CRS','CLD'), palette=my_pal, linewidth=2, ax=ax[i])
    ax[i].spines[['right', 'top']].set_visible(False)
    ax[i].spines[['left', 'bottom']].set_linewidth(1)
    ax[i].tick_params(axis='both', labelsize=13, width=1, length=3, direction='in')
    ax[i].set_xticks(tick_positions)
    ax[i].set_xticklabels(tick_labels)
    ax[i].margins(0.05, 0.05)
    ax[i].set_ylim([0, 100])
    ax[i].set_title(m, fontsize=14, fontweight='bold')
    ax[i].set_xlabel('Time from go cue [ms]', fontsize=14)
    if i == 0:
        ax[i].set_ylabel('Performance [%]', fontsize=14)
        ax[i].get_legend().remove()
    else:
        ax[i].legend(loc='upper right', frameon=False)

# Pooling both monkeys
plt.figure(figsize=(4, 3))
ax = plt.gca()
ax.axvline(color=[0.3, 0.3, 0.3], linewidth=1) 
sns.lineplot(data=data_plt, x='time', y='score', hue='uncertainty', 
                hue_order=('CRS','CLD'), palette=my_pal, linewidth=2)
ax.spines[['right', 'top']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(1)
ax.tick_params(axis='both', labelsize=13, width=1, length=3, direction='in')
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)
ax.margins(0.05, 0.05)
ax.set_ylim([0, 100])
ax.set_title('Within-level', fontsize=14, fontweight='bold')
ax.set_xlabel('Time from go cue [ms]', fontsize=14)
ax.set_ylabel('Performance [%]', fontsize=14)
ax.legend(loc='upper right', frameon=False)

#%% Between-condition classification
clf = make_pipeline(StandardScaler(), svm.SVC(kernel='linear', random_state=42))

mean_scores_list = []

df['density_go'] = df['density'].apply(lambda x: x[:, 10:40])

for m in monkey:
    print(f"Monkey: {m}")
    df_mky = df.loc[df['monkey'] == m]

    for file_idx in df_mky['file_index'].unique():
        print(f"File index: {file_idx}")
        df_file = df_mky.loc[df_mky['file_index'] == file_idx]

        mean_scores_same, mean_scores_cross = classify_neural_cross(
            df_file.loc[df_file['uncertainty'] == False], 
            df_file.loc[df_file['uncertainty'] == True],
            'density_go', clf)
        
        mean_scores_list.append({
                    'monkey': m,
                    'file_index': file_idx,
                    'score_same': mean_scores_same*100,
                    'score_cross': mean_scores_cross*100})
        
scores_df = pd.DataFrame(mean_scores_list)
scores_df['score_diff'] = scores_df['score_cross'] - scores_df['score_same']

#%% PLOT: Between-condition classifier performance difference
# For each monkey 
plt.figure(figsize=(3, 4))
ax = plt.gca()
ax.axhline(color=(0.3, 0.3, 0.3), linewidth=1) 
sns.barplot(data=scores_df, x='monkey', y='score_diff', 
            color=my_pal['CRS'], width=0.6, capsize=0.1, err_kws={'linewidth': 1.5})
sns.stripplot(data=scores_df, x='monkey', y='score_diff',
              color='k', size=4, alpha=0.6, 
              jitter=0.15, legend=False, dodge=True)
ax.spines[['right', 'top']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(1)
ax.tick_params(axis='both', labelsize=13, width=1, length=3, direction='in')
ax.margins(0.2, 0.2)
#ax.set_ylim([-30, 20])
ax.set_title('Cross-level', fontsize=14, fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('Performance diff. [%]', fontsize=14)
