#!/usr/bin/env python3
# Script with BCI progress analysis

# Author: Virginia Casasnovas
# Date: 2024-02-07

#%% --- SECTION: Imports ---
import numpy as np
import pandas as pd
import os, glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors   
import seaborn as sns 
import pickle
from utils import *
from neural_functions import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- SECTION: Inits ---
# Path definition
data_root = '/Users/virginia/Desktop/'

meta = {
    'project' : '',
    'task' : 'BMI',
    'taskAlias' : 'pool_py',
    'subject': 'zep'
        }
meta['superfolder_pool'] = os.path.join(data_root, meta['task'], meta['taskAlias'], 'pool')
meta['superfolder'] = os.path.join(data_root, meta['task'], meta['taskAlias'], meta['subject'])
monkey = [meta['subject']]

# Saving figures
meta['figfolder'] = os.path.join(data_root, meta['task'], meta['taskAlias'], 'Figs')
save_figs = 0
fmt = 'pdf'

# Color palettes
my_pal = {"CRS": [1, 94/255, 105/255], "CLD": [0, 169/255, 1], "TGT": [7/255, 182/255, 75/255]}
my_pal2 = {"CRS": [190/255, 48/255, 54/255], "CLD": [44/255, 117/255, 179/255]}
cm = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
cm = [mcolors.to_rgb(i) for i in cm]

ctm = 1/2.54
    
#%% Load data paths
file_spikes = sorted(glob.glob(os.path.join(meta['superfolder'], "*_spikes_s.csv")))
file_decoder = sorted(glob.glob(os.path.join(meta['superfolder'], "*_calibrated_values.mat")))
file_tracker = sorted(glob.glob(os.path.join(meta['superfolder'], "*_BCI_trackers.txt")))
file_task = sorted(glob.glob(os.path.join(meta['superfolder'], "*_BCI_Task.txt")))

# Load and select task and trackers data
task_dfs = [task_selection(file, idx) for idx, file in enumerate(file_task)]
pos_bci, vel_bci = zip(*[trackers_selection(file) for file in file_tracker])

# Reconstruct position using KF parameters and process task dataframe        
pos_true, pos_rec, uncert, task_dfs_proc = zip(*[reconstruct_KF(t, s, d, p, v) for t, s, d, p, v in zip(task_dfs, file_spikes, file_decoder, pos_bci, vel_bci)])

# Steady-state Kalman gain
Ks = [get_kss(file) for file in file_decoder]   

# Get progress metrics
df_a = [get_progress_ss(df, spk, ks) for df, spk, ks in zip(task_dfs_proc, file_spikes, Ks)]
df_a = [get_progress_time(df, spk, dec) for df, spk, dec in zip(task_dfs_proc, file_spikes, file_decoder)]
df_a = [get_progress_angles(df) for df in df_a]
df_a = [bin_traj(df) for df in df_a]

# Change monkey name
df_a = [change_mky_name(df) for df in df_a]

# First submovement detection
df_first_sbm = [get_submov_arc_df(df) for df in df_a]
df_first_sbm = pd.concat(df_first_sbm, axis=0, ignore_index=True)

# Submovement detection and neighboring points
df_a = [get_submov_spd_df(df) for df in df_a]
df_submov_nn = [get_submov_nn(df) for df in df_a]
df_submov_nn = pd.concat(df_submov_nn, axis=0, ignore_index=True)

# Tortuosity and timings (reaction time, movement time, late movement time), and binning by median
df_behav = [bin_tortuosity(df) for df in df_a]
df_behav = [bin_RT_MT(df) for df in df_behav]
df_behav = pd.concat(df_behav, axis=0, ignore_index=True)

monkey = list(df_behav['monkey'].unique())

#%% Save dataframe
with open(os.path.join(meta['superfolder'], 'df_MZ.pkl'), 'wb') as file: 
    pickle.dump([df_a, df_first_sbm, df_submov_nn, df_behav], file) 
    
#%% Load and merge dataframes
with open(os.path.join(data_root, meta['task'], meta['taskAlias'], 'yod', 'df_MY.pkl'), 'rb') as file:
        df_a_y, df_first_sbm_y, df_submov_nn_y, df_behav_y = pickle.load(file)
        
with open(os.path.join(data_root, meta['task'], meta['taskAlias'], 'zep', 'df_MZ.pkl'), 'rb') as file:
        df_a_z, df_first_sbm_z, df_submov_nn_z, df_behav_z = pickle.load(file)

df_a_m = df_a_y + df_a_z
df_first_sbm_m = pd.concat([df_first_sbm_y, df_first_sbm_z], ignore_index=True)
df_submov_nn_m = pd.concat([df_submov_nn_y, df_submov_nn_z], ignore_index=True)
df_behav_m = pd.concat([df_behav_y, df_behav_z], ignore_index=True)
        
#%% Save merged dataframe
with open(os.path.join(meta['superfolder_pool'], 'df_m.pkl'), 'wb') as file: 
    pickle.dump([df_a_m, df_first_sbm_m, df_submov_nn_m, df_behav_m], file) 
    
#%% Load merged dataframe
with open(os.path.join(meta['superfolder_pool'], 'df_m.pkl'),'rb') as file:
        df_a, df_first_sbm, df_submov_nn, df_behav = pd.read_pickle(file)

monkey = ['MY','MZ']

#%% --- SECTION: Single-trial plots ---
#%% Pos real and reconstructed, check
for f in range(0, len(pos_true)):
    plt.figure(figsize=(8, 8))
    for i in range(0, len(pos_true[f])):  
        plt.axis('square')
        plt.plot(pos_true[f][i][:, 0], pos_true[f][i][:, 1], color='r')
        plt.plot(pos_rec[f][i][:, 0], pos_rec[f][i][:, 1], color='b')

#%% Trajectories (2D plot) with neural projection and cursor-to-target vector
hit_num = 360
for f in range(0, 9):
    plt.figure(figsize=(8, 8))
    
    hit_sel = df_a[f]['hit_number'] == hit_num
    
    if any(hit_sel):
        px = df_a[f].loc[hit_sel, 'pos_x'].values
        py = df_a[f].loc[hit_sel, 'pos_y'].values
        
        pxp = px + df_a[f].loc[hit_sel, 'proj_x'].values/10
        pyp = py + df_a[f].loc[hit_sel, 'proj_y'].values/10
        
        pxct = px + df_a[f].loc[hit_sel, 'ct_x'].values*5
        pyct = py + df_a[f].loc[hit_sel, 'ct_y'].values*5
        
        tx = df_a[f].loc[hit_sel, 'tgt_x'].values[0]
        ty = df_a[f].loc[hit_sel, 'tgt_y'].values[0]
        
        t = df_a[f].loc[hit_sel, 'time'].values
        
        plt.plot(px, py, color='k', label='Trajectory')
        plt.scatter(tx, ty, color='k', marker=('x'), label='Target')
        
        tgt = plt.Circle((tx, ty), radius=22.5, fill=False, color='k')
        tgt2 = plt.Circle((tx, ty), radius=15, fill=False, color='k')
        plt.gca().add_patch(tgt)
        plt.gca().add_patch(tgt2)
    
        for i in range(0, len(px)):
            plt.plot([px[i], pxp[i]], [py[i], pyp[i]], color='b',)
            plt.plot([px[i], pxct[i]], [py[i], pyct[i]], color='g')
        
        plt.title(f'Session {f}, Hit #{hit_num}')
        plt.axis('square')
    
#%% Trajectories (2D plot), speed in time and progress in time (1D plots)
hit_num = 280
for f in range(0, len(df_a)):   
    hit_sel = df_a[f]['hit_number'] == hit_num
    
    if any(hit_sel):
        unc = df_a[f].loc[hit_sel, 'uncertainty'].values[0]
        spd = df_a[f].loc[hit_sel, 'speed'].values
        px = df_a[f].loc[hit_sel, 'pos_x'].values
        py = df_a[f].loc[hit_sel, 'pos_y'].values
        t = df_a[f].loc[hit_sel, 'time'].values
        tx = df_a[f].loc[hit_sel, 'tgt_x'].values[0]
        ty = df_a[f].loc[hit_sel, 'tgt_y'].values[0]
        tgt_traj_ang = df_a[f].loc[hit_sel, 'tgt_traj_ang'].values
        dist = np.sqrt(px**2+py**2)
        arclen = dist*np.deg2rad(tgt_traj_ang)

        pxp = px + df_a[f].loc[hit_sel, 'proj_x'].values/10
        pyp = py + df_a[f].loc[hit_sel, 'proj_y'].values/10
        
        pxct = px + df_a[f].loc[hit_sel, 'ct_x'].values*5
        pyct = py + df_a[f].loc[hit_sel, 'ct_y'].values*5
        
        pg = df_a[f].loc[hit_sel, 'prog'].values

        sbm_sel = df_a[f]['is_sbm']
        
        pxs = df_a[f].loc[(hit_sel & sbm_sel), 'pos_x'].values
        pys = df_a[f].loc[(hit_sel & sbm_sel), 'pos_y'].values
        
        ts = df_a[f].loc[(hit_sel & sbm_sel), 'time'].values
        spds = df_a[f].loc[(hit_sel & sbm_sel), 'speed'].values 
        pgs = df_a[f].loc[(hit_sel & sbm_sel), 'progn'].values

        
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.scatter(px, py, c=spd)
        plt.scatter(tx, ty, color='k', marker=('x'))
        plt.scatter(pxs, pys, marker='x', color='r')
        tgt = plt.Circle((tx, ty), radius=22.5, fill=False, color='k')
        plt.gca().add_patch(tgt)
        plt.axis('square')   
        for i in range(0, len(px)):
            plt.plot([px[i], pxp[i]], [py[i], pyp[i]], color='b')
            plt.plot([px[i], pxct[i]], [py[i], pyct[i]], color='g')
        plt.ylabel(unc)

        ax = plt.subplot(1, 3, 2)
        plt.plot(t, spd, color='k', zorder=1)
        plt.scatter(t, spd, c=spd, zorder=2)                        
        plt.scatter(ts, spds, marker='x', color='r', zorder=3)
        ax.spines[['right', 'top']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(1)
        ax.tick_params(axis='both', labelsize=12)
        plt.xlabel('Time from go cue [ms]', fontsize=14)
        plt.ylabel('Speed [mm/s]', fontsize=14)
        
        ax = plt.subplot(1, 3, 3)
        plt.plot(t, pg, color='k', zorder=1)
        plt.scatter(t, pg, c=spd, zorder=2)                        
        plt.scatter(ts, pgs, marker='x', color='r', zorder=3)
        ax.spines[['right', 'top']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(1)
        ax.tick_params(axis='both', labelsize=12)
        plt.xlabel('Time from go cue [ms]', fontsize=14)
        plt.ylabel('Progress [mm/s]', fontsize=14)

#%% EXAMPLE (yod): Trajectory (2D plot), speed in time 
# and angle between trajectory and target direction in time (1D plots)
f = 7
hit_sel = df_a[f]['hit_number'] == 208
    
unc = df_a[f].loc[hit_sel, 'fb_uncert'].values[0]
spd = df_a[f].loc[hit_sel, 'speed'].values
px = df_a[f].loc[hit_sel, 'pos_x'].values
py = df_a[f].loc[hit_sel, 'pos_y'].values
t = df_a[f].loc[hit_sel, 'time'].values
tx = df_a[f].loc[hit_sel, 'tgt_x'].values[0]
ty = df_a[f].loc[hit_sel, 'tgt_y'].values[0]
tgt_traj_ang = df_a[f].loc[hit_sel, 'tgt_traj_ang'].values

pxp = df_a[f].loc[hit_sel, 'proj_x'].values/10
pyp =  df_a[f].loc[hit_sel, 'proj_y'].values/10

pxct = df_a[f].loc[hit_sel, 'ct_x'].values*6
pyct = df_a[f].loc[hit_sel, 'ct_y'].values*6

sbm_sel = df_a[f]['is_sbm']
sbm_sel_d = df_a[f]['is_sbm_d']   
mov_sel = df_a[f]['is_mov']        

pxs = df_a[f].loc[(hit_sel & sbm_sel), 'pos_x'].values
pys = df_a[f].loc[(hit_sel & sbm_sel), 'pos_y'].values

ts = df_a[f].loc[(hit_sel & sbm_sel), 'time'].values
spds = df_a[f].loc[(hit_sel & sbm_sel), 'speed'].values

tm = df_a[f].loc[(hit_sel & mov_sel), 'time'].values
spdm = df_a[f].loc[(hit_sel & mov_sel), 'speed'].values
tgt_traj_angm = df_a[f].loc[(hit_sel & mov_sel), 'tgt_traj_ang'].values
tgt_traj_angms = df_a[f].loc[(hit_sel & mov_sel & sbm_sel), 'tgt_traj_ang'].values

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.scatter(px, py, c=spd, s=10, zorder=3)
plt.scatter(tx, ty, color='k', marker=('x'))
plt.scatter(pxs, pys, c='r', marker=('x'), s=10, zorder=4)
tgt = plt.Circle((tx, ty), radius=22.5, fill=False, color='k')
plt.gca().add_patch(tgt)
plt.xlabel('px [mm]', fontsize=14)
plt.ylabel('py [mm]', fontsize=14)
plt.axis('square')   
for i in range(0,len(px)):
    plt.arrow(px[i], py[i], pxp[i], pyp[i], color=[112/256, 48/256, 161/256], length_includes_head=True, head_width=0.8)
    plt.arrow(px[i], py[i], pxct[i], pyct[i], color=[0.7, 0.7, 0.7], length_includes_head=True, head_width=0.8)

ax = plt.subplot(1, 3, 2)
plt.plot(t, spd, color='k', zorder=1)
plt.scatter(t, spd, c=spd, zorder=2)                        
plt.scatter(ts, spds, marker='x', color='r', zorder=3)
ax.spines[['right', 'top']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(1)
ax.tick_params(axis='both', labelsize=12)
plt.xlabel('Time from go cue [ms]', fontsize=14)
plt.ylabel('Speed [mm/s]', fontsize=14)

ax = plt.subplot(1, 3, 3)
plt.plot(tm, tgt_traj_angm, color='k', zorder=1)
plt.scatter(tm, tgt_traj_angm, c=spdm, zorder=2)                        
plt.scatter(ts, tgt_traj_angms, marker='x', color='r', zorder=3)
ax.spines[['right', 'top']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(1)
ax.tick_params(axis='both', labelsize=12)
plt.xlabel('Time from go cue [ms]', fontsize=14)
plt.ylabel('Cursor to target direction angle [deg]', fontsize=14) 

if save_figs:
    fig_path = os.path.join(meta['figfolder'], f"traj_ex_{meta['subject'][0].upper()}.{fmt}")
    plt.savefig(fig_path, format=fmt, dpi=1200)

#%% EXAMPLE (yod): Trajectory (2D plot)
f = 7
hit_sel = df_a[f]['hit_number'] == 208
   
unc = df_a[f].loc[hit_sel, 'fb_uncert'].values[0]
px = df_a[f].loc[hit_sel, 'pos_x'].values
py = df_a[f].loc[hit_sel, 'pos_y'].values
tx = df_a[f].loc[hit_sel, 'tgt_x'].values[0]
ty = df_a[f].loc[hit_sel, 'tgt_y'].values[0]

pxp = df_a[f].loc[hit_sel, 'proj_x'].values/10
pyp =  df_a[f].loc[hit_sel, 'proj_y'].values/10

pxct = df_a[f].loc[hit_sel, 'ct_x'].values*6
pyct = df_a[f].loc[hit_sel, 'ct_y'].values*6

plt.figure(figsize=(8, 8))
plt.scatter(px, py, c='k', s=8, zorder=3)
plt.scatter(tx, ty, color='k', marker=('x'))
tgt = plt.Circle((tx, ty), radius=22.5, fill=False, color='k')
plt.gca().add_patch(tgt)
plt.xlabel('px [mm]', fontsize=14)
plt.ylabel('py [mm]', fontsize=14)
plt.axis('square')   
for i in range(0, len(px)):
    plt.arrow(px[i], py[i], pxp[i], pyp[i], color=[112/256, 48/256, 161/256], length_includes_head=True, head_width=0.8)
    plt.arrow(px[i], py[i], pxct[i], pyct[i], color=[0.5, 0.5, 0.5], length_includes_head=True, head_width=0.8)

if save_figs:
    fig_path = os.path.join(meta['figfolder'], f"traj_ex_2_{meta['subject'][0].upper()}.{fmt}")
    plt.savefig(fig_path, format=fmt, dpi=1200)

#%% --- SECTION: Main progress analysis ---
df_fn = pd.concat(df_a, axis=0, ignore_index=True)
meas = ['prog', 'progn', 'prog_mag', 'prog_ang']

# Clean: remove last 2 samples of each trial (cursor within target)
df_fn = df_fn.drop(df_fn.groupby(['monkey', 'uncertainty', 'target_direction', 'file_index', 'hit_number']).tail(2).index, axis=0)

# Process variables
df_fn = df_fn.assign(
    prog_anga = abs(df_fn['prog_ang']),
    #traj_ang = abs(df_fn['traj_ang']),
    prog_mag = np.sqrt(df_fn['proj_x']**2+df_fn['proj_y']**2),
    dist_tgt = np.sqrt((df_fn['pos_x']-df_fn['tgt_x'])**2+(df_fn['pos_y']-df_fn['tgt_y'])**2),
    proj_ang = np.rad2deg(np.arctan2(df_fn['proj_y'], df_fn['proj_x'])),
    proj_ang_d = angle_between(np.column_stack((df_fn['proj_x'].values, df_fn['proj_y'].values)), 
                               np.column_stack((df_fn['tgt_x'].values, df_fn['tgt_y'].values))),
    prog_angs = abs(df_fn['prog_angs']),
    #traj_angs = abs(df_fn['traj_angs']),
    prog_mags = np.sqrt(df_fn['proj_xs']**2+df_fn['proj_ys']**2),
    tgt_traj_anga = abs(df_fn['tgt_traj_ang'])
)

proj_ang_df = df_fn.groupby(by=['is_mov', 'target_direction'], as_index=False)[
    ['proj_x', 'proj_y', 'tgt_x', 'tgt_y']].mean()
proj_ang_df = proj_ang_df.assign(
    proj_ang_m = angle_wrap(np.rad2deg(np.arctan2(proj_ang_df['proj_y'], proj_ang_df['proj_x']))),
    proj_ang_d = angle_between(np.column_stack((proj_ang_df['proj_x'].values, proj_ang_df['proj_y'].values)), np.column_stack((proj_ang_df['tgt_x'].values, proj_ang_df['tgt_y'].values)))
)
print(proj_ang_df[['target_direction', 'is_mov', 'proj_ang_m', 'proj_ang_d']])

# Group-level aggregations
prog_cond = df_fn.groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index'], as_index=False)[meas].mean()
prog_trial = df_fn.groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index', 'hit_number'], as_index=False)[meas].mean()
prog_time = df_fn.groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index', 'time'], as_index=False)[meas].mean()

# Paired t-test between uncertainty conditions for progress measures
for m in monkey:
    print(f"Monkey: {m}")
    for i in meas:
        print(i)
        stat_test(prog_cond.loc[prog_cond.monkey == m, i], prog_cond.loc[prog_cond.monkey == m, 'uncertainty'], 1)

#%% --- SECTION: Intermediate analysis and plotting ---
#%% PLOT: Speed in time, aligned to go cue, average across trials for conditions
spd_time = df_fn.groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index', 'time'], as_index=False)['speed'].mean()

t_end = 1000
times = np.arange(0, t_end+50, 50)
spd_time = spd_time.loc[(spd_time['time'] <= t_end),:]

fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True, sharey=True)
for i, m in enumerate(monkey):
    sns.lineplot(data=spd_time.loc[spd_time.monkey == m], x='time', y='speed', hue='uncertainty', palette=my_pal, ax=ax[i], estimator='mean', errorbar=('se'), linewidth=2)
    ax[i].set_title(m, fontsize=14)
    ax[i].set_xlabel('Time from go cue [ms]', fontsize=14)
    ax[i].set_ylabel('Speed [mm/s]', fontsize=14)
    ax[i].set_xlim([0, t_end])
    ax[i].spines[['right', 'top']].set_visible(False)
    ax[i].spines[['left', 'bottom']].set_linewidth(1)
    ax[i].tick_params(axis='both', labelsize=12)
    ax[i].get_legend().remove()

#%% PLOT: Firing rate (average across units) in time, aligned to go cue, average across trials for conditions
fr_time = df_fn.groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index', 'time'], as_index=False)['FR'].mean()

t_end = 1000
times = np.arange(0, t_end+50, 50)
fr_time = fr_time.loc[(fr_time['time'] <= t_end),:]

fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
for i, m in enumerate(monkey):
    sns.lineplot(data=fr_time.loc[fr_time.monkey == m], x='time', y='FR', hue='uncertainty',   
                 palette=my_pal, ax=ax[i], estimator='mean', errorbar=('se'), linewidth=2)
    ax[i].set_title(m, fontsize=14)
    ax[i].set_xlabel('Time from go cue [ms]', fontsize=14)
    ax[i].set_ylabel('Firing rate [spk/s]', fontsize=14)
    ax[i].set_xlim([0, t_end])
    ax[i].spines[['right','top']].set_visible(False)
    ax[i].spines[['left','bottom']].set_linewidth(1)
    ax[i].tick_params(axis='both', labelsize=12)
    ax[i].get_legend().remove()
    
#%% PLOT: Histogram neural projection angles
mky_sel = 'MY'
df_plt = df_fn.loc[(df_fn.monkey == mky_sel) & (df_fn.is_mov == 1)]
g = sns.displot(kind='hist', data=df_plt, x='proj_ang',
                hue='uncertainty', palette=my_pal, hue_order=('CRS','CLD'), 
                legend=False, binwidth=20, col='target_direction', 
                col_wrap=4, height=4, aspect=0.8, edgecolor=None)
g.map(add_median,'proj_ang')
g.set_axis_labels('Neural projection angle [deg]', 'Count')
g.set_titles('Target direction = {col_name}')

g = sns.displot(data=df_plt, x='proj_ang', hue='uncertainty', 
                palette=my_pal, hue_order=('CRS','CLD'), legend=False,
                binwidth=20, col='file_index', col_wrap=5, height=4, aspect=0.8, edgecolor=None)
g.map(add_median,'proj_ang')
g.set_axis_labels('Neural projection angle [deg]', 'Count')
g.set_titles('File index = {col_name}')

#%% PLOT: Progress, magnitude and angle, average across time and trials for conditions
meas_plt = ['prog', 'prog_mag', 'prog_anga']
df_plt = df_fn[df_fn['dist'] > 10].groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index'], as_index=False)[meas_plt].mean()

fig, ax = plt.subplots(1, 3, figsize=(20*ctm, 8*ctm), constrained_layout=True)
ym_lab = ["Progress [mm/s]", "Neural proj magnitude [mm/s]", "Neural proj angle [°]"]

for i in range(3):
    sns.pointplot(data=df_plt, hue="uncertainty", x="monkey", y=meas_plt[i], 
                  palette=my_pal2,  hue_order=('CRS','CLD'), errorbar='ci', dodge=0.4, linestyle='none', ax=ax[i])
    sns.stripplot(data=df_plt,  hue="uncertainty", x="monkey", y=meas_plt[i], 
                  palette=my_pal2, hue_order=('CRS','CLD'), size=3, alpha=0.3, jitter=0.1, dodge=True, ax=ax[i])
    ax[i].set_xlabel('', fontsize=14)
    ax[i].set_ylabel(ym_lab[i], fontsize=14)
    ax[i].spines[['right', 'top']].set_visible(False)
    ax[i].spines[['left', 'bottom']].set_linewidth(1)
    ax[i].tick_params(axis='both', labelsize=13, width=1, length=3, direction='in')
    ax[i].set_xlim([-0.5+round(0), round(1)+0.5])
    ax[i].get_legend().remove()
    
if save_figs:
    plt.savefig(os.path.join(meta['figfolder'],f'prog_pma_m.{fmt}'), format=fmt) 
  
#%% PLOT: Progress in time, aligned to go cue, average across trials for conditions
meas_plt = 'prog'
t_end = 1000
times = np.arange(0, t_end+50, 50)
df_plt = prog_time.loc[prog_time['time'] <= t_end]

fig, ax = plt.subplots(1, 2, figsize=(16*ctm, 8*ctm), constrained_layout=True, sharey=True)
for i, m in enumerate(monkey):
    sns.lineplot(data=df_plt.loc[df_plt.monkey == m], x='time', y=meas_plt, hue='uncertainty', 
                 palette=my_pal2, ax=ax[i], estimator='mean', errorbar=('se'), linewidth=2)
    ax[i].set_title(m, fontsize=14, fontweight='bold')
    ax[i].set_xlabel('Time from go cue [ms]', fontsize=14)
    if i == 0:
        ax[i].set_ylabel('Progress [mm/s]', fontsize=14)
    else:
        ax[i].set_ylabel('')
    ax[i].set_xlim([0, t_end])
    ax[i].spines[['right', 'top']].set_visible(False)
    ax[i].spines[['left', 'bottom']].set_linewidth(1)
    ax[i].tick_params(axis='both', labelsize=13, width=1, length=3, direction='in')
    ax[i].get_legend().remove()

if save_figs:
    plt.savefig(os.path.join(meta['figfolder'],'prog_tm_g.%s' % fmt), format=fmt) 

#%% PLOT: Progress in time, aligned to movement onset, average across trials for conditions
meas_plt = 'prog'
prog_mov = df_fn.loc[(df_fn['dist'] > 10)].copy()
prog_mov['time_mov'] = prog_mov.groupby(['monkey', 'file_index', 'hit_number'])['time'].transform(lambda x: x - x.iloc[0])

prog_mov = prog_mov.groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index', 'time_mov'], as_index=False)[[meas_plt]].mean()

t_end = 600
times = np.arange(0, t_end+50, 50)
df_plt = prog_mov.loc[(prog_mov['time_mov'] <= t_end),:]

fig, ax = plt.subplots(1, 2, figsize=(16*ctm, 8*ctm), constrained_layout=True, sharey=True)
for i,m in enumerate(monkey):
    sns.lineplot(data=df_plt.loc[df_plt.monkey == m], x='time_mov', y=meas_plt, hue='uncertainty', 
                 palette=my_pal2, ax=ax[i], estimator='mean', errorbar=('se'), linewidth=2)
    ax[i].set_title(m, fontsize=14, fontweight='bold')
    ax[i].set_xlabel('Time from mov onset [ms]', fontsize=14)
    if i == 0:
        ax[i].set_ylabel('Progress [mm/s]', fontsize=14)
    else:
        ax[i].set_ylabel('')
    ax[i].set_xlim([0, t_end])
    ax[i].spines[['right', 'top']].set_visible(False)
    ax[i].spines[['left', 'bottom']].set_linewidth(1)
    ax[i].tick_params(axis='both', labelsize=13, width=1, length=3, direction='in')
    ax[i].get_legend().remove()

if save_figs:
    plt.savefig(os.path.join(meta['figfolder'],'prog_tm_m.%s' % fmt), format=fmt) 

#%% Progress aligned to go cue and movement onset, average in +200ms time window and across trials for conditions
meas_plt = 'prog'
prog_mov = df_fn.loc[(df_fn['dist'] > 10)].copy()
prog_mov['time_mov'] = prog_mov.groupby(['monkey', 'file_index', 'hit_number'])['time'].transform(lambda x: x - x.iloc[0])

prog_mov = prog_mov.loc[(prog_mov['time_mov'] <= 200)].groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index'], as_index=False)[[meas_plt]].mean()

prog_go = df_fn.loc[(df_fn['dist'] <= 10) & (df_fn['time'] <= 200)].groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index'], as_index=False)[[meas_plt]].mean()

for m in monkey:
    print(f"Monkey: {m}")
    
    print("Go cue +200ms")
    stat_test(prog_go.loc[prog_go.monkey == m, meas_plt], prog_go.loc[prog_go.monkey == m, 'uncertainty'], 1)
    
    print("Mov init +200ms")
    stat_test(prog_mov.loc[prog_mov.monkey == m, meas_plt], prog_mov.loc[prog_mov.monkey == m,'uncertainty'], 1)


fig, ax = plt.subplots(1, 2, figsize=(13*ctm, 8*ctm), constrained_layout=True)

sns.pointplot(data=prog_go, hue="uncertainty", x="monkey", y=meas_plt, palette=my_pal2,  hue_order=('CRS','CLD'), errorbar='ci', dodge=0.4, linestyle='none', ax=ax[0])
sns.stripplot(data=prog_go,  hue="uncertainty", x="monkey", y=meas_plt, palette=my_pal2, hue_order=('CRS','CLD'), size=3, alpha=0.3, jitter=0.1, ax=ax[0], dodge=True)

sns.pointplot(data=prog_mov, hue="uncertainty", x="monkey", y=meas_plt, palette=my_pal2, hue_order=('CRS','CLD'), errorbar='ci', dodge=0.4, linestyle='none', ax=ax[1])
sns.stripplot(data=prog_mov,  hue="uncertainty", x="monkey", y=meas_plt, palette=my_pal2, hue_order=('CRS','CLD'), size=3, alpha=0.3, jitter=0.1, ax=ax[1], dodge=True) 
    
for i in range(2):
    ax[i].set_xlabel('', fontsize=14)
    if i==0:
        ax[i].set_title('go cue +200ms', fontsize=13, fontweight='bold')
        ax[i].set_ylabel('Progress [mm/s]', fontsize=14)
    else:
        ax[i].set_title('mov init +200ms', fontsize=13, fontweight='bold')
        ax[i].set_ylabel('')
    ax[i].spines[['right', 'top']].set_visible(False)
    ax[i].spines[['left', 'bottom']].set_linewidth(1)
    ax[i].tick_params(axis='both', labelsize=13, width=1, length=3, direction='in')
    ax[i].set_xlim([-0.5+round(0), round(1)+0.5])
    ax[i].get_legend().remove()
    #sns.despine(offset=10, trim=True)
    ax[i].margins(0.1)
    
if save_figs:
    plt.savefig(os.path.join(meta['figfolder'],'prog_p_gm.%s' % fmt), format=fmt) 
     
#%% PLOT: Progress, averaged across trials and time for conditions
meas_plt = 'prog'

plt.figure(figsize=(10*ctm, 8*ctm))
sns.pointplot(data=prog_cond, x="monkey", y=meas_plt, hue="uncertainty",  
              palette=my_pal,  errorbar='ci', hue_order=('CRS','CLD'), dodge=0.4)
sns.stripplot(data=prog_cond, x="monkey", y=meas_plt, hue="uncertainty", 
              palette=my_pal, hue_order=('CRS','CLD'), size=4, alpha=0.5, jitter=0.05, legend=False, dodge=True)
plt.ylabel('Progress [mm/s]', fontsize=12)
plt.xlabel('Monkey', fontsize=12)
ax = plt.gca()
ax.spines[['right', 'top']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(1)
ax.tick_params(axis='both', labelsize=12, width=1, length=3, direction='in')
ax.legend(loc='upper right', frameon=False, fontsize=12)

# Split by target direction
g = sns.FacetGrid(data=prog_cond, col="monkey", height=4, aspect=1.5)
g.map_dataframe(sns.barplot, x="target_direction", y=meas_plt, hue="uncertainty", 
            palette=my_pal,  errorbar='ci', width=0.6, dodge=True, hue_order=('CRS','CLD'), alpha=0.6)
g.map_dataframe(sns.stripplot,  x="target_direction", y=meas_plt, hue="uncertainty", 
              palette=my_pal, hue_order=('CRS','CLD'), size=4, jitter=0.05, legend=False, dodge=True)
plt.xlabel('Target direction [deg]', fontsize=12)
plt.ylabel('Progress [mm/s]', fontsize=12)
ax = plt.gca()
ax.spines[['right', 'top']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(1)
ax.tick_params(axis='both', labelsize=12, width=1, length=3, direction='in')
ax.legend(loc='upper right', frameon=False, fontsize=12)

#%% Neural projection in 2D plane, average across trials and single-trial, average across time
meas_plt = ['proj_x', 'proj_y']
proj_xy = df_fn.groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index'], as_index=False)[meas_plt].mean()
proj_xy2 = df_fn.groupby(by=['monkey', 'file_index'], as_index=False)[meas_plt].mean()

proj_xy2.rename(columns={meas_plt[0]: 'proj_xm', meas_plt[1]: 'proj_ym'}, inplace=True)
proj_xy = pd.merge(proj_xy, proj_xy2, how='right', on=['monkey', 'file_index'])

mky_sel = 'MY'
df_plt = proj_xy.loc[proj_xy.monkey == mky_sel]
g = sns.FacetGrid(data=df_plt, col='file_index', col_wrap=2, sharex=False, sharey=False, aspect=1)
g.map_dataframe(sns.scatterplot, x=meas_plt[0], y=meas_plt[1], hue='target_direction', style='uncertainty', palette=sns.color_palette("husl", 8))
g.map_dataframe(sns.scatterplot, x='proj_xm', y='proj_ym', color='k', marker=('+'))
g.add_legend()
g.set_xlabels('x-projection')
g.set_ylabels('y-projection')
g.set_titles('File index = {col_name}')
for legend in g._legend.get_texts():
    text = legend.get_text()
    if text == 'target_direction':
        legend.set_text('Target direction')
    elif text == 'uncertainty':
        legend.set_text('Uncertainty')

# Trial average
proj_xy_m = df_plt.groupby(by=['monkey', 'uncertainty', 'target_direction'], as_index=False)[meas_plt].mean()
plt.figure()
sns.scatterplot(data=proj_xy_m,  x=meas_plt[0], y=meas_plt[1], hue="target_direction", style="uncertainty", palette=sns.color_palette("husl", 8), sizes=14, legend=False)
plt.axis('square')
plt.xlabel('x-projection')
plt.ylabel('y-projection')
plt.title(mky_sel)

#%% Bin progress by distance from center, average across trials and time for conditions
bin_x = [-1, 20, 80, 120]
bin_l = [1, 2, 3]
df_fn['dist_bin'] = pd.cut(df_fn['dist'], bins=bin_x, labels=bin_l)
meas_plt = 'prog'

df_fn_a = df_fn.dropna(subset=['dist_bin'])
df_fn_a.loc[:, 'dist_bin'] = df_fn_a['dist_bin'].cat.codes+1
prog_dist = df_fn_a.groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index', 'dist_bin'], as_index=False)[[meas_plt]].mean()

for m in monkey:
    print(f"Monkey: {m}")
    for i in bin_l:
        print(f"Bin: {i}")
        sel = prog_dist.loc[(prog_dist['dist_bin'] == i) & (prog_dist['monkey'] == m)]
        sel = equal_cond_df(sel)
        stat_test(sel[meas_plt], sel['uncertainty'], 1)

#%% PLOT: Binned progress by distance from center
# Split by target directions
g = sns.catplot(kind='bar', data=prog_dist, x="target_direction", y=meas_plt, hue="uncertainty", col='dist_bin', row='monkey', palette=my_pal, errorbar='ci', width=0.5, hue_order=('CRS','CLD'))
g.set_axis_labels("Target direction [deg]", "Progress [mm/ms]")
g.set_titles('{row_name} | Distance bin = {col_name}')
 
fig, ax = plt.subplots(1, 2, figsize=(13*ctm, 8*ctm), constrained_layout=True, sharey=True)
for i in range(2):
    sns.pointplot(data=prog_dist.loc[prog_dist.dist_bin == i+1], hue="uncertainty", x="monkey", y=meas_plt, palette=my_pal, hue_order=('CRS','CLD'), errorbar='ci', dodge=0.4, linestyle='none', ax=ax[i])
    sns.stripplot(data=prog_dist.loc[prog_dist.dist_bin == i+1],  hue="uncertainty", x="monkey", y=meas_plt, palette=my_pal, hue_order=('CRS','CLD'), size=3, alpha=0.3, jitter=0.1, ax=ax[i], dodge=True) 
    ax[i].set_xlabel('', fontsize=14)
    if i == 0:
        ax[i].set_title('20-80 mm', fontsize=13, fontweight='bold')
        ax[i].set_ylabel('Progress [mm/s]', fontsize=14)
    else:
        ax[i].set_title('80-120 mm', fontsize=13, fontweight='bold')
        ax[i].set_ylabel('')
    ax[i].spines[['right', 'top']].set_visible(False)
    ax[i].spines[['left', 'bottom']].set_linewidth(1)
    ax[i].tick_params(axis='both', labelsize=13, width=1, length=3, direction='in')
    ax[i].set_xlim([-0.5+round(0), round(1)+0.5])
    ax[i].get_legend().remove()
    ax[i].margins(0.1)

#%% PLOT: Progess binned by trajectory direction angle (angle between center-to-cursor vector and target direction)
bin_x = [-1, 10, 30, 60]
bin_l = [1, 2, 3]
df_fn['tgt_traj_bin'] = pd.cut(abs(df_fn['tgt_traj_ang']), bins=bin_x, labels=bin_l)
meas_plt = 'prog'

df_fn_a = df_fn.dropna(subset=['tgt_traj_bin'])
df_fn_a.loc[:, 'tgt_traj_bin'] = df_fn_a['tgt_traj_bin'].cat.codes+1
prog_tt_ang = df_fn_a.loc[df_fn_a['dist'] > 10].groupby(by=['monkey', 'uncertainty','target_direction','file_index','tgt_traj_bin'], as_index=False)[[meas_plt]].mean()

for m in monkey:
    print(f"Monkey: {m}")
    for i in bin_l:
        print(f"Bin: {i}")
        sel = prog_tt_ang.loc[(prog_tt_ang['tgt_traj_bin'] == i) & (prog_tt_ang['monkey'] == m)]
        sel = equal_cond_df(sel)
        stat_test(sel[meas_plt], sel['uncertainty'], 1)

#%% PLOT: Binned progress by trajectory direction angle
# Split by target directions
g = sns.catplot(kind='bar', data=prog_tt_ang, x="target_direction", y=meas_plt, hue="uncertainty", col='tgt_traj_bin', row='monkey', palette=my_pal, errorbar='ci', width=0.5, hue_order=('CRS','CLD'))
g.set_axis_labels("Target direction [deg]", "Progress [mm/ms]")
g.set_titles('{row_name} | Trajectory angle bin = {col_name}')
 
fig, ax = plt.subplots(1, 2, figsize=(13*ctm, 8*ctm), constrained_layout=True, sharey=True)
for i in range(2):
    sns.pointplot(data=prog_tt_ang.loc[prog_tt_ang.tgt_traj_bin == i+1], hue="uncertainty", x="monkey", y=meas_plt, palette=my_pal, hue_order=('CRS','CLD'), errorbar='ci', dodge=0.4, linestyle='none', ax=ax[i])
    sns.stripplot(data=prog_tt_ang.loc[prog_tt_ang.tgt_traj_bin == i+1],  hue="uncertainty", x="monkey", y=meas_plt, palette=my_pal, hue_order=('CRS','CLD'), size=3, alpha=0.3, jitter=0.1, ax=ax[i], dodge=True) 
    ax[i].set_xlabel('', fontsize=14)
    if i == 0:
        ax[i].set_title('10-30 deg', fontsize=13, fontweight='bold')
        ax[i].set_ylabel('Progress [mm/s]', fontsize=14)
    else:
        ax[i].set_title('30-60 deg', fontsize=13, fontweight='bold')
        ax[i].set_ylabel('')
    ax[i].spines[['right', 'top']].set_visible(False)
    ax[i].spines[['left', 'bottom']].set_linewidth(1)
    ax[i].tick_params(axis='both', labelsize=13, width=1, length=3, direction='in')
    ax[i].set_xlim([-0.5+round(0), round(1)+0.5])
    ax[i].get_legend().remove()
    ax[i].margins(0.1)

#%% Bin progress by tortuosity, averaged across trials and time
meas_plt = 'prog'
prog_tort = df_behav.groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index', 'tort_bin'], as_index=False)[[meas_plt]].mean()
prog_tortu = df_behav.groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index', 'tort_binu'], as_index=False)[[meas_plt]].mean()

prog_tort = equal_cond_df(prog_tort)
prog_tortu = equal_cond_df(prog_tortu)

for m in monkey:
    print(f"Monkey: {m}")
    for i in range(1, 3):
        print(f"Bin: {i}")
        stat_test(prog_tort.loc[(prog_tort['monkey'] == m) & (prog_tort['tort_bin'] == i), meas_plt], prog_tort.loc[(prog_tort['monkey'] == m) & (prog_tort['tort_bin'] == i),'uncertainty'], 1)
        stat_test(prog_tortu.loc[(prog_tort['monkey'] == m) & (prog_tortu['tort_binu'] == i), meas_plt], prog_tortu.loc[(prog_tort['monkey'] == m) & (prog_tortu['tort_binu'] == i),'uncertainty'], 1)

#%% PLOT: Binned progress by tortuosity
# Split by target directions
g = sns.catplot(kind='bar', data=prog_tort, x="target_direction", y=meas_plt, hue="uncertainty", col='tort_bin', row='monkey', palette=my_pal, errorbar='ci', width=0.5, hue_order=('CRS','CLD'))
g.set_axis_labels("Target direction [deg]", "Progress [mm/ms]")
g.set_titles('{row_name} | Tortuosity bin = {col_name}')
 
fig, ax = plt.subplots(1, 2, figsize=(13*ctm, 8*ctm), constrained_layout=True, sharey=True)
for i in range(2):
    sns.pointplot(data=prog_tort.loc[prog_tort.tort_bin == i+1], hue="uncertainty", x="monkey", y=meas_plt, palette=my_pal, hue_order=('CRS','CLD'), errorbar='ci', dodge=0.4, linestyle='none', ax=ax[i])
    sns.stripplot(data=prog_tort.loc[prog_tort.tort_bin == i+1],  hue="uncertainty", x="monkey", y=meas_plt, palette=my_pal, hue_order=('CRS','CLD'), size=3, alpha=0.3, jitter=0.1, ax=ax[i], dodge=True) 
    ax[i].set_xlabel('', fontsize=14)
    ax[i].set_title('Tort bin ' + str(i+1), fontsize=13)
    if i == 0:
        ax[i].set_ylabel('Progress [mm/s]', fontsize=14)
    else:
        ax[i].set_ylabel('')
    ax[i].spines[['right', 'top']].set_visible(False)
    ax[i].spines[['left', 'bottom']].set_linewidth(1)
    ax[i].tick_params(axis='both', labelsize=13, width=1, length=3, direction='in')
    ax[i].set_xlim([-0.5+round(0), round(1)+0.5])
    ax[i].get_legend().remove()
    ax[i].margins(0.1)

#%% Bin progress by reaction and movement and late movement time, averaged across trials and time
meas_plt = 'prog'
prog_MT = df_behav.groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index', 'MT_bin'], as_index=False)[[meas_plt]].mean()
prog_MT_late = df_behav.groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index', 'MT_late_bin'], as_index=False)[[meas_plt]].mean()
prog_RT = df_behav.groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index', 'RT_bin'], as_index=False)[[meas_plt]].mean()

prog_MT = equal_cond_df(prog_MT)
prog_MT_late = equal_cond_df(prog_MT_late)
prog_RT = equal_cond_df(prog_RT)

for m in monkey:
    print(f"Monkey: {m}")
    for i in range(1, 3):
        print(f"Bin: {i}")    
        print("RT") 
        stat_test(prog_RT.loc[(prog_RT['monkey'] == m) & (prog_RT['RT_bin'] == i), meas_plt], prog_RT.loc[(prog_RT['monkey'] == m) & (prog_RT['RT_bin'] == i),'uncertainty'], 1)
        print("MT") 
        stat_test(prog_MT.loc[(prog_MT['monkey'] == m) & (prog_MT['MT_bin'] == i), meas_plt], prog_MT.loc[(prog_MT['monkey'] == m) & (prog_MT['MT_bin'] == i),'uncertainty'], 1)
        print("Late MT") 
        stat_test(prog_MT_late.loc[(prog_MT_late['monkey'] == m) & (prog_MT_late['MT_late_bin'] == i), meas_plt], prog_MT_late.loc[(prog_MT_late['monkey'] == m) & (prog_MT_late['MT_late_bin'] == i),'uncertainty'], 1)

#%% PLOT: Progress binned by reaction time
# Split by target directions
g = sns.catplot(kind='bar', data=prog_RT, x="target_direction", y=meas_plt, hue="uncertainty", col='RT_bin', row='monkey', palette=my_pal, errorbar='ci', width=0.5, hue_order=('CRS','CLD'))
g.set_axis_labels("Target direction [deg]", "Progress [mm/ms]")
g.set_titles('{row_name} | RT bin = {col_name}')
 
fig, ax = plt.subplots(1, 2, figsize=(13*ctm, 8*ctm), constrained_layout=True, sharey=True)
for i in range(2):
    sns.pointplot(data=prog_RT.loc[prog_RT.RT_bin == i+1], hue="uncertainty", x="monkey", y=meas_plt, palette=my_pal, hue_order=('CRS','CLD'), errorbar='ci', dodge=0.4, linestyle='none', ax=ax[i])
    sns.stripplot(data=prog_RT.loc[prog_RT.RT_bin == i+1],  hue="uncertainty", x="monkey", y=meas_plt, palette=my_pal, hue_order=('CRS','CLD'), size=3, alpha=0.3, jitter=0.1, ax=ax[i], dodge=True) 
    ax[i].set_xlabel('', fontsize=14)
    ax[i].set_title('RT bin ' + str(i+1), fontsize=13)
    if i == 0:
        ax[i].set_ylabel('Progress [mm/s]', fontsize=14)
    else:
        ax[i].set_ylabel('')
    ax[i].spines[['right', 'top']].set_visible(False)
    ax[i].spines[['left', 'bottom']].set_linewidth(1)
    ax[i].tick_params(axis='both', labelsize=13, width=1, length=3, direction='in')
    ax[i].set_xlim([-0.5+round(0), round(1)+0.5])
    ax[i].get_legend().remove()
    ax[i].margins(0.1)

#%% PLOT: Progress binned by movement time
# Split by target directions
g = sns.catplot(kind='bar', data=prog_MT, x="target_direction", y=meas_plt, hue="uncertainty", col='MT_bin', row='monkey', palette=my_pal, errorbar='ci', width=0.5, hue_order=('CRS','CLD'))
g.set_axis_labels("Target direction [deg]", "Progress [mm/ms]")
g.set_titles('{row_name} | RT bin = {col_name}')
 
fig, ax = plt.subplots(1, 2, figsize=(13*ctm, 8*ctm), constrained_layout=True, sharey=True)
for i in range(2):
    sns.pointplot(data=prog_MT.loc[prog_MT.MT_bin == i+1], hue="uncertainty", x="monkey", y=meas_plt, palette=my_pal, hue_order=('CRS','CLD'), errorbar='ci', dodge=0.4, linestyle='none', ax=ax[i])
    sns.stripplot(data=prog_MT.loc[prog_MT.MT_bin == i+1],  hue="uncertainty", x="monkey", y=meas_plt, palette=my_pal, hue_order=('CRS','CLD'), size=3, alpha=0.3, jitter=0.1, ax=ax[i], dodge=True) 
    ax[i].set_xlabel('', fontsize=14)
    ax[i].set_title('MT bin ' + str(i+1), fontsize=13)
    if i == 0:
        ax[i].set_ylabel('Progress [mm/s]', fontsize=14)
    else:
        ax[i].set_ylabel('')
    ax[i].spines[['right', 'top']].set_visible(False)
    ax[i].spines[['left', 'bottom']].set_linewidth(1)
    ax[i].tick_params(axis='both', labelsize=13, width=1, length=3, direction='in')
    ax[i].set_xlim([-0.5+round(0), round(1)+0.5])
    ax[i].get_legend().remove()
    ax[i].margins(0.1)
    
#%% PLOT: Progress binned by late movement time
# Split by target directions
g = sns.catplot(kind='bar', data=prog_MT_late, x="target_direction", y=meas_plt, hue="uncertainty", col='MT_late_bin', row='monkey', palette=my_pal, errorbar='ci', width=0.5, hue_order=('CRS','CLD'))
g.set_axis_labels("Target direction [deg]", "Progress [mm/ms]")
g.set_titles('{row_name} | RT bin = {col_name}')
 
fig, ax = plt.subplots(1, 2, figsize=(13*ctm, 8*ctm), constrained_layout=True, sharey=True)
for i in range(2):
    sns.pointplot(data=prog_MT_late.loc[prog_MT_late.MT_late_bin == i+1], hue="uncertainty", x="monkey", y=meas_plt, palette=my_pal, hue_order=('CRS','CLD'), errorbar='ci', dodge=0.4, linestyle='none', ax=ax[i])
    sns.stripplot(data=prog_MT_late.loc[prog_MT_late.MT_late_bin == i+1],  hue="uncertainty", x="monkey", y=meas_plt, palette=my_pal, hue_order=('CRS','CLD'), size=3, alpha=0.3, jitter=0.1, ax=ax[i], dodge=True) 
    ax[i].set_xlabel('', fontsize=14)
    ax[i].set_title('Late MT bin ' + str(i+1), fontsize=13)
    if i == 0:
        ax[i].set_ylabel('Progress [mm/s]', fontsize=14)
    else:
        ax[i].set_ylabel('')
    ax[i].spines[['right', 'top']].set_visible(False)
    ax[i].spines[['left', 'bottom']].set_linewidth(1)
    ax[i].tick_params(axis='both', labelsize=13, width=1, length=3, direction='in')
    ax[i].set_xlim([-0.5+round(0), round(1)+0.5])
    ax[i].get_legend().remove()
    ax[i].margins(0.1)

#%% Progress at submovement timepoints
meas = 'prog'

prog_sbm = df_fn.loc[df_fn['is_sbm'] == 1].groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index', 'is_sbm_d'], as_index=False)[[meas]].mean()
prog_sbm = equal_cond_df(prog_sbm)

lab = ['Correction', 'Non-correction']
for m in monkey:
    print(f"Monkey: {m}")
    for i in range(2):
        print(lab[i])
        stat_test(prog_sbm.loc[(prog_sbm['monkey'] == m) & (prog_sbm['is_sbm_d'] == i), meas], prog_sbm.loc[(prog_sbm['monkey'] == m) & (prog_sbm['is_sbm_d'] == i), 'uncertainty'], 1)

prog_sbm_m = df_fn.loc[df_fn['is_sbm'] == 1,:].groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index', 'is_sbm'], as_index=False)[[meas]].mean()
prog_sbm_m = equal_cond_df(prog_sbm_m)

sbm_c = df_fn.groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index', 'hit_number'], as_index=False)[['is_sbm']].sum()
sbm_c = sbm_c.groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index'], as_index=False)[['is_sbm']].mean()

for m in monkey:
    print(f"Monkey: {m}")

    print('Progress at submovement start')
    stat_test(prog_sbm_m.loc[(prog_sbm_m['monkey'] == m), meas], prog_sbm_m.loc[(prog_sbm_m['monkey'] == m), 'uncertainty'], 1)

    print('Number of submovements per trial')
    stat_test(sbm_c.loc[(sbm_c['monkey'] == m), 'is_sbm'], sbm_c.loc[(sbm_c['monkey'] == m), 'uncertainty'], 1)

prog_sbm_NN = df_submov_nn.groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index', 'sbm_idx', 'sbm_idx_d'], as_index=False)[[meas]].mean()
prog_sbm_NN['sbm_t'] = 50*prog_sbm_NN['sbm_idx'].astype(int)

prog_sbm_NN_m = df_submov_nn.groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index', 'sbm_idx'], as_index=False)[[meas]].mean()
prog_sbm_NN_m['sbm_t'] = 50*prog_sbm_NN_m['sbm_idx'].astype(int)

#%% PLOT: Trajectory (2D plot) for trials without (DIR) and with submovements (Non-DIR)
mky_sel = 'MY'
file_sel = 2
df_plt = df_fn.loc[(df_fn.monkey == mky_sel) & (df_fn.file_index == file_sel)]

df_sb = df_plt.groupby(by=['monkey', 'file_index', 'hit_number', 'uncertainty', 'target_direction'], as_index=False)[['is_sbm']].any()
df_sb = pd.merge(df_plt, df_sb, how ='right', on=['monkey', 'file_index', 'hit_number', 'uncertainty', 'target_direction'])

tgt_uni = [0, 45, 90, 135, 180, 225, 270, 315]

fig, ax = plt.subplots(2, 2, figsize=(8, 8), squeeze=True)

for i in df_sb.loc[df_sb.file_index == 2, 'hit_number'].unique():
    sel = (df_sb.file_index == 2) & (df_sb.hit_number == i)
    color_idx = np.where(np.array(tgt_uni) == df_sb.loc[sel, 'target_direction'].values[0])[0][0]
    color = cm[color_idx]

    for j, (unc, is_sbm) in enumerate([('CRS', False), ('CLD', False), ('CRS', True), ('CLD', True)]):
        row, col = divmod(j, 2)
        sel_plot = sel & (df_sb.uncertainty == unc) & (df_sb.is_sbm_y == is_sbm)
        ax[row, col].plot(df_sb.loc[sel_plot, 'pos_x'], df_sb.loc[sel_plot, 'pos_y'], color=color)
    
lab = ['CRS + DIR', 'CLD + DIR', 'CRS + Non-DIR', 'CLD + Non-DIR',]
c = 0
for j in range(2):
    for k in range(2):
        for t in tgt_uni:   
            tgt = plt.Circle((120*np.cos(np.deg2rad(t)), 120*np.sin(np.deg2rad(t))),radius=22.5, fill=False, color='k')
            ax[j, k].add_patch(tgt)
        
        ax[j, k].set_xlim([-180, 180])
        ax[j, k].set_ylim([-180, 180])
        ax[j, k].spines[['right', 'top', 'left', 'bottom']].set_visible(False)
        ax[j, k].set_xticks([])
        ax[j, k].set_yticks([])
        ax[j, k].set_title(lab[c])
        c += 1

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

#%% PLOT: Progress at submovement index, grouped by corrective and uncorrective
meas_plt = 'prog'

fig, ax = plt.subplots(1, 2, figsize=(13*ctm, 8*ctm), constrained_layout=True, sharey=True)
for i, m in enumerate(monkey):
    sns.pointplot(data=prog_sbm.loc[prog_sbm.monkey == m], hue="uncertainty", x="is_sbm_d", y=meas_plt, palette=my_pal, hue_order=('CRS','CLD'), errorbar='ci', dodge=0.4, linestyle='none', ax=ax[i])
    sns.stripplot(data=prog_sbm.loc[prog_sbm.monkey == m],  hue="uncertainty", x="is_sbm_d", y=meas_plt, palette=my_pal, hue_order=('CRS','CLD'), size=3, alpha=0.3, jitter=0.1, ax=ax[i], dodge=True) 
    ax[i].set_xlabel('Correction', fontsize=14)
    ax[i].set_title(m, fontsize=13)
    if i == 0:
        ax[i].set_ylabel('Progress [mm/s]', fontsize=14)
    else:
        ax[i].set_ylabel('')
    ax[i].spines[['right', 'top']].set_visible(False)
    ax[i].spines[['left', 'bottom']].set_linewidth(1)
    ax[i].tick_params(axis='both', labelsize=13, width=1, length=3, direction='in')
    ax[i].set_xlim([-0.5+round(0), round(1)+0.5])
    ax[i].get_legend().remove()
    ax[i].margins(0.1)

#%% PLOT: Progress at submovement index and neighbouring timepoints
fig, ax = plt.subplots(2, 2, figsize=(20, 5), sharey=True, sharex=True)
for j, m in enumerate(monkey):
    for i in range(2):
        sns.pointplot(ax=ax[j, i], data=prog_sbm_NN.loc[(prog_sbm_NN['monkey'] == m) &(prog_sbm_NN['sbm_idx_d'] == i),:], x="sbm_t", y=meas, hue="uncertainty", dodge=0.2, palette=my_pal, hue_order=('CRS','CLD'))
        ax[j, i].set_xlabel('Time relative to submovement [ms]', fontsize=14)
        ax[j, i].set_ylabel('Progress [mm/s]', fontsize=14)
        ax[j, i].spines[['right', 'top']].set_visible(False)
        ax[j, i].spines[['left', 'bottom']].set_linewidth(1)
        ax[j, i].tick_params(axis='both', labelsize=12)
        ax[j, i].get_legend().remove()

#%% PLOT: Progress in trials with (non-direct) and without (direct) submovements (based on speed), average for conditions
df_sb = df_fn.groupby(by=['monkey', 'file_index', 'hit_number', 'uncertainty', 'target_direction'], as_index=False)[['is_sbm']].any()
df_sb = pd.merge(df_fn, df_sb, how ='right', on=['monkey', 'file_index', 'hit_number', 'uncertainty', 'target_direction'])

meas = 'prog'
prog_sbm_c = df_sb.groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index', 'is_sbm_y'], as_index=False)[['hit_number']].nunique()
prog_sbm = df_sb.groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index', 'is_sbm_y'], as_index=False)[[meas]].mean()
prog_sbm = equal_cond_df(prog_sbm)

lab = ['Non-direct', 'Direct']
for m in monkey:
    print(f"Monkey: {m}")
    for i in range(2):
        print(lab[i])
        stat_test(prog_sbm.loc[(prog_sbm['monkey'] == m) & (prog_sbm['is_sbm_y'] == i), meas],prog_sbm.loc[((prog_sbm['monkey'] == m) & prog_sbm['is_sbm_y'] == i), 'uncertainty'], 1)


fig, ax = plt.subplots(1, 2, figsize=(13*ctm, 8*ctm), constrained_layout=True)

for i in range(2):
    sns.pointplot(data=prog_sbm[(prog_sbm['is_sbm_y'] == i)], hue="uncertainty", x="monkey", y=meas_plt, palette=my_pal2,  
              hue_order=('CRS','CLD'), errorbar='ci', dodge=0.4, linestyle='none', ax=ax[i])
    sns.stripplot(data=prog_sbm[(prog_sbm['is_sbm_y'] == i)],  hue="uncertainty", x="monkey", y=meas_plt, palette=my_pal2, 
              hue_order=('CRS','CLD'), size=3, alpha=0.3, jitter=0.1, ax=ax[i], dodge=True)
    ax[i].set_xlabel('', fontsize=14)        
    if i==0:
        ax[i].set_ylabel('Progress [mm/s]', fontsize=14)
    else:
        ax[i].set_ylabel('')
    ax[i].set_title(lab[i], fontsize=13, fontweight='bold')
    ax[i].spines[['right', 'top']].set_visible(False)
    ax[i].spines[['left', 'bottom']].set_linewidth(1)
    ax[i].tick_params(axis='both', labelsize=13, width=1, length=3, direction='in')
    ax[i].set_xlim([-0.5+round(0), round(1)+0.5])
    ax[i].get_legend().remove()
    ax[i].margins(0.1)

#%% PLOT: Progress in trials with (non-direct) and without (direct) submovements (based on arc, filtered for movement period & MT < 1500), average for conditions
df_sb = df_first_sbm.groupby(by=['monkey', 'file_index', 'hit_number', 'uncertainty', 'target_direction'], as_index=False)[['is_sgt']].any()
df_sb = pd.merge(df_fn, df_sb, how ='right', on=['monkey', 'file_index', 'hit_number', 'uncertainty', 'target_direction'])

meas = 'prog'
prog_sbm_c = df_sb.groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index', 'is_sgt'], as_index=False)[['hit_number']].nunique()
prog_sbm = df_sb[(df_sb['dist'] > 10) & (df_sb['time'] < 1500)].groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index', 'is_sgt'], as_index=False)[[meas]].mean()
prog_sbm = equal_cond_df(prog_sbm)

lab = ['Non-direct', 'Direct']
for i in range(2):
    print(lab[i])
    stat_test(prog_sbm.loc[(prog_sbm['is_sgt'] == i), meas], prog_sbm.loc[(prog_sbm['is_sgt'] == i), 'uncertainty'], 1)

fig, ax = plt.subplots(1, 2, figsize=(13*ctm, 8*ctm), constrained_layout=True)

for i in range(2):
    sns.pointplot(data=prog_sbm[(prog_sbm['is_sgt'] == i)], hue="uncertainty", x="monkey", y=meas_plt, palette=my_pal2, hue_order=('CRS','CLD'), errorbar='ci', dodge=0.4, linestyle='none', ax=ax[i])
    sns.stripplot(data=prog_sbm[(prog_sbm['is_sgt'] == i)],  hue="uncertainty", x="monkey", y=meas_plt, palette=my_pal2, hue_order=('CRS','CLD'), size=3, alpha=0.3, jitter=0.1, ax=ax[i], dodge=True)
    ax[i].set_xlabel('', fontsize=14)        
    if i==0:
        ax[i].set_ylabel('Progress [mm/s]', fontsize=14)
    else:
        ax[i].set_ylabel('')
    ax[i].set_title(lab[i], fontsize=13, fontweight='bold')
    ax[i].spines[['right', 'top']].set_visible(False)
    ax[i].spines[['left', 'bottom']].set_linewidth(1)
    ax[i].tick_params(axis='both', labelsize=13, width=1, length=3, direction='in')
    ax[i].set_xlim([-0.5+round(0), round(1)+0.5])
    ax[i].get_legend().remove()
    ax[i].margins(0.1)
    
if save_figs:
    plt.savefig(os.path.join(meta['figfolder'],'prog_sbm.%s' % fmt), format=fmt) 

#%% Number of trials with submovements (based on arc)
df_fs = df_first_sbm[df_first_sbm.is_sgt == 0].groupby(['monkey', 'uncertainty', 'target_direction', 'file_index'])['hit_number'].nunique().reset_index(name='n_trials')

df_fs = equal_cond_df(df_fs)
for m in monkey:
    print(f"Monkey: {m}")
    stat_test(df_fs.loc[df_fs.monkey == m, 'n_trials'], df_fs.loc[df_fs.monkey == m,'uncertainty'], 1)

fig = plt.figure(figsize=(8*ctm, 8*ctm), constrained_layout=True)
sns.pointplot(data=df_fs, x='monkey', y='n_trials', hue="uncertainty", 
              palette=my_pal,  errorbar='ci', hue_order=('CRS','CLD'), linestyle='none', dodge=0.4)
sns.stripplot(data=df_fs, x='monkey', y='n_trials', hue="uncertainty", 
              palette=my_pal, hue_order=('CRS','CLD'), size=4, alpha=0.5, jitter=0.05, legend=False, dodge=True)
ax = fig.gca()
ax.set_xlabel('', fontsize=14)        
ax.set_ylabel('Number of trials', fontsize=14)
ax.spines[['right', 'top']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(1)
ax.tick_params(axis='both', labelsize=13, width=1, length=3, direction='in')
ax.margins(0.1)

#%% PLOT: First submovement trajectory (2D)
meas = 'prog'
df_fs = df_first_sbm.drop(df_first_sbm.groupby(['monkey', 'uncertainty', 'target_direction', 'file_index', 'hit_number']).tail(2).index, axis=0)
df_fs = df_fs.loc[df_fs['dist'] >= 10].groupby(by=['monkey', 'uncertainty', 'target_direction', 'file_index', 'is_sgt'], as_index=False)[[meas]].mean()

lab = ['Non-direct', 'Direct']
for m in monkey:
    print(f"Monkey: {m}")
    for i in range(2):
        print(lab[i])  
        sel = df_fs.loc[(df_fs['is_sgt'] == i) & (df_fs['monkey'] == m),:]
        sel = equal_cond_df(sel)
        stat_test(sel[meas], sel['uncertainty'], 1)
  

titles = ["CRS + sgt", "CLD + sgt",
    "CRS + non-sgt", "CLD + non-sgt"]
unc_lab = ['CRS', 'CLD']
sgt_id = [True, False]

fig, ax = plt.subplots(2, 2, figsize=(12, 12))
for i in range(2):
    for j in range(2):
        sel = ((df_first_sbm.file_index == 0) &
            (df_first_sbm.uncertainty == unc_lab[j]) &
            (df_first_sbm.is_sgt == sgt_id[i]))
        ax[i, j].scatter(df_first_sbm.loc[sel].pos_x, df_first_sbm.loc[sel].pos_y)
        ax[i, j].axis('square')
        ax[i, j].set_title(titles[i*2+j])
        ax[i, j].set_xlabel('x-pos [mm]')
        ax[i, j].set_ylabel('y-pos [mm]')
plt.tight_layout()
plt.show()

#%% Heatmap with progress difference between uncertainty conditions, binned by time and angle between trajectory and target direction
sel_mky = 'MY'
df_hm = df_fn.loc[(df_fn['time'] < 1000) & (df_fn['dist'] < 100) & 
                 (df_fn['time'] > 0) & (df_fn['monkey'] == sel_mky)].copy() #& (dfa2['monkey']=='MY')

time_bins = np.arange(0, 900, 100)
tang_bins = np.arange(0, 60, 10)
#speed_bins = np.round(np.arange(0, 0.4, 0.05), 2)
#dist_bins = np.round(np.arange(0, 130, 10), 2)
#progn_bins = np.round(np.arange(-1, 1.2, 0.2), 2)

df_hm['time_bin'] = pd.cut(df_hm['time'], bins=time_bins, labels=time_bins[:-1])
df_hm['tang_bin'] = pd.cut(df_hm['tgt_traj_anga'], bins=tang_bins, labels=tang_bins[:-1])
#df_hm['speed_bin'] = pd.cut(df_hm['speed'], bins=speed_bins, labels=speed_bins[:-1])
#df_hm['dist_bin'] = pd.cut(df_hm['dist'], bins=dist_bins, labels=dist_bins[:-1])
#df_hm['progn_bin'] = pd.cut(df_hm['progn'], bins=progn_bins, labels=progn_bins[:-1])

meas = 'prog'
x_bin = 'time_bin'
y_bin = 'tang_bin'
prog_cond = df_hm.groupby(by=['uncertainty', 'file_index', x_bin, y_bin], as_index=False, observed=False)[[meas]].mean()

crs = prog_cond[prog_cond.uncertainty == 'CRS']
cld = prog_cond[prog_cond.uncertainty == 'CLD']

diff_df = crs.copy()
diff_df[meas] = crs[meas].values - cld[meas].values

diff_df = diff_df.groupby(by=[x_bin, y_bin], as_index=False, observed=False)[[meas]].mean()
diff_df = diff_df.pivot_table(index=y_bin, columns=x_bin, values=meas, observed=False)

#% PLOT: Heatmap diff
plt.figure(figsize=(ctm*14, ctm*8))
ax = sns.heatmap(diff_df, cmap=plt.get_cmap('RdBu_r'), center=0)
plt.xlabel('Time bin [ms]', fontsize=14)
plt.ylabel('Trajectory angle bin [°]', fontsize=14)
plt.xticks(ticks=np.arange(0, len(time_bins), 1), labels=time_bins, fontsize=12)
plt.yticks(ticks=np.arange(0, len(tang_bins), 1), labels=tang_bins, fontsize=12)
ax.set_title(sel_mky, fontsize=14, fontweight='bold')
ax.invert_yaxis()
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)

if save_figs:
    plt.savefig(os.path.join(meta['figfolder'], f"prog_hmap_{meta['subject'][0].upper()}.{fmt}"), format=fmt, dpi=1200)
