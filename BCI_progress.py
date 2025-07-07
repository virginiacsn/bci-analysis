# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:44:02 2024

@author: VCasasnovas
"""
#%% --- SECTION: Imports ---
import numpy as np
import os, glob
import scipy.io
import scipy.signal
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
import seaborn as sns 
from utils import *
import random as rnd
from numpy.linalg import inv
import sklearn
import pickle
import matplotlib.colors as mcolors       

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

# --- SECTION: Inits ---
# Path definition
data_root = '/Users/virginia/Desktop/'

meta = {
    'project' : '', # for the folder in data-preproc
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

#%% --- SECTION: Utility Functions ---
def task_selection(file_path, file_index):
    """
    Loads and processes task data file (txt) into filtered DataFrame.

    Parameters
    ----------
    file_path : str
        Path to task data file
    file_index : int
        Index to label file

    Returns
    -------
    filtered_df : pd.Dataframe
        DataFrame containing selected task variables (cols) and trials (rows) 
    """
    
    df = txt2df(file_path)
    df['monkey'] = file_path.split(os.path.sep)[-2]
    
    bool_cols = ['BCI_on', 'IDLE_on', 'Hit', 'Fb_uncert']
    df[bool_cols] = df[bool_cols].apply(str2bool)

    float_cols = ['Time', 'Trial', 'tgt_x', 'tgt_y', 'Rot_ang', 'SC']
    df[float_cols] = df[float_cols].astype(float)
    
    df['file_index'] = file_index
    df['target_direction'] = angle_wrap(np.rad2deg(np.arctan2(df['tgt_y'], df['tgt_x']))).astype(int)
    
    # Compute hit number based on reward stage for trial re-indexing
    df['rew_sel'] = df['Stage'] == 'REWARD_BCI_1'
    dfg = df.groupby(by='Trial')[['rew_sel']].any().rename(columns={'rew_sel': 'hit_sel'})
    dfg['hit_number'] = np.cumsum(dfg['hit_sel'].astype(int))
    
    df = pd.merge(df, dfg, on='Trial', how='right')
    df = df[df['hit_sel']]
    
    # Select trials after decoding calibration
    filtered_df = df[(df['hit_number'] >= 192) & (df['SC'] == 0)]
    
    return filtered_df
 
def trackers_selection(file_path):
    """
    Extracts cursor position and velocity from tracker file (txt).

    Parameters
    ----------
    file_path : str
        Path to tracker data file

    Returns
    -------
    pos: np.ndarray
        Array containing time, x- and y-position of cursor
    vel: np.ndarray
        Array containing time, x- and y-velocity of cursor
    """

    data = np.loadtxt(file_path, skiprows=1)
    pos = data[:, [0, 5, 6]]
    vel = data[:, [7, 8]]
    
    return pos, vel

def reconstruct_KF(task_df, spike_file, decoder_file, pos, vel):
    """
    Reconstructs cursor position using Kalman filter (KF) decoder based on spike data 
    and filters task DataFrame to include times from go-cue to movement offset and 
    well-reconstructed-position trials to simulate experimental conditions.

    Parameters
    ----------
    task_df : pd.DataFrame
        Task data DataFrame
    spike_file : str
        Path to spike data file (csv)
    decoder_file : str
        Path to decoder parameters file (mat)
    pos : np.ndarray
        True cursor position array [time, px, py]
    vel : np.ndarray
        True xursor velocity array [vx, vy].

    Returns
    -------
    pos_true : list of np.ndarray
        True cursor position per trial
    pos_rec : list of np.ndarray
        KF-reconstructed positions per trial
    uncert_ids : list of np.ndarray
        Feedback uncertainty flags per trial (0 for CRS, low uncert and 1 for CLD, high uncert)
    df : pd.DataFrame
        DataFrame including selected trial timings (from go-cue to movement offset) and
        well-reconstructed-position trials
    """
    
    decoder = scipy.io.loadmat(decoder_file)
    H = decoder['dH_stored']
    A = decoder['A_stored']
    Q = decoder['Q_stored']
    W = decoder['W_stored']
    
    spike_data = pd.read_csv(spike_file).to_numpy()
    spike_time = spike_data[:, 0]
    firing = spike_data[:, 1:].T/0.05
    Z = firing

    hit_number = np.unique(task_df['hit_number'])
    start_stage = 'LEAVE_FIX_MEM_BCI_1' # From go-cue to movement onset
    end_stage = 'ACQUIRE_MEM_TGT_BCI_1' # From movement onset to movement offset
    dim = 2
    
    pos_true = []
    pos_rec = []
    uncert_ids = []
    
    df = pd.DataFrame()
    
    for i in hit_number:
        
        trial_sel = (task_df['hit_number'] == i) & (
            (task_df['Stage'] == start_stage) | (task_df['Stage'] == end_stage))
        trial_times = task_df.loc[trial_sel, 'Time'].values
        
        if len(trial_times) < 2:
            continue

        t_start = trial_times[0]
        idx_spk_start = np.argmin(abs(spike_time-t_start))
        idx_pos_start = np.argmin(abs(pos[:,0]-t_start))    

        P = np.zeros(W.shape)
        xr = np.zeros((dim*2+1, len(trial_times)+1))
        xr[0, :] = 1
        
        for t in range(len(trial_times)):
            xp = A @ xr[:, t]
    
            P = A @ P @ A.T + W
            P[0:3, :] = 0
            P[3:5, 0:3] = 0
        
            K = P @ H.T @ inv(H @ P @ H.T + Q)
            xr[:, t+1] = xp + K @ (Z[:, idx_spk_start+t] - H @ xp)
            
            P = (np.eye(W.shape[0]) - K @ H) @ P
            
        pos_trial = pos[idx_pos_start:idx_pos_start+len(trial_times), 1:]

        if not(np.isnan(pos_trial).any()):  
            reconst = np.asarray(xr[1:3, :-1].T)
            score_r2 = sklearn.metrics.r2_score(pos_trial, reconst)
            if score_r2 > 0.6:   
                uncert_id = task_df.loc[trial_sel,'Fb_uncert'].values
                uncert_label = np.asarray(['CRS']*len(trial_times))
                uncert_label[uncert_id == 1] = 'CLD'
                
                pos_true.append(pos_trial)
                pos_rec.append(xr[1:3, :].T)
                uncert_ids.append(uncert_id)

                speed = np.linalg.norm(vel[idx_pos_start:idx_pos_start + len(trial_times)]/1000, axis=1)
                dist = np.linalg.norm(pos_trial, axis=1)

                cols = {
                    'monkey': task_df.loc[trial_sel, 'monkey'].values,
                    'file_index': task_df.loc[trial_sel, 'file_index'].values,
                    'hit_number': task_df.loc[trial_sel, 'hit_number'].values,
                    'target_direction': task_df.loc[trial_sel, 'target_direction'].values,
                    'tgt_x': task_df.loc[trial_sel, 'tgt_x'].values,
                    'tgt_y': task_df.loc[trial_sel, 'tgt_y'].values,
                    'fb_uncert': uncert_id,
                    'uncertainty': uncert_label,
                    'is_mov': task_df.loc[trial_sel,'Stage'] != start_stage,
                    'time': np.arange(0, 50*len(trial_times), 50),
                    'pos_x': pos_trial[:, 0],
                    'pos_y': pos_trial[:, 1],
                    'dist': dist,
                    'vel_x': vel[idx_pos_start:idx_pos_start+len(trial_times), 0]/1000,
                    'vel_y': vel[idx_pos_start:idx_pos_start+len(trial_times), 1]/1000,
                    'speed': speed,
                    'spk_sel': list(range(idx_spk_start, idx_spk_start+len(trial_times))),
                    'FR': np.asarray(np.mean(Z[:, idx_spk_start:idx_spk_start+len(trial_times)], axis=0)).flatten()
                    }
                
                df = pd.concat([df, pd.DataFrame(cols)], ignore_index=True)
            else:
                print(f"File: {task_df['file_index'].values[0]}, Trial {i}: R² < 0.6")
        else:
            print(f"File: {task_df['file_index'].values[0]}, Trial {i}: Contains NaNs")
    
    return pos_true, pos_rec, uncert_id, df

def get_kss(decoder_file):
    """
    Computes steady-state Kalman gain based on ().

    Parameters
    ----------
    decoder_file : str
        Path to decoder parameters file (mat)

    Returns
    -------
    Ks : np.ndarray
        Steady-state Kalman gain
    """
    
    decoder = scipy.io.loadmat(decoder_file)
    H = decoder['dH_stored']
    A = decoder['A_stored']
    Q = decoder['Q_stored']
    W = decoder['W_stored']

    A1 = A[2:, 2:]
    A1[0, 2] = 0
    H1 =  np.delete(H, [1, 2], axis=1)
    W1 = W[2:, 2:]
    
    try:
        P = scipy.linalg.solve_discrete_are(A1, H1.T, W1, Q)   
    except np.linalg.LinAlgError:
        P = np.full([3, 3], np.nan)
        print('DARE failed to converge.')
        
    Ks = P @ H1.T @ inv(H1 @ P @ H1.T + Q) 
    Ks = Ks[1:, :]

    return Ks
    
def get_progress_ss(task_df, spike_file, Ks):
    """
    Computes progress based on steady-stake Kalman gain.

    Parameters
    ----------
    task_df : pd.DataFrame
        Task DataFrame
    spike_file : str
        Path to spike data file (csv)
    Ks: np.ndarray
        Steady-state Kalman gain

    Returns
    -------
    task_df : pd.DataFrame
        Task DataFrame including progress variables:
            - proj_xs, proj_ys: x- and y-component of neural vector (spiking vector into behavioral coord)
            - progs: Projection of neural vector onto instantaneous cursor-to-target vector
            - progns: Projection of normalized neural vector onto instantaneous cursor-to-target vector 
    """
    
    spike_data = pd.read_csv(spike_file).to_numpy()

    firing = spike_data[:, 1:]/0.05
    Z = firing
    Z_sel = Z[task_df['spk_sel'].astype(int).values,:]

    # Compute neural vector as projection of firing rates onto Kalman gain 
    proj = Ks @ Z_sel.T

    proj_x = np.asarray(proj[0, :]).flatten()
    proj_y = np.asarray(proj[1, :]).flatten()
    
    proj_norm = np.linalg.norm(np.column_stack((proj_x, proj_y)), axis=1)
    proj_xn = proj_x/proj_norm
    proj_yn = proj_y/proj_norm

    # Projection vector (x,y) 
    task_df['proj_xs'] = proj_x
    task_df['proj_ys'] = proj_y
    
    # Cursor-to-target vector (x,y) 
    ct_x = task_df['tgt_x']-task_df['pos_x']
    ct_y = task_df['tgt_y']-task_df['pos_y']

    ct_norm = np.linalg.norm(np.column_stack((ct_x, ct_y)), axis=1)
    ct_xn = ct_x/ct_norm
    ct_yn = ct_y/ct_norm
    
    # Normalized cursor-to-target vector (x,y) 
    task_df['ct_x'] = ct_xn
    task_df['ct_y'] = ct_yn
    
    # Progress as projection of neural vector onto cursor-to-target vector
    prog = proj_x*ct_xn + proj_y*ct_yn
    task_df['progs'] = prog
    
    # Normalized progress
    progn = proj_xn*ct_xn + proj_yn*ct_yn
    task_df['progns'] = progn
    
    return task_df

def get_progress_time(task_df, spike_file, decoder_file):
    """
    Computes progress based on time-dependent/experimental Kalman gain.

    Parameters
    ----------
    task_df : pd.DataFrame
        Task DataFrame
    spike_file : str
        Path to spike data file (csv)
    decoder_file: str
        Path to decoder parameters file (mat)

    Returns
    -------
    task_df : pd.DataFrame
        Task DataFrame including progress variables:
            - proj_x, proj_y: x- and y-component of neural vector (spiking vector into behavioral coord)
            - prog: Projection of neural vector onto cursor-to-target vector
            - progn: Projection of normalized neural vector onto cursor-to-target vector 
    """

    spike_data = pd.read_csv(spike_file).to_numpy()
    firing = spike_data[:, 1:].T/0.05
    Z = firing
    Z_sel = Z[:, task_df['spk_sel'].astype(int).values]

    decoder = scipy.io.loadmat(decoder_file)
    H = decoder['dH_stored']
    A = decoder['A_stored']
    Q = decoder['Q_stored']
    W = decoder['W_stored']
    
    dim = 2
    num_samples = Z_sel.shape[1]

    proj = np.empty((5, num_samples))

    hit_aux = -1
    for j in range(0, num_samples):
        if task_df.iloc[j,task_df.columns.get_loc('hit_number')] != hit_aux:
            P = np.zeros(W.shape)
            xr = np.zeros((dim*2+1, num_samples))
            xr[0, :] = 1
            hit_aux = task_df.iloc[j, task_df.columns.get_loc('hit_number')]     
            c = 0
                
        xp = A @ xr[:, c]
        
        P = A @ P @ A.T + W
        P[0:3, :] = 0
        P[3:5, 0:3] = 0
            
        K = P @ H.T @ inv(H @ P @ H.T + Q)
        xr[:, c+1] = xp + K @ (Z_sel[:,j] - H @ xp)      
        
        P = (np.eye(W.shape[0]) - K @ H) @ P
        c += 1
    
        proj[:, j] = K @ (Z_sel[:,j] - H[:,0])
        
    proj_x = proj[3, :]
    proj_y = proj[4, :]

    proj_norm = np.linalg.norm(np.column_stack((proj_x, proj_y)), axis=1)
    proj_xn = proj_x/proj_norm
    proj_yn = proj_y/proj_norm

    task_df['proj_x'] = proj_x
    task_df['proj_y'] = proj_y

    ct_x = task_df['tgt_x']-task_df['pos_x']
    ct_y = task_df['tgt_y']-task_df['pos_y']
    
    ct_norm = np.linalg.norm(np.column_stack((ct_x, ct_y)), axis=1)
    ct_xn = ct_x/ct_norm
    ct_yn = ct_y/ct_norm
     
    prog = proj_x * ct_xn + proj_y * ct_yn
    task_df['prog'] = prog
    
    progn = proj_xn * ct_xn + proj_yn * ct_yn
    task_df['progn'] = progn
    
    return task_df

def get_progress_angles(task_df):
    """
    Computes angle between projection vector (from steady-state and time-dependent Kalman gain)
    and cursor-to-target vector. Computes angle between cursor position direction and target direction (both referenced from workspace center (0, 0)). 

    Parameters
    ----------
    task_df : pd.DataFrame
        Task DataFrame
    spike_file : str
        Path to spiking data file (csv)
    decoder_file: str
        Path to decoder parameter data file (mat)

    Returns
    -------
    task_df : pd.DataFrame
        Task DataFrame including angle variables:
            - prog_ang, prog_angs: angle between neural vector and cursor-to-target vector
            - tgt_traj_ang: angle between cursor position and target direction
    """
    
    proj_x = task_df['proj_x']
    proj_y = task_df['proj_y']
    
    proj_xs = task_df['proj_xs']
    proj_ys = task_df['proj_ys']
    
    ct_xn = task_df['ct_x']
    ct_yn = task_df['ct_y']
    
    px = task_df['pos_x']
    py = task_df['pos_y']
    
    tgt_x = task_df['tgt_x']
    tgt_y = task_df['tgt_y']
    
    pxd = np.append(np.diff(px), 0)
    pyd = np.append(np.diff(py), 0)
    
    pxdn = pxd/(np.linalg.norm(np.column_stack((pxd, pyd)), axis=1)+0.0001)
    pydn = pyd/(np.linalg.norm(np.column_stack((pxd, pyd)), axis=1)+0.0001)
    
    task_df['pos_xd'] = pxdn
    task_df['pos_yd'] = pydn

    task_df['prog_ang'] = angle_between(np.column_stack((proj_x, proj_y)),np.column_stack((ct_xn, ct_yn)))
    #task_df['traj_ang'] = angle_between(np.column_stack((proj_x, proj_y)),np.column_stack((pxdn, pydn)))
    task_df['prog_angs'] = angle_between(np.column_stack((proj_xs, proj_ys)),np.column_stack((ct_xn, ct_yn)))
    #task_df['traj_angs'] = angle_between(np.column_stack((proj_xs, proj_ys)),np.column_stack((pxdn, pydn)))
    task_df['tgt_traj_ang'] = angle_between(np.column_stack((tgt_x, tgt_y)),np.column_stack((px, py)))

    return task_df
    
def trim_df_idx(task_df, start_idx, end_idx):
    """
    Trims each trial from task DataFrame based on start and end index.

    Parameters
    ----------
    task_df : pd.DataFrame
        Task DataFrame

    Returns
    -------
    df : pd.DataFrame
        Trimmed task DataFrame
    """  
    
    df = task_df.copy()
    
    if start_idx > 0:
        df.drop(df.groupby(['hit_number'])['time'].head(start_idx).index, inplace=True)
        
    if end_idx > 0:        
        df.drop(df.groupby(['hit_number'])['time'].tail(end_idx).index, inplace=True)

    return df.reset_index(drop=True)

def tortuosity(x, y):
    """
    Computes tortuosity of trajectory.

    Parameters
    ----------
    x, y : np.ndarray
        x- and y-position of cursor trajectory from a single trial

    Returns
    -------
    tort : float
        Tortuosity measure for trial
    """     

    path_len = np.cumsum(np.linalg.norm([x[1:]-x[-1], y[1:]-y[-1]], axis=0))
    sgt_dist = np.linalg.norm([x-x[0], y-y[0]], axis=0)
    tort = path_len[-1]/sgt_dist[-1]
    
    return tort
    
def bin_tortuosity(task_df):
    """
    Computes tortuosity for each trial and bins trials into low- and high-tortuosity based on overall median and within-uncertainty median.

    Parameters
    ----------
    task_df : pd.DataFrame
        Task DataFrame

    Returns
    -------
    df : Task DataFrame including tortuosity variables:
        - tort: Tortuosity of cursor trajectory
        - tort_bin: Trial bin based on overall tortuosity median (1: lower, 2: higher)
        - tort_binu: Trial bin based on within-uncertainty tortuosity median (1: lower, 2: higher)
    """      
    
    df = task_df.copy()
    df['tort'] = np.nan
    
    for i in df['hit_number'].unique(): 
        trial_sel = (df['hit_number'] == i)
    
        px = df.loc[trial_sel, 'pos_x'].values
        py = df.loc[trial_sel, 'pos_y'].values
        
        tort = tortuosity(px, py) 
        df.loc[trial_sel, 'tort'] = tort
        
    tort_median = df['tort'].median()
    df['tort_bin'] = np.where(df['tort'] < tort_median, 1, 2)
    
    df['tort_binu'] = np.nan
    for u in [0, 1]:
        unc_sel = df['fb_uncert'] == u
        median_u = df.loc[unc_sel, 'tort'].median()
        df.loc[unc_sel & (df['tort'] < median_u), 'tort_binu'] = 1
        df.loc[unc_sel & (df['tort'] >= median_u), 'tort_binu'] = 2

    return df

def bin_RT_MT(task_df):
    """
    Computes reaction, movement time and late movement time for each trial. Reaction and 
    movement times based on speed threshold crossing (0.05 mm/ms) as in experiment. Late movement time defined as time starting from cursor crossing 60mm of distance from the workspace center until movement offset.
    For each variable, computes trial bin based on median value for each target direction.

    Parameters
    ----------
    task_df : pd.DataFrame
        Task DataFrame

    Returns
    -------
    df : Task DataFrame including tortuosity variables:
        - RT, MT, MT_late: reaction, movement and late movement timings
        - RT_bin, MT_bin, MT_late_bin: Trial bin based on median for each target direction (1: lower, 2: higher)
    """  

    df = task_df.copy()
    df['RT'] = np.nan
    df['MT'] = np.nan
    df['MT_late'] = np.nan

    for i in df['hit_number'].unique():
        
        trial_sel = (df['hit_number'] == i)
        trial_mov_sel = ((df['hit_number'] == i) & (df['is_mov'] == True))

        time = df.loc[trial_mov_sel, 'time'].values
        
        dist = np.sqrt(df.loc[trial_mov_sel,'pos_x']**2+df.loc[trial_mov_sel,'pos_y']**2)
        idx_late = np.where(dist >= 60)[0]
        
        df.loc[trial_sel, 'RT'] = time[0] - 50
        df.loc[trial_sel, 'MT'] = time[-1] - time[0]
        
        if idx_late.size > 0:
            df.loc[trial_sel, 'MT_late'] = time[-1] - time[idx_late[0]]
    
    df['RT_bin'] = 1
    df['MT_bin'] = 1 
    df['MT_late_bin'] = 1

    rt_median = df.groupby(['target_direction'])['RT'].median()
    mt_median = df.groupby(['target_direction'])['MT'].median()
    mt_late_median = df.groupby(['target_direction'])['MT_late'].median()

    for d in df['target_direction'].unique():
        dir_sel = df['target_direction'] == d
        df.loc[dir_sel & (df['RT'] >= rt_median[d]),'RT_bin'] = 2
        df.loc[dir_sel & (df['MT'] >= mt_median[d]),'MT_bin'] = 2        
        df.loc[dir_sel & (df['MT_late'] >= mt_late_median[d]),'MT_late_bin'] = 2
        
    return df
  
def get_submov_spd(task_df, trial):
    """
    Computes submovement indexes for each trial based on trajectory speed. First detects indexes finding speed minimums, then refines selection based on trajectory change (only keeps indexes with whose trajectory angle change is higher than 15 deg). Checks trajectory change for neighoring indexes and keeps index with the maximum change.

    Parameters
    ----------
    task_df : pd.DataFrame
        Task DataFrame
    trial: int
        Trial number

    Returns
    -------
    submov_all : list
        Indexes for submovement start.
    """     

    df = task_df.copy()
    trial_sel = df['hit_number'] == trial
    submov_all = [] 

    if not any(trial_sel):
        return np.array([])
    
    spd = df.loc[trial_sel, 'speed'].values
    px = df.loc[trial_sel, 'pos_x'].values
    py = df.loc[trial_sel, 'pos_y'].values

    valley_idx, _ = scipy.signal.find_peaks(-spd, height=0.4*(min(-spd)))
    valley_idx = valley_idx[valley_idx > 2]
    
    submov_min = [] 
    sbm_min_ang = []

    for j in valley_idx:
        sweep_idx = [j-1, j, j+1]
        ang = []
    
        for k in sweep_idx:
            if k < len(px)-2:   
                p_1 = np.array([px[k]-px[k-1], py[k]-py[k-1]])[np.newaxis, :]        
                p_2 = np.array([px[k+1]-px[k], py[k+1]-py[k]])[np.newaxis, :]     
                ang.append(abs(angle_between(p_1, p_2)))
        
        if ang:
            ang_max = max(ang)
            ang_max_idx = np.argmax(ang)
            if ang_max > 15:   
                submov_min.append(sweep_idx[ang_max_idx])    
                sbm_min_ang.append(ang_max)    
    
    if not submov_min:
        return np.array([])

    submov_min = np.array(submov_min)
    sbm_min_ang = np.array(sbm_min_ang)
    submov_all, u_idx = np.unique(submov_min, return_index=True)
    ang_all = sbm_min_ang
    ang_all = ang_all[u_idx]
    
    idx_close = np.where(np.diff(submov_all) == 1)[0]
    idx_out = []
    for l in idx_close:
        idx = [l, l+1]
        idx_out.append(idx[np.argmin(ang_all[l:l+2])])
        
    submov_all = np.delete(submov_all, idx_out).astype(int)
        
    return submov_all

def get_submov_spd_df(task_df):
    """
    Computes whether index is a submovement start based on trajectory speed and whether submovement is corrective or not based on trajectory angle change. 

    Parameters
    ----------
    task_df : pd.DataFrame
        Task DataFrame

    Returns
    -------
    df : pd.DataFrame
        Task DataFrame including submovement variables:
            - is_sbm: bool of whether index is submovement start
            - is_sbm_d: bool of whether submovement is corrective or not
    """     

    df = task_df.copy()
    df['is_sbm'] = False
    df['is_sbm_d'] = False

    for i in df['hit_number'].unique():      
        trial_sel = df['hit_number'] == i
        trial_idx = df.index[trial_sel]
        
        submov_all = get_submov_spd(df, i)
        if len(submov_all) == 0:
            continue
        
        sbm_idx = trial_idx[submov_all]
        df.loc[sbm_idx, 'is_sbm'] = True
        df.loc[~df['is_mov'], 'is_sbm'] = False
        
        future_idx = sbm_idx+2
        future_idx = future_idx[future_idx < len(df)]

        ang = abs(df.loc[future_idx, 'tgt_traj_ang'].values)-abs(df.loc[sbm_idx, 'tgt_traj_ang'].values)
       
        df.loc[sbm_idx[ang <= 0], 'is_sbm_d'] = True
        df.loc[~df['is_mov'], 'is_sbm_d'] = False
            
    return df  

def get_submov_nn(task_df):
    """
    Assigns bin number relative to submovement start to neighboring indexes within window [-2, 2]. Assigns whether submovement is corrective or not to neighboring indexes.

    Parameters
    ----------
    task_df : pd.DataFrame
        Task DataFrame

    Returns
    -------
    df1 : pd.DataFrame
        Task DataFrame with rows from submovement start or neighboring indexes and additional submovement variables:
            - sbm_idx: bin number from -2 to 2 relative to submovement start
            - sbm_idx_d: bool of whether submovement is corrective or not, assigned to submovement start and neighboring indexes
    """      

    df = task_df.copy()
    df1 = pd.DataFrame()

    df['sbm_idx'] = np.nan 
    df.loc[df['is_sbm'], 'sbm_idx'] = 0
    df['sbm_idx_d'] = False 
    
    for i in df['hit_number'].unique():      
        trial_sel = df['hit_number'] == i
        dft = df.loc[trial_sel]
        trial_min = dft.index[0]
        
        if dft['is_sbm'].any():
            sbm_sel = np.where(dft['sbm_idx'] == 0)[0]
            
            for j in sbm_sel:
                for k in range(-2, 3):
                    if j+k < trial_sel.sum():
                        df.at[j+k+trial_min, 'sbm_idx'] = k
                        if df.at[j+trial_min, 'is_sbm_d']:
                            df.at[j+k+trial_min, 'sbm_idx_d'] = True
                        df1 = pd.concat([df1, df.iloc[[j+k+trial_min]]], ignore_index=True)           
    return df1  

def get_submov_arc(task_df, trial):
    """
    Computes submovement indexes for each trial based on changes in arc length of cursor trajectory. First determines whether there is a submovement based on a difference in arc length larger than 20 mm over the trial. Then computes the indexes of submovement start based on maximums and minimums of arc length. 

    Parameters
    ----------
    task_df : pd.DataFrame
        Task DataFrame
    trial: int
        Trial number

    Returns
    -------
    submov_all : list
        Indexes for submovement start.
    """  

    trial_sel = task_df['hit_number'] == trial

    if not trial_sel.any():
        return np.array([])
    
    px = task_df.loc[trial_sel,'pos_x'].values
    py = task_df.loc[trial_sel,'pos_y'].values
    tgt_ang = task_df.loc[trial_sel,'tgt_traj_ang'].values
    dist = np.linalg.norm([px, py], axis=0)
    arclen = dist*np.deg2rad(tgt_ang)
    
    submov_all = np.array([]) 

    if (abs(max(arclen)-min(arclen)) > 20):
        valley_idx, _ = scipy.signal.find_peaks(-arclen, height=0.5*(min(-arclen)))
        valley_idx = valley_idx[valley_idx > 2]
    
        peak_idx, _ = scipy.signal.find_peaks(arclen, height=0.5*(max(arclen)))
        peak_idx = peak_idx[peak_idx > 2]
    
        submov_all = np.append(valley_idx, peak_idx)
        
    return submov_all
              
def get_submov_arc_df(task_df):
    """
    Identifies indexes of trajectory from the first submovement (straight segment). First submovement is detected by changes in trajectory arc length. 

    Parameters
    ----------
    task_df : pd.DataFrame
        Task DataFrame

    Returns
    -------
    df : pd.DataFrame
        Task DataFrame with rows from first submovement and additional submovement variable:
            - is_sgt: bool of whether trial trajectory is straight (only one submovment) or not (corrective submovements were made)
    """  

    df = pd.DataFrame()
    
    for i in task_df['hit_number'].unique():      
        trial_sel = task_df['hit_number'] == i
        
        submov_all = get_submov_arc(task_df, i)
        
        if len(submov_all) > 0:         
            trial_min = np.min(np.where(trial_sel == True)[0])
            idx = submov_all+trial_min
            
            df = pd.concat([df, task_df.iloc[range(trial_min, idx[0])]], ignore_index=True)    
            df.loc[df['hit_number']==i, 'is_sgt'] = False

        else:
            df = pd.concat([df, task_df.loc[trial_sel, :]], ignore_index=True)       
            df.loc[df['hit_number']==i, 'is_sgt'] = True
   
    return df
    
def bin_traj(task_df, bin_start=1, bin_end=10.5):
    """
    Bins trial and computes trial length.

    Parameters
    ----------
    task_df : pd.DataFrame
        Task DataFrame

    Returns
    -------
    task_df : pd.DataFrame
        Task DataFrame with additional variables:
            - traj_bin: Bin number [1:10]
            - trial_len: number of timepoints in trial
    """  

    for i in task_df['hit_number'].unique():      
        trial_sel = task_df['hit_number'] == i
        trial_len = sum(trial_sel)
        
        bin_vec = np.linspace(1, 10.5, num = trial_len)
        
        task_df.loc[trial_sel, 'traj_bin'] = np.round(bin_vec*2)/2
        task_df.loc[trial_sel, 'trial_len'] = trial_len

    return task_df
        
def change_mky_name(df):
    """
    Changes name of monkey in DataFrame.
    """  

    name_map = {'yod': 'MY', 'zep': 'MZ'}
    df['monkey'] = df['monkey'].replace(name_map)
    
    return df
    
#%%
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
