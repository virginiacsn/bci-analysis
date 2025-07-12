# Script containing bci/neural analysis functions

# Author: Virginia Casasnovas
# Date: 2021-10-07

import os
import numpy as np
import pandas as pd
import sklearn
from utils import *

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
    filtered_df = filtered_df.drop(columns=['Trial', 'BCI_on', 'IDLE_on', 'Rot_ang', 'SC', 'rew_sel', 'hit_sel'], axis=1)
    
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
        
            K = P @ H.T @ np.linalg.inv(H @ P @ H.T + Q)
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
        
    Ks = P @ H1.T @ np.linalg.inv(H1 @ P @ H1.T + Q) 
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
            
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + Q)
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