"""
Kinematic Behavior Classification (KBC) Module

This module provides functionality for analyzing and classifying movement patterns
from motion capture or video data. It includes methods for processing time series
data, performing kinematic analysis, and clustering movement patterns.
"""

import numpy as np
import pandas as pd
import os
import json
from typing import Dict, List, Tuple, Optional, Any, Set
import utils_cleaned as utils
from utils_cleaned import ERR9, ERR9999, ERR999, SUBDIRS  # Add SUBDIRS to imports
import logging
from datetime import datetime
import argparse  # Add at the top with other imports
import random
from tqdm import tqdm  # Add tqdm for progress bars

# Constants
FPS = 30.0
DT = 1 / FPS
STRIDE = int(FPS/2)
WINDOW_SIZE = int(FPS/2)

# Standard points for data standardization
STANDARD_POINTS = [12, 16]

# Region of Interest (ROI) definitions
ALL_PARTS = ['LH', 'RH', 'Head', 'LE']  # Left Hand, Right Hand, Head, Lower Extremity
CLASSES = ALL_PARTS

# ROI mapping for consistent naming
ROI_MAP = {
    'LH': 'left_hand',
    'RH': 'right_hand',
    'Head': 'head',
    'LE': 'lower_extremity'
}

# List of emotions to process
EMOTIONS = [
    'admiration',
    'anger',
    'despair',
    'disgust',
    'envy',
    'fear',
    'guilt',
    'happiness',  # Primary name for happy/happiness
    'pride',
    'sadness',    # Primary name for sad/sadness
    'shame',
    'surprise'
]

# Emotion name mapping for variations and misspellings
EMOTION_MAP = {
    # Happiness variations
    'happy': 'happiness',
    'Happy': 'happiness',
    'Happiness': 'happiness',
    'happiness': 'happiness',
    
    # Sadness variations
    'sad': 'sadness',
    'Sad': 'sadness',
    'Sadness': 'sadness',
    'sadness': 'sadness',
    'sad - part (2)': 'sadness',
    'sad - part 1': 'sadness',
    
    # Despair variations (including common misspellings)
    'despair': 'despair',
    'Despair': 'despair',
    'DESPAIR': 'despair',
    'dispair': 'despair',  # common misspelling
    'Dispair': 'despair',  # common misspelling with capital
    'dispar': 'despair',   # another common misspelling
    'Dispar': 'despair',   # another common misspelling with capital
    
    # Other potential misspellings found
    'admiraton': 'admiration',  # misspelling of admiration
    'guilit': 'guilt',  # misspelling of guilt
}

# Set up logging first
# Create logs directory
logs_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Create results directory structure
results_dir = os.path.join(os.getcwd(), 'results')
for emotion in EMOTIONS:
    emotion_dir = os.path.join(results_dir, emotion)
for subdir in ['time_series', 'dtw', 'clustering', 'statistical']:
    os.makedirs(os.path.join(emotion_dir, subdir), exist_ok=True)
    if subdir == 'dtw':  # DTW outputs remain ROI-specific
        for roi in [roi.lower() for roi in CLASSES]:
            os.makedirs(os.path.join(emotion_dir, subdir, roi), exist_ok=True)

# Create data directories
data_dir = os.path.join(os.getcwd(), 'data')
os.makedirs(data_dir, exist_ok=True)

# Set up debug logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(logs_dir, f'debug_log_{timestamp}.txt')

debug_logger = logging.getLogger('debug')
debug_logger.setLevel(logging.DEBUG)

# File handler - keep all debug logs in file
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
debug_logger.addHandler(file_handler)

# Console handler - only show warnings and errors
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Changed from INFO to WARNING
console_handler.setFormatter(formatter)
debug_logger.addHandler(console_handler)

def debug_data_structure(data: Dict, prefix: str = "") -> None:
    """Helper function to log data structure information"""
    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, list)):
                debug_logger.debug(f"{new_prefix} (type: {type(value).__name__}, size: {len(value)})")
                debug_data_structure(value, new_prefix)
            else:
                debug_logger.debug(f"{new_prefix} = {type(value).__name__}")
    elif isinstance(data, list):
        debug_logger.debug(f"{prefix} (list length: {len(data)})")
        if data and len(data) > 0:
            debug_logger.debug(f"{prefix}[0] type: {type(data[0]).__name__}")

def verify_data_structure(data, stage):
    """Verify the structure of data at different stages of processing."""
    debug_logger.debug(f"Data Structure Verification - {stage}")
    if isinstance(data, dict):
        debug_logger.debug(f"Dictionary structure with keys: {list(data.keys())}")
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                debug_logger.debug(f"Key '{key}' contains {len(value)} items")
    elif isinstance(data, list):
        debug_logger.debug(f"List structure with {len(data)} items")
        if data and isinstance(data[0], (dict, list)):
            debug_logger.debug(f"First item contains {len(data[0])} elements")
    else:
        debug_logger.debug(f"Data is of type {type(data)}")

def save_time_series_data(v_total: Dict[str, List[float]], 
                         a_total: Dict[str, List[float]], 
                         time_series_data: Dict[str, Any]) -> None:
    """
    Saves velocity, acceleration, and time series data to disk.
    
    Args:
        v_total: Dictionary of velocity data
        a_total: Dictionary of acceleration data
        time_series_data: Dictionary containing time series data
    """
    for emotion in EMOTIONS:
        emotion_dir = os.path.join(results_dir, emotion, 'time_series')
        os.makedirs(emotion_dir, exist_ok=True)
        
        emotion_variations = [emotion]
        for mapped_key, mapped_value in EMOTION_MAP.items():
            if mapped_value == emotion:
                emotion_variations.append(mapped_key)
        emotion_variations = list(dict.fromkeys(emotion_variations))
        emotion_variations_lc = [e.lower() for e in emotion_variations]
        
        v_emotion = {k: v for k, v in v_total.items()
                     if any(var in k.lower() for var in emotion_variations_lc)}
        a_emotion = {k: v for k, v in a_total.items()
                     if any(var in k.lower() for var in emotion_variations_lc)}
        
        if v_emotion:
            np.savez_compressed(os.path.join(emotion_dir, 'v_total.npz'), **v_emotion)
        if a_emotion:
            np.savez_compressed(os.path.join(emotion_dir, 'a_total.npz'), **a_emotion)

        time_series_data_converted = {
            metric: {
                roi: {
                    pid: val.tolist()
                    for pid, val in roi_data.items()
                    if any(var in pid.lower() for var in emotion_variations)
                }
                for roi, roi_data in metric_data.items()
            }
            for metric, metric_data in time_series_data.items()
        }

        has_data = any(
            any(roi_dict for roi_dict in metric_dict.values())
            for metric_dict in time_series_data_converted.values()
        )

        if has_data:
            with open(os.path.join(emotion_dir, 'time_series_data.json'), 'w') as f:
                json.dump(time_series_data_converted, f, indent=4)

        if v_emotion or a_emotion or has_data:
            print(f"Time series data saved successfully for {emotion}.")

def load_time_series_data() -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Loads precomputed time series data from disk.
    
    Returns:
        Tuple containing v_total, a_total, and time_series_data if files exist,
        otherwise returns (None, None, None)
    """
    v_total = {}
    a_total = {}
    time_series_data = {
        'velocity': {roi.lower(): {} for roi in CLASSES},
        'acceleration': {roi.lower(): {} for roi in CLASSES}
    }
    
    data_found = False
    
    for emotion in EMOTIONS:
        emotion_dir = os.path.join(results_dir, emotion, 'time_series')
        v_total_file = f"{emotion_dir}/v_total.npz"
        a_total_file = f"{emotion_dir}/a_total.npz"
        time_series_file = f"{emotion_dir}/time_series_data.json"
        
        if all(os.path.exists(f) for f in [v_total_file, a_total_file, time_series_file]):
            print(f"📂 Loading time series data for {emotion}...")
            data_found = True

            # Load velocity and acceleration data
            v_emotion = dict(np.load(v_total_file, allow_pickle=True))
            a_emotion = dict(np.load(a_total_file, allow_pickle=True))
            v_total.update(v_emotion)
            a_total.update(a_emotion)

            # Load time series data
            with open(time_series_file, "r") as f:
                time_series_data_loaded = json.load(f)

            # Convert lists back to NumPy arrays and update the main dictionary
            for metric in ['velocity', 'acceleration']:
                for roi in time_series_data_loaded[metric]:
                    for pid, val in time_series_data_loaded[metric][roi].items():
                        time_series_data[metric][roi][pid] = np.array(val)

    if data_found:
        return v_total, a_total, time_series_data

    print("⚠️ Time series data not found, needs to be recomputed.")
    return None, None, None

def kinematic_analysis(v_total: Dict[str, List[float]], 
                      a_total: Dict[str, List[float]], 
                      max_num_win: int = 0,
                      actor_filter: Optional[set] = None) -> int:
    """
    Performs kinematic analysis on movement data.
    
    Args:
        v_total: Dictionary to store velocity data
        a_total: Dictionary to store acceleration data
        max_num_win: Maximum number of windows
        
    Returns:
        Updated maximum number of windows
    """
    debug_logger.debug("\nStarting kinematic analysis...")
    debug_logger.debug(f"Number of text files to process: {len(text_files)}")
    
    # First pass: collect all data to find global maxima
    debug_logger.debug("\nCalculating global maxima for consistent plotting...")
    global_v_max = 0
    global_a_max = 0
    
    print("\n📈 Calculating global maxima...")
    for file_path in tqdm(text_files, desc="First pass", leave=False):
        debug_logger.debug(f"\nProcessing file for global maxima: {file_path}")
        data, x, y, missing_numbers, act, actorNumber = utils.read_data(file_path)
        # If filtering to specific actors, skip others
        if actor_filter is not None:
            try:
                if str(actorNumber) not in {str(a) for a in actor_filter}:
                    continue
            except Exception:
                if str(actorNumber) not in {str(a) for a in actor_filter}:
                    continue
        if len(data) > 0 and (act in EMOTIONS or act in EMOTION_MAP):  # Only process if emotion is in our list or mapped
            # Standardize data
            ref_dist = utils.calc_ref(x, y, [0, 1, 2], STANDARD_POINTS)
            L = len(x)
            utils.standardize(data, x, y, ref_dist, L)

            # Compute number of windows
            num_windows = int(np.floor((L - WINDOW_SIZE) / STRIDE) + 1)
            
            # Process each ROI to find maxima
            for j in CLASSES:
                try:
                    (v_frame, a_frame, _, _, _, _) = utils.findVelAcc(
                        data, x, y, [j], missing_numbers, num_windows)
                    
                    if v_frame == ERR9 or a_frame == ERR9:
                        continue
                    
                    # Validate and update global maxima
                    if isinstance(v_frame, (list, np.ndarray)) and len(v_frame) > 0:
                        current_v_max = max(v_frame)
                        debug_logger.debug(f"\nVelocity values for Actor {actorNumber} {act} - {j}:")
                        debug_logger.debug(f"Min: {min(v_frame):.4f}")
                        debug_logger.debug(f"Max: {current_v_max:.4f}")
                        debug_logger.debug(f"Mean: {np.mean(v_frame):.4f}")
                        debug_logger.debug(f"Std: {np.std(v_frame):.4f}")
                        
                        if current_v_max > 1000:  # Sanity check for unrealistic values
                            debug_logger.debug(f"Warning: Unrealistic velocity value {current_v_max} found in Actor {actorNumber} {act} for {j}")
                            continue
                        global_v_max = max(global_v_max, current_v_max)
                        
                    if isinstance(a_frame, (list, np.ndarray)) and len(a_frame) > 0:
                        current_a_max = max(a_frame)
                        debug_logger.debug(f"\nAcceleration values for Actor {actorNumber} {act} - {j}:")
                        debug_logger.debug(f"Min: {min(a_frame):.4f}")
                        debug_logger.debug(f"Max: {current_a_max:.4f}")
                        debug_logger.debug(f"Mean: {np.mean(a_frame):.4f}")
                        debug_logger.debug(f"Std: {np.std(a_frame):.4f}")
                        
                        if current_a_max > 1000:  # Sanity check for unrealistic values
                            debug_logger.debug(f"Warning: Unrealistic acceleration value {current_a_max} found in Actor {actorNumber} {act} for {j}")
                            continue
                        global_a_max = max(global_a_max, current_a_max)
                        
                except Exception as e:
                    debug_logger.debug(f"Error calculating maxima for Actor {actorNumber} {act} - {j}: {e}")
                    continue
    
    debug_logger.debug(f"\nGlobal maxima calculated:")
    debug_logger.debug(f"Maximum velocity: {global_v_max}")
    debug_logger.debug(f"Maximum acceleration: {global_a_max}")
    
    # Second pass: process and plot with consistent y-axis limits
    print("\n📊 Processing files and generating plots...")
    for file_path in tqdm(text_files, desc="Second pass", leave=False):
        debug_logger.debug(f"\nProcessing file: {file_path}")
        data, x, y, missing_numbers, act, actorNumber = utils.read_data(file_path)
        # If filtering to specific actors, skip others
        if actor_filter is not None:
            try:
                if str(actorNumber) not in {str(a) for a in actor_filter}:
                    continue
            except Exception:
                if str(actorNumber) not in {str(a) for a in actor_filter}:
                    continue
        if len(data) > 0 and (act in EMOTIONS or act in EMOTION_MAP):  # Only process if emotion is in our list or mapped
            debug_logger.debug(f"Data loaded successfully for actor {actorNumber}, action {act}")
            debug_logger.debug(f"Number of frames: {len(x)}")
            
            # Standardize data
            ref_dist = utils.calc_ref(x, y, [0, 1, 2], STANDARD_POINTS)
            L = len(x)
            utils.standardize(data, x, y, ref_dist, L)

            # Compute number of windows
            num_windows = int(np.floor((L - WINDOW_SIZE) / STRIDE) + 1)
            max_num_win = max(max_num_win, num_windows)
            debug_logger.debug(f"Number of windows: {num_windows}")

            # Initialize DataFrame for results
            df = pd.DataFrame({
                'v_auc': [None] * len(CLASSES),
                'v_mean': [None] * len(CLASSES),
                'v_max': [None] * len(CLASSES),
                'v_ent': [None] * len(CLASSES),
                'a_auc': [None] * len(CLASSES),
                'a_mean': [None] * len(CLASSES),
                'a_max': [None] * len(CLASSES),
                'a_ent': [None] * len(CLASSES)
            })
            df.index = CLASSES

            v_f = []
            a_f = []

            v_total[f"{actorNumber} {act}"] = []
            a_total[f"{actorNumber} {act}"] = []

            for j in CLASSES:
                debug_logger.debug(f"\nProcessing {j}...")
                try:
                    (v_frame, a_frame, xcenter, ycenter, x_center_f, y_center_f) = utils.findVelAcc(
                        data, x, y, [j], missing_numbers, num_windows)
                    
                    if v_frame == ERR9 or a_frame == ERR9:
                        debug_logger.debug(f"  - Error processing {j}, skipping...")
                        continue
                    
                    # Validate data before processing
                    if not isinstance(v_frame, (list, np.ndarray)) or not isinstance(a_frame, (list, np.ndarray)):
                        debug_logger.debug(f"  - Invalid data type for {j}: v_frame={type(v_frame)}, a_frame={type(a_frame)}")
                        continue
                        
                    if len(v_frame) == 0 or len(a_frame) == 0:
                        debug_logger.debug(f"  - Empty data for {j}: v_frame length={len(v_frame)}, a_frame length={len(a_frame)}")
                        continue
                        
                    # Additional validation for unrealistic values
                    if max(v_frame) > 1000 or max(a_frame) > 1000:
                        debug_logger.debug(f"  - Warning: Unrealistic values detected for {j}, skipping...")
                        continue
                        
                    (out, v, a) = utils.gen_output_sq(v_frame, a_frame, xcenter, ycenter)

                    v_f.append(v)
                    a_f.append(a)
                    
                    for key, val in out.items():
                        df.at[j, key] = val
                        
                    debug_logger.debug(f"  - Successfully processed {j}")
                    
                    # Plot kinematic values with consistent y-axis limits
                    try:
                        utils.plot_kinematic_values(v_frame, a_frame, j, actorNumber, act, 
                                                 v_max=global_v_max * 1.1,  # Add 10% padding
                                                 a_max=global_a_max * 1.1)  # Add 10% padding
                    except Exception as e:
                        debug_logger.debug(f"  - Error plotting {j}: {str(e)}")
                    
                except Exception as e:
                    debug_logger.debug(f"  - Error processing {j}: {e}")

            if v_f and a_f:  # Only add if we have valid data
                v_total[f"{actorNumber} {act}"] = v_f
                a_total[f"{actorNumber} {act}"] = a_f
                debug_logger.debug(f"Added velocity and acceleration data for {actorNumber} {act}")
            else:
                debug_logger.debug(f"No valid data for {actorNumber} {act}, skipping...")

            # Standardize emotion name before saving to avoid duplicate folders
            try:
                def _std_emotion(name: str) -> str:
                    n = (name or "").strip()
                    # Try direct mapping, then lowercase mapping, else pass-through lowercase
                    if n in EMOTION_MAP:
                        return EMOTION_MAP[n]
                    ln = n.lower()
                    if ln in EMOTION_MAP:
                        return EMOTION_MAP[ln]
                    return ln

                std_act = _std_emotion(act)
            except Exception:
                std_act = act

            utils.save_result(v_f, a_f, df, std_act, actorNumber, None)
        else:
            debug_logger.debug(f"No data found in {file_path} or emotion {act} not in EMOTIONS list")

    debug_logger.debug(f"\nKinematic analysis completed. Max number of windows: {max_num_win}")
    return max_num_win

def analyze_and_cluster(time_series_data: Dict[str, Any], 
                       selected_k: List[int], 
                       regenerate_dtw: bool = False, 
                       regenerate_clustering: bool = False,
                       demographic_file: str = None,
                       emotions: Optional[List[str]] = None,
                       rois: Optional[List[str]] = None,
                       methods: Optional[List[str]] = None,
                       sample_n: Optional[int] = None,
                       no_plots: bool = False,
                       no_social: bool = False,
                       max_series_length: Optional[int] = None,
                       seed: int = 42) -> None:
    """
    Analyzes and clusters movement data using multiple distance metrics.
    
    Args:
        time_series_data: Dictionary containing time series data
        selected_k: List of k values for clustering
        regenerate_dtw: Whether to regenerate DTW matrices
        regenerate_clustering: Whether to regenerate clustering results
        demographic_file: Path to demographic file
        emotions: Optional subset of emotions to run
        rois: Optional subset of ROI names (lowercase) to run
        methods: Optional subset of distance methods: ['DTW','Soft-DTW','Euclidean','Cosine']
        sample_n: Optional number of participants to sample per emotion/ROI
        no_plots: If True, skip plotting to save time
        no_social: If True, skip social score analysis
        max_series_length: If provided, truncate each time series to this many frames
        seed: Random seed for sampling
    """
    debug_logger.debug("\nAnalyzing and clustering data...")
    debug_logger.debug("Time series data structure:")
    debug_logger.debug(f"Available metrics: {list(time_series_data.keys())}")
    rnd = random.Random(seed)
    
    # Load demographic data
    try:
        demographic_data = pd.read_csv(demographic_file)
        debug_logger.debug(f"Loaded demographic data with shape: {demographic_data.shape}")
        debug_logger.debug(f"Demographic data columns: {demographic_data.columns.tolist()}")
    except Exception as e:
        debug_logger.error(f"Error loading demographic data: {str(e)}")
        demographic_data = None
    
    run_emotions = emotions if emotions else EMOTIONS
    for emotion in run_emotions:
        debug_logger.debug(f"\nProcessing emotion: {emotion}")
        
        # Filter data for current emotion and its variations
        # For each emotion, we need to check both the emotion name and any mapped variations
        emotion_variations = [emotion]
        
        # Add all variations that map to this emotion
        for mapped_key, mapped_value in EMOTION_MAP.items():
            if mapped_value == emotion:
                emotion_variations.append(mapped_key)
        
        # Remove duplicates while preserving order
        emotion_variations = list(dict.fromkeys(emotion_variations))
        
        emotion_data = {
            metric: {
                roi: {pid: val for pid, val in roi_data.items() 
                     if any(var in pid.lower() for var in emotion_variations)}
                for roi, roi_data in metric_data.items()
            }
            for metric, metric_data in time_series_data.items()
        }
        
        for metric, ROIs in emotion_data.items():
            debug_logger.debug(f"\nProcessing {metric} data for {emotion}:")
            debug_logger.debug(f"Available ROIs: {list(ROIs.keys())}")
            
            for ROI, series_dict in ROIs.items():
                if rois and ROI not in [r.lower() for r in rois]:
                    continue
                debug_logger.debug(f"\nProcessing {metric} - {ROI} for {emotion}...")
                debug_logger.debug(f"Number of series: {len(series_dict)}")
                
                if not series_dict:
                    debug_logger.debug(f"No data available for {metric} - {ROI} in {emotion}")
                    continue
                    
                # Truncate series and optionally sample participants
                series_items = list(series_dict.items())
                if sample_n is not None and len(series_items) > sample_n:
                    series_items = rnd.sample(series_items, k=sample_n)
                series_dict_prepped = {}
                for pid, val in series_items:
                    try:
                        if max_series_length and hasattr(val, '__len__'):
                            series_dict_prepped[pid] = val[:max_series_length]
                        else:
                            series_dict_prepped[pid] = val
                    except Exception:
                        series_dict_prepped[pid] = val

                # Compute distance matrices
                try:
                    dtw_matrix, soft_dtw_matrix, euclidean_matrix, cosine_matrix, participants = utils.get_distance_matrices(
                        series_dict_prepped, metric, ROI, regenerate=regenerate_dtw,
                        output_dir=os.path.join(results_dir, emotion, 'dtw', ROI)
                    )

                    distance_matrices = {
                        "DTW": dtw_matrix,
                        "Soft-DTW": soft_dtw_matrix,
                        "Euclidean": euclidean_matrix,
                        "Cosine": cosine_matrix
                    }

                    if methods:
                        distance_matrices = {k: v for k, v in distance_matrices.items() if k in methods}

                    # Perform clustering for each distance metric
                    for method, matrix in distance_matrices.items():
                        debug_logger.debug(f"\nClustering with {method} distance matrix for {emotion}...")
                        try:
                            clustering_dir = os.path.join(
                                results_dir,
                                emotion,
                                'clustering',
                                f"{metric}_{ROI}"
                            )
                            results_df = utils.get_clustering_analysis(
                                distance_matrix=matrix,
                                participants=participants,
                                metric=metric,
                                ROI=ROI,
                                method=method,
                                emotion=emotion,
                                demographic_data=demographic_data,
                                k_values=selected_k,
                                output_dir=clustering_dir
                            )
                            
                            debug_logger.debug(f"Clustering completed for {method}")
                            debug_logger.debug(f"Results shape: {results_df.shape}")
                            debug_logger.debug(f"Results columns: {results_df.columns.tolist()}")
                            
                            # Analyze social scores
                            if not no_social:
                                debug_logger.debug(f"\nAnalyzing social scores for {method} in {emotion}...")
                                utils.analyze_cluster_social_scores(
                                    results_df, demographic_data, metric, ROI, method,
                                    output_dir=os.path.join(results_dir, emotion, 'statistical', ROI),
                                    debug_logger=debug_logger
                                )
                                debug_logger.debug(f"Social score analysis completed for {method}")

                            # Generate additional clustering artifacts (membership + heatmap)
                            if not no_plots:
                                try:
                                    utils.visualize_cluster_membership(
                                        results_df=results_df,
                                        metric=metric,
                                        ROI=ROI,
                                        method=method,
                                        emotion=emotion
                                    )
                                except Exception as e:
                                    debug_logger.error(f"Error creating cluster membership viz for {method}: {e}")

                            if not no_plots:
                                try:
                                    heatmap_path = os.path.join(
                                        clustering_dir,
                                        f'heatmap_{method}.png'
                                    )
                                    utils.plot_distance_matrix(
                                        matrix,
                                        title=f'{metric} {ROI} ({method})',
                                        participants=participants,
                                        save_path=heatmap_path
                                    )
                                except Exception as e:
                                    debug_logger.error(f"Error creating distance heatmap for {method}: {e}")
                            
                            # Produce cluster interpretation artifacts (sizes, exemplars, summaries, ARI)
                            try:
                                utils.interpret_clusters(
                                    results_df=results_df,
                                    participants=participants,
                                    distance_matrix=matrix,
                                    metric=metric,
                                    ROI=ROI,
                                    method=method,
                                    emotion=emotion,
                                    debug_logger=debug_logger
                                )
                            except Exception as e:
                                debug_logger.error(f"Error creating interpretation artifacts for {method}: {e}")
                            
                        except Exception as e:
                            debug_logger.error(f"Error in clustering/social score analysis for {method} in {emotion}: {str(e)}")
                            continue

                except Exception as e:
                    debug_logger.error(f"Error processing {metric} - {ROI} for {emotion}: {str(e)}")
                    continue

def saving_time_series(v_total: Dict[str, List[float]], 
                      a_total: Dict[str, List[float]], 
                      time_series_data: Dict[str, Any]) -> None:
    """
    Prepares time series data structure from velocity and acceleration data.
    
    Args:
        v_total: Dictionary of velocity data
        a_total: Dictionary of acceleration data
        time_series_data: Dictionary to store time series data
    """
    debug_logger.debug("\nProcessing velocity and acceleration data...")
    debug_logger.debug(f"Number of velocity entries: {len(v_total)}")
    debug_logger.debug(f"Number of acceleration entries: {len(a_total)}")
    
    # Initialize the data structure with all ROIs
    if 'velocity' not in time_series_data:
        time_series_data['velocity'] = {roi.lower(): {} for roi in CLASSES}
    if 'acceleration' not in time_series_data:
        time_series_data['acceleration'] = {roi.lower(): {} for roi in CLASSES}
    
    # Print initial structure
    debug_logger.debug("\nInitial time_series_data structure:")
    debug_logger.debug(f"Velocity keys: {list(time_series_data['velocity'].keys())}")
    debug_logger.debug(f"Acceleration keys: {list(time_series_data['acceleration'].keys())}")
    
    # Process velocity data
    for key, v_data in v_total.items():
        debug_logger.debug(f"\nProcessing velocity data for {key}")
        debug_logger.debug(f"Data type: {type(v_data)}")
        debug_logger.debug(f"Data length: {len(v_data) if isinstance(v_data, (list, np.ndarray)) else 'N/A'}")
        
        try:
            # Split only once: actorNumber and the rest as action (handles multi-word)
            parts = key.split(' ', 1)
            actor = parts[0]
            action = parts[1] if len(parts) > 1 else ''
            debug_logger.debug(f"Processing actor {actor}, action {action}")
            
            for i, roi in enumerate(CLASSES):
                roi_key = roi.lower()
                debug_logger.debug(f"\nProcessing ROI: {roi} -> {roi_key}")
                
                if roi_key not in time_series_data['velocity']:
                    time_series_data['velocity'][roi_key] = {}
                    debug_logger.debug(f"Created new velocity entry for {roi_key}")
                
                if isinstance(v_data, (list, np.ndarray)) and len(v_data) > i:
                    data_point = v_data[i]
                    debug_logger.debug(f"Data point type: {type(data_point)}")
                    debug_logger.debug(f"Data point length: {len(data_point) if isinstance(data_point, (list, np.ndarray)) else 'N/A'}")
                    
                    if isinstance(data_point, (list, np.ndarray)) and len(data_point) > 0:
                        participant_key = f"{actor}_{action}"
                        time_series_data['velocity'][roi_key][participant_key] = np.array(data_point)
                        debug_logger.debug(f"  - Added {roi_key} data for {participant_key} (length: {len(data_point)})")
                        debug_logger.debug(f"  - Current keys in {roi_key}: {list(time_series_data['velocity'][roi_key].keys())}")
                    else:
                        debug_logger.debug(f"  - Skipping {roi_key} for {actor}_{action}: Invalid data point")
                else:
                    debug_logger.debug(f"  - Skipping {roi_key} for {actor}_{action}: Invalid data structure")
        except Exception as e:
            debug_logger.debug(f"Error processing velocity data for {key}: {e}")
            debug_logger.debug(f"Current time_series_data structure: {time_series_data}")

    # Process acceleration data
    for key, a_data in a_total.items():
        debug_logger.debug(f"\nProcessing acceleration data for {key}")
        debug_logger.debug(f"Data type: {type(a_data)}")
        debug_logger.debug(f"Data length: {len(a_data) if isinstance(a_data, (list, np.ndarray)) else 'N/A'}")
        
        try:
            parts = key.split(' ', 1)
            actor = parts[0]
            action = parts[1] if len(parts) > 1 else ''
            debug_logger.debug(f"Processing actor {actor}, action {action}")
            
            for i, roi in enumerate(CLASSES):
                roi_key = roi.lower()
                debug_logger.debug(f"\nProcessing ROI: {roi} -> {roi_key}")
                
                if roi_key not in time_series_data['acceleration']:
                    time_series_data['acceleration'][roi_key] = {}
                    debug_logger.debug(f"Created new acceleration entry for {roi_key}")
                
                if isinstance(a_data, (list, np.ndarray)) and len(a_data) > i:
                    data_point = a_data[i]
                    debug_logger.debug(f"Data point type: {type(data_point)}")
                    debug_logger.debug(f"Data point length: {len(data_point) if isinstance(data_point, (list, np.ndarray)) else 'N/A'}")
                    
                    if isinstance(data_point, (list, np.ndarray)) and len(data_point) > 0:
                        participant_key = f"{actor}_{action}"
                        time_series_data['acceleration'][roi_key][participant_key] = np.array(data_point)
                        debug_logger.debug(f"  - Added {roi_key} data for {participant_key} (length: {len(data_point)})")
                        debug_logger.debug(f"  - Current keys in {roi_key}: {list(time_series_data['acceleration'][roi_key].keys())}")
                    else:
                        debug_logger.debug(f"  - Skipping {roi_key} for {actor}_{action}: Invalid data point")
                else:
                    debug_logger.debug(f"  - Skipping {roi_key} for {actor}_{action}: Invalid data structure")
        except Exception as e:
            debug_logger.debug(f"Error processing acceleration data for {key}: {e}")
            debug_logger.debug(f"Current time_series_data structure: {time_series_data}")

    # Verify data structure
    debug_logger.debug("\nFinal data structure verification:")
    for metric in ['velocity', 'acceleration']:
        debug_logger.debug(f"\n{metric.capitalize()} data:")
        for roi in [roi.lower() for roi in CLASSES]:
            if roi in time_series_data[metric]:
                num_series = len(time_series_data[metric][roi])
                debug_logger.debug(f"  - {roi}: {num_series} time series")
                if num_series > 0:
                    first_key = next(iter(time_series_data[metric][roi]))
                    series_data = time_series_data[metric][roi][first_key]
                    debug_logger.debug(f"    First series length: {len(series_data)}")
                    debug_logger.debug(f"    First series type: {type(series_data)}")
                    debug_logger.debug(f"    First series shape: {series_data.shape if hasattr(series_data, 'shape') else 'N/A'}")
                    debug_logger.debug(f"    Keys in {roi}: {list(time_series_data[metric][roi].keys())}")
            else:
                debug_logger.debug(f"  - {roi}: No data")

def consolidate_emotion_folders() -> None:
    """
    Consolidates emotion folders based on EMOTION_MAP.
    Moves content from variant folders (e.g., 'sad', 'happy', 'guilit') 
    into their standardized folders (e.g., 'sadness', 'happiness', 'guilt').
    """
    debug_logger.debug("\nStarting emotion folder consolidation...")
    
    # Get all existing emotion folders in results directory
    if not os.path.exists(results_dir):
        debug_logger.debug("Results directory does not exist, skipping consolidation")
        return
    
    existing_folders = [f for f in os.listdir(results_dir) 
                       if os.path.isdir(os.path.join(results_dir, f))]
    
    debug_logger.debug(f"Found existing folders: {existing_folders}")
    
    # Process each emotion mapping
    for variant_name, standard_name in EMOTION_MAP.items():
        variant_path = os.path.join(results_dir, variant_name)
        standard_path = os.path.join(results_dir, standard_name)

        if not os.path.exists(variant_path) or variant_name == standard_name:
            debug_logger.debug(f"No variant folder found for: {variant_name}")
            continue

        debug_logger.debug(f"Found variant folder: {variant_name} -> {standard_name}")

        # Handle case-only differences on case-insensitive filesystems
        if os.path.normcase(variant_path) == os.path.normcase(standard_path):
            if os.path.basename(variant_path) != os.path.basename(standard_path):
                temp_path = variant_path + "__tmpcase"
                counter = 1
                while os.path.exists(temp_path):
                    temp_path = f"{variant_path}__tmpcase{counter}"
                    counter += 1
                try:
                    os.rename(variant_path, temp_path)
                    os.rename(temp_path, standard_path)
                    debug_logger.debug(f"  - Renamed {variant_name} to {standard_name}")
                except Exception as e:
                    debug_logger.error(f"Error renaming {variant_name} -> {standard_name}: {str(e)}")
                continue
            else:
                debug_logger.debug(f"Variant and standard paths identical for {variant_name}, skipping")
                continue

        # Ensure standard folder exists
        os.makedirs(standard_path, exist_ok=True)

        # Move all content from variant to standard folder
        try:
            import shutil
            for item in os.listdir(variant_path):
                src = os.path.join(variant_path, item)
                dst = os.path.join(standard_path, item)

                if os.path.exists(dst):
                    try:
                        if os.path.samefile(src, dst):
                            debug_logger.debug(f"  - {item} already consolidated, skipping")
                            continue
                    except FileNotFoundError:
                        pass
                    except OSError:
                        pass

                    # If destination exists, merge the contents
                    if os.path.isdir(src) and os.path.isdir(dst):
                        # Merge directories
                        for subitem in os.listdir(src):
                            sub_src = os.path.join(src, subitem)
                            sub_dst = os.path.join(dst, subitem)
                            if os.path.exists(sub_dst):
                                debug_logger.debug(f"  - Merging {subitem} from {variant_name} into {standard_name}")
                                if os.path.isdir(sub_src):
                                    shutil.copytree(sub_src, sub_dst, dirs_exist_ok=True)
                                else:
                                    if os.path.normcase(sub_src) == os.path.normcase(sub_dst):
                                        continue
                                    shutil.copy2(sub_src, sub_dst)
                            else:
                                debug_logger.debug(f"  - Moving {subitem} from {variant_name} to {standard_name}")
                                shutil.move(sub_src, sub_dst)
                    else:
                        # Overwrite file
                        debug_logger.debug(f"  - Overwriting {item} in {standard_name}")
                        shutil.copy2(src, dst)
                else:
                    # Move item to standard folder
                    debug_logger.debug(f"  - Moving {item} from {variant_name} to {standard_name}")
                    shutil.move(src, dst)

            # Remove empty variant folder
            if not os.listdir(variant_path):
                os.rmdir(variant_path)
                debug_logger.debug(f"  - Removed empty folder: {variant_name}")
            else:
                debug_logger.debug(f"  - Warning: {variant_name} folder not empty after consolidation")

        except Exception as e:
            debug_logger.error(f"Error consolidating {variant_name} -> {standard_name}: {str(e)}")
    
    debug_logger.debug("Emotion folder consolidation completed")

def should_regenerate(metric: str, ROI: str, method: str, emotion: str) -> bool:
    """
    Check if we need to regenerate results for a specific analysis.
    
    Args:
        metric: Type of metric (velocity/acceleration)
        ROI: Region of interest
        method: Distance method used
        emotion: Emotion being analyzed
        
    Returns:
        bool: True if regeneration is needed, False otherwise
    """
    # Check if social score analysis files exist in standardized path
    social_score_dir = os.path.join(results_dir, emotion, 'statistical', 'social_scores', ROI)
    if not os.path.exists(social_score_dir):
        debug_logger.debug(f"Social score directory not found: {social_score_dir}")
        return True
        
    # Check for required files
    required_files = [
        f'{method}_social_scores_analysis.json',
        f'{method}_social_scores_summary.csv'
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(social_score_dir, file)):
            debug_logger.debug(f"Missing required file: {file}")
            return True
            
    # Optional: Check for presence of at least one boxplot per score (k-aware naming)
    import glob
    social_scores = ['AQ', 'RMET', 'Alexithymia']  # Add other scores as needed
    for score in social_scores:
        pattern = os.path.join(social_score_dir, f'{method}_{score}_k*_boxplot.png')
        if not glob.glob(pattern):
            debug_logger.debug(f"Missing boxplots matching: {pattern}")
            return True
            
    return False

def main():
    """Main function to run the analysis pipeline."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run kinematic behavior classification analysis')
    parser.add_argument('--force', action='store_true', help='Force regeneration of all results')
    parser.add_argument('--force-social', action='store_true', help='Force regeneration of social score analysis only')
    # Quick testing controls
    parser.add_argument('--test-quick', action='store_true', help='Run a quick test: 1 emotion, DTW only, small k, sampled participants, no plots')
    parser.add_argument('--test-emotion', type=str, help='Limit run to a single emotion (e.g., happiness)')
    parser.add_argument('--emotions', type=str, help='Comma-separated list of emotions to run')
    parser.add_argument('--rois', type=str, help='Comma-separated list of ROIs to run (head, lh, rh, le)')
    parser.add_argument('--methods', type=str, help='Comma-separated distance methods (DTW,Soft-DTW,Euclidean,Cosine)')
    parser.add_argument('--k', type=str, help='Comma-separated k values for clustering (e.g., 3,4,6)')
    parser.add_argument('--sample-n', type=int, help='Sample N participants per emotion/ROI before computing distances')
    parser.add_argument('--no-plots', action='store_true', help='Skip plotting to save time')
    parser.add_argument('--no-social', action='store_true', help='Skip social score analysis')
    parser.add_argument('--max-series-length', type=int, help='Truncate time series to this many frames before distance calc')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--actors', type=str, help='Comma-separated list or ranges of actors to update (e.g., 40-49,51)')
    parser.add_argument('--ingest-src', type=str, help='Folder containing new raw data to ingest before processing')
    parser.add_argument('--process-new', action='store_true', help='Detect missing actors/emotions and process them automatically')
    args = parser.parse_args()
    
    debug_logger.debug("\nStarting main function...")
    debug_logger.debug(f"Force flag: {args.force}")
    debug_logger.debug(f"Force social flag: {args.force_social}")

    auto_actor_filter: Set[str] = set()
    detected_emotions: Set[str] = set()
    pending_map: Dict[str, Set[str]] = {}

    if args.ingest_src:
        print(f"\n📥 Ingesting new data from {args.ingest_src}...")
        try:
            ingest_mapping = utils.ingest_new_data(args.ingest_src, data_root=data_dir, logger=debug_logger)
            if ingest_mapping:
                auto_actor_filter.update(str(v) for v in ingest_mapping.values())
        except Exception as e:
            debug_logger.error(f"Error ingesting data from {args.ingest_src}: {e}")
            print(f"⚠️ Ingestion failed: {e}")
    
    # Consolidate emotion folders first
    print("\n📁 Consolidating emotion folders...")
    consolidate_emotion_folders()

    if args.process_new or auto_actor_filter:
        pending_map = utils.detect_pending_actors(data_root=data_dir, results_root=results_dir, logger=debug_logger)
        if pending_map:
            auto_actor_filter.update(pending_map.keys())
            detected_emotions.update({em for ems in pending_map.values() for em in ems})
            summary = ', '.join(
                f"Actor {actor}: {', '.join(sorted(ems))}" for actor, ems in sorted(pending_map.items())
            )
            print(f"\n🔎 Pending actors detected -> {summary}")
        elif args.process_new:
            print("\n✅ No pending actors found")
    
    # Initialize variables
    v_total = {}
    a_total = {}
    max_num_win = 0
    
    # Try loading precomputed time series data
    loaded_v_total, loaded_a_total, time_series_data = load_time_series_data()
    
    # Parse actor filter if provided or detected
    def _parse_actor_spec(spec: str) -> Set[str]:
        out: Set[str] = set()
        for token in [t.strip() for t in spec.split(',') if t.strip()]:
            if '-' in token:
                try:
                    a, b = token.split('-', 1)
                    a, b = int(a.strip()), int(b.strip())
                    if a <= b:
                        out.update(str(x) for x in range(a, b + 1))
                    else:
                        out.update(str(x) for x in range(b, a + 1))
                    continue
                except Exception:
                    pass
            try:
                out.add(str(int(token)))
            except Exception:
                out.add(token)
        return out

    actor_filter_set: Set[str] = set()
    if args.actors:
        actor_filter_set.update(_parse_actor_spec(args.actors))
    if auto_actor_filter:
        actor_filter_set.update(auto_actor_filter)
    actor_filter = actor_filter_set if actor_filter_set else None

    if args.force and not actor_filter:
        print("\n🔄 Regenerating all results...")
        # Step 1: Compute kinematic analysis
        debug_logger.debug("\nStarting kinematic analysis...")
        max_num_win = kinematic_analysis(v_total, a_total, max_num_win)
        verify_data_structure(v_total, "velocity_data")
        verify_data_structure(a_total, "acceleration_data")

        # Step 2: Prepare time series structure with all ROIs
        debug_logger.debug("\nInitializing time series structure...")
        time_series_data = {
            'velocity': {roi.lower(): {} for roi in CLASSES},
            'acceleration': {roi.lower(): {} for roi in CLASSES}
        }
        verify_data_structure(time_series_data, "time_series_data")
        
        debug_logger.debug("\nSaving time series data...")
        saving_time_series(v_total, a_total, time_series_data)
        verify_data_structure(time_series_data, "time_series_data")

        # Step 3: Save results for future runs
        debug_logger.debug("\nSaving results to disk...")
        save_time_series_data(v_total, a_total, time_series_data)
    else:
        # If data is missing, recompute
        if loaded_v_total is None or loaded_a_total is None or time_series_data is None:
            print("\n🔄 No precomputed data found, computing from scratch...")
            # Step 1: Compute kinematic analysis
            debug_logger.debug("\nStarting kinematic analysis...")
            max_num_win = kinematic_analysis(v_total, a_total, max_num_win, actor_filter=actor_filter)
            verify_data_structure(v_total, "velocity_data")
            verify_data_structure(a_total, "acceleration_data")

            # Step 2: Prepare time series structure with all ROIs
            debug_logger.debug("\nInitializing time series structure...")
            time_series_data = {
                'velocity': {roi.lower(): {} for roi in CLASSES},
                'acceleration': {roi.lower(): {} for roi in CLASSES}
            }
            verify_data_structure(time_series_data, "time_series_data")
            
            debug_logger.debug("\nSaving time series data...")
            saving_time_series(v_total, a_total, time_series_data)
            verify_data_structure(time_series_data, "time_series_data")

            # Step 3: Save results for future runs
            debug_logger.debug("\nSaving results to disk...")
            save_time_series_data(v_total, a_total, time_series_data)
        else:
            print("\n✅ Using precomputed data")
            # Merge new actors (if any) into existing structures
            v_total = loaded_v_total
            a_total = loaded_a_total
            if actor_filter:
                debug_logger.debug(f"Updating kinematics for actors: {sorted(actor_filter)}")
                max_num_win = kinematic_analysis(v_total, a_total, max_num_win, actor_filter=actor_filter)
                # Update the in-memory time_series_data with new entries
                saving_time_series(v_total, a_total, time_series_data)
            verify_data_structure(time_series_data, "time_series_data")

    # Step 4: Aggregate kinematic features with demographic data
    print("\n📊 Aggregating kinematic features...")
    try:
        aggregated_df = utils.aggregate_kinematic_features("data.csv")
        debug_logger.debug(f"Successfully aggregated features for {len(aggregated_df)} participants")
    except Exception as e:
        debug_logger.error(f"Error aggregating features: {str(e)}")

    # Run analysis and clustering
    print("\n🔍 Starting clustering analysis...")
    # Build test/run configuration
    selected_k = [4, 6, 8]
    if args.k:
        try:
            selected_k = [int(x) for x in args.k.split(',') if x.strip()]
        except Exception:
            debug_logger.warning(f"Invalid --k value '{args.k}', using default {selected_k}")
    methods = None
    if args.methods:
        raw_methods = [m.strip() for m in args.methods.split(',') if m.strip()]
        method_map = {
            'dtw': 'DTW', 'softdtw': 'Soft-DTW', 'soft-dtw': 'Soft-DTW',
            'euclidean': 'Euclidean', 'cosine': 'Cosine'
        }
        methods = []
        for m in raw_methods:
            key = m.lower()
            methods.append(method_map.get(key, m))
    run_emotions = None
    if args.emotions:
        tmp = [e.strip() for e in args.emotions.split(',') if e.strip()]
        run_emotions = []
        for e in tmp:
            canon = EMOTION_MAP.get(e, EMOTION_MAP.get(e.lower(), e.lower()))
            if canon not in run_emotions:
                run_emotions.append(canon)
    elif args.test_emotion:
        e = args.test_emotion.strip()
        run_emotions = [EMOTION_MAP.get(e, EMOTION_MAP.get(e.lower(), e.lower()))]
    run_rois = None
    if args.rois:
        run_rois = [r.strip().lower() for r in args.rois.split(',') if r.strip()]
    # Apply quick-test defaults
    if args.test_quick:
        if not run_emotions:
            run_emotions = [EMOTION_MAP.get('happy', 'happiness')]  # default to happiness
        if not methods:
            methods = ['DTW']
        selected_k = [3]
        if args.sample_n is None:
            args.sample_n = 8
        args.no_plots = True if not args.no_plots else args.no_plots
        if args.max_series_length is None:
            args.max_series_length = 600  # ~20s at 30 FPS

    if detected_emotions:
        if not run_emotions:
            run_emotions = sorted(detected_emotions)
        else:
            for em in sorted(detected_emotions):
                if em not in run_emotions:
                    run_emotions.append(em)
    
    # Determine if we need to regenerate DTW/clustering
    regenerate_dtw = args.force or bool(actor_filter)
    regenerate_clustering = args.force or args.force_social or bool(actor_filter)
    
    # For each metric and ROI, check if social score analysis needs regeneration
    debug_logger.debug("\nChecking which analyses need regeneration...")
    for metric in time_series_data.keys():
        debug_logger.debug(f"\nChecking metric: {metric}")
        for ROI in time_series_data[metric].keys():
            debug_logger.debug(f"Checking ROI: {ROI}")
            for method in ["DTW", "Soft-DTW", "Euclidean", "Cosine"]:
                debug_logger.debug(f"Checking method: {method}")
                # Check each emotion for regeneration
                for emotion in EMOTIONS:
                    if args.force_social or should_regenerate(metric, ROI, method, emotion):
                        debug_logger.debug(f"Regenerating social score analysis for {metric} - {ROI} using {method} for {emotion}")
                        regenerate_clustering = True
                        break
                if regenerate_clustering:
                    break
            if regenerate_clustering:
                break
        if regenerate_clustering:
            break
    
    debug_logger.debug(f"\nFinal regeneration flags:")
    debug_logger.debug(f"regenerate_dtw: {regenerate_dtw}")
    debug_logger.debug(f"regenerate_clustering: {regenerate_clustering}")
    
    # Load demographic data
    try:
        demographic_data = pd.read_csv("data.csv")
        debug_logger.debug(f"Loaded demographic data with shape: {demographic_data.shape}")
        debug_logger.debug(f"Demographic data columns: {demographic_data.columns.tolist()}")
    except Exception as e:
        debug_logger.error(f"Error loading demographic data: {str(e)}")
        demographic_data = None
    
    analyze_and_cluster(time_series_data, selected_k, 
                       regenerate_dtw=regenerate_dtw,
                       regenerate_clustering=regenerate_clustering,
                       demographic_file="data.csv",
                       emotions=run_emotions,
                       rois=run_rois,
                       methods=methods,
                       sample_n=args.sample_n,
                       no_plots=args.no_plots,
                       no_social=args.no_social,
                       max_series_length=args.max_series_length,
                       seed=args.seed)
    print("\n✅ Analysis completed!")

    # Final consolidation to ensure no variant emotion folders remain
    try:
        consolidate_emotion_folders()
    except Exception as e:
        debug_logger.error(f"Final consolidation failed: {e}")

if __name__ == "__main__":
    debug_logger.debug("\n" + "="*50)
    debug_logger.debug("Starting program execution")
    debug_logger.debug("="*50)
    
    try:
        # Initialize global variables
        debug_logger.debug(f"Current working directory: {os.getcwd()}")
        debug_logger.debug(f"Temp data directory: {data_dir}")
        
        text_files = utils.find_text_files(data_dir)
        debug_logger.debug(f"Found {len(text_files)} text files")
        
        main()
    except Exception as e:
        debug_logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise 
