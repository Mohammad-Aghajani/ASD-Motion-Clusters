"""
Utility functions for Kinematic Behavior Classification (KBC)

This module provides utility functions for processing movement data, calculating
distances, performing clustering, and visualizing results.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import simpson
from scipy.stats import trapezoid
from scipy import signal
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats import f_oneway, entropy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from dtaidistance import dtw
from tslearn.metrics import SoftDTW
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import csv
import math
import json
import re
import logging
import shutil
from tqdm import tqdm

# Constants
FPS = 30.0
DT = 1 / FPS
STRIDE = int(FPS/2)
WINDOW_SIZE = int(FPS/2)

# Emotions to analyze (combined similar emotions)
EMOTIONS = ['anger', 'admiration', 'happy', 'sad', 'fear', 'disgust', 'guilt', 'pride', 'shame', 'surprise', 'envy', 'despair']

# Metrics to analyze
METRICS = ['velocity', 'acceleration']

# Standard points for data standardization - primary and backup points
STANDARD_POINTS = {
    'primary': [12, 16],  # Original reference points
    'backup': [0, 12],    # Backup reference points if primary are missing
    'fallback': [0, 4]    # Final fallback if both primary and backup are missing
}

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

# Canonical emotion mapping shared with ingestion/detection helpers
EMOTION_STANDARD_MAP = {
    'happy': 'happiness',
    'happiness': 'happiness',
    'sad': 'sadness',
    'sadness': 'sadness',
    'sad - part (2)': 'sadness',
    'sad - part 1': 'sadness',
    'despair': 'despair',
    'DESPAIR': 'despair',
    'Despair': 'despair',
    'dispair': 'despair',
    'Dispair': 'despair',
    'dispar': 'despair',
    'Dispar': 'despair',
    'admiraton': 'admiration',
    'guilit': 'guilt'
}

STANDARD_EMOTIONS = {
    'admiration', 'anger', 'despair', 'disgust', 'envy', 'fear',
    'guilt', 'happiness', 'pride', 'sadness', 'shame', 'surprise'
}

# Error codes
ERR9 = -9      # Error for missing more than 1 point in a region
ERR9999 = -9999  # Error for filling input data
ERR999 = -999   # General error

# Directory structure
RESULTS_DIR = "./results"
SUBDIRS = {
    "time_series": "time_series",  # Raw time series data
    "dtw": "dtw",                 # Distance matrices
    "clustering": "clustering",    # Clustering results
    "statistical": "statistical"   # Statistical analysis results
}

# Remove 'visualizations' from SUBDIRS
define_subdirs = False
if 'visualizations' in SUBDIRS:
    SUBDIRS.pop('visualizations')

def find_text_files(directory: str) -> List[str]:
    """
    Recursively finds all text files in a directory.
    
    Args:
        directory: Path to search for text files
        
    Returns:
        List of paths to text files
    """
    text_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt') and not file.endswith('Zone.Identifier'):
                text_files.append(os.path.join(root, file))
    return text_files

def read_data(file_path):
    """
    Read and process movement data from a text file.
    Handles missing points by tracking them but still processing valid data.
    
    Args:
        file_path (str): Path to the text file containing movement data.
        
    Returns:
        tuple: (data_dict, x_list, y_list, missing_numbers, action_type, actor_number)
    """
    print(f"\nProcessing file: {file_path}")
    
    # Extract actor number and action type from file path
    try:
        path_parts = file_path.split('\\')
        file_name = path_parts[-1]
        
        # Try to extract actor number from the directory name first
        actor_number = None
        for part in path_parts:
            if 'Actor' in part:
                try:
                    actor_number = int(part.split('Actor')[1].strip())
                    break
                except (ValueError, IndexError):
                    continue
        
        # If not found in directory, try from filename
        if actor_number is None:
            if 'actor' in file_name.lower():
                # Extract actor number from filename
                actor_part = [p for p in file_name.split('_') if 'actor' in p.lower()][0]
                try:
                    actor_number = int(actor_part.lower().replace('actor', '').strip())
                except ValueError:
                    # If no numeric actor number, use the full actor part as identifier
                    actor_number = actor_part.lower().replace('actor', '').strip()
        
        # Extract action type (emotion)
        action_type = file_name.split('_')[-1].split('.')[0].strip()
        
        print(f"Extracted actor identifier: {actor_number}, action type: {action_type}")
        
    except (IndexError, ValueError) as e:
        print(f"Error extracting actor/action info from path: {str(e)}")
        # Instead of raising an error, use a default identifier
        actor_number = "unknown"
        action_type = "unknown"
        print(f"Using default values: actor={actor_number}, action={action_type}")
    
    data = {}
    x_list = []
    y_list = []
    missing_numbers = []
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                print("Warning: File is empty")
                return data, x_list, y_list, missing_numbers, action_type, actor_number
            
            # Validate and process header
            try:
                num_people = int(lines[0].strip())
                print(f"Number of people in file: {num_people}")
            except ValueError:
                print("Warning: First line is not a valid number of people")
                num_people = 1
            
            # Get point names from second line
            header_line = lines[1].strip()
            point_names = [name.strip() for name in header_line.split(',') if name.strip()]
            num_points = len(point_names) // 2
            print(f"Number of points detected: {num_points}")
            print(f"Point names: {point_names}")
            
            # Initialize data dictionary
            for i in range(num_points):
                data[f'x{i}'] = []
                data[f'y{i}'] = []
            
            # Process data lines
            valid_frames = 0
            invalid_frames = 0
            for line_num, line in enumerate(lines[2:], start=3):
                try:
                    # Split by comma and remove empty strings
                    values = [val.strip() for val in line.strip().split(',') if val.strip()]
                    
                    # Check for missing values
                    if len(values) != num_points * 2:
                        print(f"Warning: Line {line_num} has {len(values)} values, expected {num_points * 2}")
                        invalid_frames += 1
                        continue
                    
                    # Process x and y coordinates
                    frame_x = []
                    frame_y = []
                    frame_missing = []
                    
                    for i in range(num_points):
                        try:
                            x_val = float(values[i * 2])
                            y_val = float(values[i * 2 + 1])
                            frame_x.append(x_val)
                            frame_y.append(y_val)
                        except (ValueError, IndexError):
                            print(f"Warning: Invalid coordinates for point {i} in line {line_num}")
                            frame_missing.append(i)
                            frame_x.append(0.0)  # Use 0.0 instead of None
                            frame_y.append(0.0)  # Use 0.0 instead of None
                    
                    # Add frame data
                    for i in range(num_points):
                        data[f'x{i}'].append(frame_x[i])
                        data[f'y{i}'].append(frame_y[i])
                    
                    x_list.append(frame_x)
                    y_list.append(frame_y)
                    valid_frames += 1
                    
                    # Track missing points
                    missing_numbers.extend(frame_missing)
                    
                except Exception as e:
                    print(f"Warning: Error processing line {line_num} in {file_path}: {str(e)}")
                    invalid_frames += 1
                    continue
            
            print(f"Successfully processed {valid_frames} frames")
            if invalid_frames > 0:
                print(f"Warning: Skipped {invalid_frames} invalid frames")
            
            # Remove duplicates from missing_numbers
            missing_numbers = list(set(missing_numbers))
            if missing_numbers:
                print(f"Warning: Missing data detected for points: {missing_numbers}")
            
    except IOError as e:
        print(f"Error reading file {file_path}: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error processing {file_path}: {str(e)}")
        raise
    
    return data, x_list, y_list, missing_numbers, action_type, actor_number

def fill_data(data_dict: Dict, n: int, inputs: List[str], missing: List[int]) -> None:
    """
    Fills data dictionary with empty lists for each point.
    
    Args:
        data_dict: Dictionary to fill
        n: Number of points
        inputs: List of input types (e.g., ['x', 'y'])
        missing: List of missing point numbers
    """
    for i in range(n):
        for k in inputs:
            key = f"{k}{i}"
            data_dict[key] = []

def calc_ref(x: List[List[float]], y: List[List[float]], frames: List[int], points: List[int]) -> float:
    """
    Calculates reference distance for standardization.
    Tries different pairs of points if standard points are missing.
    
    Args:
        x: X coordinates
        y: Y coordinates
        frames: Frame indices to use
        points: Point indices to use (not used anymore, kept for compatibility)
        
    Returns:
        Average distance between points
    """
    if not x or not y:
        raise ValueError("Empty coordinate lists provided")
    
    if len(frames) == 0:
        raise ValueError("No frames provided for reference calculation")
    
    # Try different pairs of reference points
    reference_pairs = [
        STANDARD_POINTS['primary'],
        STANDARD_POINTS['backup'],
        STANDARD_POINTS['fallback']
    ]
    
    for ref_points in reference_pairs:
        try:
            dist = 0
            valid_frames = 0
            
            for frame in frames:
                if frame >= len(x) or frame >= len(y):
                    print(f"Warning: Frame {frame} is out of range")
                    continue
                
                try:
                    x1 = x[frame][ref_points[0]]
                    y1 = y[frame][ref_points[0]]
                    x2 = x[frame][ref_points[1]]
                    y2 = y[frame][ref_points[1]]
                    
                    # Check for None values or invalid data
                    if None in [x1, y1, x2, y2]:
                        continue
                    
                    frame_dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if frame_dist > 0:  # Only count non-zero distances
                        dist += frame_dist
                        valid_frames += 1
                        
                except (IndexError, TypeError) as e:
                    continue
            
            if valid_frames > 0:
                print(f"Using reference points {ref_points} with {valid_frames} valid frames")
                return dist/valid_frames
                
        except Exception as e:
            print(f"Warning: Failed to calculate reference distance with points {ref_points}: {str(e)}")
            continue
    
    # If we get here, none of the reference point pairs worked
    raise ValueError("Could not calculate reference distance with any available point pairs")

def standardize(data: Dict, x: List[List[float]], y: List[List[float]], ref_dist: float, num_frames: int) -> None:
    """
    Standardizes data using reference distance.
    All data should be valid (no None values) at this point.
    
    Args:
        data: Data dictionary
        x: X coordinates
        y: Y coordinates
        ref_dist: Reference distance
        num_frames: Number of frames
    """
    if ref_dist <= 0:
        print("Warning: Invalid reference distance, skipping standardization")
        return
    
    # Standardize data dictionary
    for key in data:
        data[key] = [val/ref_dist for val in data[key]]
    
    # Standardize coordinate lists
    for i in range(len(x)):  # Use actual length instead of num_frames
        x[i] = [val/ref_dist for val in x[i]]
        y[i] = [val/ref_dist for val in y[i]]

def findVelAcc(data: Dict, x: List[List[float]], y: List[List[float]], 
              roi: List[str], missing_p: List[str], num_windows: int) -> Tuple:
    """
    Finds velocity and acceleration for a region of interest.
    
    Args:
        data: Data dictionary
        x: X coordinates
        y: Y coordinates
        roi: Region of interest
        missing_p: Missing points
        num_windows: Number of windows
        
    Returns:
        Tuple containing velocity, acceleration, and center coordinates
    """
    # Map ROI to point numbers
    roi_map = {
        'LF': ['12', '13', '14', '15'],
        'RF': ['16', '17', '18', '19'],
        'LA': ['4', '5', '6', '7', '21', '22'],
        'RA': ['8', '9', '10', '11', '23', '24'],
        'LH': ['6', '7', '21', '22'],
        'RH': ['10', '11', '23', '24'],
        'ALL': [str(i) for i in range(25)],
        'Head': ['2', '3', '20'],
        'Hip': ['0', '12', '16'],
        'LE': ['0', '12', '13', '14', '15', '16', '17', '18', '19']
    }
    
    ROI = roi_map.get(roi[0], roi)
    common_elements = [item for item in missing_p if item in ROI]
    
    if len(common_elements) >= 2:
        return (ERR9, ERR9, [ERR9]*num_windows, [ERR9]*num_windows, [ERR9], [ERR9])
    elif len(common_elements) == 1:
        ROI.remove(common_elements[0])
    
    xcenter, ycenter = [], []
    for win in range(1, num_windows+1):
        start_idx = (win-1)*STRIDE
        x_center_win, y_center_win = [], []
        
        for i in range(start_idx, start_idx+WINDOW_SIZE):
            tempx, tempy, tempMass = [], [], []
            for r in ROI:
                try:
                    tempx.append(data['x' + r][i])
                    tempy.append(data['y' + r][i])
                    tempMass.append(1)
                except (KeyError, IndexError):
                    continue
            if len(tempx) > 0:  # Only calculate center if we have points
                xc, yc = cntr_mass(tempx, tempy, tempMass)
                x_center_win.append(xc)
                y_center_win.append(yc)
        
        if len(x_center_win) > 0:  # Only add to centers if we have valid windows
            xcenter.append(np.mean(x_center_win))
            ycenter.append(np.mean(y_center_win))
    
    x_center_f, y_center_f = [], []
    for d in range(len(x)):
        tempx, tempy, tempMass = [], [], []
        for r in ROI:
            try:
                tempx.append(x[d][int(r)])
                tempy.append(y[d][int(r)])
                tempMass.append(1)
            except (IndexError, ValueError):
                continue
        if len(tempx) > 0:  # Only calculate center if we have points
            xc, yc = cntr_mass(tempx, tempy, tempMass)
            x_center_f.append(xc)
            y_center_f.append(yc)
    
    if len(x_center_f) < 2:  # Need at least 2 points for velocity calculation
        return (ERR9, ERR9, [ERR9]*num_windows, [ERR9]*num_windows, [ERR9], [ERR9])
    
    # Apply trapezoidal window to center coordinates
    t = np.linspace(0, 1, len(x_center_f))
    trap = trapezoid.pdf(t, 0.1, 0.9)
    x_center_f = np.array(x_center_f) * trap
    y_center_f = np.array(y_center_f) * trap
    
    vx_f = calc_velo(x_center_f, DT)
    vy_f = calc_velo(y_center_f, DT)
    v_frame = [np.sqrt(vx_f[i]**2 + vy_f[i]**2) for i in range(len(vx_f))]
    
    ax_f = calc_velo(vx_f, DT)
    ay_f = calc_velo(vy_f, DT)
    a_frame = [np.sqrt(ax_f[i]**2 + ay_f[i]**2) for i in range(len(ay_f))]
    
    return v_frame, a_frame, xcenter, ycenter, x_center_f, y_center_f

def cntr_mass(x: List[float], y: List[float], mass: List[float]) -> Tuple[float, float]:
    """
    Calculates center of mass.
    
    Args:
        x: X coordinates
        y: Y coordinates
        mass: Mass values
        
    Returns:
        Tuple of (x, y) center coordinates
    """
    totalmass = sum(mass)
    totalx = sum(x)
    totaly = sum(y)
    return (totalx/totalmass, totaly/totalmass)

def calc_velo(x: List[float], time: float) -> List[float]:
    """
    Calculates velocity from position data.
    
    Args:
        x: Position data
        time: Time step
        
    Returns:
        List of velocities
    """
    return [(x[i]-x[i-1])/time for i in range(1,len(x))]

def gen_output_sq(v_frame: List[float], a_frame: List[float], 
                 xcenter: List[float], ycenter: List[float]) -> Tuple[Dict, List[float], List[float]]:
    """
    Generates output metrics from velocity and acceleration data.
    
    Args:
        v_frame: Velocity data
        a_frame: Acceleration data
        xcenter: X center coordinates
        ycenter: Y center coordinates
        
    Returns:
        Tuple containing metrics dictionary, velocity, and acceleration
    """
    e_vel = sample_entropy_wiki(v_frame, 3, np.std(v_frame)*0.2)
    e_accel = sample_entropy_wiki(a_frame, 3, np.std(a_frame)*0.2)
    
    vx = calc_velo(xcenter, WINDOW_SIZE*DT)
    vy = calc_velo(ycenter, WINDOW_SIZE*DT)
    
    ax = calc_velo(vx, WINDOW_SIZE*DT)
    ay = calc_velo(vy, WINDOW_SIZE*DT)
    
    v = [np.sqrt(vx[i]**2 + vy[i]**2) for i in range(len(vy))]
    v_w = len(v)
    t = np.linspace(0,1, v_w)
    trap_v = trapezoid.pdf(t, 0.1, 0.9)
    v_f = v * trap_v
    
    a = [np.sqrt(ax[i]**2 + ay[i]**2) for i in range(len(ay))]
    a_w = len(a)
    t = np.linspace(0,1, a_w)
    trap_a = trapezoid.pdf(t, 0.1, 0.9)
    a_f = a * trap_a
    
    v_auc = np.trapz(v_f, dx=1)
    v_max = np.max(v_f)
    v_mean = np.mean(v_f)
    
    a_auc = np.trapz(a_f, dx=1)
    a_max = np.max(a_f)
    a_mean = np.mean(a_f)
    
    return ({
        'v_auc': v_auc, 'v_mean': v_mean, 'v_max': v_max,
        'a_auc': a_auc, 'a_mean': a_mean, 'a_max': a_max,
        'v_ent': e_vel, 'a_ent': e_accel
    }, v_f, a_f)

def sample_entropy_wiki(timeseries_data: List[float], window_size: int, r: float) -> float:
    """
    Calculate sample entropy of a time series.
    
    Args:
        timeseries_data: List of float values representing the time series
        window_size: Size of the window for template matching
        r: Tolerance for matching
        
    Returns:
        float: Sample entropy value
    """
    def _construct_templates(data: List[float], m: int) -> List[List[float]]:
        return [data[i:i+m] for i in range(len(data)-m+1)]
        
    def _get_matches(templates: List[List[float]], r: float) -> int:
        matches = 0
        for i in range(len(templates)):
            for j in range(i+1, len(templates)):
                if all(abs(templates[i][k] - templates[j][k]) < r for k in range(len(templates[i]))):
                    matches += 1
        return matches

    templates = _construct_templates(timeseries_data, window_size)
    matches = _get_matches(templates, r)
    return -np.log(matches / (len(templates) * (len(templates) - 1) / 2)) if matches > 0 else 0


def canonical_emotion(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    key = name.strip()
    if not key:
        return None
    if key in EMOTION_STANDARD_MAP:
        return EMOTION_STANDARD_MAP[key]
    lower = key.lower()
    if lower in EMOTION_STANDARD_MAP:
        return EMOTION_STANDARD_MAP[lower]
    if lower in STANDARD_EMOTIONS:
        return lower
    if lower in EMOTIONS:
        mapped = EMOTION_STANDARD_MAP.get(lower)
        return mapped if mapped else lower
    return lower if lower in STANDARD_EMOTIONS else None


def ingest_new_data(src_root: str,
                    data_root: str = "data",
                    logger: Optional[logging.Logger] = None) -> Dict[str, int]:
    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"Source directory not found: {src_root}")
    os.makedirs(data_root, exist_ok=True)

    existing_ids = []
    for item in os.listdir(data_root):
        match = re.match(r"^Actor\s*(\d+)$", item)
        if match:
            try:
                existing_ids.append(int(match.group(1)))
            except ValueError:
                continue
    next_actor_id = max(existing_ids) + 1 if existing_ids else 1

    mapping: Dict[str, int] = {}
    copied = 0
    skipped = 0

    for root, _, files in os.walk(src_root):
        for fname in files:
            if not fname.lower().endswith('.txt'):
                continue
            src = os.path.join(root, fname)

            actor_key, emotion = _extract_actor_and_emotion(src)
            if emotion is None:
                skipped += 1
                msg = f"Skipping (unrecognized emotion): {src}"
                if logger:
                    logger.debug(msg)
                else:
                    print(msg)
                continue

            source_id = actor_key or os.path.dirname(src)
            if source_id not in mapping:
                mapping[source_id] = next_actor_id
                next_actor_id += 1

            assigned_id = mapping[source_id]
            dest_dir = os.path.join(data_root, f"Actor {assigned_id}")
            os.makedirs(dest_dir, exist_ok=True)
            dest_name = f"actor{assigned_id}_{emotion}.txt"
            dest = os.path.join(dest_dir, dest_name)

            if os.path.exists(dest):
                try:
                    if os.path.getsize(dest) == os.path.getsize(src):
                        skipped += 1
                        msg = f"Exists (same size), skipping: {dest}"
                        if logger:
                            logger.debug(msg)
                        else:
                            print(msg)
                        continue
                except OSError:
                    pass

            shutil.copy2(src, dest)
            copied += 1
            msg = f"Copied -> {dest}"
            if logger:
                logger.debug(msg)
            else:
                print(msg)

    summary_msg = f"Ingestion completed. Copied: {copied}, Skipped: {skipped}"
    if logger:
        logger.info(summary_msg)
    else:
        print(f"Done. Copied: {copied}, Skipped: {skipped}")

    return {str(k): v for k, v in mapping.items()}


def detect_pending_actors(data_root: str = "data",
                          results_root: str = "results",
                          logger: Optional[logging.Logger] = None) -> Dict[str, Set[str]]:
    pending: Dict[str, Set[str]] = {}
    if not os.path.isdir(data_root):
        return pending

    for item in os.listdir(data_root):
        match = re.match(r"^Actor\s*(\d+)$", item)
        if not match:
            continue
        actor_id = match.group(1)
        actor_dir = os.path.join(data_root, item)
        if not os.path.isdir(actor_dir):
            continue

        for fname in os.listdir(actor_dir):
            if not fname.lower().endswith('.txt'):
                continue
            emotion = canonical_emotion(os.path.splitext(fname)[0].split('_')[-1])
            if emotion is None or emotion not in STANDARD_EMOTIONS:
                continue

            dest_dir = os.path.join(results_root, emotion, 'statistical', 'kinematic_analysis', f'Actor{actor_id}')
            summary_file = os.path.join(dest_dir, 'summary_metrics.json')
            if os.path.exists(summary_file):
                continue

            pending.setdefault(actor_id, set()).add(emotion)

    if pending and logger:
        logger.debug(f"Pending actors for processing: {pending}")

    return pending


def _extract_actor_and_emotion(path: str) -> Tuple[Optional[str], Optional[str]]:
    fname = os.path.basename(path)

    match_dir = re.search(r"(?:\\|/)Actor\s*(\d+)(?:\\|/)", path)
    actor_from_dir = match_dir.group(1) if match_dir else None

    match_name = re.search(r"actor\s*(\d+)", fname, re.IGNORECASE)
    actor_from_name = match_name.group(1) if match_name else None

    if actor_from_dir:
        actor_key = actor_from_dir
    elif actor_from_name:
        actor_key = actor_from_name
    else:
        match_token = re.search(r"(^|[^\d])(\d{2,3})([^\d]|$)", fname)
        actor_key = match_token.group(2) if match_token else None

    emotion = canonical_emotion(os.path.splitext(fname)[0].split('_')[-1])

    return actor_key, emotion

def get_distance_matrices(time_series_dict: Dict[str, List[float]], 
                        metric: str, 
                        ROI: str, 
                        regenerate: bool = False,
                        output_dir: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Computes various distance matrices for time series data.
    Saves matrices in both NPY and CSV formats.
    
    Args:
        time_series_dict: Dictionary of time series data
        metric: Type of metric (velocity/acceleration)
        ROI: Region of interest
        regenerate: Whether to regenerate matrices
        output_dir: Directory to save matrices
        
    Returns:
        Tuple containing DTW, Soft-DTW, Euclidean, and Cosine distance matrices, and participant list
    """
    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get list of participants
    participants = sorted(time_series_dict.keys(), key=natural_sort_key)
    n = len(participants)
    
    print(f"\nComputing distance matrices for {metric} - {ROI}")
    print(f"Number of participants: {n}")
    print(f"Participants: {', '.join(participants)}")
    
    # Initialize matrices
    dtw_matrix = np.zeros((n, n))
    soft_dtw_matrix = np.zeros((n, n))
    euclidean_matrix = np.zeros((n, n))
    cosine_matrix = np.zeros((n, n))
    
    # Check if matrices already exist
    if not regenerate and output_dir:
        try:
            print("\nChecking for existing distance matrices...")
            dtw_matrix = np.load(os.path.join(output_dir, 'dtw_matrix.npy'))
            soft_dtw_matrix = np.load(os.path.join(output_dir, 'soft_dtw_matrix.npy'))
            euclidean_matrix = np.load(os.path.join(output_dir, 'euclidean_matrix.npy'))
            cosine_matrix = np.load(os.path.join(output_dir, 'cosine_matrix.npy'))
            print("✅ Successfully loaded existing matrices")
            return dtw_matrix, soft_dtw_matrix, euclidean_matrix, cosine_matrix, participants
        except:
            print("❌ No existing matrices found, computing new ones...")
    
    # Normalize time series lengths
    print("\nNormalizing time series...")
    normalized_series = {}
    target_length = 100
    
    for pid, series in time_series_dict.items():
        if len(series) > 0:
            # Resample the series to target length
            x_old = np.linspace(0, 1, len(series))
            x_new = np.linspace(0, 1, target_length)
            normalized_series[pid] = np.interp(x_new, x_old, series)
            print(f"  Normalized {pid}: {len(series)} points → {target_length} points")
    
    # Compute matrices
    print("\nComputing distance matrices...")
    total_pairs = (n * (n-1)) // 2  # Number of unique pairs
    with tqdm(total=total_pairs, desc="Computing distances") as pbar:
        for i in range(n):
            for j in range(i+1, n):
                series1 = normalized_series[participants[i]]
                series2 = normalized_series[participants[j]]
                
                # DTW
                dtw_matrix[i,j] = dtw_matrix[j,i] = dtw.distance(series1, series2)
                
                # Soft-DTW
                soft_dtw_matrix[i,j] = soft_dtw_matrix[j,i] = compute_soft_dtw(series1, series2)
                
                # Euclidean
                euclidean_matrix[i,j] = euclidean_matrix[j,i] = np.linalg.norm(series1 - series2)
                
                # Cosine
                cosine_matrix[i,j] = cosine_matrix[j,i] = cosine_distances(
                    series1.reshape(1, -1), 
                    series2.reshape(1, -1)
                )[0,0]
                
                pbar.update(1)
    
    # Save matrices if output directory is provided
    if output_dir:
        print("\nSaving distance matrices...")
        # Save as NPY files
        np.save(os.path.join(output_dir, 'dtw_matrix.npy'), dtw_matrix)
        np.save(os.path.join(output_dir, 'soft_dtw_matrix.npy'), soft_dtw_matrix)
        np.save(os.path.join(output_dir, 'euclidean_matrix.npy'), euclidean_matrix)
        np.save(os.path.join(output_dir, 'cosine_matrix.npy'), cosine_matrix)
        
        # Save as CSV files
        pd.DataFrame(dtw_matrix, index=participants, columns=participants).to_csv(
            os.path.join(output_dir, 'dtw_matrix.csv'))
        pd.DataFrame(soft_dtw_matrix, index=participants, columns=participants).to_csv(
            os.path.join(output_dir, 'soft_dtw_matrix.csv'))
        pd.DataFrame(euclidean_matrix, index=participants, columns=participants).to_csv(
            os.path.join(output_dir, 'euclidean_matrix.csv'))
        pd.DataFrame(cosine_matrix, index=participants, columns=participants).to_csv(
            os.path.join(output_dir, 'cosine_matrix.csv'))
        
        # Save participant list
        pd.DataFrame({'participant': participants}).to_csv(
            os.path.join(output_dir, 'participants.csv'), index=False)
        print("✅ Successfully saved all matrices and participant list")
    
    # Print matrix statistics
    print("\nDistance Matrix Statistics:")
    print(f"DTW Matrix: min={dtw_matrix.min():.3f}, max={dtw_matrix.max():.3f}, mean={dtw_matrix.mean():.3f}")
    print(f"Soft-DTW Matrix: min={soft_dtw_matrix.min():.3f}, max={soft_dtw_matrix.max():.3f}, mean={soft_dtw_matrix.mean():.3f}")
    print(f"Euclidean Matrix: min={euclidean_matrix.min():.3f}, max={euclidean_matrix.max():.3f}, mean={euclidean_matrix.mean():.3f}")
    print(f"Cosine Matrix: min={cosine_matrix.min():.3f}, max={cosine_matrix.max():.3f}, mean={cosine_matrix.mean():.3f}")
    
    return dtw_matrix, soft_dtw_matrix, euclidean_matrix, cosine_matrix, participants

def compute_soft_dtw(series1: List[float], series2: List[float], gamma: float = 1.0) -> float:
    """
    Computes Soft-DTW distance between two time series.
    
    Args:
        series1: First time series
        series2: Second time series
        gamma: Smoothing parameter
        
    Returns:
        Soft-DTW distance
    """
    D = np.array([[np.abs(a - b) for b in series2] for a in series1])
    return SoftDTW(D, gamma=gamma).compute()

def extract_features(time_series: List[float]) -> np.ndarray:
    """
    Extracts features from time series data.
    
    Args:
        time_series: Time series data
        
    Returns:
        Array of extracted features
    """
    velocity = np.gradient(time_series)
    acceleration = np.gradient(velocity)
    jerk = np.gradient(acceleration)
    
    features = [
        np.mean(velocity), np.max(velocity), np.std(velocity),
        np.mean(acceleration), np.max(acceleration), np.std(acceleration),
        np.mean(jerk), np.std(jerk),
        entropy(np.abs(velocity) + 1e-8)
    ]
    return np.array(features)

def natural_sort_key(s: str) -> List:
    """
    Generate a key for natural sorting of strings containing numbers.
    
    Args:
        s: String to generate key for
        
    Returns:
        List of parts for sorting
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', s)]

def plot_distance_matrix(matrix: np.ndarray, 
                        title: str, 
                        participants: List[str], 
                        save_path: str) -> None:
    """
    Plots and saves a distance matrix heatmap.
    
    Args:
        matrix: Distance matrix
        title: Plot title
        participants: Participant labels
        save_path: Path to save the plot
    """
    # Sort participants using natural sort
    sorted_indices = sorted(range(len(participants)), key=lambda k: natural_sort_key(participants[k]))
    sorted_participants = [participants[i] for i in sorted_indices]
    sorted_matrix = matrix[sorted_indices][:, sorted_indices]
    
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(sorted_matrix, cmap="coolwarm", annot=False)
    
    n_labels = len(sorted_participants)
    step = max(1, n_labels // 20)
    
    ax.set_xticks(range(n_labels))
    ax.set_xticklabels([sorted_participants[i] if i % step == 0 else "" for i in range(n_labels)], 
                      fontsize=8, rotation=90, ha='right')
    
    ax.set_yticks(range(n_labels))
    ax.set_yticklabels([sorted_participants[i] if i % step == 0 else "" for i in range(n_labels)], 
                      fontsize=8)
    
    plt.title(title, fontsize=14)
    plt.xlabel("Participants", fontsize=12)
    plt.ylabel("Participants", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_mds(matrix: np.ndarray, 
            labels: List[int], 
            title: str, 
            save_path: str,
            participants: List[str] = None) -> None:
    """
    Plots and saves MDS visualization.
    
    Args:
        matrix: Distance matrix
        labels: Cluster labels
        title: Plot title
        save_path: Path to save the plot
        participants: Optional list of participant labels
    """
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords = mds.fit_transform(matrix)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="viridis", alpha=0.7)
    
    if participants:
        # Sort points by natural order of participant labels
        sorted_indices = sorted(range(len(participants)), key=lambda k: natural_sort_key(participants[k]))
        for idx in sorted_indices:
            # Extract only the actor number from participant ID
            actor_num = participants[idx].split('actor')[1].split('_')[0] if 'actor' in participants[idx].lower() else participants[idx].split('_')[0]
            plt.annotate(actor_num, (coords[idx, 0], coords[idx, 1]), 
                        fontsize=8, alpha=0.7)
    
    plt.title(title)
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.colorbar(scatter, label='Cluster')
    
    plt.savefig(save_path)
    plt.close()

def compute_silhouette(matrix: np.ndarray, labels: List[int]) -> float:
    """
    Computes silhouette score for clustering.
    
    Args:
        matrix: Distance matrix
        labels: Cluster labels
        
    Returns:
        Silhouette score
    """
    return silhouette_score(matrix, labels, metric="precomputed")

def save_result(v_f: List[float], 
                a_f: List[float], 
                df: pd.DataFrame, 
                act: str, 
                actorNumber: str, 
                output_folder: str) -> None:
    """
    Saves kinematic analysis results in a hierarchical structure within results/[emotion]/statistical.
    Saves data in both NPY and CSV formats.
    
    Args:
        v_f: Velocity data
        a_f: Acceleration data
        df: DataFrame with metrics
        act: Action type (emotion)
        actorNumber: Actor number
        output_folder: Output directory (not used, kept for compatibility)
    """
    # Create base directory structure
    base_dir = os.path.join(RESULTS_DIR, act, SUBDIRS['statistical'], 'kinematic_analysis')
    actor_dir = os.path.join(base_dir, f"Actor{actorNumber}")
    
    # Create directories
    os.makedirs(actor_dir, exist_ok=True)
    
    # Save velocity and acceleration data for each ROI
    for i, roi in enumerate(CLASSES):
        roi_dir = os.path.join(actor_dir, roi.lower())
        os.makedirs(roi_dir, exist_ok=True)
        
        # Save velocity and acceleration time series in both NPY and CSV formats
        if i < len(v_f):
            # Save as NPY
            np.save(os.path.join(roi_dir, "velocity.npy"), v_f[i])
            # Save as CSV
            pd.DataFrame({
                'frame': range(len(v_f[i])),
                'velocity': v_f[i]
            }).to_csv(os.path.join(roi_dir, "velocity.csv"), index=False)
            
        if i < len(a_f):
            # Save as NPY
            np.save(os.path.join(roi_dir, "acceleration.npy"), a_f[i])
            # Save as CSV
            pd.DataFrame({
                'frame': range(len(a_f[i])),
                'acceleration': a_f[i]
            }).to_csv(os.path.join(roi_dir, "acceleration.csv"), index=False)
        
        # Save metrics
        if roi in df.index:
            metrics = df.loc[roi].to_dict()
            # Save as JSON
            with open(os.path.join(roi_dir, "metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=4)
            # Save as CSV
            pd.DataFrame([metrics]).to_csv(os.path.join(roi_dir, "metrics.csv"), index=False)
        
        # Create and save visualization for this ROI
        if i < len(v_f) and i < len(a_f):
            try:
                # Create time axes
                time_v = np.arange(len(v_f[i])) * DT
                time_a = np.arange(len(a_f[i])) * DT
                
                # Create figure
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                fig.suptitle(f'Kinematic Analysis - Actor {actorNumber} {act} - {roi}', fontsize=14)
                
                # Plot velocity
                ax1.plot(time_v, v_f[i], 'b-', label='Velocity')
                ax1.set_ylabel('Velocity (units/s)')
                ax1.set_title('Velocity Time Series')
                ax1.grid(True)
                ax1.legend()
                
                # Plot acceleration
                ax2.plot(time_a, a_f[i], 'r-', label='Acceleration')
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Acceleration (units/s²)')
                ax2.set_title('Acceleration Time Series')
                ax2.grid(True)
                ax2.legend()
                
                # Save plot in the ROI directory
                plot_path = os.path.join(roi_dir, f'actor{actorNumber}_{act}_{roi.lower()}_kinematic.png')
                plt.tight_layout()
                plt.savefig(plot_path, dpi=300)
                plt.close()
                
            except Exception as e:
                print(f"Error creating plot for {roi}: {str(e)}")
    
    # Save summary metrics for the actor
    summary_metrics = {
        roi: df.loc[roi].to_dict() 
        for roi in CLASSES 
        if roi in df.index
    }
    # Save as JSON
    with open(os.path.join(actor_dir, "summary_metrics.json"), 'w') as f:
        json.dump(summary_metrics, f, indent=4)
    # Save as CSV
    summary_df = pd.DataFrame(summary_metrics).T
    summary_df.to_csv(os.path.join(actor_dir, "summary_metrics.csv"))

def plot_kinematic_values(v_frame: List[float], a_frame: List[float], 
                         roi: str, actor_number: int, action_type: str,
                         save_dir: str = None,
                         v_max: float = None, a_max: float = None) -> None:
    """
    Plots and saves velocity and acceleration time series for a given ROI.
    """
    try:
        print(f"\nAttempting to create plots for {roi}...")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # Default save_dir is statistical/kinematic_analysis/[Actor##]/[ROI]/
        if save_dir is None:
            save_dir = os.path.join(RESULTS_DIR, action_type, SUBDIRS['statistical'], 'kinematic_analysis', f'Actor{actor_number}', roi.lower())
        os.makedirs(save_dir, exist_ok=True)
        print(f"Directory exists: {os.path.exists(save_dir)}")
        
        # Create time axes for both velocity and acceleration
        time_v = np.arange(len(v_frame)) * DT
        time_a = np.arange(len(a_frame)) * DT
        
        print(f"Time points for velocity: {len(time_v)}")
        print(f"Time points for acceleration: {len(time_a)}")
        print(f"Velocity points: {len(v_frame)}")
        print(f"Acceleration points: {len(a_frame)}")
        
        # Calculate max values if not provided
        if v_max is None:
            v_max = max(v_frame) * 1.1  # Add 10% padding
        if a_max is None:
            a_max = max(a_frame) * 1.1  # Add 10% padding
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'Kinematic Analysis - Actor {actor_number} {action_type} - {roi}', fontsize=14)
        
        # Plot velocity
        ax1.plot(time_v, v_frame, 'b-', label='Velocity')
        ax1.set_ylabel('Velocity (units/s)')
        ax1.set_title('Velocity Time Series')
        ax1.set_ylim(0, v_max)  # Set consistent y-axis limit
        ax1.grid(True)
        ax1.legend()
        
        # Plot acceleration
        ax2.plot(time_a, a_frame, 'r-', label='Acceleration')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Acceleration (units/s²)')
        ax2.set_title('Acceleration Time Series')
        ax2.set_ylim(0, a_max)  # Set consistent y-axis limit
        ax2.grid(True)
        ax2.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save plot
        save_path = os.path.join(save_dir, f'actor{actor_number}_{action_type}_{roi.lower()}_kinematic.png')
        print(f"Attempting to save plot to: {save_path}")
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        # Verify the file was created
        if os.path.exists(save_path):
            print(f"✅ Successfully saved plot to {save_path}")
        else:
            print(f"❌ Failed to save plot to {save_path}")
            
    except Exception as e:
        print(f"❌ Error creating plot: {str(e)}")
        raise

def aggregate_kinematic_features(demographic_file: str = "data.csv") -> Dict:
    """
    Aggregates kinematic features with demographic information in a hierarchical structure.
    Saves results in both JSON and CSV formats.
    
    Args:
        demographic_file: Path to the CSV file containing demographic information
        
    Returns:
        Dictionary containing aggregated features organized by actors and emotions
    """
    # Read demographic data
    demo_df = pd.read_csv(demographic_file)
    
    # Initialize the hierarchical structure
    output = {}
    # Initialize flattened structure for CSV
    flattened_data = []
    
    # Process each actor's data
    for actor in demo_df['Actor']:
        actor_data = {}
        
        # Add demographic information
        actor_demo = demo_df[demo_df['Actor'] == actor].iloc[0]
        demographics = {
            col: int(actor_demo[col]) if isinstance(actor_demo[col], np.int64) else actor_demo[col]
            for col in demo_df.columns 
            if col not in ['Actor', 'Unnamed: 0']
        }
        actor_data['demographics'] = demographics
        
        # Initialize emotions dictionary
        actor_data['emotions'] = {}
        
        # Process each emotion
        for emotion in sorted(STANDARD_EMOTIONS):
            emotion_data = {}
            
            # Path to actor's emotion directory
            base_dir = os.path.join(RESULTS_DIR, emotion, SUBDIRS['statistical'], 'kinematic_analysis')
            actor_dir = os.path.join(base_dir, f"Actor{actor}")
            
            if not os.path.exists(actor_dir):
                continue
                
            # Process each ROI
            for roi in CLASSES:
                roi_dir = os.path.join(actor_dir, roi.lower())
                
                if not os.path.exists(roi_dir):
                    continue
                    
                # Load metrics from JSON
                metrics_file = os.path.join(roi_dir, "metrics.json")
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                        # Convert any remaining NumPy types to Python native types
                        metrics = {k: int(v) if isinstance(v, np.int64) else float(v) if isinstance(v, np.float64) else v 
                                 for k, v in metrics.items()}
                        emotion_data[roi] = metrics
                
                # Load velocity and acceleration data from .npy files
                v_file = os.path.join(roi_dir, "velocity.npy")
                a_file = os.path.join(roi_dir, "acceleration.npy")
                
                if os.path.exists(v_file) and os.path.exists(a_file):
                    v_data = np.load(v_file)
                    a_data = np.load(a_file)
                    
                    if roi not in emotion_data:
                        emotion_data[roi] = {}
                    emotion_data[roi]['velocity_series'] = v_data.tolist()
                    emotion_data[roi]['acceleration_series'] = a_data.tolist()
            
            if emotion_data:  # Only add if we have data for this emotion
                actor_data['emotions'][emotion] = emotion_data
                
                # Create flattened entry for CSV
                flattened_entry = {
                    'actor': actor,
                    'emotion': emotion,
                    **demographics  # Add all demographic fields
                }
                
                # Add metrics for each ROI
                for roi, metrics in emotion_data.items():
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            if metric_name not in ['velocity_series', 'acceleration_series']:
                                flattened_entry[f'{roi}_{metric_name}'] = value
                
                flattened_data.append(flattened_entry)
        
        if actor_data['emotions']:  # Only add if we have data for this actor
            output[str(actor)] = actor_data  # Convert actor number to string for JSON compatibility
    
    # Create aggregated results directory
    aggregated_dir = os.path.join(RESULTS_DIR, 'aggregated')
    os.makedirs(aggregated_dir, exist_ok=True)
    
    # Save aggregated results in JSON format
    output_file = os.path.join(aggregated_dir, 'aggregated_features.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)
    
    # Save flattened results in CSV format
    csv_file = os.path.join(aggregated_dir, 'aggregated_features.csv')
    if flattened_data:  # Only save if we have data
        flattened_df = pd.DataFrame(flattened_data)
        flattened_df.to_csv(csv_file, index=False)
        
        # Also save a summary statistics file
        summary_file = os.path.join(aggregated_dir, 'aggregated_summary.csv')
        summary_stats = flattened_df.describe()
        summary_stats.to_csv(summary_file)
        
        # Save correlation matrix
        corr_file = os.path.join(aggregated_dir, 'aggregated_correlations.csv')
        numeric_cols = flattened_df.select_dtypes(include=[np.number]).columns
        correlations = flattened_df[numeric_cols].corr()
        correlations.to_csv(corr_file)
    
    return output

def visualize_clustering_results(clustering_results: pd.DataFrame, 
                               distance_matrix: np.ndarray,
                               metric: str,
                               ROI: str,
                               method: str,
                               emotion: str) -> None:
    """
    Visualizes clustering results using MDS and heatmaps.
    """
    try:
        print(f"\nVisualizing clustering results for {metric} - {ROI} using {method}...")
        output_dir = os.path.join(RESULTS_DIR, emotion, SUBDIRS['clustering'], f"{metric}_{ROI}")
        os.makedirs(output_dir, exist_ok=True)
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        mds_coords = mds.fit_transform(distance_matrix)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(mds_coords[:, 0], mds_coords[:, 1], 
                            c=clustering_results['Cluster'],
                            cmap='viridis',
                            alpha=0.7)
        # Add labels for some points (only actor number)
        for i, (x, y) in enumerate(mds_coords):
            pid = clustering_results['Participant'].iloc[i]
            actor_num = pid.split('actor')[1].split('_')[0] if 'actor' in pid.lower() else pid.split('_')[0]
            plt.annotate(actor_num, (x, y), fontsize=8)
        plt.title(f'MDS Plot - {metric} {ROI} ({method})')
        plt.colorbar(scatter, label='Cluster')
        plt.savefig(os.path.join(output_dir, f'mds_{method}.png'))
        plt.close()
        
        # 2. Distance Matrix Heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(distance_matrix, 
                   cmap='viridis',
                   square=True,
                   xticklabels=clustering_results['Participant'],
                   yticklabels=clustering_results['Participant'])
        
        plt.title(f'Distance Matrix - {metric} {ROI} ({method})')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'heatmap_{method}.png'))
        plt.close()
        
        # 3. Cluster Distribution Bar Plot
        plt.figure(figsize=(8, 6))
        cluster_counts = clustering_results['Cluster'].value_counts().sort_index()
        plt.bar(cluster_counts.index, cluster_counts.values)
        plt.title(f'Cluster Distribution - {metric} {ROI} ({method})')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Participants')
        plt.savefig(os.path.join(output_dir, f'cluster_dist_{method}.png'))
        plt.close()
        
        print(f"Visualizations saved to statistical/kinematic_analysis/[Actor##]/[ROI]/ (see per-actor folders)")
        
    except Exception as e:
        print(f"Error creating clustering visualizations: {str(e)}")
        raise

def visualize_cluster_membership(results_df: pd.DataFrame, 
                               metric: str, 
                               ROI: str, 
                               method: str,
                               emotion: str) -> None:
    """
    Creates a visualization showing which subjects belong to each cluster.
    Uses the smallest available k value for plotting and exports membership
    assignments for all k values.
    
    Args:
        results_df: DataFrame containing clustering results
        metric: Type of metric (velocity/acceleration)
        ROI: Region of interest
        method: Distance method used
        emotion: Emotion being analyzed
    """
    output_dir = os.path.join(RESULTS_DIR, emotion, SUBDIRS['clustering'], f"{metric}_{ROI}")
    os.makedirs(output_dir, exist_ok=True)

    participant_col = 'participant_id' if 'participant_id' in results_df.columns else 'Participant'
    cluster_cols = sorted([c for c in results_df.columns if c.startswith('cluster_k')], key=lambda x: int(x.split('cluster_k')[1]))
    if not cluster_cols:
        print('No cluster columns found; skipping membership export.')
        return

    primary_col = cluster_cols[0]
    clusters = results_df.groupby(primary_col)[participant_col].apply(list)

    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(clusters))
    cluster_sizes = [len(members) for members in clusters]
    plt.barh(y_pos, cluster_sizes, align='center')

    for i, (cluster, members) in enumerate(clusters.items()):
        sorted_members = sorted(members, key=natural_sort_key)
        plt.text(0, i, 'Cluster ' + str(cluster) + ': ' + ', '.join(sorted_members),
                 va='center', ha='left', fontsize=10)

    plt.yticks([])
    plt.xlabel('Number of Participants')
    plt.title(f'Cluster Membership - {metric} {ROI} ({method})')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'cluster_membership_{method}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    csv_path = os.path.join(output_dir, f'cluster_membership_{method}.csv')
    records = []
    for _, row in results_df.iterrows():
        participant = row.get(participant_col)
        for col in cluster_cols:
            try:
                k_val = int(col.split('cluster_k')[1])
            except Exception:
                continue
            cluster_val = row[col]
            records.append({
                'Participant': participant,
                'Metric': metric,
                'ROI': ROI,
                'Method': method,
                'k': k_val,
                'Cluster': cluster_val
            })
    pd.DataFrame(records).to_csv(csv_path, index=False)

    print(f'Cluster membership visualization saved to {save_path}')
    print(f'Cluster membership data saved to {csv_path}')

def get_clustering_analysis(distance_matrix: np.ndarray,
                          participants: List[str],
                          metric: str,
                          ROI: str,
                          method: str,
                          emotion: str,
                          demographic_data: pd.DataFrame = None,
                          k_values: List[int] = None,
                          output_dir: str = None) -> pd.DataFrame:
    """
    Performs clustering analysis on distance matrix and merges with demographic data.
    
    Args:
        distance_matrix: Distance matrix for clustering
        participants: List of participant IDs
        metric: Type of metric (velocity/acceleration)
        ROI: Region of interest
        method: Distance method used
        emotion: Emotion being analyzed
        demographic_data: DataFrame containing demographic data
        k_values: List of k values for clustering
        output_dir: Directory to save results
        
    Returns:
        DataFrame containing clustering results and demographic data
    """
    # Adjust k values based on number of participants
    n_participants = len(participants)
    if k_values is None:
        if n_participants <= 3:
            k_values = [2]  # Use only k=2 for very small samples
        elif n_participants <= 5:
            k_values = [2, 3]  # Use k=2,3 for small samples
        else:
            k_values = [4, 6, 8]  # Use default values for larger samples
    
    # Filter out k values that are too large
    k_values = [k for k in k_values if k < n_participants]
    if not k_values:
        print(f"Warning: No valid k values for {n_participants} participants")
        return pd.DataFrame({'participant_id': participants})
        
    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results DataFrame with participant IDs
    results_df = pd.DataFrame({'participant_id': participants})
    
    # Extract actor numbers from participant IDs
    def extract_actor_number(pid):
        # Handle different formats of participant IDs
        if 'actor' in pid.lower():
            # Format: "Exp1a_TSD_actor87_anger"
            return int(pid.lower().split('actor')[1].split('_')[0])
        else:
            # Format: "87_anger" or similar
            return int(pid.split('_')[0])
    
    results_df['actor_number'] = results_df['participant_id'].apply(extract_actor_number)
    
    # Perform clustering for each k value
    for k in k_values:
        # Perform k-means on a distance matrix is not ideal, but retain behavior
        # If this is a precomputed distance matrix, KMeans will still run on it as if it were features
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(distance_matrix)
        
        # Add cluster assignments to results
        results_df[f'cluster_k{k}'] = cluster_labels
        
        # Compute silhouette score
        # Use precomputed distances for silhouette on distance matrices
        silhouette_avg = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
        results_df[f'silhouette_k{k}'] = silhouette_avg
        
        # Create MDS plot
        if output_dir:
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            mds_coords = mds.fit_transform(distance_matrix)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(mds_coords[:, 0], mds_coords[:, 1], c=cluster_labels, cmap='viridis')
            plt.title(f'MDS Plot - {method} (k={k})\n{metric} - {ROI}\nSilhouette Score: {silhouette_avg:.3f}')
            plt.colorbar(scatter, label='Cluster')
            
            # Add participant labels
            for i, participant in enumerate(participants):
                actor_num = participant.split('actor')[1].split('_')[0] if 'actor' in participant.lower() else participant.split('_')[0]
                plt.annotate(actor_num, (mds_coords[i, 0], mds_coords[i, 1]))
            
            plot_path = os.path.join(output_dir, f'{method}_k{k}_mds.png')
            plt.savefig(plot_path)
            plt.close()
    
    # Merge with demographic data if available
    if demographic_data is not None:
        # Create a copy of demographic data to avoid modifying the original
        demo_data = demographic_data.copy()
        
        # Rename 'Actor' column to 'actor_number' for merging
        demo_data = demo_data.rename(columns={'Actor': 'actor_number'})
        
        # Convert actor numbers to integers for matching
        demo_data['actor_number'] = demo_data['actor_number'].astype(int)
        results_df['actor_number'] = results_df['actor_number'].astype(int)
        
        # Merge on actor number
        results_df = pd.merge(
            results_df,
            demo_data,
            on='actor_number',
            how='left'
        )
    
    return results_df

def interpret_clusters(results_df: pd.DataFrame,
                       participants: List[str],
                       distance_matrix: Optional[np.ndarray],
                       metric: str,
                       ROI: str,
                       method: str,
                       emotion: str,
                       debug_logger: logging.Logger = None) -> None:
    """
    Produce interpretation artifacts for clustering results:
    - cluster_sizes_{method}.csv
    - exemplars_{method}.csv (medoids and intra-cluster spread)
    - cluster_feature_summary_{method}.csv (means/stds for ROI features)
    - stability_ari_{method}.csv (pairwise ARI across ks)
    Saved under results/<emotion>/clustering/<metric>_<ROI>/
    """
    try:
        out_dir = os.path.join(RESULTS_DIR, emotion, SUBDIRS['clustering'], f"{metric}_{ROI}")
        os.makedirs(out_dir, exist_ok=True)

        # 1) Cluster sizes per k
        size_rows = []
        k_cols = [c for c in results_df.columns if c.startswith('cluster_k')]
        for col in k_cols:
            k = int(col.split('_k')[1])
            counts = results_df[col].value_counts().sort_index()
            for cluster_id, cnt in counts.items():
                size_rows.append({'k': k, 'cluster': int(cluster_id), 'size': int(cnt)})
        if size_rows:
            pd.DataFrame(size_rows).sort_values(['k','cluster']).to_csv(
                os.path.join(out_dir, f'cluster_sizes_{method}.csv'), index=False)

        # 2) Exemplars (medoids) per k/cluster using distance matrix if available
        if distance_matrix is not None and participants is not None and len(participants) == distance_matrix.shape[0]:
            pid_to_idx = {pid: i for i, pid in enumerate(participants)}
            ex_rows = []
            for col in k_cols:
                k = int(col.split('_k')[1])
                for cluster_id in sorted(results_df[col].unique()):
                    members = results_df.loc[results_df[col] == cluster_id, 'participant_id'].tolist()
                    idxs = [pid_to_idx[m] for m in members if m in pid_to_idx]
                    if len(idxs) == 0:
                        continue
                    subD = distance_matrix[np.ix_(idxs, idxs)]
                    # medoid = row with minimal sum distance
                    sums = subD.sum(axis=1)
                    medoid_local_idx = int(np.argmin(sums))
                    medoid_global_idx = idxs[medoid_local_idx]
                    medoid_pid = participants[medoid_global_idx]
                    actor_num = results_df.loc[results_df['participant_id'] == medoid_pid, 'actor_number']
                    actor_num = int(actor_num.iloc[0]) if len(actor_num) else None
                    avg_intra = float(np.mean(sums / max(len(idxs) - 1, 1)))
                    ex_rows.append({
                        'k': k,
                        'cluster': int(cluster_id),
                        'exemplar_participant': medoid_pid,
                        'exemplar_actor': actor_num,
                        'members': len(idxs),
                        'avg_intra_distance': avg_intra
                    })
            if ex_rows:
                pd.DataFrame(ex_rows).sort_values(['k','cluster']).to_csv(
                    os.path.join(out_dir, f'exemplars_{method}.csv'), index=False)

        # 3) Feature summaries per cluster (for the same ROI)
        try:
            agg_csv = os.path.join(RESULTS_DIR, 'aggregated', 'aggregated_features.csv')
            if not os.path.exists(agg_csv):
                # Best-effort: build aggregated features
                aggregate_kinematic_features('data.csv')
            if os.path.exists(agg_csv):
                agg_df = pd.read_csv(agg_csv)
                # map ROI to aggregated prefix
                roi_prefix_map = {'lh': 'LH', 'rh': 'RH', 'head': 'Head', 'le': 'LE'}
                roi_prefix = roi_prefix_map.get(ROI.lower(), ROI)
                roi_cols = [c for c in agg_df.columns if c.startswith(f'{roi_prefix}_')]
                if roi_cols:
                    # Join on actor and emotion
                    tmp = results_df.copy()
                    tmp = tmp.rename(columns={'actor_number': 'actor'})
                    tmp['actor'] = tmp['actor'].astype(int)
                    # Filter aggregated to this emotion
                    agg_f = agg_df[agg_df['emotion'].str.lower() == str(emotion).lower()].copy()
                    merged = pd.merge(tmp[['participant_id','actor'] + k_cols],
                                      agg_f[['actor','emotion'] + roi_cols],
                                      on='actor', how='left')
                    # Long-form summary per k/cluster
                    sum_rows = []
                    for col in k_cols:
                        k = int(col.split('_k')[1])
                        for cluster_id, group in merged.groupby(col):
                            valid = group[roi_cols].select_dtypes(include=[np.number]).dropna(how='all')
                            if valid.empty:
                                continue
                            means = valid.mean(numeric_only=True)
                            stds = valid.std(numeric_only=True)
                            for feat in roi_cols:
                                if feat in means.index:
                                    sum_rows.append({
                                        'k': k,
                                        'cluster': int(cluster_id),
                                        'feature': feat,
                                        'mean': float(means[feat]),
                                        'std': float(stds[feat]) if not np.isnan(stds[feat]) else 0.0,
                                        'n': int(len(valid))
                                    })
                    if sum_rows:
                        pd.DataFrame(sum_rows).sort_values(['k','cluster','feature']).to_csv(
                            os.path.join(out_dir, f'cluster_feature_summary_{method}.csv'), index=False)
        except Exception as e:
            if debug_logger:
                debug_logger.warning(f"Feature summary failed: {e}")

        # 4) Stability across ks (ARI)
        try:
            ari_rows = []
            ks = sorted([int(c.split('_k')[1]) for c in k_cols])
            for i in range(len(ks)):
                for j in range(i+1, len(ks)):
                    k1, k2 = ks[i], ks[j]
                    l1 = results_df[f'cluster_k{k1}']
                    l2 = results_df[f'cluster_k{k2}']
                    ari = adjusted_rand_score(l1, l2)
                    ari_rows.append({'k1': k1, 'k2': k2, 'ARI': float(ari)})
            if ari_rows:
                pd.DataFrame(ari_rows).to_csv(os.path.join(out_dir, f'stability_ari_{method}.csv'), index=False)
        except Exception as e:
            if debug_logger:
                debug_logger.warning(f"ARI computation failed: {e}")
    except Exception as e:
        if debug_logger:
            debug_logger.error(f"interpret_clusters failed: {e}")
        else:
            print(f"interpret_clusters failed: {e}")

def analyze_and_cluster(test_mode: bool = False,
                       sample_size: int = None,
                       emotions: List[str] = None,
                       metrics: List[str] = None,
                       rois: List[str] = None,
                       regenerate_dtw: bool = False,
                       regenerate_clustering: bool = False,
                       regenerate_social_scores: bool = False,
                       debug_logger: logging.Logger = None) -> None:
    """
    Main function to analyze and cluster time series data.
    
    Args:
        test_mode: If True, run in test mode with limited data
        sample_size: Number of participants to sample (for test mode)
        emotions: List of emotions to analyze
        metrics: List of metrics to analyze
        rois: List of ROIs to analyze
        regenerate_dtw: If True, regenerate DTW matrices
        regenerate_clustering: If True, regenerate clustering results
        regenerate_social_scores: If True, regenerate social score analysis
        debug_logger: Logger instance for debugging
    """
    if debug_logger:
        debug_logger.info("Starting analyze_and_cluster function")
    
    # Load demographic data
    demographic_data = load_demographic_data()
    if demographic_data is None:
        if debug_logger:
            debug_logger.error("Failed to load demographic data")
        return
    
    if debug_logger:
        debug_logger.info(f"Loaded demographic data with shape: {demographic_data.shape}")
    
    # Set default values if not provided
    if emotions is None:
        emotions = ['happy', 'sad', 'admiration', 'anger', 'disgust', 'fear']
    if metrics is None:
        metrics = ['velocity', 'acceleration']
    if rois is None:
        rois = ['Head', 'RightHand', 'LeftHand', 'RightFoot', 'LeftFoot']
    
    # Convert ROIs to lowercase for consistency
    rois = [roi.lower() for roi in rois]
    
    if debug_logger:
        debug_logger.info(f"Processing emotions: {emotions}")
        debug_logger.info(f"Processing metrics: {metrics}")
        debug_logger.info(f"Processing ROIs: {rois}")
    
    # Process each emotion
    for emotion in emotions:
        if debug_logger:
            debug_logger.info(f"\nProcessing emotion: {emotion}")
        
        # Create emotion directory
        emotion_dir = os.path.join(RESULTS_DIR, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        
        # Load time series data
        time_series_data = load_time_series_data(emotion, metrics, rois, debug_logger)
        if not time_series_data:
            if debug_logger:
                debug_logger.error(f"No time series data found for emotion: {emotion}")
            continue
        
        # Sample participants if in test mode
        if test_mode and sample_size:
            if debug_logger:
                debug_logger.info(f"Test mode: Sampling {sample_size} participants")
            sampled_participants = list(time_series_data.keys())[:sample_size]
            time_series_data = {p: data for p, data in time_series_data.items() if p in sampled_participants}
        
        # Process each metric and ROI
        for metric in metrics:
            for roi in rois:
                if debug_logger:
                    debug_logger.info(f"\nProcessing {metric} - {roi}")
                
                # Get distance matrices
                distance_matrices = get_distance_matrices(
                    time_series_data,
                    metric,
                    roi,
                    emotion,
                    regenerate=regenerate_dtw,
                    debug_logger=debug_logger
                )
                
                if not distance_matrices:
                    if debug_logger:
                        debug_logger.error(f"Failed to get distance matrices for {metric} - {roi}")
                    continue
                
                # Get clustering analysis
                clustering_results = get_clustering_analysis(
                    distance_matrices,
                    metric,
                    roi,
                    emotion,
                    regenerate=regenerate_clustering,
                    debug_logger=debug_logger
                )
                
                if clustering_results is None:
                    if debug_logger:
                        debug_logger.error(f"Failed to get clustering analysis for {metric} - {roi}")
                    continue
                
                # Analyze social scores (aggregate across participants at emotion level)
                if not test_mode or regenerate_social_scores:
                    stat_dir = os.path.join(emotion_dir, SUBDIRS['statistical'])
                    os.makedirs(stat_dir, exist_ok=True)
                    analyze_cluster_social_scores(
                        clustering_results,
                        demographic_data,
                        metric,
                        roi,
                        'dtw',
                        output_dir=stat_dir,
                        debug_logger=debug_logger
                    )
                
                # Generate visualizations if not in test mode
                if not test_mode:
                    generate_visualizations(
                        distance_matrices,
                        clustering_results,
                        metric,
                        roi,
                        emotion,
                        debug_logger=debug_logger
                    )
    
    if debug_logger:
        debug_logger.info("Analysis completed successfully")

def analyze_cluster_social_scores(cluster_results: pd.DataFrame, 
                                demographic_data: pd.DataFrame,
                                metric: str,
                                ROI: str,
                                method: str,
                                output_dir: str = None,
                                debug_logger: logging.Logger = None) -> None:
    """
    Analyzes the relationship between social scores and cluster assignments.
    """
    if debug_logger:
        debug_logger.debug(f"Starting social score analysis for {metric} - {ROI} using {method}")
    if demographic_data is None:
        if debug_logger:
            debug_logger.error("No demographic data provided for social score analysis")
        return
    # Always save to /statistical/social_scores/[ROI]/
    if output_dir:
        # output_dir is .../statistical/[ROI] or .../statistical/kinematic_analysis/ActorX/[ROI]
        # Go up to .../statistical/
        stat_dir = output_dir
        while os.path.basename(stat_dir) not in ['statistical', 'statistical/'] and stat_dir != '' and stat_dir != os.path.dirname(stat_dir):
            stat_dir = os.path.dirname(stat_dir)
        # Now stat_dir is .../statistical
        output_dir = os.path.join(stat_dir, 'social_scores', ROI)
        os.makedirs(output_dir, exist_ok=True)
    # List of available social scores to analyze
    social_scores = [
        'AQ', 'AQ_SocialSkills', 'AQ_AttnSwitch', 'AQ_AttntoDetail',
        'AQ_Communication', 'AQ_Imagination', 'RMET', 'Alexithymia',
        'Cambridge_Behavior', 'PerceivedSocialSupport',
        'TIPI_EXT', 'TIPI_AGR', 'TIPI_CON', 'TIPI_NEU', 'TIPI_OM'
    ]
    # Get available scores from demographic data
    available_scores = [score for score in social_scores if score.lower().strip() in [col.lower().strip() for col in demographic_data.columns]]
    if debug_logger:
        debug_logger.debug(f"Available social scores: {available_scores}")
    if not available_scores:
        if debug_logger:
            debug_logger.error("No social scores found in the demographic data. Columns found: " + str(list(demographic_data.columns)))
        return
    results = {
        'anova': {},
        'correlations': {},
        'descriptive_stats': {}
    }
    for col in cluster_results.columns:
        if col.startswith('cluster_k'):
            k = int(col.split('_k')[1])
            cluster_col = col
            if cluster_col not in cluster_results.columns:
                if debug_logger:
                    debug_logger.warning(f"Cluster column {cluster_col} not found in clustering results.")
                continue
            for score in available_scores:
                if debug_logger:
                    debug_logger.debug(f"\nAnalyzing {score} for k={k}...")
                valid_data = cluster_results[cluster_results[score] > -50].copy()
                if debug_logger:
                    debug_logger.debug(f"Valid data points for {score}: {len(valid_data)}")
                if len(valid_data) < 4:
                    if debug_logger:
                        debug_logger.warning(f"Not enough valid data for {score} (need >=4, got {len(valid_data)})")
                    continue
                groups = [group for _, group in valid_data.groupby(cluster_col)[score]]
                if len(groups) < 2:
                    if debug_logger:
                        debug_logger.warning(f"Not enough clusters for ANOVA for {score} (need >=2, got {len(groups)})")
                    continue
                f_stat, p_val = f_oneway(*groups)
                results['anova'][f'{score}_k{k}'] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_val),
                    'significant': bool(p_val < 0.05)
                }
                correlation = valid_data[score].corr(valid_data[cluster_col])
                results['correlations'][f'{score}_k{k}'] = {
                    'correlation': float(correlation),
                    'significant': bool(abs(correlation) > 0.3)
                }
                stats = valid_data.groupby(cluster_col)[score].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ])
                stats_dict = {}
                for cluster in stats.index:
                    stats_dict[str(cluster)] = {
                        'count': int(stats.loc[cluster, 'count']),
                        'mean': float(stats.loc[cluster, 'mean']),
                        'std': float(stats.loc[cluster, 'std']),
                        'min': float(stats.loc[cluster, 'min']),
                        'max': float(stats.loc[cluster, 'max'])
                    }
                results['descriptive_stats'][f'{score}_k{k}'] = stats_dict
                if output_dir:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x=cluster_col, y=score, data=valid_data)
                    plt.title(f'Distribution of {score} by Cluster (k={k})\n{metric} - {ROI} ({method})')
                    plt.xlabel('Cluster')
                    plt.ylabel(score)
                    plot_path = os.path.join(output_dir, f'{method}_{score}_k{k}_boxplot.png')
                    plt.savefig(plot_path)
                    plt.close()
    if output_dir and results['anova']:
        results_path = os.path.join(output_dir, f'{method}_social_scores_analysis.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        # Build summary for whatever k values were actually computed
        summary_data = []
        ks = sorted({int(col.split('_k')[1]) for col in cluster_results.columns if col.startswith('cluster_k')})
        for k in ks:
            for score in available_scores:
                key = f'{score}_k{k}'
                if key in results['anova']:
                    summary_data.append({
                        'Score': score,
                        'k': k,
                        'ANOVA_F': f"{results['anova'][key]['f_statistic']:.3f}",
                        'ANOVA_p': f"{results['anova'][key]['p_value']:.3f}",
                        'ANOVA_Sig': '*' if results['anova'][key]['significant'] else '',
                        'Correlation': f"{results['correlations'][key]['correlation']:.3f}",
                        'Corr_Sig': '*' if results['correlations'][key]['significant'] else ''
                    })
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(output_dir, f'{method}_social_scores_summary.csv')
            summary_df.to_csv(summary_path, index=False)

def save_time_series_data(v_total: Dict[str, List[float]], 
                         a_total: Dict[str, List[float]], 
                         time_series_data: Dict[str, Any]) -> None:
    """
    Saves time series data in a structured format.
    
    Args:
        v_total: Dictionary of velocity time series
        a_total: Dictionary of acceleration time series
        time_series_data: Dictionary containing all time series data
    """
    # Save velocity and acceleration data
    np.savez(os.path.join(RESULTS_DIR, 'v_total.npz'), **v_total)
    np.savez(os.path.join(RESULTS_DIR, 'a_total.npz'), **a_total)
    
    # Save complete time series data
    with open(os.path.join(RESULTS_DIR, 'time_series_data.json'), 'w') as f:
        json.dump(time_series_data, f, indent=4)
    
    # Save data for each emotion
    for emotion in sorted(STANDARD_EMOTIONS):
        emotion_dir = os.path.join(RESULTS_DIR, emotion, SUBDIRS['time_series'])
        os.makedirs(emotion_dir, exist_ok=True)
        
        # Save emotion-specific data
        emotion_data = {
            metric: {
                roi: data 
                for roi, data in roi_data.items() 
                if emotion in roi
            }
            for metric, roi_data in time_series_data.items()
        }
        
        # Only save if there's data for this emotion
        if any(emotion_data.values()):
            with open(os.path.join(emotion_dir, 'time_series_data.json'), 'w') as f:
                json.dump(emotion_data, f, indent=4)

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
    
    for emotion in sorted(STANDARD_EMOTIONS):
        emotion_dir = os.path.join(RESULTS_DIR, emotion, SUBDIRS['time_series'])
        v_total_file = os.path.join(emotion_dir, 'v_total.npz')
        a_total_file = os.path.join(emotion_dir, 'a_total.npz')
        time_series_file = os.path.join(emotion_dir, 'time_series_data.json')
        
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
                if metric in time_series_data_loaded:
                    for roi in time_series_data_loaded[metric]:
                        if roi in time_series_data[metric]:
                            for pid, val in time_series_data_loaded[metric][roi].items():
                                time_series_data[metric][roi][pid] = np.array(val)

    if data_found:
        return v_total, a_total, time_series_data

    print("⚠️ Time series data not found, needs to be recomputed.")
    return None, None, None

def load_demographic_data(file_path: str = "data.csv") -> Optional[pd.DataFrame]:
    """
    Loads demographic data from a CSV file.
    
    Args:
        file_path: Path to the CSV file containing demographic information
        
    Returns:
        DataFrame containing demographic data if file exists, None otherwise
    """
    try:
        if not os.path.exists(file_path):
            print(f"Warning: Demographic data file {file_path} not found")
            return None
            
        demo_df = pd.read_csv(file_path)
        print(f"Loaded demographic data with shape: {demo_df.shape}")
        print(f"Demographic data columns: {demo_df.columns.tolist()}")
        return demo_df
        
    except Exception as e:
        print(f"Error loading demographic data: {str(e)}")
        return None 
