import os
import shutil
from typing import Dict, List

# Consolidate result folders into a single canonical folder per emotion
# and normalize the internal structure for downstream loading.

# Map messy names to canonical emotion names
EMOTION_MAP: Dict[str, str] = {
    # happiness variants
    'happy': 'happiness', 'happiness': 'happiness', 'Happiness': 'happiness', 'Happy': 'happiness',
    # sadness variants
    'sad': 'sadness', 'sadness': 'sadness', 'Sad': 'sadness', 'Sadness': 'sadness',
    'sad - part (2)': 'sadness', 'sad - part 1': 'sadness', 'sad - part 2': 'sadness',
    # despair variants
    'despair': 'despair', 'dispair': 'despair', 'Dispair': 'despair', 'dispar': 'despair', 'Dispar': 'despair',
    # typos
    'admiration': 'admiration', 'admiraton': 'admiration',
    'guilt': 'guilt', 'guilit': 'guilt',
}

# Standardized ROI names mapping
ROI_MAP: Dict[str, str] = {
    'left_hand': 'lh', 'LH': 'lh', 'lh': 'lh', 'lefthand': 'lh',
    'right_hand': 'rh', 'RH': 'rh', 'rh': 'rh', 'righthand': 'rh',
    'head': 'head', 'Head': 'head',
    'le': 'le', 'lower_extremity': 'le', 'LowerExtremity': 'le', 'Lower_Extremity': 'le', 'lowerextremity': 'le'
}

RESULTS_DIR = os.path.join(os.getcwd(), 'results')

STANDARD_SUBDIRS = {
    'time_series': os.path.join('time_series'),
    'dtw': os.path.join('dtw'),
    'clustering': os.path.join('clustering'),
    'stat_kinematic': os.path.join('statistical', 'kinematic_analysis'),
    'stat_social': os.path.join('statistical', 'social_scores'),
}

def norm_emotion_name(name: str) -> str:
    n = name.strip().strip("/\\")
    # direct map or lowercase lookup
    if n in EMOTION_MAP:
        return EMOTION_MAP[n]
    low = n.lower()
    if low in EMOTION_MAP:
        return EMOTION_MAP[low]
    # default to lowercase token
    return low

def ensure_dirs(base: str, subpaths: List[str]) -> None:
    for sp in subpaths:
        os.makedirs(os.path.join(base, sp), exist_ok=True)

def move_tree(src: str, dst: str) -> None:
    if os.path.normcase(os.path.abspath(src)) == os.path.normcase(os.path.abspath(dst)):
        return
    os.makedirs(dst, exist_ok=True)
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target_root = os.path.join(dst, rel) if rel != '.' else dst
        os.makedirs(target_root, exist_ok=True)
        for f in files:
            sfile = os.path.join(root, f)
            dfile = os.path.join(target_root, f)
            try:
                if os.path.exists(dfile):
                    try:
                        os.remove(dfile)
                    except Exception:
                        pass
                shutil.move(sfile, dfile)
            except Exception:
                try:
                    shutil.copy2(sfile, dfile)
                    os.remove(sfile)
                except Exception:
                    pass

def normalize_roi_dirname(name: str) -> str:
    key = name.replace('-', '').replace(' ', '').replace('_', '')
    # try direct and lowercase keys
    return ROI_MAP.get(name, ROI_MAP.get(key, ROI_MAP.get(key.lower(), name.lower())))

def fix_roi_layer(path: str) -> None:
    """Normalize ROI directory names one level under given path."""
    if not os.path.isdir(path):
        return
    try:
        for entry in list(os.scandir(path)):
            if entry.is_dir():
                normalized = normalize_roi_dirname(entry.name)
                if normalized != entry.name:
                    src = entry.path
                    dst = os.path.join(path, normalized)
                    if not os.path.exists(dst):
                        try:
                            os.rename(src, dst)
                        except Exception:
                            pass
    except Exception:
        pass

def flatten_extra_emotion_layer(stat_kin_dir: str, emotion_name: str) -> None:
    """Some runs created an extra emotion folder under ActorX/. Move ROI dirs up."""
    if not os.path.isdir(stat_kin_dir):
        return
    for actor_entry in list(os.scandir(stat_kin_dir)):
        if not actor_entry.is_dir():
            continue
        nested_emotion = os.path.join(actor_entry.path, emotion_name)
        if os.path.isdir(nested_emotion):
            # Move all first-level ROI directories up
            for roi_entry in list(os.scandir(nested_emotion)):
                if roi_entry.is_dir():
                    target = os.path.join(actor_entry.path, normalize_roi_dirname(roi_entry.name))
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    move_tree(roi_entry.path, target)
            # Try to remove the now-empty extra layer
            try:
                for root, dirs, files in os.walk(nested_emotion, topdown=False):
                    for f in files:
                        try:
                            os.remove(os.path.join(root, f))
                        except Exception:
                            pass
                    for d in dirs:
                        try:
                            os.rmdir(os.path.join(root, d))
                        except Exception:
                            pass
                os.rmdir(nested_emotion)
            except Exception:
                pass

def relocate_social_scores(stat_dir: str) -> None:
    """Ensure social scores live in statistical/social_scores/<roi>. Move from misplaced locations if needed."""
    desired_root = os.path.join(stat_dir, 'social_scores')
    os.makedirs(desired_root, exist_ok=True)
    # Search for any social score artifacts under statistical/** and move them under social_scores/<roi>
    for root, dirs, files in os.walk(stat_dir):
        # Skip the correct destination
        if os.path.normcase(os.path.abspath(root)).startswith(os.path.normcase(os.path.abspath(desired_root))):
            continue
        # Identify files that look like social score outputs
        for f in files:
            if 'social_scores' in f or f.endswith('_social_scores_summary.csv') or f.endswith('_social_scores_analysis.json') or f.endswith('_boxplot.png'):
                roi = 'unknown'
                parts = root.replace('\\', '/').split('/')
                # Heuristic: ROI is the last path part that matches known ROI names
                for p in reversed(parts):
                    cand = normalize_roi_dirname(p)
                    if cand in ['lh', 'rh', 'head', 'le']:
                        roi = cand
                        break
                target_dir = os.path.join(desired_root, roi)
                os.makedirs(target_dir, exist_ok=True)
                try:
                    shutil.move(os.path.join(root, f), os.path.join(target_dir, f))
                except Exception:
                    try:
                        shutil.copy2(os.path.join(root, f), os.path.join(target_dir, f))
                        os.remove(os.path.join(root, f))
                    except Exception:
                        pass

def ensure_emotion_structure(emotion_dir: str, emotion_name: str) -> None:
    # Ensure standard subfolders
    ensure_dirs(emotion_dir, list(STANDARD_SUBDIRS.values()))
    # Normalize ROI dirnames under dtw and clustering
    fix_roi_layer(os.path.join(emotion_dir, STANDARD_SUBDIRS['dtw']))
    fix_roi_layer(os.path.join(emotion_dir, STANDARD_SUBDIRS['clustering']))
    # Fix extra nested emotion layer under statistical/kinematic_analysis/ActorX/<emotion>/roi
    stat_kin = os.path.join(emotion_dir, STANDARD_SUBDIRS['stat_kinematic'])
    flatten_extra_emotion_layer(stat_kin, emotion_name)
    # Move any misplaced social scores into statistical/social_scores/<roi>
    stat_dir = os.path.join(emotion_dir, 'statistical')
    relocate_social_scores(stat_dir)

def consolidate() -> None:
    if not os.path.isdir(RESULTS_DIR):
        print(f"Results dir not found: {RESULTS_DIR}")
        return

    # First, consolidate all emotion folders into canonical names
    entries = [e for e in os.scandir(RESULTS_DIR) if e.is_dir()]
    for e in entries:
        canon = norm_emotion_name(e.name)
        if canon != e.name:
            src = os.path.join(RESULTS_DIR, e.name)
            dst = os.path.join(RESULTS_DIR, canon)
            print(f"Consolidating {e.name} -> {canon}")
            move_tree(src, dst)
            # try removing now-empty src tree
            try:
                for root, dirs, files in os.walk(src, topdown=False):
                    for f in files:
                        try:
                            os.remove(os.path.join(root, f))
                        except Exception:
                            pass
                    for d in dirs:
                        try:
                            os.rmdir(os.path.join(root, d))
                        except Exception:
                            pass
                os.rmdir(src)
            except Exception:
                pass

    # Now enforce standardized internal structure per emotion
    emotions = [d.name for d in os.scandir(RESULTS_DIR) if d.is_dir()]
    for em in emotions:
        em_dir = os.path.join(RESULTS_DIR, em)
        ensure_emotion_structure(em_dir, em)
        print(f"Standardized structure for emotion: {em}")

if __name__ == '__main__':
    consolidate()
