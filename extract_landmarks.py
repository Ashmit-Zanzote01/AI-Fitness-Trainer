import cv2
import mediapipe as mp
import numpy as np
import os
import json
import logging
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import time
import warnings
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['GLOG_minloglevel'] = '3'  # Suppress MediaPipe glog

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('feature_extraction.log'), logging.StreamHandler()]
)

# Exercise alias mapping for standardization
EXERCISE_ALIAS = {
    "barbell biceps curl": "bicep_curl",
    "push up": "push_up",
    "shoulder press": "shoulder_press",
    # Removed squat as per requirement #1
}

# Exercise configurations for required landmarks
EXERCISE_CONFIG = {
    "bicep_curl": {
        "visibility_threshold": 0.2,
        "required_landmarks": {
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
        },
        "optional_landmarks": {
            mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
            mp.solutions.pose.PoseLandmark.RIGHT_ELBOW
        }
    },
    "push_up": {
        "visibility_threshold": 0.1,  # Lower threshold for push-ups
        "required_landmarks": {
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
        },
        "optional_landmarks": {
            mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
            mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
            mp.solutions.pose.PoseLandmark.LEFT_WRIST,
            mp.solutions.pose.PoseLandmark.RIGHT_WRIST
        }
    },
    "shoulder_press": {
        "visibility_threshold": 0.2,
        "required_landmarks": {
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
        },
        "optional_landmarks": {
            mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
            mp.solutions.pose.PoseLandmark.RIGHT_ELBOW
        }
    }
    # Removed squat config as per requirement #1
}

# Default configuration for fallback
DEFAULT_CONFIG = {
    "visibility_threshold": 0.2,
    "required_landmarks": {
        mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
    },
    "optional_landmarks": {
        mp.solutions.pose.PoseLandmark.LEFT_HIP,
        mp.solutions.pose.PoseLandmark.RIGHT_HIP
    }
}

# Constants
TARGET_FRAME_WIDTH = 640
MIN_FRAME_HEIGHT = 480
MIN_VALID_FRAMES = 5  # Minimum frames needed for a valid sequence
FRAME_SKIP = 3  # Fixed frame skip for consistent sampling
SLIDING_WINDOW_SIZE = 20  # Window size for sequences (requirement #6)
WINDOW_STRIDE = 10  # Stride between consecutive windows

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose

# Define landmark indices for feature extraction
LANDMARK_INDICES = {
    'LEFT_SHOULDER': mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    'RIGHT_SHOULDER': mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    'LEFT_ELBOW': mp_pose.PoseLandmark.LEFT_ELBOW.value,
    'RIGHT_ELBOW': mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    'LEFT_WRIST': mp_pose.PoseLandmark.LEFT_WRIST.value,
    'RIGHT_WRIST': mp_pose.PoseLandmark.RIGHT_WRIST.value,
    'LEFT_HIP': mp_pose.PoseLandmark.LEFT_HIP.value,
    'RIGHT_HIP': mp_pose.PoseLandmark.RIGHT_HIP.value,
    'LEFT_KNEE': mp_pose.PoseLandmark.LEFT_KNEE.value,
    'RIGHT_KNEE': mp_pose.PoseLandmark.RIGHT_KNEE.value,
    'LEFT_ANKLE': mp_pose.PoseLandmark.LEFT_ANKLE.value,
    'RIGHT_ANKLE': mp_pose.PoseLandmark.RIGHT_ANKLE.value
}

relevant_landmarks_indices = list(LANDMARK_INDICES.values())

# Core geometric calculation functions
def calculate_angle(a, b, c):
    """Calculate 3D angle between three points"""
    try:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cosine_angle))
    except:
        return 180.0  # Neutral value for missing angles (requirement #5)

def calculate_distance(a, b):
    """Calculate 3D Euclidean distance between two points"""
    try:
        return np.linalg.norm(np.array(a) - np.array(b))
    except:
        return 0.0  # Neutral value for missing distances (requirement #5)

def calculate_y_distance(a, b):
    """Calculate vertical (Y-axis) distance between two points"""
    try:
        return abs(a[1] - b[1])
    except:
        return 0.0  # Neutral value for missing vertical distances (requirement #5)

def apply_sharpening(frame):
    """Applies simple sharpening filter"""
    kernel = np.array([[-1, -1, -1], 
                       [-1, 9, -1], 
                       [-1, -1, -1]])
    return cv2.filter2D(frame, -1, kernel)

def detect_body_orientation(keypoints):
    """Detect if body is facing camera, sideways, or away"""
    try:
        indices = {name: relevant_landmarks_indices.index(value) 
                 for name, value in LANDMARK_INDICES.items()}
        
        ls = keypoints[indices['LEFT_SHOULDER']]
        rs = keypoints[indices['RIGHT_SHOULDER']]
        
        # Check if valid coordinates
        if -1 in ls or -1 in rs:
            return "unknown"
            
        # Use z-coordinates to determine orientation
        if ls[2] < -0.15 and rs[2] < -0.15:
            return "facing_away"
        elif ls[2] > 0.15 and rs[2] > 0.15:
            return "facing_camera"
        else:
            return "side_view"
    except:
        return "unknown"

def adaptive_preprocess_frame(frame):
    """
    Simple frame preprocessing that only resizes the frame for optimal detection
    """
    if frame is None:
        return None
        
    # Get frame dimensions
    h, w = frame.shape[:2]
    aspect_ratio = w / h
    
    # Calculate new dimensions while preserving aspect ratio
    new_w = TARGET_FRAME_WIDTH
    new_h = int(new_w / aspect_ratio)
    
    if new_h < MIN_FRAME_HEIGHT:
        new_h = MIN_FRAME_HEIGHT
        new_w = int(new_h * aspect_ratio)
    
    # Resize the frame (only transformation needed for efficiency)
    return cv2.resize(frame, (new_w, new_h))

def get_robust_normalization_factor(keypoints, fallback_torso_height=1.0):
    """
    Calculate normalization factor from torso dimensions with fallbacks for partial visibility
    Consistent normalization strategy as per requirement #4
    """
    try:
        # Get indices from ordered list
        ls_idx = relevant_landmarks_indices.index(LANDMARK_INDICES['LEFT_SHOULDER'])
        rs_idx = relevant_landmarks_indices.index(LANDMARK_INDICES['RIGHT_SHOULDER'])
        lh_idx = relevant_landmarks_indices.index(LANDMARK_INDICES['LEFT_HIP'])
        rh_idx = relevant_landmarks_indices.index(LANDMARK_INDICES['RIGHT_HIP'])
        
        # 3D coordinates
        ls = keypoints[ls_idx]
        rs = keypoints[rs_idx]
        lh = keypoints[lh_idx]
        rh = keypoints[rh_idx]
        
        # Check if we have the main torso points
        valid_left_torso = -1 not in np.concatenate([ls, lh])
        valid_right_torso = -1 not in np.concatenate([rs, rh])
        valid_shoulders = -1 not in np.concatenate([ls, rs])
        valid_hips = -1 not in np.concatenate([lh, rh])
        
        factors = []
        
        # Use available measurements with fallbacks
        if valid_left_torso:
            factors.append(calculate_distance(ls, lh))
        if valid_right_torso:
            factors.append(calculate_distance(rs, rh))
        if valid_shoulders:
            factors.append(calculate_distance(ls, rs))
        if valid_hips:
            factors.append(calculate_distance(lh, rh))
            
        # If we have at least one valid measurement, use the average
        if factors:
            return sum(factors) / len(factors)
        else:
            return fallback_torso_height
    except:
        return fallback_torso_height

def calculate_angles_3d(keypoints, exercise_type=None, orientation="unknown"):
    """Calculate 8 key joint angles in 3D with neutral values for missing landmarks"""
    try:
        indices = {name: relevant_landmarks_indices.index(value) 
                 for name, value in LANDMARK_INDICES.items()}

        angles = []
        
        # Arms angles
        left_arm_angle = compute_safe_angle(
            keypoints[indices['LEFT_SHOULDER']],
            keypoints[indices['LEFT_ELBOW']],
            keypoints[indices['LEFT_WRIST']]
        )
        right_arm_angle = compute_safe_angle(
            keypoints[indices['RIGHT_SHOULDER']],
            keypoints[indices['RIGHT_ELBOW']],
            keypoints[indices['RIGHT_WRIST']]
        )
        
        # If push-up with missing arm angles, try to estimate
        if exercise_type == "push_up" and (left_arm_angle == 180.0 or right_arm_angle == 180.0):
            # For push-ups, we can estimate arm angle from shoulder height relative to body
            if orientation == "facing_away" or orientation == "unknown":
                # When facing away or unknown, estimate arm bend from Y position
                if left_arm_angle == 180.0 and -1 not in keypoints[indices['LEFT_SHOULDER']]:
                    y_shoulder = keypoints[indices['LEFT_SHOULDER']][1]
                    # Estimate bend based on shoulder height (lower = more bent)
                    left_arm_angle = max(90 - y_shoulder * 180, 0)  # Rough approximation
                
                if right_arm_angle == 180.0 and -1 not in keypoints[indices['RIGHT_SHOULDER']]:
                    y_shoulder = keypoints[indices['RIGHT_SHOULDER']][1]
                    right_arm_angle = max(90 - y_shoulder * 180, 0)  # Rough approximation
        
        angles.append(left_arm_angle)
        angles.append(right_arm_angle)
        
        # Legs angles
        angles.append(compute_safe_angle(
            keypoints[indices['LEFT_HIP']],
            keypoints[indices['LEFT_KNEE']],
            keypoints[indices['LEFT_ANKLE']]
        ))
        
        angles.append(compute_safe_angle(
            keypoints[indices['RIGHT_HIP']],
            keypoints[indices['RIGHT_KNEE']],
            keypoints[indices['RIGHT_ANKLE']]
        ))
        
        # Hips angles
        angles.append(compute_safe_angle(
            keypoints[indices['LEFT_SHOULDER']],
            keypoints[indices['LEFT_HIP']],
            keypoints[indices['LEFT_KNEE']]
        ))
        
        angles.append(compute_safe_angle(
            keypoints[indices['RIGHT_SHOULDER']],
            keypoints[indices['RIGHT_HIP']],
            keypoints[indices['RIGHT_KNEE']]
        ))
        
        # Shoulder angles
        angles.append(compute_safe_angle(
            keypoints[indices['LEFT_HIP']],
            keypoints[indices['LEFT_SHOULDER']],
            keypoints[indices['LEFT_ELBOW']]
        ))
        
        angles.append(compute_safe_angle(
            keypoints[indices['RIGHT_HIP']],
            keypoints[indices['RIGHT_SHOULDER']],
            keypoints[indices['RIGHT_ELBOW']]
        ))
        
        # If we couldn't calculate all angles, pad with neutral values
        if len(angles) < 8:
            angles.extend([180.0] * (8 - len(angles)))
            
        return angles
    except:
        return [180.0] * 8  # Return neutral angles if calculation fails

def compute_safe_angle(a, b, c):
    """Calculate angle with safety checks for invalid points"""
    if -1 in a or -1 in b or -1 in c:
        return 180.0  # Neutral value for missing angles (requirement #5)
    try:
        angle = calculate_angle(a, b, c)
        return angle if np.isfinite(angle) else 180.0
    except:
        return 180.0

def calculate_distances_3d(keypoints, norm_factor=1.0, exercise_type=None):
    """Calculate 12 normalized 3D distances with neutral values for missing landmarks"""
    if norm_factor is None or norm_factor <= 0:
        norm_factor = 1.0
        
    try:
        indices = {name: relevant_landmarks_indices.index(value) 
                 for name, value in LANDMARK_INDICES.items()}

        distance_pairs = [
            # Horizontal distances
            (indices['LEFT_SHOULDER'], indices['RIGHT_SHOULDER']),
            (indices['LEFT_HIP'], indices['RIGHT_HIP']),
            # Limb lengths
            (indices['LEFT_HIP'], indices['LEFT_KNEE']),
            (indices['RIGHT_HIP'], indices['RIGHT_KNEE']),
            # Torso connections
            (indices['LEFT_SHOULDER'], indices['LEFT_HIP']),
            (indices['RIGHT_SHOULDER'], indices['RIGHT_HIP']),
            # Cross-body connections
            (indices['LEFT_ELBOW'], indices['LEFT_KNEE']),
            (indices['RIGHT_ELBOW'], indices['RIGHT_KNEE']),
            # Arm extensions (wrist to shoulder)
            (indices['LEFT_WRIST'], indices['LEFT_SHOULDER']),
            (indices['RIGHT_WRIST'], indices['RIGHT_SHOULDER']),
            # Body-wrist connections
            (indices['LEFT_WRIST'], indices['LEFT_HIP']),
            (indices['RIGHT_WRIST'], indices['RIGHT_HIP'])
        ]

        distances = []
        for i, j in distance_pairs:
            # Normal case processing
            if -1 in keypoints[i] or -1 in keypoints[j]:
                # Use an average neutral distance (requirement #5)
                distances.append(0.5)  # Normalized neutral value
                continue
                
            dist = calculate_distance(keypoints[i], keypoints[j])
            norm_dist = dist / norm_factor if dist > 0 else 0.5  # Use neutral value if calculation fails
            distances.append(norm_dist)
        
        # Ensure we have exactly 12 distances
        if len(distances) < 12:
            distances.extend([0.5] * (12 - len(distances)))
            
        return distances[:12]
    except:
        return [0.5] * 12  # Return neutral distances if calculation fails

def calculate_z_features(keypoints, norm_factor=1.0):
    """Calculate depth-specific features with neutral values for missing landmarks"""
    if norm_factor is None or norm_factor <= 0:
        norm_factor = 1.0
        
    try:
        indices = {name: relevant_landmarks_indices.index(value) 
                 for name, value in LANDMARK_INDICES.items()}

        z_pairs = [
            # Shoulder depth comparison
            (indices['LEFT_SHOULDER'], indices['RIGHT_SHOULDER']),
            # Hip-knee depth
            (indices['LEFT_HIP'], indices['LEFT_KNEE']),
            (indices['RIGHT_HIP'], indices['RIGHT_KNEE']),
            # Vertical arm depth
            (indices['LEFT_WRIST'], indices['LEFT_SHOULDER'])
        ]

        z_features = []
        for i, j in z_pairs:
            if -1 in keypoints[i] or -1 in keypoints[j]:
                z_features.append(0.0)  # Neutral value for z-features (requirement #5)
                continue
            # Calculate absolute depth difference
            z_diff = abs(keypoints[i][2] - keypoints[j][2])
            norm_z = z_diff / norm_factor
            z_features.append(norm_z)

        # Ensure we have exactly 4 z-features
        if len(z_features) < 4:
            z_features.extend([0.0] * (4 - len(z_features)))
            
        return z_features[:4]
    except:
        return [0.0] * 4  # Return neutral z-features if calculation fails

def calculate_y_distances(keypoints, norm_factor=1.0):
    """Calculate vertical distance features with neutral values for missing landmarks"""
    if norm_factor is None or norm_factor <= 0:
        norm_factor = 1.0
        
    try:
        indices = {name: relevant_landmarks_indices.index(value) 
                 for name, value in LANDMARK_INDICES.items()}

        y_pairs = [
            (indices['LEFT_ELBOW'], indices['LEFT_SHOULDER']),
            (indices['RIGHT_ELBOW'], indices['RIGHT_SHOULDER']),
            (indices['LEFT_HIP'], indices['LEFT_ANKLE']),
            (indices['RIGHT_HIP'], indices['RIGHT_ANKLE'])
        ]

        y_distances = []
        for i, j in y_pairs:
            if -1 in keypoints[i] or -1 in keypoints[j]:
                y_distances.append(0.5)  # Neutral normalized value (requirement #5)
                continue
            y_dist = calculate_y_distance(keypoints[i], keypoints[j])
            norm_dist = y_dist / norm_factor if y_dist > 0 else 0.5
            y_distances.append(norm_dist)
        
        # Ensure we have exactly 4 y-distances
        if len(y_distances) < 4:
            y_distances.extend([0.5] * (4 - len(y_distances)))
            
        return y_distances[:4]
    except:
        return [0.5] * 4  # Return neutral y-distances if calculation fails

def interpolate_missing_features(frames):
    """Interpolate missing feature values between valid frames"""
    if not frames or len(frames) < 2:
        return frames
        
    # Process each feature type
    feature_types = ['angles', 'distances', 'y_distances', 'z_features']
    
    # Find frames with complete data to use as reference
    complete_frames = []
    for i, frame in enumerate(frames):
        is_complete = True
        for ft in feature_types:
            if -1 in frame[ft]:
                is_complete = False
                break
        if is_complete:
            complete_frames.append((i, frame))
    
    # If we don't have at least 2 complete frames, we can't interpolate
    if len(complete_frames) < 2:
        return frames
    
    # For each incomplete frame, interpolate from nearest complete frames
    for i in range(len(frames)):
        frame = frames[i]
        
        # Skip already complete frames
        all_complete = True
        for ft in feature_types:
            if -1 in frame[ft]:
                all_complete = False
                break
        if all_complete:
            continue
            
        # Find nearest complete frames before and after
        prev_complete = None
        next_complete = None
        
        for j, complete_frame in complete_frames:
            if j < i:
                if prev_complete is None or j > prev_complete[0]:
                    prev_complete = (j, complete_frame)
            elif j > i:
                if next_complete is None or j < next_complete[0]:
                    next_complete = (j, complete_frame)
                    
        # If we don't have both before and after frames, use what we have
        if prev_complete is None and next_complete is not None:
            prev_complete = next_complete
        elif next_complete is None and prev_complete is not None:
            next_complete = prev_complete
        
        # If we still don't have reference frames, skip this frame
        if prev_complete is None or next_complete is None:
            continue
            
        # Calculate interpolation weight based on frame position
        if prev_complete[0] == next_complete[0]:
            weight = 0.5
        else:
            weight = (i - prev_complete[0]) / (next_complete[0] - prev_complete[0])
        
        # Interpolate each feature value
        for ft in feature_types:
            for j in range(len(frame[ft])):
                if frame[ft][j] == -1:
                    # Linear interpolation between prev and next
                    prev_value = prev_complete[1][ft][j]
                    next_value = next_complete[1][ft][j]
                    
                    # Only interpolate if both reference values are valid
                    if prev_value != -1 and next_value != -1:
                        frame[ft][j] = prev_value + weight * (next_value - prev_value)
                    else:
                        # Use neutral values instead of -1
                        if ft == 'angles':
                            frame[ft][j] = 180.0
                        elif ft == 'distances' or ft == 'y_distances':
                            frame[ft][j] = 0.5
                        elif ft == 'z_features':
                            frame[ft][j] = 0.0
    
    return frames

def create_sliding_windows(features_data, window_size=SLIDING_WINDOW_SIZE, stride=WINDOW_STRIDE):
    """Create sliding windows from features data for sequence processing (requirement #6)"""
    if not features_data or len(features_data) < window_size:
        return []
    
    windows = []
    feature_names = ['angles', 'distances', 'y_distances', 'z_features']
    
    for start_idx in range(0, len(features_data) - window_size + 1, stride):
        window = []
        for i in range(start_idx, start_idx + window_size):
            frame_features = []
            for feature_name in feature_names:
                frame_features.extend(features_data[i][feature_name])
            window.append(frame_features)
        windows.append(window)
    
    return windows

def extract_features(video_path, exercise_type=None):
    """
    Optimized feature extraction function that processes frames efficiently
    Extracts full 28-dimensional feature vector (requirement #3)
    """
    # Get exercise-specific settings or use defaults
    if exercise_type in EXERCISE_CONFIG:
        config = EXERCISE_CONFIG[exercise_type]
    else:
        config = DEFAULT_CONFIG
        
    visibility_threshold = config["visibility_threshold"]
    required_landmarks = config["required_landmarks"]
    optional_landmarks = config.get("optional_landmarks", set())
    
    # Create pose object specifically for this thread/process
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # Using 1 instead of 2 for speed
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Could not open video: {video_path}")
            return None, None, 0

        # Get video parameters (once)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        all_data = []
        frame_metadata = []
        video_norm_factors = []
        
        skipped_frames_count = 0
        processed_frames_count = 0
        frame_count = 0
        
        # Process frames with fixed skip rate
        pbar = tqdm(total=total_frames//FRAME_SKIP, 
                    desc=f"Processing {os.path.basename(video_path)}", 
                    leave=False)
        
        while cap.isOpened() and frame_count < total_frames:
            # Use grab() for faster frame skipping
            success = cap.grab()
            if not success:
                break
            
            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            frame_count += 1
            
            # Process only every FRAME_SKIP frames
            if frame_id % FRAME_SKIP != 0:
                continue
                
            # Always process first, middle and last frames
            is_key_frame = (frame_id < 5 or 
                          frame_id > total_frames - 5 or 
                          abs(frame_id - total_frames * 0.5) < 3)
                
            if not is_key_frame and frame_id % FRAME_SKIP != 0:
                continue

            # Retrieve the frame
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                skipped_frames_count += 1
                continue

            processed_frames_count += 1
            pbar.update(1)
                
            # Process the frame
            try:
                # Simple preprocessing - just resize
                processed_frame = adaptive_preprocess_frame(frame)
                if processed_frame is None:
                    skipped_frames_count += 1
                    continue

                # Convert to RGB before detection
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                # Try sharpening if detection quality is poor
                if not results.pose_landmarks or (results.pose_landmarks and 
                   any(lm.visibility < visibility_threshold for lm in results.pose_landmarks.landmark)):
                    # Apply sharpening
                    processed_frame = apply_sharpening(processed_frame)
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)

                if not results.pose_landmarks:
                    skipped_frames_count += 1
                    continue

                # Extract 3D landmarks with visibility check
                landmarks = []
                required_landmarks_present = True
                optional_landmarks_present = len(optional_landmarks) == 0  # True if no optional landmarks defined
                
                for idx in relevant_landmarks_indices:
                    lm = results.pose_landmarks.landmark[idx]
                    
                    # Check if this landmark is required or optional
                    is_required = mp_pose.PoseLandmark(idx) in required_landmarks
                    is_optional = mp_pose.PoseLandmark(idx) in optional_landmarks
                    
                    # Apply visibility threshold
                    vis_ok = lm.visibility >= visibility_threshold
                    
                    if is_required and not vis_ok:
                        required_landmarks_present = False
                    
                    if is_optional and vis_ok:
                        optional_landmarks_present = True
                    
                    # Store coordinates or -1 if not visible
                    landmarks.extend([lm.x, lm.y, lm.z] if vis_ok else [-1, -1, -1])
                
                # Skip frame if required landmarks not visible
                # But only if we're not processing a push-up (which needs special handling)
                if not required_landmarks_present and exercise_type != "push_up":
                    skipped_frames_count += 1
                    continue
                
                # For push-ups, be more lenient - only need required AND at least one optional
                if exercise_type == "push_up" and not (required_landmarks_present and optional_landmarks_present):
                    skipped_frames_count += 1
                    continue
                    
                # Reshape landmarks into keypoints array
                keypoints = np.array(landmarks).reshape(-1, 3)
                
                # Detect body orientation for better feature extraction
                orientation = detect_body_orientation(keypoints)
                
                # Get normalization factor for this frame
                norm_factor = get_robust_normalization_factor(keypoints)
                
                # Store valid normalization factor
                if norm_factor > 0:
                    video_norm_factors.append(norm_factor)
                    
                # Calculate features with exercise-specific adaptations
                angles = calculate_angles_3d(keypoints, exercise_type, orientation)
                distances = calculate_distances_3d(keypoints, 1.0, exercise_type)
                y_distances = calculate_y_distances(keypoints, 1.0)
                z_features = calculate_z_features(keypoints, 1.0)

                features = {
                    'frame_id': frame_id,
                    'timestamp': frame_id / fps,
                    'angles': angles,
                    'distances': distances,
                    'y_distances': y_distances,
                    'z_features': z_features,
                    'orientation': orientation
                }

                # NO ROI MASKING - removed per requirement #2
                # All features are kept intact without zeroing out any joint

                # For push-ups, be more lenient with missing features
                if exercise_type == "push_up":
                    # Just ensure we have at least some valid data
                    has_some_valid_angles = any(a != 180.0 for a in angles)
                    has_some_valid_distances = any(d != 0.5 for d in distances)
                    if not (has_some_valid_angles and has_some_valid_distances):
                        skipped_frames_count += 1
                        continue

                # Store the data
                all_data.append(features)
                
                # Store metadata about the frame
                frame_metadata.append({
                    'frame_id': frame_id,
                    'timestamp': frame_id / fps,
                    'orientation': orientation,
                    'norm_factor': norm_factor
                })
                
            except Exception as e:
                logging.warning(f"Error processing frame {frame_count} of {video_path}: {str(e)}")
                skipped_frames_count += 1
                continue
                
        pbar.close()
        cap.release()
        
        logging.info(f"Video {os.path.basename(video_path)}: Processed {processed_frames_count} frames, skipped {skipped_frames_count}")
        
        # We need at least MIN_VALID_FRAMES to consider this a valid sequence
        if len(all_data) < MIN_VALID_FRAMES:
            logging.warning(f"Too few valid frames ({len(all_data)}) in {video_path}")
            return None, None, 0
            
        # Compute final normalization using average of all valid factors
        if video_norm_factors:
            avg_norm_factor = sum(video_norm_factors) / len(video_norm_factors)
        else:
            avg_norm_factor = 1.0  # Default if no valid norm factors

        # Normalize all features with same factor for consistency
        for features in all_data:
            # Renormalize with the average factor
            features['distances'] = [d / avg_norm_factor for d in features['distances']]
            features['y_distances'] = [d / avg_norm_factor for d in features['y_distances']]
            features['z_features'] = [z / avg_norm_factor for z in features['z_features']]
            
        # Interpolate missing features between valid frames
        all_data = interpolate_missing_features(all_data)
        
        # Create sliding windows with 20-frame sequences (requirement #6)
        windows = create_sliding_windows(all_data)
        
        return windows, all_data, processed_frames_count
        
    except Exception as e:
        logging.error(f"Error extracting features from {video_path}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None, 0
    finally:
        # Ensure we release resources
        if 'cap' in locals() and cap is not None:
            cap.release()
        # Clean up the pose object
        pose.close()

def process_video_batch(batch, output_dir):
    """Process a batch of videos and save the extracted features"""
    results = []
    for video_path in batch:
        try:
            # Extract exercise type from the video's parent folder name
            parent_folder = os.path.basename(os.path.dirname(video_path))
            exercise_type = EXERCISE_ALIAS.get(parent_folder.lower(), parent_folder.lower())

            # Skip squats as per requirement #1
            if exercise_type == "squat":
                logging.info(f"Skipping squat video: {video_path}")
                continue

            # Extract features
            _, raw_data, processed_frames = extract_features(video_path, exercise_type)

            if not raw_data:
                logging.warning(f"No valid data extracted from {video_path}")
                continue

            # Get video dimensions for JSON metadata
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Create exercise-specific subfolder
            exercise_output_dir = os.path.join(output_dir, exercise_type)
            os.makedirs(exercise_output_dir, exist_ok=True)

            # Use the original video filename (with .json extension)
            video_filename = os.path.basename(video_path)
            video_name = os.path.splitext(video_filename)[0]
            json_filename = f"{video_name}.json"
            json_path = os.path.join(exercise_output_dir, json_filename)

            # Save JSON file
            output_data = {
                "video": video_filename,
                "label": exercise_type,
                "metadata": {
                    "original_resolution": f"{width}x{height}",
                    "total_frames": total_frames,
                    "processed_frames": processed_frames,
                    "orientations": list(set([f['orientation'] for f in raw_data if 'orientation' in f]))
                },
                "data": raw_data
            }

            with open(json_path, "w") as f:
                json.dump(output_data, f, indent=2)

            results.append({
                'video': video_path,
                'exercise': exercise_type,
                'json_path': json_path,
                'frame_count': processed_frames
            })

            logging.info(f"Processed {video_path}, saved JSON to {json_path}")

        except Exception as e:
            logging.error(f"Error processing {video_path}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())

    return results
def collect_videos(data_dir):
    """Collect all video files from the data directory structure"""
    video_files = []
    
    # Common video extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    # Walk through directory
    for root, dirs, files in os.walk(data_dir):
        # Skip directories that contain "squat" (requirement #1)
        if "squat" in root.lower():
            logging.info(f"Skipping squat directory: {root}")
            continue
            
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                # Check if file name contains "squat" (extra precaution)
                if "squat" in file.lower():
                    logging.info(f"Skipping squat video: {file}")
                    continue
                    
                video_path = os.path.join(root, file)
                video_files.append(video_path)
                
    return video_files

def batch_videos(video_files, batch_size=8):
    """Split videos into batches for parallel processing"""
    for i in range(0, len(video_files), batch_size):
        yield video_files[i:i+batch_size]

def main():
    # Hardcoded paths - modify these as needed
    data_dir = r"D:\final_year_project\Final_Dataset\Train_Data"  # Directory containing exercise videos
    output_dir = r"D:\final_year_project\Final_Dataset\Extracted_Features"  # Output directory for feature files
    workers = 4  # Number of worker processes
    batch_size = 8  # Videos per batch
    
    start_time = time.time()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all videos
    video_files = collect_videos(data_dir)
    logging.info(f"Found {len(video_files)} videos for processing")
    
    # Process in batches with parallelism
    all_results = []
    batches = list(batch_videos(video_files, batch_size))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_video_batch, batch, output_dir) for batch in batches]
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing batches"):
            batch_results = future.result()
            if batch_results:
                all_results.extend(batch_results)
    
    # Generate summary statistics
    exercise_counts = {}
    for result in all_results:
        exercise = result['exercise']
        exercise_counts[exercise] = exercise_counts.get(exercise, 0) + 1
        
    summary = {
        'total_videos': len(all_results),
        'exercise_distribution': exercise_counts,
        'processing_time_minutes': (time.time() - start_time) / 60
    }
    
    # Save summary
    summary_path = os.path.join(output_dir, "processing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    logging.info(f"Processing complete! Extracted features from {len(all_results)} videos")
    logging.info(f"Distribution: {exercise_counts}")
    logging.info(f"Results saved to {output_dir}")
    logging.info(f"Total processing time: {summary['processing_time_minutes']:.2f} minutes")

if __name__ == "__main__":
    main()