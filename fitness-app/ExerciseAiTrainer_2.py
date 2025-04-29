import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import base64
import logging
import os
import joblib
from functools import lru_cache

mp_pose = mp.solutions.pose
logging.basicConfig(level=logging.INFO)

LANDMARK_INDICES = {
    'LEFT_SHOULDER': mp_pose.PoseLandmark.LEFT_SHOULDER,
    'RIGHT_SHOULDER': mp_pose.PoseLandmark.RIGHT_SHOULDER,
    'LEFT_ELBOW': mp_pose.PoseLandmark.LEFT_ELBOW,
    'RIGHT_ELBOW': mp_pose.PoseLandmark.RIGHT_ELBOW,
    'LEFT_WRIST': mp_pose.PoseLandmark.LEFT_WRIST,
    'RIGHT_WRIST': mp_pose.PoseLandmark.RIGHT_WRIST,
    'LEFT_HIP': mp_pose.PoseLandmark.LEFT_HIP,
    'RIGHT_HIP': mp_pose.PoseLandmark.RIGHT_HIP,
    'LEFT_KNEE': mp_pose.PoseLandmark.LEFT_KNEE,
    'RIGHT_KNEE': mp_pose.PoseLandmark.RIGHT_KNEE,
    'LEFT_ANKLE': mp_pose.PoseLandmark.LEFT_ANKLE,
    'RIGHT_ANKLE': mp_pose.PoseLandmark.RIGHT_ANKLE
}

FORM_THRESHOLDS = {
    'push_up': {'elbow_angle_down': (70, 100), 'elbow_angle_up': (160, 190), 'body_alignment': 0.15},
    'squat': {'knee_angle_down': (80, 120), 'knee_angle_up': (160, 190), 'hip_knee_alignment': 0.2},
    'bicep_curl': {'elbow_angle_down': (30, 50), 'elbow_angle_up': (160, 180), 'shoulder_stability': 0.1},
    'shoulder_press': {'elbow_angle_down': (80, 100), 'elbow_angle_up': (170, 190), 'spine_alignment': 0.15}
}

class ExerciseProcessor:
    def __init__(self):
        # Reduced model complexity for better performance
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
        self.feature_window = deque(maxlen=30)
        self.form_score = 100
        self.counters = {'push_up': 0, 'squat': 0, 'bicep_curl': 0, 'shoulder_press': 0}
        self.stages = {'push_up': None, 'squat': None, 'bicep_curl': None, 'shoulder_press': None}
        self.load_models()

    def load_models(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.bi_lstm_model = tf.keras.models.load_model(os.path.join(base_dir, 'models', 'final_exercise_model_3d.keras'))
        self.scaler = joblib.load(os.path.join(base_dir, 'models', 'scaler.pkl'))
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(['push_up', 'squat', 'bicep_curl', 'shoulder_press'])
        logging.info(f"Label encoder classes: {self.label_encoder.classes_}")  # Add this line

    def process_frame(self, frame_data):
        try:
            img = self.decode_base64(frame_data)
            # Add image preprocessing
            img = cv2.resize(img, (640, 480))  # Standardize input size
            img = cv2.GaussianBlur(img, (3, 3), 0)  # Reduce noise

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.pose.process(img_rgb)
            
            if not results.pose_landmarks:
                return {"error": "No human detected"}

            landmarks = self.get_normalized_landmarks(results, img.shape)
            angles = self.calculate_joint_angles(landmarks)
            features = self.extract_features(landmarks)
            self.feature_window.append(features)
            
            if len(self.feature_window) < 30:
                return {"exercise_type": "collecting_data"}

            exercise_type = self.classify_exercise()
            feedback, stage = self.generate_feedback(exercise_type, landmarks)
            self.update_form_score(len(feedback))
            self.update_counters(exercise_type, landmarks)

            return {
                "exercise_type": exercise_type,
                "form_score": self.form_score,
                "feedback": feedback,
                "counters": self.counters,
                "counter": self.counters.get(exercise_type, 0),  # Added for frontend compatibility
                "stage": self.stages.get(exercise_type, ""),     # Added proper stage tracking
                "landmarks": landmarks,
                "angles": angles,
                "angle_thresholds": FORM_THRESHOLDS.get(exercise_type, {})  # Added for ideal form comparison
            }

        except Exception as e:
            logging.error(f"Processing error: {str(e)}")
            return {"error": str(e)}

    def decode_base64(self, frame_data):
        try:
            if 'base64' in frame_data:
                header, data = frame_data.split(',', 1)
            else:
                data = frame_data
            img_bytes = base64.b64decode(data)
            return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            logging.error(f"Decode error: {str(e)}")
            raise

    def get_normalized_landmarks(self, results, img_shape):
        return [[lm.x, lm.y, lm.z, lm.visibility] 
                for lm in results.pose_landmarks.landmark]

    def extract_features(self, landmarks):
        features = []
        features.extend(self.calculate_joint_angles(landmarks))
        features.extend(self.calculate_distances(landmarks))
        features.extend(self.calculate_y_distances(landmarks))
        features.extend(self.calculate_z_features(landmarks))
        assert len(features) == 30, f"Expected 30 features, got {len(features)}"
        return np.array(features)

    def calculate_joint_angles(self, landmarks):
        return [
            self.get_angle(landmarks[LANDMARK_INDICES['LEFT_SHOULDER'].value],
                          landmarks[LANDMARK_INDICES['LEFT_ELBOW'].value],
                          landmarks[LANDMARK_INDICES['LEFT_WRIST'].value]),
            self.get_angle(landmarks[LANDMARK_INDICES['RIGHT_SHOULDER'].value],
                          landmarks[LANDMARK_INDICES['RIGHT_ELBOW'].value],
                          landmarks[LANDMARK_INDICES['RIGHT_WRIST'].value]),
            self.get_angle(landmarks[LANDMARK_INDICES['LEFT_HIP'].value],
                          landmarks[LANDMARK_INDICES['LEFT_KNEE'].value],
                          landmarks[LANDMARK_INDICES['LEFT_ANKLE'].value]),
            self.get_angle(landmarks[LANDMARK_INDICES['RIGHT_HIP'].value],
                          landmarks[LANDMARK_INDICES['RIGHT_KNEE'].value],
                          landmarks[LANDMARK_INDICES['RIGHT_ANKLE'].value]),
            self.get_angle(landmarks[LANDMARK_INDICES['LEFT_SHOULDER'].value],
                          landmarks[LANDMARK_INDICES['LEFT_HIP'].value],
                          landmarks[LANDMARK_INDICES['LEFT_KNEE'].value]),
            self.get_angle(landmarks[LANDMARK_INDICES['RIGHT_SHOULDER'].value],
                          landmarks[LANDMARK_INDICES['RIGHT_HIP'].value],
                          landmarks[LANDMARK_INDICES['RIGHT_KNEE'].value]),
            self.get_angle(landmarks[LANDMARK_INDICES['LEFT_HIP'].value],
                          landmarks[LANDMARK_INDICES['LEFT_SHOULDER'].value],
                          landmarks[LANDMARK_INDICES['LEFT_ELBOW'].value]),
            self.get_angle(landmarks[LANDMARK_INDICES['RIGHT_HIP'].value],
                          landmarks[LANDMARK_INDICES['RIGHT_SHOULDER'].value],
                          landmarks[LANDMARK_INDICES['RIGHT_ELBOW'].value])
        ]

    def get_angle(self, a, b, c):
        ba = np.array([a[0]-b[0], a[1]-b[1]])
        bc = np.array([c[0]-b[0], c[1]-b[1]])
        return np.degrees(np.arccos(np.clip(np.dot(ba, bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)), -1, 1)))

    def calculate_distances(self, landmarks):
        return [np.linalg.norm(np.array(landmarks[j1.value][:2])-np.array(landmarks[j2.value][:2])) 
                for j1,j2 in [(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                              (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
                              (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_HIP),
                              (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_HIP),
                              (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER),
                              (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                              (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_HIP),
                              (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_HIP),
                              (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_KNEE),
                              (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_KNEE),
                              (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
                              (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
                              (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                              (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)]]

    def calculate_y_distances(self, landmarks):
        return [
            abs(landmarks[LANDMARK_INDICES['LEFT_ELBOW'].value][1] - 
            landmarks[LANDMARK_INDICES['LEFT_SHOULDER'].value][1]),
            abs(landmarks[LANDMARK_INDICES['RIGHT_ELBOW'].value][1] - 
            landmarks[LANDMARK_INDICES['RIGHT_SHOULDER'].value][1]),
            abs(landmarks[LANDMARK_INDICES['LEFT_HIP'].value][1] - 
            landmarks[LANDMARK_INDICES['LEFT_ANKLE'].value][1]),
            abs(landmarks[LANDMARK_INDICES['RIGHT_HIP'].value][1] - 
            landmarks[LANDMARK_INDICES['RIGHT_ANKLE'].value][1])
        ]

    def calculate_z_features(self, landmarks):
        return [
            abs(landmarks[LANDMARK_INDICES['LEFT_SHOULDER'].value][2] - 
            landmarks[LANDMARK_INDICES['RIGHT_SHOULDER'].value][2]),
            abs(landmarks[LANDMARK_INDICES['LEFT_HIP'].value][2] - 
            landmarks[LANDMARK_INDICES['LEFT_KNEE'].value][2]),
            abs(landmarks[LANDMARK_INDICES['RIGHT_HIP'].value][2] - 
            landmarks[LANDMARK_INDICES['RIGHT_KNEE'].value][2]),
            abs(landmarks[LANDMARK_INDICES['LEFT_WRIST'].value][2] - 
            landmarks[LANDMARK_INDICES['LEFT_SHOULDER'].value][2])
        ]

    def classify_exercise(self):
        if len(self.feature_window) < 30:
            return "collecting_data"
        
        sequence = np.array(self.feature_window)
        # Ensure the sequence has shape (1, 30, 30) for the model
        if sequence.shape != (30, 30):
            logging.error(f"Invalid sequence shape: {sequence.shape}")
            return "unknown"
        
        scaled = self.scaler.transform(sequence.reshape(-1, 30))
        scaled = scaled.reshape(1, 30, 30)  # Correct shape: (samples, timesteps, features)
        prediction = self.bi_lstm_model.predict(scaled)
        exercise_idx = int(np.argmax(prediction))
        confidence = np.max(prediction)
        logging.info(f"Predicted index: {exercise_idx}, Confidence: {confidence:.2f}")
        return self.label_encoder.inverse_transform([exercise_idx])[0]

    def generate_feedback(self, exercise_type, landmarks):
        feedback = []

        if exercise_type not in FORM_THRESHOLDS:
            return feedback, ""
            
        thresholds = FORM_THRESHOLDS[exercise_type]
        angles = self.calculate_joint_angles(landmarks)
        distances = self.calculate_distances(landmarks)

        # Add debug logging
        logging.info(f"Exercise: {exercise_type} | Angles: {angles} | Distances: {distances}")

        # Enhanced feedback logic
        if exercise_type == 'push_up':
            elbow_angle = angles[0]
            shoulder_alignment = distances[0]  # e.g., distance between shoulders

            if elbow_angle < thresholds['elbow_angle_down'][0]:
                feedback.append("Lower completely (Current: {:.1f}°)".format(elbow_angle))
            elif elbow_angle > thresholds['elbow_angle_down'][1]:
                feedback.append("Don't overextend (Current: {:.1f}°)".format(elbow_angle))

            if shoulder_alignment > thresholds['body_alignment']:
                feedback.append("Keep shoulders aligned")

        elif exercise_type == 'squat':
            knee_angle = angles[2]
            hip_distance = distances[1]  # Example: distance between hips/feet for alignment

            if knee_angle < thresholds['knee_angle_down'][0]:
                feedback.append("Deeper squat (Current: {:.1f}°)".format(knee_angle))
            elif knee_angle > thresholds['knee_angle_down'][1]:
                feedback.append("Reduce knee bend (Current: {:.1f}°)".format(knee_angle))

            if hip_distance < thresholds.get('hip_width_min', 0.1):  # optional
                feedback.append("Widen stance for balance")

        elif exercise_type == 'bicep_curl':
            elbow_angle = angles[0]
            wrist_alignment = distances[2]  # Example: wrist-to-shoulder distance

            if elbow_angle < thresholds['elbow_angle_down'][0]:
                feedback.append("Curl higher (Current: {:.1f}°)".format(elbow_angle))
            elif elbow_angle > thresholds['elbow_angle_down'][1]:
                feedback.append("Control the descent (Current: {:.1f}°)".format(elbow_angle))

            if wrist_alignment > thresholds.get('wrist_alignment_max', 0.15):
                feedback.append("Keep elbows close to body")

        elif exercise_type == 'shoulder_press':
            elbow_angle = angles[0]
            arm_symmetry = distances[3]  # Example: difference in arm height

            if elbow_angle < thresholds['elbow_angle_down'][0]:
                feedback.append("Press higher (Current: {:.1f}°)".format(elbow_angle))
            elif elbow_angle > thresholds['elbow_angle_down'][1]:
                feedback.append("Lower weights fully (Current: {:.1f}°)".format(elbow_angle))

            if arm_symmetry > thresholds.get('arm_symmetry_max', 0.1):
                feedback.append("Keep arms aligned")

        return feedback[:3], self.stages[exercise_type]

    def update_form_score(self, error_count):
        self.form_score = max(0, min(100, self.form_score + (1 if error_count == 0 else -error_count*3)))

    def update_counters(self, exercise_type, landmarks):
        # Enhanced counter logic for all exercise types
        if exercise_type == 'push_up':
            angle = self.get_angle(
                landmarks[LANDMARK_INDICES['LEFT_SHOULDER'].value],
                landmarks[LANDMARK_INDICES['LEFT_ELBOW'].value],
                landmarks[LANDMARK_INDICES['LEFT_WRIST'].value]
            )
            if angle < 90 and self.stages['push_up'] != "down":
                self.stages['push_up'] = "down"
            elif angle > 160 and self.stages['push_up'] == "down":
                self.counters['push_up'] += 1
                self.stages['push_up'] = "up"
        
        elif exercise_type == 'squat':
            angle = self.get_angle(
                landmarks[LANDMARK_INDICES['LEFT_HIP'].value],
                landmarks[LANDMARK_INDICES['LEFT_KNEE'].value],
                landmarks[LANDMARK_INDICES['LEFT_ANKLE'].value]
            )
            if angle < 120 and self.stages['squat'] != "down":
                self.stages['squat'] = "down"
            elif angle > 160 and self.stages['squat'] == "down":
                self.counters['squat'] += 1
                self.stages['squat'] = "up"
        
        elif exercise_type == 'bicep_curl':
            angle = self.get_angle(
                landmarks[LANDMARK_INDICES['LEFT_SHOULDER'].value],
                landmarks[LANDMARK_INDICES['LEFT_ELBOW'].value],
                landmarks[LANDMARK_INDICES['LEFT_WRIST'].value]
            )
            if angle < 50 and self.stages['bicep_curl'] != "up":
                self.stages['bicep_curl'] = "up"
            elif angle > 160 and self.stages['bicep_curl'] == "up":
                self.counters['bicep_curl'] += 1
                self.stages['bicep_curl'] = "down"
        
        elif exercise_type == 'shoulder_press':
            angle = self.get_angle(
                landmarks[LANDMARK_INDICES['LEFT_SHOULDER'].value],
                landmarks[LANDMARK_INDICES['LEFT_ELBOW'].value],
                landmarks[LANDMARK_INDICES['LEFT_WRIST'].value]
            )
            if angle < 100 and self.stages['shoulder_press'] != "down":
                self.stages['shoulder_press'] = "down"
            elif angle > 170 and self.stages['shoulder_press'] == "down":
                self.counters['shoulder_press'] += 1
                self.stages['shoulder_press'] = "up"

@lru_cache(maxsize=1)
def create_exercise_processor():
    return ExerciseProcessor()

def classify_exercise(frame_data):
    processor = create_exercise_processor()
    return processor.process_frame(frame_data)