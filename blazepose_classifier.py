import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import pickle
import math
from typing import List, Dict, Tuple, Optional
import time
from enum import Enum

class ExerciseType(Enum):
    SQUAT = 0
    PUSHUP = 1
    SITUP = 2
    JUMPING_JACK = 3
    LUNGE = 4
    NONE = 5

class RepetitionState(Enum):
    STARTING = 0
    DESCENDING = 1
    BOTTOM = 2
    ASCENDING = 3
    TOP = 4

class FeatureExtractor:
    """Extracts features from BlazePose landmarks"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def extract_landmarks(self, image):
        """Extract landmarks from a single image"""
        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5) as pose:
            
            # Convert image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            return results.pose_landmarks
            
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def extract_static_features(self, landmarks):
        """Extract static features (angles, distances) from landmarks"""
        features = {}
        
        if landmarks is None:
            return None
        
        # Key joint angles
        # Right arm angle
        r_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        r_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        r_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        features['right_arm_angle'] = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
        
        # Left arm angle
        l_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        l_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        l_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        features['left_arm_angle'] = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
        
        # Right leg angle
        r_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        r_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        r_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        features['right_leg_angle'] = self.calculate_angle(r_hip, r_knee, r_ankle)
        
        # Left leg angle
        l_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        l_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
        l_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        features['left_leg_angle'] = self.calculate_angle(l_hip, l_knee, l_ankle)
        
        # Torso angle (right side)
        r_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        r_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        r_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        features['torso_angle_right'] = self.calculate_angle(r_shoulder, r_hip, r_knee)
        
        # Torso angle (left side)
        l_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        l_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        l_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
        features['torso_angle_left'] = self.calculate_angle(l_shoulder, l_hip, l_knee)
        
        # Add more advanced features here...
        
        return features
    
    def extract_dynamic_features(self, feature_window: List[Dict]):
        """Extract dynamic features from a window of static features"""
        if not feature_window or len(feature_window) < 2:
            return {}
            
        dynamic_features = {}
        
        # Calculate velocity and acceleration for key angles
        for key in ['right_arm_angle', 'left_arm_angle', 'right_leg_angle', 'left_leg_angle']:
            if key not in feature_window[0]:
                continue
                
            values = [frame[key] for frame in feature_window if key in frame]
            
            # Velocity (average change per frame)
            if len(values) >= 2:
                deltas = [values[i+1] - values[i] for i in range(len(values)-1)]
                dynamic_features[f'{key}_velocity_mean'] = np.mean(deltas)
                dynamic_features[f'{key}_velocity_max'] = np.max(deltas) if deltas else 0
                
                # Acceleration (change in velocity)
                if len(deltas) >= 2:
                    acc = [deltas[i+1] - deltas[i] for i in range(len(deltas)-1)]
                    dynamic_features[f'{key}_acceleration_mean'] = np.mean(acc)
        
        return dynamic_features
    
    def extract_all_features(self, landmarks, feature_window=None):
        """Extract all features, both static and dynamic if window provided"""
        static_features = self.extract_static_features(landmarks)
        
        if static_features is None:
            return None
            
        features = static_features
        
        if feature_window is not None:
            # Add current frame to window and maintain window size
            feature_window.append(static_features)
            if len(feature_window) > 15:  # Keep last 15 frames
                feature_window.pop(0)
                
            dynamic_features = self.extract_dynamic_features(feature_window)
            features.update(dynamic_features)
        
        return features

class RepCounter:
    """Counts repetitions for exercises"""
    
    def __init__(self, exercise_type: ExerciseType):
        self.exercise_type = exercise_type
        self.rep_count = 0
        self.state = RepetitionState.STARTING
        self.key_metric = self._get_key_metric_for_exercise()
        self.threshold_high = self._get_high_threshold()
        self.threshold_low = self._get_low_threshold()
        self.last_peak_time = 0
        self.min_rep_time = 0.5  # Minimum time between reps (seconds)
    
    def _get_key_metric_for_exercise(self):
        """Get the key metric to track for the given exercise type"""
        metrics = {
            ExerciseType.SQUAT: "right_leg_angle",
            ExerciseType.PUSHUP: "right_arm_angle",
            ExerciseType.SITUP: "torso_angle_right",
            ExerciseType.JUMPING_JACK: "right_arm_angle",
            ExerciseType.LUNGE: "right_leg_angle"
        }
        return metrics.get(self.exercise_type, "right_leg_angle")
    
    def _get_high_threshold(self):
        """Get the high threshold for the given exercise type"""
        thresholds = {
            ExerciseType.SQUAT: 160,
            ExerciseType.PUSHUP: 160,
            ExerciseType.SITUP: 150,
            ExerciseType.JUMPING_JACK: 160,
            ExerciseType.LUNGE: 160
        }
        return thresholds.get(self.exercise_type, 160)
    
    def _get_low_threshold(self):
        """Get the low threshold for the given exercise type"""
        thresholds = {
            ExerciseType.SQUAT: 80,
            ExerciseType.PUSHUP: 80,
            ExerciseType.SITUP: 60,
            ExerciseType.JUMPING_JACK: 30,
            ExerciseType.LUNGE: 90
        }
        return thresholds.get(self.exercise_type, 90)
    
    def update(self, features):
        """Update rep counter with new features"""
        if not features or self.key_metric not in features:
            return self.rep_count
            
        current_value = features[self.key_metric]
        current_time = time.time()
        
        # State machine for rep counting
        if self.state == RepetitionState.STARTING:
            if current_value > self.threshold_high:
                self.state = RepetitionState.TOP
        
        elif self.state == RepetitionState.TOP:
            if current_value < self.threshold_low:
                self.state = RepetitionState.BOTTOM
        
        elif self.state == RepetitionState.BOTTOM:
            if current_value > self.threshold_high:
                time_diff = current_time - self.last_peak_time
                if time_diff >= self.min_rep_time:
                    self.rep_count += 1
                    self.last_peak_time = current_time
                self.state = RepetitionState.TOP
        
        return self.rep_count

class ExerciseClassifier:
    """Classifies exercises based on pose landmarks"""
    
    def __init__(self, model_path=None):
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.feature_window = []
        self.rep_counter = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load a trained classifier model"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def predict(self, image):
        """Predict exercise type from a single image"""
        if self.model is None:
            print("Model not loaded. Use load_model() first.")
            return None, None
            
        landmarks = self.feature_extractor.extract_landmarks(image)
        
        if landmarks is None:
            return None, None
            
        features = self.feature_extractor.extract_all_features(landmarks, self.feature_window)
        
        if features is None:
            return None, None
            
        # Convert dict to DataFrame for prediction
        feature_df = pd.DataFrame([features])
        
        # Fill missing values that might be in the model but not in current features
        for col in self.model.feature_names_:
            if col not in feature_df.columns:
                feature_df[col] = 0
        
        # Select only columns that the model knows about
        feature_df = feature_df[self.model.feature_names_]
        
        # Predict exercise type
        prediction = self.model.predict(feature_df)[0]
        exercise_type = ExerciseType(prediction)
        
        # Initialize or update rep counter
        if self.rep_counter is None or self.rep_counter.exercise_type != exercise_type:
            self.rep_counter = RepCounter(exercise_type)
        
        rep_count = self.rep_counter.update(features)
        
        return exercise_type, rep_count
    
    def visualize_prediction(self, image, exercise_type, rep_count):
        """Visualize prediction on image"""
        if exercise_type is None:
            return image
            
        h, w, c = image.shape
        
        # Draw prediction text
        cv2.putText(
            image, f"Exercise: {exercise_type.name}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        
        if rep_count is not None:
            cv2.putText(
                image, f"Reps: {rep_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
        
        return image

def train_model(data_path, save_path=None):
    """Train a random forest classifier on the dataset"""
    # Load dataset
    try:
        df = pd.read_csv(data_path)
        print(f"Dataset loaded: {df.shape[0]} samples with {df.shape[1]} features")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Prepare features and target
    X = df.drop('exercise_type', axis=1)
    y = df['exercise_type']
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    
    model.fit(X, y)
    print(f"Model trained with accuracy: {model.score(X, y):.4f}")
    
    # Save model if path provided
    if save_path:
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model saved to {save_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    return model

def demo_live_classification(model_path):
    """Demo for live classification using webcam"""
    classifier = ExerciseClassifier(model_path)
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)
        
        # Classify the exercise
        exercise_type, rep_count = classifier.predict(image)
        
        # Visualize the result
        image = classifier.visualize_prediction(image, exercise_type, rep_count)
        
        # Show the image
        cv2.imshow('BlazePose Exercise Classifier', image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC key
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Set paths
    DATASET_PATH = "exercise_features_dataset.csv"
    MODEL_PATH = "exercise_classifier_model.pkl"
    
    # Train and save model if needed
    # model = train_model(DATASET_PATH, MODEL_PATH)
    
    # Demo live classification
    demo_live_classification(MODEL_PATH)