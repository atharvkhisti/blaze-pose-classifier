import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os
import time
from blazepose_classifier import FeatureExtractor, ExerciseType

class DatasetGenerator:
    """Generates dataset from videos or webcam for BlazePose exercise classifier"""
    
    def __init__(self, output_path="exercise_features_dataset.csv"):
        self.feature_extractor = FeatureExtractor()
        self.output_path = output_path
        self.collected_data = []
        self.exercise_counts = {ex_type: 0 for ex_type in ExerciseType}
    
    def process_video(self, video_path, exercise_type: ExerciseType):
        """Process a video and extract features"""
        cap = cv2.VideoCapture(video_path)
        feature_window = []
        frame_count = 0
        
        print(f"Processing video: {video_path} for exercise: {exercise_type.name}")
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
                
            # Process every 3rd frame to reduce redundancy
            if frame_count % 3 == 0:
                landmarks = self.feature_extractor.extract_landmarks(image)
                
                if landmarks is not None:
                    features = self.feature_extractor.extract_all_features(landmarks, feature_window)
                    
                    if features is not None:
                        features['exercise_type'] = exercise_type.value
                        self.collected_data.append(features)
                        
                        if len(self.collected_data) % 50 == 0:
                            print(f"Collected {len(self.collected_data)} samples...")
            
            frame_count += 1
            
            # Display progress
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
        
        cap.release()
        self.exercise_counts[exercise_type] += 1
        print(f"Completed processing video. Added {frame_count//3} samples for {exercise_type.name}")
    
    def collect_from_webcam(self, exercise_type: ExerciseType, duration_seconds=10):
        """Collect data from webcam for a specified duration"""
        cap = cv2.VideoCapture(0)
        feature_window = []
        start_time = time.time()
        samples_collected = 0
        
        print(f"Collecting {exercise_type.name} data from webcam for {duration_seconds} seconds...")
        print("Press 'ESC' to stop early.")
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"Starting in {i}...")
            time.sleep(1)
        
        print("GO! Perform the exercise now.")
        
        while cap.isOpened() and (time.time() - start_time) < duration_seconds:
            success, image = cap.read()
            if not success:
                print("Failed to capture image from webcam.")
                break
            
            # Flip the image horizontally for a selfie-view display
            image = cv2.flip(image, 1)
            
            landmarks = self.feature_extractor.extract_landmarks(image)
            
            if landmarks is not None:
                features = self.feature_extractor.extract_all_features(landmarks, feature_window)
                
                if features is not None:
                    features['exercise_type'] = exercise_type.value
                    self.collected_data.append(features)
                    samples_collected += 1
                    
                    # Draw landmarks on image
                    mp_drawing = mp.solutions.drawing_utils
                    mp_pose = mp.solutions.pose
                    mp_drawing.draw_landmarks(
                        image,
                        landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )
                    
                    # Add text with time remaining and samples collected
                    time_left = int(duration_seconds - (time.time() - start_time))
                    cv2.putText(
                        image, f"Exercise: {exercise_type.name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                    cv2.putText(
                        image, f"Time left: {time_left}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                    cv2.putText(
                        image, f"Samples: {samples_collected}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
            
            # Show the image
            cv2.imshow('Data Collection', image)
            if cv2.waitKey(5) & 0xFF == 27:  # ESC key
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        self.exercise_counts[exercise_type] += 1
        print(f"Collected {samples_collected} samples for {exercise_type.name}")
    
    def save_dataset(self):
        """Save collected data to CSV file"""
        if not self.collected_data:
            print("No data to save.")
            return
            
        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(self.collected_data)
        
        # Save to CSV
        df.to_csv(self.output_path, index=False)
        print(f"Dataset saved to {self.output_path} with {len(df)} samples.")
        
        # Print class distribution
        class_counts = df['exercise_type'].value_counts()
        print("Class distribution:")
        for ex_type in ExerciseType:
            count = class_counts.get(ex_type.value, 0)
            print(f"  {ex_type.name}: {count} samples")

def main():
    generator = DatasetGenerator()
    
    # Example usage: Collect data for each exercise type
    for exercise_type in [ExerciseType.SQUAT, ExerciseType.PUSHUP, 
                          ExerciseType.SITUP, ExerciseType.JUMPING_JACK,
                          ExerciseType.LUNGE]:
        
        print(f"\n=== Collecting data for {exercise_type.name} ===")
        input("Press Enter when you're ready to start...")
        
        # Collect from webcam for 20 seconds
        generator.collect_from_webcam(exercise_type, duration_seconds=20)
        
        # Option to collect more data for this exercise
        more_data = input("Do you want to collect more data for this exercise? (y/n): ")
        if more_data.lower() == 'y':
            generator.collect_from_webcam(exercise_type, duration_seconds=20)
    
    # Save the collected dataset
    generator.save_dataset()

if __name__ == "__main__":
    main()