import subprocess
import os
import sys
import platform
import argparse

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    required_version = (3, 7)
    current_version = sys.version_info
    
    if current_version < required_version:
        print(f"Error: Python {required_version[0]}.{required_version[1]} or higher is required.")
        print(f"Current version: Python {current_version[0]}.{current_version[1]}")
        return False
    
    print(f"Python version check passed: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    print("\nInstalling required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All required packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return False

def check_camera():
    """Check if camera is accessible"""
    print("\nChecking camera availability...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Warning: Could not access camera. Video capture may not work.")
            return False
        
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Warning: Could not read frame from camera.")
            cap.release()
            return False
        
        h, w = frame.shape[:2]
        print(f"Camera check passed: Accessed camera with resolution {w}x{h}")
        cap.release()
        return True
    except Exception as e:
        print(f"Warning: Error checking camera: {e}")
        return False

def check_mediapipe():
    """Check if MediaPipe pose works correctly"""
    print("\nChecking MediaPipe pose functionality...")
    try:
        import mediapipe as mp
        import numpy as np
        import cv2
        
        # Create a blank image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Initialize pose detector
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            min_detection_confidence=0.5) as pose:
            
            # Process image
            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        print("MediaPipe pose check passed: Successfully initialized pose detector")
        return True
    except Exception as e:
        print(f"Warning: Error checking MediaPipe pose functionality: {e}")
        return False

def create_directories():
    """Create required directories"""
    print("\nCreating required directories...")
    directories = ["figures", "results", "models", "data"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")
    
    return True

def create_sample_data():
    """Create a sample dataset file if none exists"""
    print("\nChecking for sample data...")
    data_file = "exercise_features_dataset.csv"
    
    if not os.path.exists(data_file):
        try:
            # Create a minimal sample dataset for demonstration
            import pandas as pd
            import numpy as np
            
            # Generate random data for demonstration
            np.random.seed(42)
            n_samples = 100
            
            # Create a DataFrame with features
            data = {
                'right_arm_angle': np.random.uniform(0, 180, n_samples),
                'left_arm_angle': np.random.uniform(0, 180, n_samples),
                'right_leg_angle': np.random.uniform(0, 180, n_samples),
                'left_leg_angle': np.random.uniform(0, 180, n_samples),
                'torso_angle_right': np.random.uniform(0, 180, n_samples),
                'torso_angle_left': np.random.uniform(0, 180, n_samples),
                'exercise_type': np.random.randint(0, 5, n_samples)  # Random exercise types
            }
            
            df = pd.DataFrame(data)
            df.to_csv(data_file, index=False)
            
            print(f"Created sample dataset: {data_file}")
            return True
        except Exception as e:
            print(f"Warning: Error creating sample data: {e}")
            return False
    else:
        print(f"Dataset file already exists: {data_file}")
        return True

def create_sample_model():
    """Create a sample model if none exists"""
    print("\nChecking for model file...")
    model_file = "exercise_classifier_model.pkl"
    
    if not os.path.exists(model_file):
        try:
            # Create a minimal sample model for demonstration
            from sklearn.ensemble import RandomForestClassifier
            import pickle
            import numpy as np
            
            # Generate a simple model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            X = np.random.rand(100, 6)  # 6 features
            y = np.random.randint(0, 5, 100)  # 5 classes
            
            # Add feature names
            feature_names = ['right_arm_angle', 'left_arm_angle', 'right_leg_angle', 
                             'left_leg_angle', 'torso_angle_right', 'torso_angle_left']
            
            # Fit model
            model.fit(X, y)
            
            # Add feature names to model
            model.feature_names_ = feature_names
            
            # Save model
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"Created sample model: {model_file}")
            return True
        except Exception as e:
            print(f"Warning: Error creating sample model: {e}")
            return False
    else:
        print(f"Model file already exists: {model_file}")
        return True

def verify_installation():
    """Run a simple test to verify the installation"""
    print("\nVerifying installation by importing main modules...")
    modules = [
        "cv2", "mediapipe", "numpy", "pandas", 
        "sklearn.ensemble", "matplotlib.pyplot", "seaborn"
    ]
    
    all_passed = True
    for module in modules:
        try:
            __import__(module)
            print(f"✓ Successfully imported {module}")
        except ImportError as e:
            print(f"✗ Failed to import {module}: {e}")
            all_passed = False
    
    return all_passed

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="BlazePose Classifier Setup")
    
    parser.add_argument("--skip-checks", action="store_true",
                        help="Skip environment checks")
    
    parser.add_argument("--skip-camera", action="store_true",
                        help="Skip camera check")
    
    parser.add_argument("--create-sample-data", action="store_true",
                        help="Force creation of sample data")
    
    return parser.parse_args()

def main():
    """Main setup function"""
    print("=" * 60)
    print("BlazePose Exercise Classifier - Setup Utility")
    print("=" * 60)
    
    args = parse_arguments()
    
    # Check environment
    if not args.skip_checks:
        if not check_python_version():
            return
        
        if not install_requirements():
            return
        
        if not args.skip_camera:
            check_camera()
        
        check_mediapipe()
    
    # Create directories and sample files
    create_directories()
    
    if args.create_sample_data:
        create_sample_data()
        create_sample_model()
    
    # Verify installation
    if verify_installation():
        print("\n✓ All checks passed! The BlazePose Classifier is ready to use.")
    else:
        print("\n⚠ Some checks failed. The system may not work correctly.")
    
    print("\nYou can now run the following commands:")
    print("  - Generate dataset:  python dataset_generator.py")
    print("  - Train models:      python model_evaluator.py")
    print("  - Run classifier:    python deploy.py")
    print("\nRefer to README.md for more information.")

if __name__ == "__main__":
    main()