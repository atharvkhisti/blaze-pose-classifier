import argparse
import cv2
import time
import pickle
import os
import sys
from blazepose_classifier import ExerciseClassifier, ExerciseType

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="BlazePose Exercise Classifier")
    
    parser.add_argument("--model", type=str, default="exercise_classifier_model.pkl",
                        help="Path to trained model file")
    
    parser.add_argument("--input", type=str, default="0",
                        help="Path to video file or camera index (default: 0)")
    
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output video file (default: None)")
    
    parser.add_argument("--display", action="store_true",
                        help="Display video output")
    
    parser.add_argument("--record-stats", action="store_true",
                        help="Record exercise statistics to CSV file")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Initialize classifier
    classifier = ExerciseClassifier(args.model)
    
    # Initialize video capture
    try:
        if args.input.isdigit():
            cap = cv2.VideoCapture(int(args.input))
        else:
            cap = cv2.VideoCapture(args.input)
    except Exception as e:
        print(f"Error opening video source: {e}")
        return
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output path is provided
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Exercise statistics
    stats = {
        "start_time": time.time(),
        "frames_processed": 0,
        "exercise_counts": {ex_type: 0 for ex_type in ExerciseType},
        "rep_counts": {ex_type: 0 for ex_type in ExerciseType},
        "processing_times": []
    }
    
    print("Starting exercise classification...")
    print("Press 'q' to quit.")
    
    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        
        # Classify exercise
        exercise_type, rep_count = classifier.predict(frame)
        
        # Update statistics
        stats["frames_processed"] += 1
        if exercise_type is not None:
            stats["exercise_counts"][exercise_type] += 1
            if rep_count is not None:
                stats["rep_counts"][exercise_type] = rep_count
        
        # Visualize prediction
        result_frame = classifier.visualize_prediction(frame, exercise_type, rep_count)
        
        # Add processing speed indicator
        process_time = time.time() - start_time
        stats["processing_times"].append(process_time)
        fps_text = f"FPS: {1/process_time:.1f}" if process_time > 0 else "FPS: N/A"
        cv2.putText(result_frame, fps_text, (width - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write frame to output video
        if writer is not None:
            writer.write(result_frame)
        
        # Display frame
        if args.display:
            cv2.imshow("BlazePose Exercise Classifier", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Clean up
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    
    # Print summary statistics
    print("\n=== Exercise Classification Summary ===")
    total_time = time.time() - stats["start_time"]
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Frames processed: {stats['frames_processed']}")
    print(f"Average processing time: {sum(stats['processing_times'])/len(stats['processing_times']):.4f} seconds per frame")
    print(f"Average FPS: {1/(sum(stats['processing_times'])/len(stats['processing_times'])):.1f}")
    
    print("\nExercise Distribution:")
    for ex_type in ExerciseType:
        if ex_type != ExerciseType.NONE:
            count = stats["exercise_counts"][ex_type]
            percentage = (count / stats["frames_processed"]) * 100 if stats["frames_processed"] > 0 else 0
            print(f"  {ex_type.name}: {count} frames ({percentage:.1f}%)")
    
    print("\nRepetition Counts:")
    for ex_type in ExerciseType:
        if ex_type != ExerciseType.NONE and stats["rep_counts"][ex_type] > 0:
            print(f"  {ex_type.name}: {stats['rep_counts'][ex_type]} reps")
    
    # Save statistics to CSV if requested
    if args.record_stats:
        import pandas as pd
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exercise_stats_{timestamp}.csv"
        
        # Create DataFrame
        stats_df = pd.DataFrame({
            "timestamp": [timestamp],
            "total_time_seconds": [total_time],
            "frames_processed": [stats["frames_processed"]],
            "average_fps": [1/(sum(stats["processing_times"])/len(stats["processing_times"]))]
        })
        
        # Add exercise counts and reps
        for ex_type in ExerciseType:
            if ex_type != ExerciseType.NONE:
                stats_df[f"{ex_type.name}_frames"] = stats["exercise_counts"][ex_type]
                stats_df[f"{ex_type.name}_reps"] = stats["rep_counts"][ex_type]
        
        # Save to CSV
        stats_df.to_csv(filename, index=False)
        print(f"\nStatistics saved to {filename}")

if __name__ == "__main__":
    main()