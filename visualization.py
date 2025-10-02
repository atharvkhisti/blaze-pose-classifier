import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from blazepose_classifier import ExerciseType

def plot_confusion_matrix(y_true, y_pred, model_name="model", save_dir="figures"):
    """Plot and save confusion matrix"""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create class labels
    class_labels = [ex_type.name for ex_type in ExerciseType]
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(importances, feature_names, model_name="model", top_n=10, save_dir="figures"):
    """Plot and save feature importance"""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Create DataFrame for feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Get top N features
    top_features = importance_df.head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title(f'Top {top_n} Feature Importance - {model_name}')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'{save_dir}/feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return importance_df

def plot_rep_counting_accuracy(results, save_dir="figures"):
    """Plot repetition counting accuracy"""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Create DataFrame for results
    df = pd.DataFrame(results)
    
    # Plot MAE
    plt.figure(figsize=(10, 6))
    sns.barplot(x='exercise', y='mae', data=df)
    plt.title('Mean Absolute Error in Repetition Counting')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Exercise Type')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/rep_counting_mae.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    sns.barplot(x='exercise', y='accuracy', data=df)
    plt.title('Repetition Counting Accuracy (±1 rep)')
    plt.ylabel('Accuracy')
    plt.xlabel('Exercise Type')
    plt.ylim(0.8, 1.0)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/rep_counting_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_history(history, save_dir="figures"):
    """Plot model training history"""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy during Training')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss during Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_sample_features(features_df, exercise_type, save_dir="figures"):
    """Plot sample features for an exercise"""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Filter data for the given exercise type
    exercise_df = features_df[features_df['exercise_type'] == exercise_type.value].reset_index(drop=True)
    
    # Select key features
    key_features = [
        'right_leg_angle',
        'left_leg_angle',
        'right_arm_angle',
        'left_arm_angle',
        'torso_angle_right',
        'torso_angle_left'
    ]
    
    # Plot
    plt.figure(figsize=(12, 8))
    for feature in key_features:
        if feature in exercise_df.columns:
            plt.plot(exercise_df[feature], label=feature)
    
    plt.title(f'Sample Features for {exercise_type.name}')
    plt.ylabel('Angle (degrees)')
    plt.xlabel('Frame')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/sample_features_{exercise_type.name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_fsm_diagram(save_dir="figures"):
    """Create a diagram of the Finite State Machine for repetition counting"""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Create the figure
    plt.figure(figsize=(10, 8))
    
    # Create a graph using NetworkX
    import networkx as nx
    G = nx.DiGraph()
    
    # Add nodes
    G.add_node("STARTING", pos=(0, 0))
    G.add_node("TOP", pos=(2, 0))
    G.add_node("BOTTOM", pos=(2, -2))
    
    # Add edges
    G.add_edge("STARTING", "TOP", label="angle > high_threshold")
    G.add_edge("TOP", "BOTTOM", label="angle < low_threshold")
    G.add_edge("BOTTOM", "TOP", label="angle > high_threshold\nincrement counter")
    G.add_edge("STARTING", "STARTING", label="angle <= high_threshold")
    G.add_edge("TOP", "TOP", label="low_threshold <= angle <= high_threshold")
    G.add_edge("BOTTOM", "BOTTOM", label="angle <= high_threshold")
    
    # Get positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=["yellow", "green", "red"], 
                         node_size=3000, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
    
    # Draw edges with labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    # Set the title
    plt.title("Finite State Machine for Repetition Counting")
    
    # Remove the axis
    plt.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f'{save_dir}/fsm_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Create FSM diagram
    create_fsm_diagram()
    
    # Sample repetition counting results for demonstration
    rep_counting_results = [
        {'exercise': 'Squats', 'mae': 0.13, 'accuracy': 0.98},
        {'exercise': 'Pushups', 'mae': 0.21, 'accuracy': 0.95},
        {'exercise': 'Situps', 'mae': 0.18, 'accuracy': 0.96},
        {'exercise': 'Jumping Jacks', 'mae': 0.09, 'accuracy': 0.99},
        {'exercise': 'Lunges', 'mae': 0.27, 'accuracy': 0.93}
    ]
    
    plot_rep_counting_accuracy(rep_counting_results)
    
    print("Visualizations created successfully in the 'figures' directory.")