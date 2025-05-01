#!/usr/bin/env python
"""
Example script demonstrating how to use the echo quality model training pipeline.
This script creates a small synthetic dataset and trains the model on it.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import from our training modules
from train_quality_model import EchoDataset, freeze_model_except_final_layers
from echo_data_augmentation import EchoVideoAugmentation, create_synthetic_low_quality
from echo_model_evaluation import evaluate_model, visualize_gradcam, plot_confusion_matrix

# Configuration
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_WEIGHTS = "video_quality_model.pt"
SAVE_DIR = "example_output"
NUM_SYNTHETIC_VIDEOS = 10  # Number of synthetic videos to create

# Create output directory
os.makedirs(SAVE_DIR, exist_ok=True)

def create_synthetic_dataset():
    """
    Create a synthetic dataset of echo videos for demonstration purposes.
    
    Returns:
        tuple: (video_paths, labels, synthetic_videos)
    """
    print("Creating synthetic dataset...")
    
    # Create synthetic videos
    synthetic_videos = []
    labels = []
    
    for i in range(NUM_SYNTHETIC_VIDEOS):
        # Create a random video tensor (3 channels, 16 frames, 112x112 resolution)
        # In a real scenario, these would be loaded from DICOM files
        video = torch.rand(3, 16, 112, 112)
        
        # Assign a random label (0 for low quality, 1 for high quality)
        label = 1 if i % 2 == 0 else 0
        
        # If it's a high-quality video, make it more structured (less random)
        if label == 1:
            # Create a more structured pattern for high-quality videos
            # Add a circular pattern that moves across frames
            for t in range(16):
                center_x = 56 + 20 * np.sin(t * np.pi / 8)
                center_y = 56 + 20 * np.cos(t * np.pi / 8)
                for h in range(112):
                    for w in range(112):
                        dist = np.sqrt((h - center_y)**2 + (w - center_x)**2)
                        if dist < 30:
                            # Create a bright circle
                            video[:, t, h, w] = 0.8 + 0.2 * torch.rand(3)
        else:
            # For low-quality videos, add noise and reduce contrast
            video = create_synthetic_low_quality(
                video, 
                noise_level=0.3, 
                blur_kernel=7, 
                contrast_reduction=0.7
            )
        
        synthetic_videos.append(video)
        labels.append(label)
    
    # Create a CSV file with paths and labels
    video_paths = [f"{SAVE_DIR}/synthetic_video_{i}.pt" for i in range(NUM_SYNTHETIC_VIDEOS)]
    
    # Save the synthetic videos
    for i, video in enumerate(synthetic_videos):
        torch.save(video, video_paths[i])
    
    # Create a CSV file
    df = pd.DataFrame({
        'video_path': video_paths,
        'quality_score': labels
    })
    
    csv_path = f"{SAVE_DIR}/synthetic_annotations.csv"
    df.to_csv(csv_path, index=False)
    print(f"Created {NUM_SYNTHETIC_VIDEOS} synthetic videos and saved annotations to {csv_path}")
    
    return video_paths, labels, synthetic_videos

class SyntheticEchoDataset(torch.utils.data.Dataset):
    """
    Dataset for synthetic echocardiogram videos.
    This is a simplified version of the EchoDataset class in train_quality_model.py.
    """
    def __init__(self, videos, labels, transform=None):
        """
        Args:
            videos (list): List of video tensors
            labels (list): List of quality labels (1 for good quality, 0 for poor quality)
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.videos = videos
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]
        label = self.labels[idx]
        
        # Apply transform if provided
        if self.transform:
            video = self.transform(video)
        
        return video, torch.tensor([label], dtype=torch.float)

def train_model(synthetic_videos, labels):
    """
    Train the model on the synthetic dataset.
    
    Args:
        synthetic_videos (list): List of synthetic video tensors
        labels (list): List of quality labels
    """
    print("Setting up training...")
    
    # Create data augmentation
    augmentation = EchoVideoAugmentation(
        brightness_range=(0.9, 1.1),
        contrast_range=(0.9, 1.1),
        rotation_range=(-5, 5),
        translation_range=(-0.05, 0.05),
        zoom_range=(0.95, 1.05),
        noise_level=(0.0, 0.02),
        temporal_crop_range=(0.9, 1.0),
        temporal_mask_prob=0.05,
        transform_prob=0.5
    )
    
    # Split data into train and validation sets
    train_size = int(0.8 * len(synthetic_videos))
    train_videos = synthetic_videos[:train_size]
    train_labels = labels[:train_size]
    val_videos = synthetic_videos[train_size:]
    val_labels = labels[train_size:]
    
    # Create datasets
    train_dataset = SyntheticEchoDataset(train_videos, train_labels, transform=augmentation)
    val_dataset = SyntheticEchoDataset(val_videos, val_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load model
    print("Loading model...")
    from torchvision.models.video import r2plus1d_18
    model = r2plus1d_18(num_classes=1)
    
    try:
        weights = torch.load(MODEL_WEIGHTS, map_location=torch.device('cpu'))
        model.load_state_dict(weights)
        print(f"Loaded weights from {MODEL_WEIGHTS}")
    except Exception as e:
        print(f"Could not load weights: {str(e)}")
        print("Initializing with random weights")
    
    # Freeze early layers
    print("Freezing early layers...")
    freeze_model_except_final_layers(model, num_layers_to_train=2)
    
    # Move model to device
    model = model.to(DEVICE)
    
    # Define loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        
        # Train
        model.train()
        train_loss = 0.0
        
        for videos, labels in tqdm(train_loader, desc="Training"):
            videos = videos.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * videos.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for videos, labels in tqdm(val_loader, desc="Validation"):
                videos = videos.to(DEVICE)
                labels = labels.to(DEVICE)
                
                # Forward pass
                outputs = model(videos)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * videos.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        
        print("-" * 50)
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
    plt.plot(range(1, EPOCHS+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'loss_plot.png'))
    
    # Evaluate the model
    print("Evaluating model...")
    metrics = evaluate_model(model, val_loader, DEVICE)
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Plot confusion matrix
    cm_fig = plot_confusion_matrix(metrics['confusion_matrix'])
    cm_fig.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'))
    
    # Visualize GradCAM for a sample video
    if len(val_videos) > 0:
        print("Generating GradCAM visualization...")
        sample_video = val_videos[0].to(DEVICE)
        visualize_gradcam(
            model, 
            sample_video, 
            target_layer_name="layer4", 
            save_path=os.path.join(SAVE_DIR, "gradcam_visualization.png")
        )
    
    print(f"Training complete! Results saved to {SAVE_DIR}")
    return model

def main():
    """Main function."""
    print("Echo Quality Model Training Example")
    print("=" * 50)
    
    # Create synthetic dataset
    video_paths, labels, synthetic_videos = create_synthetic_dataset()
    
    # Train model
    model = train_model(synthetic_videos, labels)
    
    print("Example completed successfully!")

if __name__ == "__main__":
    main()
