import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r2plus1d_18
import mlflow
import os
import glob
import numpy as np
import pandas as pd
import cv2
import pydicom
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Import functions from EchoPrime_qc.py
from EchoPrime_qc import mask_outside_ultrasound, crop_and_scale

# Configuration
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANNOTATIONS_CSV = "echo_annotations.csv"  # Path to your CSV annotations
VIDEOS_DIR = "echo_videos"  # Directory containing your 100 echo videos
MODEL_WEIGHTS = "video_quality_model.pt"  # Path to your current model weights
SAVE_DIR = "trained_models"  # Directory to save trained models
USE_MLFLOW = True  # Set to False to disable MLflow tracking
NUM_UNFROZEN_LAYERS = 2  # Number of layers to unfreeze for fine-tuning

# Set up MLflow tracking if enabled
if USE_MLFLOW:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Constants for video processing (same as in EchoPrime_qc.py)
frames_to_take = 32
frame_stride = 2
video_size = 112
mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)

class EchoDataset(Dataset):
    """
    Dataset for echocardiogram videos with quality annotations.
    """
    def __init__(self, video_paths, labels, transform=None):
        """
        Args:
            video_paths (list): List of paths to DICOM files
            labels (list): List of quality labels (1 for good quality, 0 for poor quality)
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Process DICOM file similar to process_dicoms function in EchoPrime_qc.py
        try:
            # Read DICOM file
            dcm = pydicom.dcmread(video_path)
            pixels = dcm.pixel_array
            
            # Handle different dimensions
            if pixels.ndim < 3 or pixels.shape[2] == 3:
                # Skip this file or handle differently
                # For now, we'll create a dummy tensor
                dummy_tensor = torch.zeros((3, frames_to_take, video_size, video_size))
                return dummy_tensor, label
            
            # If single channel, repeat to 3 channels
            if pixels.ndim == 3:
                pixels = np.repeat(pixels[..., None], 3, axis=3)
            
            # Mask everything outside ultrasound region
            filename = os.path.basename(video_path)
            pixels = mask_outside_ultrasound(pixels, filename)
            
            # Model specific preprocessing
            x = np.zeros((len(pixels), video_size, video_size, 3))
            for i in range(len(x)):
                x[i] = crop_and_scale(pixels[i])
            
            # Convert to tensor and permute dimensions
            x = torch.as_tensor(x, dtype=torch.float).permute([3, 0, 1, 2])
            
            # Normalize
            x.sub_(mean).div_(std)
            
            # If not enough frames, add padding
            if x.shape[1] < frames_to_take:
                padding = torch.zeros(
                    (
                        3,
                        frames_to_take - x.shape[1],
                        video_size,
                        video_size,
                    ),
                    dtype=torch.float,
                )
                x = torch.cat((x, padding), dim=1)
            
            # Apply stride and take required frames
            start = 0
            x = x[:, start: (start + frames_to_take): frame_stride, :, :]
            
            # Apply any additional transforms
            if self.transform:
                x = self.transform(x)
            
            return x, torch.tensor([label], dtype=torch.float)
        
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            # Return a dummy tensor in case of error
            dummy_tensor = torch.zeros((3, frames_to_take, video_size, video_size))
            return dummy_tensor, torch.tensor([label], dtype=torch.float)

def load_annotations(csv_path):
    """
    Load annotations from CSV file.
    
    Args:
        csv_path (str): Path to CSV file with annotations
        
    Returns:
        tuple: (video_paths, labels)
    """
    # This is a template function. Adjust according to your CSV format.
    df = pd.read_csv(csv_path)
    
    # Assuming CSV has columns 'video_path' and 'quality_score'
    # Adjust column names and processing based on your actual CSV format
    video_paths = df['video_path'].tolist()
    
    # Convert quality scores to binary labels (1 for good quality, 0 for poor quality)
    # Adjust threshold based on your annotation scheme
    labels = (df['quality_score'] >= 0.5).astype(int).tolist()
    
    return video_paths, labels

def freeze_model_except_final_layers(model, num_layers_to_train=2):
    """
    Freeze all layers except the final few layers.
    
    Args:
        model (nn.Module): The model to freeze
        num_layers_to_train (int): Number of final layers to keep trainable
    """
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Get list of all modules
    all_modules = list(model.children())
    
    # Unfreeze the final layers
    trainable_layers = all_modules[-num_layers_to_train:]
    for layer in trainable_layers:
        for param in layer.parameters():
            param.requires_grad = True
    
    # Print summary of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params} ({trainable_params/total_params:.2%} of total)")

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): The model to train
        dataloader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for videos, labels in tqdm(dataloader, desc="Training"):
        videos = videos.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(videos)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * videos.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_accuracy

def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    
    Args:
        model (nn.Module): The model to validate
        dataloader (DataLoader): Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        tuple: (val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for videos, labels in tqdm(dataloader, desc="Validation"):
            videos = videos.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * videos.size(0)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= 0.5).astype(float)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(dataloader.dataset)
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    val_recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    
    # Calculate AUC if possible (requires probabilities, not just binary predictions)
    try:
        val_auc = roc_auc_score(all_labels, all_probs)
    except:
        val_auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Confusion Matrix:\n{cm}")
    
    return val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc

def main():
    """
    Main training function.
    """
    # Load annotations
    print("Loading annotations...")
    video_paths, labels = load_annotations(ANNOTATIONS_CSV)
    
    # Split data into train, validation, and test sets
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        video_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"Training set: {len(train_paths)} videos")
    print(f"Validation set: {len(val_paths)} videos")
    print(f"Test set: {len(test_paths)} videos")
    
    # Create datasets
    train_dataset = EchoDataset(train_paths, train_labels)
    val_dataset = EchoDataset(val_paths, val_labels)
    test_dataset = EchoDataset(test_paths, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Load model
    print("Loading model...")
    model = r2plus1d_18(num_classes=1)
    weights = torch.load(MODEL_WEIGHTS, map_location=torch.device('cpu'))
    model.load_state_dict(weights)
    
    # Freeze early layers
    print("Freezing early layers...")
    freeze_model_except_final_layers(model, num_layers_to_train=NUM_UNFROZEN_LAYERS)
    
    # Move model to device
    model = model.to(DEVICE)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    
    # MLflow tracking
    if USE_MLFLOW:
        mlflow.start_run(run_name="EchoQuality_Training")
        mlflow.log_params({
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "unfrozen_layers": NUM_UNFROZEN_LAYERS,
            "train_size": len(train_paths),
            "val_size": len(val_paths),
            "test_size": len(test_paths)
        })
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = os.path.join(SAVE_DIR, "best_model.pt")
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc = validate(
            model, val_loader, criterion, DEVICE
        )
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        print(f"Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics with MLflow
        if USE_MLFLOW:
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "val_auc": val_auc,
                "learning_rate": optimizer.param_groups[0]['lr']
            }, step=epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")
            
            if USE_MLFLOW:
                mlflow.log_artifact(best_model_path)
        
        # Save checkpoint
        checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        
        print("-" * 50)
    
    # Test best model
    print("Testing best model...")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_precision, test_recall, test_f1, test_auc = validate(
        model, test_loader, criterion, DEVICE
    )
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")
    print(f"Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}")
    
    # Log test metrics with MLflow
    if USE_MLFLOW:
        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "test_auc": test_auc
        })
        
        # End MLflow run
        mlflow.end_run()
    
    print("Training complete!")

if __name__ == "__main__":
    main()
