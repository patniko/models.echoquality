import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import mlflow
import os
from tqdm import tqdm
from torchvision.models.video import r2plus1d_18

class GradCAM:
    """
    Class for generating Grad-CAM visualizations for video models.
    Adapted for 3D convolutions used in video classification models.
    """
    def __init__(self, model, target_layer):
        """
        Initialize GradCAM with a model and target layer.
        
        Args:
            model (nn.Module): The model to visualize
            target_layer (nn.Module): The target layer to compute gradients for
        """
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
        # Set model to evaluation mode
        self.model.eval()
    
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        # Forward hook
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        # Backward hook
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Register hooks
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
        
        # Store handles for removal
        self.hooks = [forward_handle, backward_handle]
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
    
    def __call__(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM for the input tensor.
        
        Args:
            input_tensor (torch.Tensor): Input tensor of shape [1, C, T, H, W]
            class_idx (int, optional): Index of the class to generate Grad-CAM for.
                If None, uses the class with the highest score.
                
        Returns:
            np.ndarray: Grad-CAM heatmap of shape [T, H, W]
        """
        # Ensure input has batch dimension
        if len(input_tensor.shape) == 4:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # If class_idx is None, use the class with the highest score
        if class_idx is None:
            if output.shape[1] > 1:
                class_idx = torch.argmax(output, dim=1).item()
            else:
                # For binary classification, use the positive class
                class_idx = 0
        
        # Clear gradients
        self.model.zero_grad()
        
        # Target for backprop
        if output.shape[1] > 1:
            # Multi-class case
            target = torch.zeros_like(output)
            target[0, class_idx] = 1
        else:
            # Binary classification case
            target = torch.ones_like(output)
        
        # Backward pass
        output.backward(gradient=target, retain_graph=True)
        
        # Global average pooling of gradients
        weights = F.adaptive_avg_pool3d(self.gradients, (1, 1, 1))[0]
        
        # Weight the activations by the gradients
        cam = torch.zeros_like(self.activations[0, 0])
        for i, w in enumerate(weights[0]):
            cam += w * self.activations[0, i]
        
        # Apply ReLU to focus on features that have a positive influence
        cam = F.relu(cam)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Convert to numpy
        cam = cam.cpu().numpy()
        
        return cam

def apply_colormap(cam, frame):
    """
    Apply a colormap to the CAM and overlay it on the frame.
    
    Args:
        cam (np.ndarray): Grad-CAM heatmap of shape [H, W]
        frame (np.ndarray): Original frame of shape [H, W, C]
        
    Returns:
        np.ndarray: Colorized CAM overlaid on the frame
    """
    # Resize CAM to frame size
    cam = cv2.resize(cam, (frame.shape[1], frame.shape[0]))
    
    # Apply jet colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    
    # Convert frame to BGR if it's not already
    if frame.shape[2] == 3:
        frame_bgr = frame
    else:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Overlay heatmap on frame
    cam_bgr = heatmap * 0.4 + frame_bgr
    
    # Normalize
    cam_bgr = cam_bgr / cam_bgr.max() * 255
    
    return np.uint8(cam_bgr)

def visualize_gradcam(model, video, target_layer_name="layer4", class_idx=None, save_path=None):
    """
    Visualize Grad-CAM for a video.
    
    Args:
        model (nn.Module): The model to visualize
        video (torch.Tensor): Video tensor of shape [C, T, H, W]
        target_layer_name (str): Name of the target layer
        class_idx (int, optional): Index of the class to generate Grad-CAM for
        save_path (str, optional): Path to save the visualization
        
    Returns:
        list: List of colorized CAM frames
    """
    # Get the target layer
    target_layer = None
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_layer = module
            break
    
    if target_layer is None:
        raise ValueError(f"Layer {target_layer_name} not found in model")
    
    # Initialize GradCAM
    grad_cam = GradCAM(model, target_layer)
    
    # Generate CAM
    cam = grad_cam(video.unsqueeze(0), class_idx)
    
    # Remove hooks
    grad_cam.remove_hooks()
    
    # Convert video to numpy for visualization
    video_np = video.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]
    
    # Apply colormap to each frame
    colored_frames = []
    for i in range(cam.shape[0]):
        frame = video_np[i]
        cam_frame = cam[i]
        colored_frame = apply_colormap(cam_frame, frame)
        colored_frames.append(colored_frame)
    
    # Save visualization if path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create a grid of frames
        n_frames = len(colored_frames)
        n_cols = min(8, n_frames)
        n_rows = (n_frames + n_cols - 1) // n_cols
        
        plt.figure(figsize=(n_cols * 3, n_rows * 3))
        for i, frame in enumerate(colored_frames):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title(f"Frame {i}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    return colored_frames

def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Evaluate a model on a dataset.
    
    Args:
        model (nn.Module): The model to evaluate
        dataloader (DataLoader): Data loader for the dataset
        device (torch.device): Device to evaluate on
        threshold (float): Threshold for binary classification
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for videos, labels in tqdm(dataloader, desc="Evaluating"):
            videos = videos.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(videos)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= threshold).astype(float)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_probs = np.array(all_probs).flatten()
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Calculate Precision-Recall curve and AUC
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve
    }
    
    return metrics

def plot_confusion_matrix(cm, classes=['Low Quality', 'High Quality'], save_path=None):
    """
    Plot a confusion matrix.
    
    Args:
        cm (np.ndarray): Confusion matrix
        classes (list): List of class names
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return plt.gcf()

def plot_roc_curve(fpr, tpr, roc_auc, save_path=None):
    """
    Plot a ROC curve.
    
    Args:
        fpr (np.ndarray): False positive rates
        tpr (np.ndarray): True positive rates
        roc_auc (float): Area under the ROC curve
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return plt.gcf()

def plot_precision_recall_curve(recall, precision, pr_auc, save_path=None):
    """
    Plot a precision-recall curve.
    
    Args:
        recall (np.ndarray): Recall values
        precision (np.ndarray): Precision values
        pr_auc (float): Area under the precision-recall curve
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return plt.gcf()

def log_evaluation_to_mlflow(metrics, prefix=""):
    """
    Log evaluation metrics to MLflow.
    
    Args:
        metrics (dict): Dictionary of evaluation metrics
        prefix (str, optional): Prefix for metric names
    """
    # Log scalar metrics
    scalar_metrics = {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'specificity': metrics['specificity'],
        'roc_auc': metrics['roc_auc'],
        'pr_auc': metrics['pr_auc']
    }
    
    for name, value in scalar_metrics.items():
        mlflow.log_metric(f"{prefix}{name}", value)
    
    # Create and log confusion matrix plot
    cm_fig = plot_confusion_matrix(metrics['confusion_matrix'])
    cm_path = f"confusion_matrix_{prefix.strip('_')}.png"
    cm_fig.savefig(cm_path, bbox_inches='tight')
    mlflow.log_artifact(cm_path)
    plt.close(cm_fig)
    
    # Create and log ROC curve plot
    roc_fig = plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'])
    roc_path = f"roc_curve_{prefix.strip('_')}.png"
    roc_fig.savefig(roc_path, bbox_inches='tight')
    mlflow.log_artifact(roc_path)
    plt.close(roc_fig)
    
    # Create and log precision-recall curve plot
    pr_fig = plot_precision_recall_curve(metrics['recall_curve'], metrics['precision_curve'], metrics['pr_auc'])
    pr_path = f"pr_curve_{prefix.strip('_')}.png"
    pr_fig.savefig(pr_path, bbox_inches='tight')
    mlflow.log_artifact(pr_path)
    plt.close(pr_fig)

def analyze_misclassifications(model, dataloader, device, threshold=0.5, save_dir="misclassifications"):
    """
    Analyze misclassified examples.
    
    Args:
        model (nn.Module): The model to evaluate
        dataloader (DataLoader): Data loader for the dataset
        device (torch.device): Device to evaluate on
        threshold (float): Threshold for binary classification
        save_dir (str): Directory to save visualizations
        
    Returns:
        tuple: (false_positives, false_negatives) lists of indices
    """
    model.eval()
    false_positives = []
    false_negatives = []
    
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (videos, labels) in enumerate(tqdm(dataloader, desc="Analyzing")):
            videos = videos.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(videos)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= threshold).astype(float)
            
            # Find misclassifications
            for j, (pred, label) in enumerate(zip(preds, labels.cpu().numpy())):
                idx = i * dataloader.batch_size + j
                
                if pred > 0 and label == 0:  # False positive
                    false_positives.append(idx)
                    
                    # Visualize with GradCAM
                    video = videos[j].cpu()
                    visualize_gradcam(
                        model, 
                        video, 
                        save_path=os.path.join(save_dir, f"fp_{idx}_gradcam.png")
                    )
                    
                elif pred == 0 and label > 0:  # False negative
                    false_negatives.append(idx)
                    
                    # Visualize with GradCAM
                    video = videos[j].cpu()
                    visualize_gradcam(
                        model, 
                        video, 
                        save_path=os.path.join(save_dir, f"fn_{idx}_gradcam.png")
                    )
    
    # Print summary
    print(f"Found {len(false_positives)} false positives and {len(false_negatives)} false negatives")
    
    return false_positives, false_negatives

def find_optimal_threshold(metrics):
    """
    Find the optimal threshold based on ROC curve.
    
    Args:
        metrics (dict): Dictionary of evaluation metrics
        
    Returns:
        float: Optimal threshold
    """
    # Find the point on the ROC curve closest to (0, 1)
    fpr, tpr, thresholds = metrics['fpr'], metrics['tpr'], metrics['thresholds']
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold

# Example usage
if __name__ == "__main__":
    # Load a model
    model = r2plus1d_18(num_classes=1)
    model.load_state_dict(torch.load("video_quality_model.pt", map_location=torch.device('cpu')))
    
    # Create a dummy video tensor
    dummy_video = torch.rand(3, 16, 112, 112)  # [C, T, H, W]
    
    # Visualize GradCAM
    gradcam_frames = visualize_gradcam(
        model, 
        dummy_video, 
        target_layer_name="layer4", 
        save_path="gradcam_visualization.png"
    )
    
    print(f"Generated {len(gradcam_frames)} GradCAM frames")
