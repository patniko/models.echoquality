#!/usr/bin/env python
"""
Script for running inference with the trained echo quality model.
This script processes DICOM files and outputs quality predictions.
"""

import os
import torch
import numpy as np
import glob
import pydicom
import cv2
import json
import argparse
from tqdm import tqdm
from torchvision.models.video import r2plus1d_18
import matplotlib.pyplot as plt

# Import from our modules
from EchoPrime_qc import mask_outside_ultrasound, crop_and_scale, get_quality_issues
from echo_model_evaluation import visualize_gradcam

# Constants for video processing (same as in EchoPrime_qc.py)
frames_to_take = 32
frame_stride = 2
video_size = 112
mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)

def process_dicom(dicom_path, save_mask_images=False):
    """
    Process a single DICOM file and prepare it for the model.
    
    Args:
        dicom_path (str): Path to the DICOM file
        save_mask_images (bool): Whether to save mask images
        
    Returns:
        torch.Tensor: Processed video tensor
    """
    try:
        # Read DICOM file
        dcm = pydicom.dcmread(dicom_path)
        pixels = dcm.pixel_array
        
        # Handle different dimensions
        if pixels.ndim < 3 or pixels.shape[2] == 3:
            print(f"Skipping {dicom_path}: Invalid dimensions {pixels.shape}")
            return None
        
        # If single channel, repeat to 3 channels
        if pixels.ndim == 3:
            pixels = np.repeat(pixels[..., None], 3, axis=3)
        
        # Mask everything outside ultrasound region
        filename = os.path.basename(dicom_path)
        pixels = mask_outside_ultrasound(pixels, filename if save_mask_images else None)
        
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
        
        return x
    
    except Exception as e:
        print(f"Error processing {dicom_path}: {str(e)}")
        return None

def run_inference(model, dicom_paths, device, threshold=0.3, save_dir=None, generate_gradcam=False):
    """
    Run inference on a list of DICOM files.
    
    Args:
        model (nn.Module): The model to use for inference
        dicom_paths (list): List of paths to DICOM files
        device (torch.device): Device to run inference on
        threshold (float): Threshold for binary classification
        save_dir (str, optional): Directory to save results and visualizations
        generate_gradcam (bool): Whether to generate GradCAM visualizations
        
    Returns:
        dict: Dictionary of results
    """
    model.eval()
    results = {}
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if generate_gradcam:
            os.makedirs(os.path.join(save_dir, "gradcam"), exist_ok=True)
    
    for dicom_path in tqdm(dicom_paths, desc="Processing"):
        filename = os.path.basename(dicom_path)
        
        # Process DICOM file
        video = process_dicom(dicom_path, save_mask_images=(save_dir is not None))
        
        if video is None:
            print(f"Skipping {filename}: Processing failed")
            continue
        
        # Run inference
        with torch.no_grad():
            video = video.unsqueeze(0).to(device)  # Add batch dimension
            output = model(video)
            probability = torch.sigmoid(output).item()
            prediction = 1 if probability >= threshold else 0
            status = "PASS" if prediction > 0 else "FAIL"
            assessment = get_quality_issues(probability)
        
        # Store results
        results[filename] = {
            "score": probability,
            "status": status,
            "assessment": assessment
        }
        
        # Generate GradCAM visualization if requested
        if generate_gradcam and save_dir:
            try:
                visualize_gradcam(
                    model, 
                    video.squeeze(0).cpu(), 
                    target_layer_name="layer4", 
                    save_path=os.path.join(save_dir, "gradcam", f"{filename.replace('.dcm', '')}_gradcam.png")
                )
            except Exception as e:
                print(f"Error generating GradCAM for {filename}: {str(e)}")
    
    # Save results to JSON if save_dir is provided
    if save_dir:
        with open(os.path.join(save_dir, "inference_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Create a summary plot
        create_summary_plot(results, save_dir)
    
    return results

def create_summary_plot(results, save_dir):
    """
    Create a summary plot of the inference results.
    
    Args:
        results (dict): Dictionary of inference results
        save_dir (str): Directory to save the plot
    """
    # Extract scores
    scores = [result["score"] for result in results.values()]
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, alpha=0.7, color='blue')
    plt.axvline(x=0.3, color='red', linestyle='--', label='Threshold (0.3)')
    plt.xlabel('Quality Score')
    plt.ylabel('Count')
    plt.title('Distribution of Echo Quality Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "score_distribution.png"))
    plt.close()
    
    # Create pie chart of pass/fail
    pass_count = sum(1 for result in results.values() if result["status"] == "PASS")
    fail_count = len(results) - pass_count
    
    plt.figure(figsize=(8, 8))
    plt.pie(
        [pass_count, fail_count], 
        labels=["PASS", "FAIL"], 
        autopct='%1.1f%%',
        colors=['#4CAF50', '#F44336'],
        explode=(0.1, 0)
    )
    plt.title('Pass/Fail Distribution')
    plt.savefig(os.path.join(save_dir, "pass_fail_distribution.png"))
    plt.close()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run inference with the echo quality model.")
    parser.add_argument("--input", type=str, required=True, help="Path to directory containing DICOM files")
    parser.add_argument("--model", type=str, default="video_quality_model.pt", help="Path to model weights")
    parser.add_argument("--output", type=str, default="inference_output", help="Directory to save results")
    parser.add_argument("--threshold", type=float, default=0.3, help="Threshold for binary classification")
    parser.add_argument("--gradcam", action="store_true", help="Generate GradCAM visualizations")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run inference on")
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = r2plus1d_18(num_classes=1)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    
    # Find DICOM files
    print(f"Finding DICOM files in {args.input}...")
    dicom_paths = glob.glob(f"{args.input}/**/*.dcm", recursive=True)
    print(f"Found {len(dicom_paths)} DICOM files")
    
    if len(dicom_paths) == 0:
        print("No DICOM files found. Exiting.")
        return
    
    # Run inference
    print(f"Running inference with threshold {args.threshold}...")
    results = run_inference(
        model, 
        dicom_paths, 
        device, 
        threshold=args.threshold, 
        save_dir=args.output,
        generate_gradcam=args.gradcam
    )
    
    # Print summary
    pass_count = sum(1 for result in results.values() if result["status"] == "PASS")
    total_count = len(results)
    pass_rate = pass_count/total_count*100 if total_count > 0 else 0
    
    print("\nQuality Assessment Results:")
    print("=" * 80)
    print(f"{'Filename':<60} {'Score':<10} {'Pass/Fail':<10} {'Assessment'}")
    print("-" * 80)
    
    for filename, result in results.items():
        # Truncate filename if too long
        short_filename = filename[:57] + "..." if len(filename) > 60 else filename.ljust(60)
        print(f"{short_filename} {result['score']:.4f}    {result['status']:<10} {result['assessment']}")
    
    print(f"\nSummary: {pass_count}/{total_count} videos passed quality check ({pass_rate:.1f}%)")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
