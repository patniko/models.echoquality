import torch
import numpy as np
import cv2
import random
from torchvision import transforms
from torch.nn.functional import interpolate

class EchoVideoTransform:
    """
    Class for applying transformations to echocardiogram videos.
    These transformations are designed to be medically relevant and preserve
    the diagnostic content of the videos while introducing variations.
    """
    def __init__(self, 
                 brightness_range=(0.8, 1.2),
                 contrast_range=(0.8, 1.2),
                 rotation_range=(-10, 10),
                 translation_range=(-0.1, 0.1),
                 zoom_range=(0.9, 1.1),
                 noise_level=(0.0, 0.05),
                 temporal_crop_range=(0.8, 1.0),
                 temporal_mask_prob=0.1,
                 prob=0.5):
        """
        Initialize the transform with ranges for various augmentations.
        
        Args:
            brightness_range (tuple): Range for brightness adjustment
            contrast_range (tuple): Range for contrast adjustment
            rotation_range (tuple): Range for rotation in degrees
            translation_range (tuple): Range for translation as fraction of image size
            zoom_range (tuple): Range for zoom factor
            noise_level (tuple): Range for Gaussian noise standard deviation
            temporal_crop_range (tuple): Range for temporal cropping factor
            temporal_mask_prob (float): Probability of masking a frame in the sequence
            prob (float): Probability of applying each transform
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.zoom_range = zoom_range
        self.noise_level = noise_level
        self.temporal_crop_range = temporal_crop_range
        self.temporal_mask_prob = temporal_mask_prob
        self.prob = prob
    
    def __call__(self, video):
        """
        Apply transformations to a video tensor.
        
        Args:
            video (torch.Tensor): Video tensor of shape [C, T, H, W]
                where C is channels, T is time/frames, H is height, W is width
                
        Returns:
            torch.Tensor: Transformed video tensor of the same shape
        """
        # Make a copy to avoid modifying the original
        video = video.clone()
        
        # Apply spatial transformations (same for all frames)
        if random.random() < self.prob:
            video = self._adjust_brightness_contrast(video)
        
        if random.random() < self.prob:
            video = self._rotate(video)
        
        if random.random() < self.prob:
            video = self._translate(video)
        
        if random.random() < self.prob:
            video = self._zoom(video)
        
        if random.random() < self.prob:
            video = self._add_noise(video)
        
        # Apply temporal transformations
        if random.random() < self.prob:
            video = self._temporal_crop(video)
        
        if random.random() < self.prob:
            video = self._temporal_mask(video)
        
        return video
    
    def _adjust_brightness_contrast(self, video):
        """Adjust brightness and contrast of the video."""
        brightness = random.uniform(*self.brightness_range)
        contrast = random.uniform(*self.contrast_range)
        
        # Apply to all frames
        # Formula: new_pixel = contrast * (pixel - mean) + mean + brightness - 1
        mean = video.mean()
        video = contrast * (video - mean) + mean + brightness - 1
        
        # Clip values to valid range [0, 1]
        video = torch.clamp(video, 0, 1)
        
        return video
    
    def _rotate(self, video):
        """Rotate the video frames."""
        angle = random.uniform(*self.rotation_range)
        
        # Get dimensions
        c, t, h, w = video.shape
        
        # Create rotation matrix
        center = (w / 2, h / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Convert to numpy for OpenCV operations
        video_np = video.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]
        
        # Apply rotation to each frame
        for i in range(t):
            frame = video_np[i]  # [H, W, C]
            video_np[i] = cv2.warpAffine(frame, rotation_matrix, (w, h), 
                                         borderMode=cv2.BORDER_REPLICATE)
        
        # Convert back to tensor
        video = torch.from_numpy(video_np).permute(3, 0, 1, 2)  # [C, T, H, W]
        
        return video
    
    def _translate(self, video):
        """Translate the video frames."""
        tx = random.uniform(*self.translation_range) * video.shape[3]  # width
        ty = random.uniform(*self.translation_range) * video.shape[2]  # height
        
        # Get dimensions
        c, t, h, w = video.shape
        
        # Create translation matrix
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        
        # Convert to numpy for OpenCV operations
        video_np = video.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]
        
        # Apply translation to each frame
        for i in range(t):
            frame = video_np[i]  # [H, W, C]
            video_np[i] = cv2.warpAffine(frame, translation_matrix, (w, h), 
                                         borderMode=cv2.BORDER_REPLICATE)
        
        # Convert back to tensor
        video = torch.from_numpy(video_np).permute(3, 0, 1, 2)  # [C, T, H, W]
        
        return video
    
    def _zoom(self, video):
        """Zoom in/out of the video frames."""
        zoom_factor = random.uniform(*self.zoom_range)
        
        # Get dimensions
        c, t, h, w = video.shape
        
        # Calculate new dimensions
        new_h = int(h * zoom_factor)
        new_w = int(w * zoom_factor)
        
        # Calculate crop/pad dimensions
        start_h = (new_h - h) // 2 if new_h > h else 0
        start_w = (new_w - w) // 2 if new_w > w else 0
        end_h = start_h + h if new_h > h else new_h
        end_w = start_w + w if new_w > w else new_w
        
        # Convert to numpy for OpenCV operations
        video_np = video.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]
        
        # Apply zoom to each frame
        zoomed_video_np = np.zeros_like(video_np)
        for i in range(t):
            frame = video_np[i]  # [H, W, C]
            
            if zoom_factor > 1:  # Zoom in
                # Resize to larger dimensions
                zoomed_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                # Crop center
                zoomed_video_np[i] = zoomed_frame[start_h:end_h, start_w:end_w]
            else:  # Zoom out
                # Create a blank frame
                zoomed_frame = np.zeros((h, w, c), dtype=frame.dtype)
                # Resize to smaller dimensions
                resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                # Calculate padding
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                # Place the resized frame in the center
                zoomed_frame[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized_frame
                zoomed_video_np[i] = zoomed_frame
        
        # Convert back to tensor
        video = torch.from_numpy(zoomed_video_np).permute(3, 0, 1, 2)  # [C, T, H, W]
        
        return video
    
    def _add_noise(self, video):
        """Add Gaussian noise to the video."""
        noise_std = random.uniform(*self.noise_level)
        
        # Generate noise
        noise = torch.randn_like(video) * noise_std
        
        # Add noise to video
        video = video + noise
        
        # Clip values to valid range [0, 1]
        video = torch.clamp(video, 0, 1)
        
        return video
    
    def _temporal_crop(self, video):
        """Randomly crop a segment of the video in the temporal dimension."""
        c, t, h, w = video.shape
        
        # Determine crop length
        crop_factor = random.uniform(*self.temporal_crop_range)
        crop_length = max(1, int(t * crop_factor))
        
        # Randomly select start frame
        if crop_length < t:
            start_frame = random.randint(0, t - crop_length)
            
            # Crop the video
            cropped_video = video[:, start_frame:start_frame+crop_length, :, :]
            
            # Resize back to original temporal length
            cropped_video = interpolate(
                cropped_video.unsqueeze(0), 
                size=(t, h, w), 
                mode='trilinear', 
                align_corners=False
            ).squeeze(0)
            
            return cropped_video
        
        return video
    
    def _temporal_mask(self, video):
        """Randomly mask frames in the video."""
        c, t, h, w = video.shape
        
        # Create a mask for frames
        mask = torch.rand(t) >= self.temporal_mask_prob
        
        # Apply mask
        for i in range(t):
            if not mask[i]:
                # If frame is masked, replace with interpolation of adjacent frames
                if i > 0 and i < t - 1:
                    video[:, i, :, :] = (video[:, i-1, :, :] + video[:, i+1, :, :]) / 2
                elif i > 0:
                    video[:, i, :, :] = video[:, i-1, :, :]
                else:
                    video[:, i, :, :] = video[:, i+1, :, :]
        
        return video


class EchoVideoAugmentation:
    """
    Class for applying augmentations to echocardiogram videos.
    This combines multiple transformations and provides a simple interface.
    """
    def __init__(self, 
                 brightness_range=(0.8, 1.2),
                 contrast_range=(0.8, 1.2),
                 rotation_range=(-10, 10),
                 translation_range=(-0.1, 0.1),
                 zoom_range=(0.9, 1.1),
                 noise_level=(0.0, 0.05),
                 temporal_crop_range=(0.8, 1.0),
                 temporal_mask_prob=0.1,
                 transform_prob=0.5):
        """
        Initialize the augmentation with ranges for various transformations.
        
        Args:
            brightness_range (tuple): Range for brightness adjustment
            contrast_range (tuple): Range for contrast adjustment
            rotation_range (tuple): Range for rotation in degrees
            translation_range (tuple): Range for translation as fraction of image size
            zoom_range (tuple): Range for zoom factor
            noise_level (tuple): Range for Gaussian noise standard deviation
            temporal_crop_range (tuple): Range for temporal cropping factor
            temporal_mask_prob (float): Probability of masking a frame in the sequence
            transform_prob (float): Probability of applying each transform
        """
        self.transform = EchoVideoTransform(
            brightness_range=brightness_range,
            contrast_range=contrast_range,
            rotation_range=rotation_range,
            translation_range=translation_range,
            zoom_range=zoom_range,
            noise_level=noise_level,
            temporal_crop_range=temporal_crop_range,
            temporal_mask_prob=temporal_mask_prob,
            prob=transform_prob
        )
    
    def __call__(self, video):
        """
        Apply augmentations to a video tensor.
        
        Args:
            video (torch.Tensor): Video tensor of shape [C, T, H, W]
                
        Returns:
            torch.Tensor: Augmented video tensor of the same shape
        """
        return self.transform(video)


def create_synthetic_low_quality(video, noise_level=0.2, blur_kernel=5, contrast_reduction=0.5):
    """
    Create a synthetic low-quality version of an echo video.
    This is useful for creating paired data for training enhancement models.
    
    Args:
        video (torch.Tensor): Video tensor of shape [C, T, H, W]
        noise_level (float): Standard deviation of Gaussian noise
        blur_kernel (int): Size of Gaussian blur kernel
        contrast_reduction (float): Factor to reduce contrast by
        
    Returns:
        torch.Tensor: Low-quality version of the input video
    """
    # Make a copy to avoid modifying the original
    lq_video = video.clone()
    
    # Convert to numpy for OpenCV operations
    c, t, h, w = lq_video.shape
    lq_video_np = lq_video.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]
    
    # Apply transformations to each frame
    for i in range(t):
        frame = lq_video_np[i]  # [H, W, C]
        
        # Reduce contrast
        mean = frame.mean()
        frame = (frame - mean) * contrast_reduction + mean
        
        # Apply Gaussian blur
        frame = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, frame.shape)
        frame = frame + noise
        
        # Clip values to valid range [0, 1]
        frame = np.clip(frame, 0, 1)
        
        lq_video_np[i] = frame
    
    # Convert back to tensor
    lq_video = torch.from_numpy(lq_video_np).permute(3, 0, 1, 2)  # [C, T, H, W]
    
    return lq_video


# Example usage
if __name__ == "__main__":
    # Create a dummy video tensor
    dummy_video = torch.rand(3, 16, 112, 112)  # [C, T, H, W]
    
    # Create augmentation
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
    
    # Apply augmentation
    augmented_video = augmentation(dummy_video)
    
    # Create synthetic low-quality version
    lq_video = create_synthetic_low_quality(dummy_video)
    
    print(f"Original video shape: {dummy_video.shape}")
    print(f"Augmented video shape: {augmented_video.shape}")
    print(f"Low-quality video shape: {lq_video.shape}")
