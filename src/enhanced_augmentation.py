"""Enhanced augmentation pipeline for PixelProof.

This module provides robust data augmentation that simulates real-world image
degradation (JPEG compression, screenshotting, resizing, noise, etc.).

WHY THIS MATTERS:
- Real AI-detector deployments see compressed/resized/noisy images
- Standard augmentation doesn't cover these well
- Enhanced augmentation → Better generalization to unseen generators
- Expected improvement: +0.01 AUC on unseen generators

USAGE:
    from enhanced_augmentation import get_augmentation_pipeline
    
    train_aug = get_augmentation_pipeline(mode='train')
    val_aug = get_augmentation_pipeline(mode='val')
    
    # Apply to images
    augmented = train_aug(image=image_array)
    image_tensor = augmented['image']
"""

from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


def get_augmentation_pipeline(mode: str = 'train', img_size: int = 224) -> A.Compose:
    """Get augmentation pipeline for training or validation.
    
    Args:
        mode: 'train' (aggressive), 'val' (minimal), or 'test' (none)
        img_size: Target image size (default 224 for ResNet/ViT)
    
    Returns:
        albumentations.Compose pipeline
    
    Modes:
        'train': Heavy augmentation with degradation simulation
        'val': Light augmentation for validation
        'test': Minimal transformation (just normalization)
    """
    
    if mode == 'train':
        return A.Compose([
            # ==================== COMPRESSION ====================
            # JPEG compression is the most common real-world degradation
            # AI images often have different compression artifacts than real photos
        #    A.ImageCompression(
        #         quality_lower=50,
        #         quality_upper=95,
        #         compression_type=cv2.IMWRITE_JPEG_QUALITY,
        #         p=0.7
        #     ),
            A.OneOf([
                A.ImageCompression(
                    quality_range=(75, 95),
                    compression_type="jpeg",
                    p=1.0
                ),
                A.ImageCompression(
                    
                    compression_type="webp",
                    p=1.0
                ),
            ], p=0.7),
            
            # ==================== RESIZE/SCALE ====================
            # Simulate screenshotting, mobile uploads, web compression
            # These change the image significantly
            A.OneOf([
                # Downscale then upscale (simulates screenshot quality loss)
                # A.Downscale(scale_min=0.5, scale_max=0.85, interpolation=cv2.INTER_LINEAR, p=1)
                 A.Downscale(
                        scale_range=(0.5, 0.85),
                        interpolation_pair={"downscale": cv2.INTER_LINEAR, "upscale": cv2.INTER_LINEAR},
                        p=1
                    ),
                # Direct resize (simulates mobile camera/web upload)
                A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR, p=1),
            ], p=0.5),
            
            # ==================== BLUR ====================
            # Real images: camera/motion blur, out-of-focus
            # AI images: sometimes suspiciously sharp or blurry
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7),sigma_limit=(0.1, 2.0),p=1),
                A.MotionBlur(blur_limit=7, p=1),
                A.MedianBlur(blur_limit=5, p=1),
            ], p=0.4),
            
            # ==================== NOISE ====================
            # Sensor noise, compression artifacts
            A.OneOf([
                # Gaussian noise (camera sensor)
               A.GaussNoise(std_range=(0.05, 0.15), p=1),
                # ISO/color noise
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
                # Gaussian blur alternative
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1),
            ], p=0.3),
            
            # ==================== BRIGHTNESS/CONTRAST ====================
            # White balance, exposure differences
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            
            # ==================== COLOR SHIFTS ====================
            # White balance, color temperature, saturation
            A.OneOf([
               A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=0, p=1),], p=0.3),
            
            # ==================== GEOMETRIC ====================
            # Perspective, crop, rotation
            A.OneOf([
                # Perspective distortion
                A.Perspective(scale=(0.05, 0.1), p=1),
                # Affine transform (slight rotation + shear)
                A.Affine(scale=(0.95, 1.05), rotate=(-10, 10), shear=(-5, 5), p=1),
            ], p=0.3),
            
            # ==================== CROPS ====================
            # Simulate partial images, cropping
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=128,
                p=0.2
            ),
            # ==================== CHANNEL SHIFTS ====================
            # Random channel manipulation
            A.ChannelShuffle(p=0.1),
            
            # ==================== FINAL NORMALIZATION ====================
            # Resize to exact size if not done yet
            A.Resize(height=img_size, width=img_size, p=0.5),
            
            # Normalize to ImageNet standards
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                p=1.0
            ),
            
            # Convert to PyTorch tensor
            ToTensorV2(p=1.0),
        ], bbox_params=None)  # No bounding boxes for classification
    
    elif mode == 'val':
        # Validation: minimal augmentation, just resize & normalize
        return A.Compose([
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
    
    elif mode == 'test':
        # Test: no augmentation, just normalize
        return A.Compose([
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
    
    else:
        raise ValueError(f"Unknown augmentation mode: {mode}. Choose 'train', 'val', or 'test'")


def get_degradation_augmentation() -> A.Compose:
    """Augmentation specifically for robustness testing (Module C).
    
    Used to test model performance under real-world degradation.
    NOT used for training - used for evaluation only.
    
    Simulates:
    - JPEG compression at various qualities
    - Screenshotting (downscale + upscale)
    - Mobile camera degradation
    - Web compression
    
    Returns:
        albumentations.Compose pipeline for degradation testing
    """
    return A.Compose([
        A.OneOf([
            # # Light JPEG compression
            A.ImageCompression(quality_range=(85, 95), compression_type="jpeg", p=1),
            # # Medium JPEG compression
            A.ImageCompression(quality_range=(60, 85), compression_type="jpeg", p=1),
            # # Heavy JPEG compression
            A.ImageCompression(quality_range=(40, 60), compression_type="jpeg", p=1),
        ], p=0.8),
        
        A.OneOf([
            # Light downscale (0.9x)
            A.Downscale(scale_range=(0.85, 0.95), p=1),
            # Medium downscale (0.75x)
            A.Downscale(scale_range=(0.65, 0.85), p=1),
            # Heavy downscale (0.5x)
            A.Downscale(scale_range=(0.4, 0.65), p=1),
        ], p=0.6),
        
        A.Resize(height=224, width=224),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


class AugmentationVariants:
    """Generate 4 variants of the same image (for CLIP feature caching).
    
    The original PixelProof code precomputes CLIP features with 4 variants:
    - clean
    - JPEG compressed
    - downscale-upscale
    - blur
    
    This class maintains that pattern while adding enhanced augmentation.
    """
    
    def __init__(self, img_size: int = 224):
        self.img_size = img_size
        
        # Variant 1: Clean (minimal processing)
        self.clean = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
        
        # Variant 2: JPEG compressed
        self.jpeg = A.Compose([
            A.ImageCompression(quality_range=(60, 80), compression_type="jpeg", p=1.0),
            A.Resize(height=img_size, width=img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
        
        # Variant 3: Downscale-upscale
        self.downscale = A.Compose([
            A.Downscale(scale_range=(0.65, 0.85), p=1.0),
            A.Resize(height=img_size, width=img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
        
        # Variant 4: Blur
        self.blur = A.Compose([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.Resize(height=img_size, width=img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
    
    def get_variants(self, image: np.ndarray) -> list:
        """Get all 4 variants of an image.
        
        Args:
            image: numpy array, (H, W, 3), uint8, RGB
        
        Returns:
            list of 4 torch tensors: [clean, jpeg, downscale, blur]
        """
        return [
            self.clean(image=image)['image'],
            self.jpeg(image=image)['image'],
            self.downscale(image=image)['image'],
            self.blur(image=image)['image'],
        ]


if __name__ == "__main__":
    # Test the augmentation pipeline
    import torch
    from PIL import Image
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


    # Test training augmentation
    train_aug = get_augmentation_pipeline(mode='train')
    augmented = train_aug(image=dummy_image)
    tensor = augmented['image']
    
    print(f"✅ Train augmentation works!")
    print(f"   Input shape: {dummy_image.shape}")
    print(f"   Output shape: {tensor.shape}")
    print(f"   Output dtype: {tensor.dtype}")
    
    # Test validation augmentation
    val_aug = get_augmentation_pipeline(mode='val')
    augmented = val_aug(image=dummy_image)
    tensor = augmented['image']
    
    print(f"✅ Validation augmentation works!")
    print(f"   Output shape: {tensor.shape}")
    
    # Test variants
    variants_gen = AugmentationVariants()
    variants = variants_gen.get_variants(dummy_image)
    
    print(f"✅ Augmentation variants work!")
    print(f"   Generated {len(variants)} variants")
    print(f"   Each shape: {variants[0].shape}")
    
    print("\n✅ All augmentation pipelines working!")
