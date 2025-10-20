#!/usr/bin/env python3
"""
Create animated GIFs showing all DINOv3 features for drones-racing.gif
Uses improved depth visualization (TURBO colormap + bilateral filtering)
"""

import cv2
import numpy as np
import torch
import sys
import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dinov3_toolkit'))

from backbone.model_backbone import DinoBackbone
from head_detection.model_head import DinoFCOSHead
from head_detection.utils import detection_inference, generate_detection_overlay
from head_segmentation.model_head import ASPPDecoder
from head_segmentation.utils import outputs_to_maps, generate_segmentation_overlay
from head_depth.model_head import DepthHeadLite
from head_depth.utils import depth_to_colormap
from head_optical_flow.model_head import LiteFlowHead
from head_optical_flow.utils import flow_to_image

print("=" * 80)
print("Creating Multi-Feature DINOv3 GIFs from drones-racing.gif")
print("=" * 80)

# Configuration
IMG_PATH = '/home/user/Downloads/drones-racing.gif'
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
DINOV3_REPO = os.path.join(os.path.dirname(__file__), 'dinov3')
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'dinov3_toolkit/backbone/weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 640  # Model input size
OUTPUT_SIZE = 128  # Output visualization size (pixelated)
EMBED_DIM = 384
N_LAYERS = 12

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n1. Loading frames from drones-racing.gif...")
cap = cv2.VideoCapture(IMG_PATH)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
print(f"   ✓ Loaded {len(frames)} frames")

# Load DINOv3 backbone
print(f"\n2. Loading DINOv3 backbone...")
dino_loader = torch.hub.load(
    DINOV3_REPO,
    'dinov3_vits16plus',
    source='local',
    pretrained=True,
    weights=WEIGHTS_PATH
)
backbone = DinoBackbone(dino_loader, n_layers=N_LAYERS).to(DEVICE)
backbone.eval()
print(f"   ✓ Backbone loaded (output: {EMBED_DIM} channels)")

# Load task-specific heads
print(f"\n3. Loading task-specific heads...")
heads_loaded = {}

# Detection
try:
    detection_classes_path = 'dinov3_toolkit/head_detection/class_names.txt'
    with open(detection_classes_path) as f:
        detection_class_names = [line.strip() for line in f]
    detection_num_classes = len(detection_class_names)

    detection_head = DinoFCOSHead(
        backbone_out_channels=EMBED_DIM,
        fpn_channels=192,
        num_classes=detection_num_classes,
        num_convs=4
    ).to(DEVICE)
    detection_weights = torch.load('dinov3_toolkit/head_detection/weights/model.pth', map_location=DEVICE, weights_only=True)
    detection_head.load_state_dict(detection_weights)
    detection_head.eval()
    heads_loaded['detection'] = detection_head
    print(f"   ✓ Detection head loaded ({detection_num_classes} classes)")
except Exception as e:
    print(f"   ⚠ Detection head failed: {str(e)[:100]}")

# Segmentation
try:
    seg_classes_path = 'dinov3_toolkit/head_segmentation/class_names.txt'
    with open(seg_classes_path) as f:
        segmentation_class_names = [line.strip() for line in f]
    segmentation_num_classes = len(segmentation_class_names)

    segmentation_head = ASPPDecoder(
        num_classes=segmentation_num_classes,
        in_ch=EMBED_DIM,
        target_size=(320, 320)
    ).to(DEVICE)
    seg_weights = torch.load('dinov3_toolkit/head_segmentation/weights/model.pth', map_location=DEVICE, weights_only=True)
    segmentation_head.load_state_dict(seg_weights)
    segmentation_head.eval()
    heads_loaded['segmentation'] = segmentation_head
    print(f"   ✓ Segmentation head loaded ({segmentation_num_classes} classes)")
except Exception as e:
    print(f"   ⚠ Segmentation head failed: {str(e)[:100]}")

# Depth - with improved visualization
try:
    depth_head = DepthHeadLite(
        in_ch=EMBED_DIM,
        out_size=(IMG_SIZE, IMG_SIZE)
    ).to(DEVICE)
    depth_weights = torch.load('dinov3_toolkit/head_depth/weights/model.pth', map_location=DEVICE, weights_only=True)
    depth_head.load_state_dict(depth_weights)
    depth_head.eval()
    heads_loaded['depth'] = depth_head
    print(f"   ✓ Depth head loaded (TURBO colormap + bilateral filter)")
except Exception as e:
    print(f"   ⚠ Depth head failed: {str(e)[:100]}")

# Optical Flow
try:
    flow_head = LiteFlowHead(
        out_size=(IMG_SIZE, IMG_SIZE),
        in_channels=EMBED_DIM,
        proj_channels=256,
        radius=4,
        fusion_channels=448,
        fusion_layers=3,
        convex_up=3,
        refinement_layers=2
    ).to(DEVICE)
    flow_weights = torch.load('dinov3_toolkit/head_optical_flow/weights/model.pth', map_location=DEVICE, weights_only=True)
    flow_head.load_state_dict(flow_weights)
    flow_head.eval()
    heads_loaded['flow'] = flow_head
    print(f"   ✓ Optical flow head loaded")
except Exception as e:
    print(f"   ⚠ Optical flow head failed: {str(e)[:100]}")

def preprocess_frame(frame):
    """Preprocess frame for DINOv3"""
    img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_norm - mean) / std
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)
    return img_tensor, img_rgb

def downscale_to_output(img):
    """Downscale image to OUTPUT_SIZE (128x128) with NEAREST interpolation for pixelated look"""
    if img.shape[0] != OUTPUT_SIZE or img.shape[1] != OUTPUT_SIZE:
        # Use NEAREST for blocky/pixelated look
        img_downscaled = cv2.resize(img, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_NEAREST)
        return img_downscaled
    return img

def add_label(img, text, color=(255, 255, 0)):
    """Add label to top of image (adjusted for OUTPUT_SIZE)"""
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    try:
        # Smaller font for 128x128
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (img.shape[1] - text_width) // 2
    y = 3  # Smaller margin

    draw.rectangle([x-3, y-2, x+text_width+3, y+text_height+2], fill=(0, 0, 0))
    draw.text((x, y), text, fill=color, font=font)
    return np.array(img_pil)

# Storage for all frames
task_frames = {
    'original': [],
    'detection': [],
    'segmentation': [],
    'depth': [],
    'depth_bw': [],  # Normalized black and white depth (0=close, 1=far)
    'flow': [],
    'attention': [],
}

prev_features = None

# Balanced temporal smoothing for 128x128 output
print(f"\n4. Processing ALL {len(frames)} frames with balanced smoothing (128x128)...")
depth_min_ema = None  # Exponential moving average for min depth
depth_max_ema = None  # Exponential moving average for max depth
prev_depth_map = None  # Previous frame's depth map for temporal smoothing
ema_alpha = 0.2  # Range smoothing: balanced (responsive but smooth)
depth_temporal_weight = 0.6  # Depth map temporal smoothing: 60% history, 40% current (BALANCED)
for i, frame in enumerate(tqdm(frames, desc="Processing")):
    img_tensor, img_rgb = preprocess_frame(frame)
    frame_rgb = img_rgb.copy()

    # Extract backbone features
    with torch.no_grad():
        features = backbone(img_tensor)

    # Compute attention map FIRST (needed for attention-adaptive depth)
    try:
        feat_mean = features.mean(dim=1).squeeze().cpu().numpy()
        h = w = int(np.sqrt(feat_mean.size))
        feat_mean = feat_mean.reshape(h, w)
        feat_mean_norm = (feat_mean - feat_mean.min()) / (feat_mean.max() - feat_mean.min() + 1e-8)

        # Resize attention to IMG_SIZE for depth processing (smooth for depth)
        attention_map = cv2.resize(feat_mean_norm, (IMG_SIZE, IMG_SIZE))

        # Create PIXELATED visualization (blocky attention regions)
        ATTENTION_PIXEL_SIZE = 64  # Make 64x64 pixelated grid (more detail than 16x16)

        # Downsample to very low res
        attention_lowres = cv2.resize(feat_mean_norm, (ATTENTION_PIXEL_SIZE, ATTENTION_PIXEL_SIZE),
                                     interpolation=cv2.INTER_AREA)

        # Upsample back with NEAREST neighbor for blocky pixels
        attention_blocky = cv2.resize(attention_lowres, (IMG_SIZE, IMG_SIZE),
                                     interpolation=cv2.INTER_NEAREST)

        # Apply colormap to blocky version
        heatmap = cv2.applyColorMap((attention_blocky * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        attention_overlay = cv2.addWeighted(frame_rgb, 0.6, heatmap_rgb, 0.4, 0)
        attention_downscaled = downscale_to_output(attention_overlay)
        task_frames['attention'].append(add_label(attention_downscaled, "Attention (Pixelated)", color=(255, 100, 255)))
    except Exception as e:
        # Fallback: uniform attention
        attention_map = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32) * 0.5
        error_frame = downscale_to_output(frame_rgb.copy())
        task_frames['attention'].append(add_label(error_frame, "Attention Error"))

    # Original - downscale to 128x128
    original_downscaled = downscale_to_output(frame_rgb.copy())
    task_frames['original'].append(add_label(original_downscaled, "Original"))

    # Object Detection
    if 'detection' in heads_loaded:
        try:
            with torch.no_grad():
                boxes, scores, labels = detection_inference(
                    heads_loaded['detection'],
                    features,
                    img_size=(IMG_SIZE, IMG_SIZE),
                    score_thresh=0.3,
                    nms_thresh=0.5
                )
            det_vis = generate_detection_overlay(frame_rgb.copy(), boxes, scores, labels, detection_class_names)
            det_vis_downscaled = downscale_to_output(det_vis)
            task_frames['detection'].append(add_label(det_vis_downscaled, "Object Detection"))
        except Exception as e:
            error_frame = downscale_to_output(frame_rgb.copy())
            task_frames['detection'].append(add_label(error_frame, "Detection Error"))

    # Semantic Segmentation
    if 'segmentation' in heads_loaded:
        try:
            with torch.no_grad():
                seg_logits = heads_loaded['segmentation'](features)
            semantic_map = outputs_to_maps(seg_logits, image_size=(IMG_SIZE, IMG_SIZE))
            seg_vis = generate_segmentation_overlay(frame_rgb.copy(), semantic_map,
                                                     class_names=segmentation_class_names, alpha=0.5)
            seg_vis_downscaled = downscale_to_output(seg_vis)
            task_frames['segmentation'].append(add_label(seg_vis_downscaled, "Semantic Segmentation", color=(0, 255, 128)))
        except Exception as e:
            error_frame = downscale_to_output(frame_rgb.copy())
            task_frames['segmentation'].append(add_label(error_frame, "Segmentation Error"))

    # Depth Estimation - ULTRA-SMOOTH for autonomous navigation
    if 'depth' in heads_loaded:
        try:
            with torch.no_grad():
                depth_output = heads_loaded['depth'](features)
            depth_map = depth_output[0, 0].cpu().numpy()

            # Temporal smoothing of depth map itself (not just range)
            if prev_depth_map is not None:
                # Blend current with previous depth map for ultra-stable output
                depth_map = depth_temporal_weight * prev_depth_map + (1 - depth_temporal_weight) * depth_map
            prev_depth_map = depth_map.copy()

            # Use percentiles instead of min/max (robust to outliers)
            current_min = np.percentile(depth_map, 5)
            current_max = np.percentile(depth_map, 95)

            # Apply exponential moving average for smooth range transitions
            if depth_min_ema is None:
                depth_min_ema = current_min
                depth_max_ema = current_max
            else:
                depth_min_ema = ema_alpha * current_min + (1 - ema_alpha) * depth_min_ema
                depth_max_ema = ema_alpha * current_max + (1 - ema_alpha) * depth_max_ema

            # ATTENTION-ADAPTIVE processing: sharpen where attention is high, blur where low

            # Create two versions: sharp and blurred
            depth_sharp = depth_map.copy()  # Keep original detail
            depth_blurred = cv2.bilateralFilter(depth_map.astype(np.float32), d=9, sigmaColor=0.1, sigmaSpace=9)

            # Use attention map as spatial weight (0=blur, 1=sharp)
            # Boost attention contrast for stronger effect
            attention_weight = np.power(attention_map, 0.5)  # Make attention regions stronger

            # Blend: high attention = use sharp, low attention = use blurred
            depth_adaptive = attention_weight * depth_sharp + (1 - attention_weight) * depth_blurred

            # Use temporally smoothed range for ultra-stable visualization
            depth_colored = depth_to_colormap(
                depth_adaptive,
                dmin=depth_min_ema,
                dmax=depth_max_ema,
                colormap=cv2.COLORMAP_TURBO,
                bgr=False,
                invert=True  # Closer = warmer colors (red/yellow) for better visibility
            )

            # Apply EXTRA sharpening to high-attention regions only
            kernel_sharpen = np.array([[-0.5, -0.5, -0.5],
                                       [-0.5,  5.0, -0.5],
                                       [-0.5, -0.5, -0.5]])
            depth_sharpened_all = cv2.filter2D(depth_colored.astype(np.float32), -1, kernel_sharpen)

            # Apply sharpening only where attention is high (>0.3)
            attention_3ch = np.stack([attention_map, attention_map, attention_map], axis=-1)
            attention_strong = np.clip((attention_3ch - 0.3) / 0.7, 0, 1)  # 0.3-1.0 mapped to 0-1

            depth_colored_sharp = attention_strong * depth_sharpened_all + (1 - attention_strong) * depth_colored.astype(np.float32)
            depth_colored_sharp = np.clip(depth_colored_sharp, 0, 255).astype(np.uint8)

            # Downscale to 128x128
            depth_downscaled = downscale_to_output(depth_colored_sharp)
            task_frames['depth'].append(add_label(depth_downscaled, "Depth Estimation", color=(255, 128, 0)))

            # Create HIGH CONTRAST normalized black and white depth
            # Use percentile-based normalization to clip outliers and boost contrast
            depth_min_perc = np.percentile(depth_map, 2)  # Clip bottom 2%
            depth_max_perc = np.percentile(depth_map, 98)  # Clip top 2%

            # Normalize with percentile clipping
            depth_normalized = (depth_map - depth_min_perc) / (depth_max_perc - depth_min_perc + 1e-8)
            depth_normalized = np.clip(depth_normalized, 0, 1)  # Clip outliers to [0, 1]

            # Convert to 8-bit
            depth_bw = (depth_normalized * 255).astype(np.uint8)

            # Apply histogram equalization for MAXIMUM contrast
            depth_bw_eq = cv2.equalizeHist(depth_bw)

            # Optional: Apply gamma correction for even more dramatic contrast (gamma < 1 = darker shadows)
            gamma = 0.7
            depth_bw_gamma = np.power(depth_bw_eq / 255.0, gamma) * 255
            depth_bw_final = depth_bw_gamma.astype(np.uint8)

            # Convert to RGB for consistency (grayscale in all channels)
            depth_bw_rgb = np.stack([depth_bw_final, depth_bw_final, depth_bw_final], axis=-1)

            # Downscale to 128x128
            depth_bw_downscaled = downscale_to_output(depth_bw_rgb)
            task_frames['depth_bw'].append(add_label(depth_bw_downscaled, "Depth (High Contrast BW)", color=(255, 255, 255)))
        except Exception as e:
            error_frame = downscale_to_output(frame_rgb.copy())
            task_frames['depth'].append(add_label(error_frame, "Depth Error"))
            task_frames['depth_bw'].append(add_label(error_frame.copy(), "Depth BW Error"))

    # Optical Flow
    if 'flow' in heads_loaded and prev_features is not None:
        try:
            with torch.no_grad():
                flow_output = heads_loaded['flow'](torch.cat([prev_features, features], dim=1))
            flow = flow_output[0].permute(1, 2, 0).cpu().numpy()
            flow_colored = flow_to_image(flow, convert_to_bgr=False)
            flow_downscaled = downscale_to_output(flow_colored)
            task_frames['flow'].append(add_label(flow_downscaled, "Optical Flow", color=(0, 255, 255)))
        except Exception as e:
            error_frame = downscale_to_output(frame_rgb.copy())
            task_frames['flow'].append(add_label(error_frame, "Flow Error"))
    elif 'flow' in heads_loaded:
        first_frame = downscale_to_output(frame_rgb.copy())
        task_frames['flow'].append(add_label(first_frame, "Optical Flow (first frame)", color=(128, 128, 128)))

    prev_features = features
    # Note: Attention heatmap already computed at top of loop for attention-adaptive depth processing

# Get original GIF duration
original_gif = Image.open(IMG_PATH)
duration = original_gif.info.get('duration', 60)

print(f"\n5. Creating output GIFs (duration={duration}ms per frame)...")

# Create individual GIFs
gifs_created = []
for task_name in ['original', 'detection', 'segmentation', 'depth', 'depth_bw', 'flow', 'attention']:
    frames_list = task_frames.get(task_name, [])
    if not frames_list:
        continue

    try:
        pil_frames = [Image.fromarray(f) if isinstance(f, np.ndarray) else f for f in frames_list]
        gif_path = os.path.join(OUTPUT_DIR, f'drones_{task_name}.gif')
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=duration, loop=0)
        size_kb = os.path.getsize(gif_path) / 1024
        print(f"   ✓ {task_name.capitalize()}: {size_kb:.1f} KB")
        gifs_created.append((task_name, size_kb))
    except Exception as e:
        print(f"   ✗ {task_name} failed: {e}")

# Create 2x3 grid GIF (128x128 tiles)
print(f"\n6. Creating combined grid GIF (128x128 per tile)...")
grid_frames = []
for i in range(len(frames)):
    grid = np.zeros((OUTPUT_SIZE * 2, OUTPUT_SIZE * 3, 3), dtype=np.uint8)

    def safe_get(task_name, idx):
        frames_list = task_frames.get(task_name, [])
        if frames_list and idx < len(frames_list):
            return frames_list[idx]
        return task_frames['original'][idx]

    # Top row: Original | Detection | Segmentation
    grid[0:OUTPUT_SIZE, 0:OUTPUT_SIZE] = task_frames['original'][i]
    grid[0:OUTPUT_SIZE, OUTPUT_SIZE:OUTPUT_SIZE*2] = safe_get('detection', i)
    grid[0:OUTPUT_SIZE, OUTPUT_SIZE*2:OUTPUT_SIZE*3] = safe_get('segmentation', i)

    # Bottom row: Depth | Flow | Attention
    grid[OUTPUT_SIZE:OUTPUT_SIZE*2, 0:OUTPUT_SIZE] = safe_get('depth', i)
    grid[OUTPUT_SIZE:OUTPUT_SIZE*2, OUTPUT_SIZE:OUTPUT_SIZE*2] = safe_get('flow', i)
    grid[OUTPUT_SIZE:OUTPUT_SIZE*2, OUTPUT_SIZE*2:OUTPUT_SIZE*3] = task_frames['attention'][i]

    grid_frames.append(Image.fromarray(grid))

grid_path = os.path.join(OUTPUT_DIR, 'drones_all_features_grid.gif')
grid_frames[0].save(grid_path, save_all=True, append_images=grid_frames[1:], duration=duration, loop=0, optimize=False)
grid_size_kb = os.path.getsize(grid_path) / 1024
print(f"   ✓ Combined Grid: {grid_size_kb:.1f} KB")

print("\n" + "=" * 80)
print(f"SUCCESS! Processed ALL {len(frames)} frames - Created {len(gifs_created) + 1} GIFs")
print("=" * 80)
print(f"\nGenerated GIFs in {OUTPUT_DIR}/:")
for name, size in gifs_created:
    print(f"  • drones_{name}.gif ({size:.0f} KB)")
print(f"  • drones_all_features_grid.gif ({grid_size_kb:.0f} KB)")
print("\nGrid layout (2x3): 128x128 pixels per tile (384x256 total)")
print("  Top row:    Original | Detection | Segmentation")
print("  Bottom row: Depth    | Flow      | Attention")
print("\nOutput settings for autonomous ring detection:")
print("  • 128x128 pixel maps (PIXELATED with NEAREST neighbor)")
print("  • TURBO colormap (inverted: closer = RED/YELLOW)")
print("  • Balanced temporal smoothing (60% history, 40% current)")
print("  • ATTENTION-ADAPTIVE depth processing:")
print("    - Sharp where DINOv3 attention is HIGH (distant rings!)")
print("    - Blurred where attention is LOW (background)")
print("    - Extra sharpening applied to high-attention regions")
print("  • Attention map: PIXELATED (64x64 grid) for detail + clarity!")
print("  • Range smoothing (EMA α=0.2) - responsive yet smooth")
print("  • Percentile-based ranges (5th-95th) - robust to outliers")
print("\nNEW: HIGH CONTRAST Normalized B&W depth (drones_depth_bw.gif):")
print("  • 0 (black) = closest depth in frame")
print("  • 1 (white) = farthest depth in frame")
print("  • Percentile clipping (2nd-98th) removes outliers")
print("  • Histogram equalization for MAXIMUM contrast")
print("  • Gamma correction (γ=0.7) for dramatic shadows")
print("  • Perfect for edge detection and thresholding!")
print("=" * 80)
