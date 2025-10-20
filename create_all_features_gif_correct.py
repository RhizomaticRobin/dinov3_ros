#!/usr/bin/env python3
"""
Create animated GIFs showing all DINOv3 features for bird.gif
CORRECTED VERSION - matches dinov3_node.py implementation
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
print("Creating Multi-Feature DINOv3 GIFs from bird.gif")
print("=" * 80)

# Configuration - matches dinov3_node defaults
IMG_PATH = '/home/user/Downloads/bird.gif'
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
DINOV3_REPO = os.path.join(os.path.dirname(__file__), 'dinov3')
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'dinov3_toolkit/backbone/weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 640
EMBED_DIM = 384  # vits16plus embedding dimension
N_LAYERS = 12

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n1. Loading frames from bird.gif...")
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

# Load task-specific heads - CORRECTED INITIALIZATION
print(f"\n3. Loading task-specific heads...")
heads_loaded = {}

# Detection - uses backbone_out_channels
try:
    # Read class names
    detection_classes_path = 'dinov3_toolkit/head_detection/class_names.txt'
    with open(detection_classes_path) as f:
        detection_class_names = [line.strip() for line in f]
    detection_num_classes = len(detection_class_names)

    detection_head = DinoFCOSHead(
        backbone_out_channels=EMBED_DIM,  # CORRECT parameter name!
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

# Segmentation - uses in_ch and requires num_classes
try:
    seg_classes_path = 'dinov3_toolkit/head_segmentation/class_names.txt'
    with open(seg_classes_path) as f:
        segmentation_class_names = [line.strip() for line in f]
    segmentation_num_classes = len(segmentation_class_names)

    segmentation_head = ASPPDecoder(
        num_classes=segmentation_num_classes,  # FIRST parameter!
        in_ch=EMBED_DIM,  # CORRECT parameter name!
        target_size=(320, 320)  # From config default
    ).to(DEVICE)
    seg_weights = torch.load('dinov3_toolkit/head_segmentation/weights/model.pth', map_location=DEVICE, weights_only=True)
    segmentation_head.load_state_dict(seg_weights)
    segmentation_head.eval()
    heads_loaded['segmentation'] = segmentation_head
    print(f"   ✓ Segmentation head loaded ({segmentation_num_classes} classes)")
except Exception as e:
    print(f"   ⚠ Segmentation head failed: {str(e)[:100]}")

# Depth - uses in_ch
try:
    depth_head = DepthHeadLite(
        in_ch=EMBED_DIM,  # CORRECT parameter name!
        out_size=(IMG_SIZE, IMG_SIZE)
    ).to(DEVICE)
    depth_weights = torch.load('dinov3_toolkit/head_depth/weights/model.pth', map_location=DEVICE, weights_only=True)
    depth_head.load_state_dict(depth_weights)
    depth_head.eval()
    heads_loaded['depth'] = depth_head
    print(f"   ✓ Depth head loaded")
except Exception as e:
    print(f"   ⚠ Depth head failed: {str(e)[:100]}")

# Optical Flow - single backbone input, not concatenated
try:
    flow_head = LiteFlowHead(
        out_size=(IMG_SIZE, IMG_SIZE),
        in_channels=EMBED_DIM,  # Single backbone features!
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
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    return img_tensor, img_rgb

def add_label(img, text, color=(255, 255, 0)):
    """Add label to top of image"""
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (img.shape[1] - text_width) // 2
    y = 20

    draw.rectangle([x-10, y-8, x+text_width+10, y+text_height+8], fill=(0, 0, 0))
    draw.text((x, y), text, fill=color, font=font)
    return np.array(img_pil)

# Storage for all frames
task_frames = {
    'original': [],
    'detection': [],
    'segmentation': [],
    'depth': [],
    'flow': [],
    'attention': [],
}

prev_features = None

print(f"\n4. Processing {len(frames)} frames...")
for i, frame in enumerate(tqdm(frames, desc="Processing")):
    img_tensor, img_rgb = preprocess_frame(frame)
    frame_rgb = img_rgb.copy()

    # Extract backbone features
    with torch.no_grad():
        features = backbone(img_tensor)

    # Original
    task_frames['original'].append(add_label(frame_rgb.copy(), "Original"))

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
            task_frames['detection'].append(add_label(det_vis, "Object Detection"))
        except Exception as e:
            task_frames['detection'].append(add_label(frame_rgb.copy(), "Detection Error"))

    # Semantic Segmentation
    if 'segmentation' in heads_loaded:
        try:
            with torch.no_grad():
                seg_logits = heads_loaded['segmentation'](features)
            semantic_map = outputs_to_maps(seg_logits, image_size=(IMG_SIZE, IMG_SIZE))
            seg_vis = generate_segmentation_overlay(frame_rgb.copy(), semantic_map,
                                                     class_names=segmentation_class_names, alpha=0.5)
            task_frames['segmentation'].append(add_label(seg_vis, "Semantic Segmentation", color=(0, 255, 128)))
        except Exception as e:
            task_frames['segmentation'].append(add_label(frame_rgb.copy(), "Segmentation Error"))

    # Depth Estimation
    if 'depth' in heads_loaded:
        try:
            with torch.no_grad():
                depth_output = heads_loaded['depth'](features)
            depth_map = depth_output[0, 0].cpu().numpy()

            # Apply bilateral filter for smoother depth while preserving edges
            depth_smooth = cv2.bilateralFilter(depth_map.astype(np.float32), d=5, sigmaColor=0.1, sigmaSpace=5)

            # Use TURBO colormap for better perceptual contrast + full range for maximum dynamic range
            depth_colored = depth_to_colormap(
                depth_smooth,
                dmin=depth_map.min(),  # Use full range instead of percentiles
                dmax=depth_map.max(),
                colormap=cv2.COLORMAP_TURBO,  # Much better contrast than MAGMA
                bgr=False
            )
            task_frames['depth'].append(add_label(depth_colored, "Depth Estimation", color=(255, 128, 0)))
        except Exception as e:
            task_frames['depth'].append(add_label(frame_rgb.copy(), "Depth Error"))

    # Optical Flow
    if 'flow' in heads_loaded and prev_features is not None:
        try:
            with torch.no_grad():
                # Flow head expects concatenated features
                flow_output = heads_loaded['flow'](torch.cat([prev_features, features], dim=1))
            flow = flow_output[0].permute(1, 2, 0).cpu().numpy()
            flow_colored = flow_to_image(flow, convert_to_bgr=False)
            task_frames['flow'].append(add_label(flow_colored, "Optical Flow", color=(0, 255, 255)))
        except Exception as e:
            task_frames['flow'].append(add_label(frame_rgb.copy(), "Flow Error"))
    elif 'flow' in heads_loaded:
        task_frames['flow'].append(add_label(frame_rgb.copy(), "Optical Flow (first frame)", color=(128, 128, 128)))

    prev_features = features

    # Attention heatmap
    try:
        feat_mean = features.mean(dim=1).squeeze().cpu().numpy()
        h = w = int(np.sqrt(feat_mean.size))
        feat_mean = feat_mean.reshape(h, w)
        feat_mean = (feat_mean - feat_mean.min()) / (feat_mean.max() - feat_mean.min() + 1e-8)
        activation_resized = cv2.resize(feat_mean, (IMG_SIZE, IMG_SIZE))
        heatmap = cv2.applyColorMap((activation_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        attention_overlay = cv2.addWeighted(frame_rgb, 0.6, heatmap_rgb, 0.4, 0)
        task_frames['attention'].append(add_label(attention_overlay, "Attention Map", color=(255, 100, 255)))
    except:
        task_frames['attention'].append(add_label(frame_rgb.copy(), "Attention Error"))

# Get original GIF duration
original_gif = Image.open(IMG_PATH)
duration = original_gif.info.get('duration', 70)

print(f"\n5. Creating output GIFs (duration={duration}ms per frame)...")

# Create individual GIFs
gifs_created = []
for task_name in ['original', 'detection', 'segmentation', 'depth', 'flow', 'attention']:
    frames_list = task_frames.get(task_name, [])
    if not frames_list:
        continue

    try:
        pil_frames = [Image.fromarray(f) if isinstance(f, np.ndarray) else f for f in frames_list]
        gif_path = os.path.join(OUTPUT_DIR, f'bird_{task_name}_fixed.gif')
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=duration, loop=0)
        size_kb = os.path.getsize(gif_path) / 1024
        print(f"   ✓ {task_name.capitalize()}: {size_kb:.1f} KB")
        gifs_created.append((task_name, size_kb))
    except Exception as e:
        print(f"   ✗ {task_name} failed: {e}")

# Create 2x3 grid GIF
print(f"\n6. Creating combined grid GIF...")
grid_frames = []
for i in range(len(frames)):
    grid = np.zeros((IMG_SIZE * 2, IMG_SIZE * 3, 3), dtype=np.uint8)

    def safe_get(task_name, idx):
        frames_list = task_frames.get(task_name, [])
        if frames_list and idx < len(frames_list):
            return frames_list[idx]
        return task_frames['original'][idx]

    # Top row: Original | Detection | Segmentation
    grid[0:IMG_SIZE, 0:IMG_SIZE] = task_frames['original'][i]
    grid[0:IMG_SIZE, IMG_SIZE:IMG_SIZE*2] = safe_get('detection', i)
    grid[0:IMG_SIZE, IMG_SIZE*2:IMG_SIZE*3] = safe_get('segmentation', i)

    # Bottom row: Depth | Flow | Attention
    grid[IMG_SIZE:IMG_SIZE*2, 0:IMG_SIZE] = safe_get('depth', i)
    grid[IMG_SIZE:IMG_SIZE*2, IMG_SIZE:IMG_SIZE*2] = safe_get('flow', i)
    grid[IMG_SIZE:IMG_SIZE*2, IMG_SIZE*2:IMG_SIZE*3] = task_frames['attention'][i]

    grid_frames.append(Image.fromarray(grid))

grid_path = os.path.join(OUTPUT_DIR, 'bird_all_features_grid_fixed.gif')
grid_frames[0].save(grid_path, save_all=True, append_images=grid_frames[1:], duration=duration, loop=0, optimize=False)
grid_size_kb = os.path.getsize(grid_path) / 1024
print(f"   ✓ Combined Grid: {grid_size_kb:.1f} KB")

print("\n" + "=" * 80)
print(f"SUCCESS! Created {len(gifs_created) + 1} GIFs with CORRECT model initialization")
print("=" * 80)
print(f"\nGenerated GIFs in {OUTPUT_DIR}/:")
for name, size in gifs_created:
    print(f"  • bird_{name}_fixed.gif ({size:.0f} KB)")
print(f"  • bird_all_features_grid_fixed.gif ({grid_size_kb:.0f} KB)")
print("\nGrid layout (2x3):")
print("  Top row:    Original | Detection | Segmentation")
print("  Bottom row: Depth    | Flow      | Attention")
print("=" * 80)
