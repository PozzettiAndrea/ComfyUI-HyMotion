import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
import comfy.utils
from typing import Dict, Any, List

class HYMotionPreview3D:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_data": ("HYMOTION_DATA",),
                "width": ("INT", {"default": 512, "min": 128, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 128, "max": 2048}),
                "elevation": ("INT", {"default": 20, "min": -90, "max": 90}),
                "azimuth": ("INT", {"default": 45, "min": -180, "max": 180}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
                "draw_skeleton": ("BOOLEAN", {"default": True}),
                "sample_index": ("INT", {"default": 0, "min": 0, "max": 64}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render_animation"
    CATEGORY = "HY-Motion/preview"

    def render_animation(self, motion_data, width, height, elevation, azimuth, scale, draw_skeleton, sample_index):
        # Extract keypoints (B, T, J, 3)
        k3d = motion_data.output_dict["keypoints3d"]
        
        # Clamp sample index
        sample_index = min(sample_index, k3d.shape[0] - 1)
        
        # Get specific sample (T, J, 3)
        sample = k3d[sample_index].cpu().numpy()
        num_frames = sample.shape[0]
        num_joints = sample.shape[1]
        
        # Define skeleton (SMPL-H 22 joints)
        skeleton = []
        if draw_skeleton:
            if num_joints == 22 or num_joints >= 22:
                # Basic SMPL skeleton (first 22 joints)
                skeleton = [
                    (0, 1), (0, 2), (0, 3), # Pelvis to L/R Hip, Spine1
                    (1, 4), (2, 5), (3, 6), # Hip to Knee, Spine2
                    (4, 7), (5, 8), (6, 9), # Knee to Ankle, Spine3
                    (7, 10), (8, 11), (9, 12), # Ankle to Foot, Neck
                    (12, 13), (12, 14), # Neck to L/R Shoulder
                    (12, 15), # Neck to Head
                    (13, 16), (14, 17), # Shoulder to Elbow
                    (16, 18), (17, 19), # Elbow to Wrist
                    (18, 20), (19, 21), # Wrist to L/R Hand
                ]
            
            # Add fingers if present (52 joints)
            if num_joints == 52:
                # Add simplified finger connections if needed
                pass

        # Prepare rendering
        images = []
        
        # Set up matplotlib figure
        dpi = 100
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        canvas = FigureCanvasAgg(fig)
        
        # Calculate global bounds for consistent scaling
        min_vals = sample.min(axis=(0, 1))
        max_vals = sample.max(axis=(0, 1))
        center = (min_vals + max_vals) / 2
        max_range = (max_vals - min_vals).max() / (2.0 / scale)

        pbar = comfy.utils.ProgressBar(num_frames)

        for f in range(num_frames):
            fig.clf()
            ax = fig.add_subplot(111, projection='3d')
            
            # Hide axes
            ax.set_axis_off()
            
            # Set angles
            ax.view_init(elev=elevation, azim=azimuth)
            
            # Set limits
            ax.set_xlim3d([center[0] - max_range, center[0] + max_range])
            ax.set_ylim3d([center[1] - max_range, center[1] + max_range])
            ax.set_zlim3d([center[2] - max_range, center[2] + max_range])
            
            # Draw joints
            joints = sample[f]
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='blue', s=10)
            
            # Draw skeleton
            if draw_skeleton:
                for j1, j2 in skeleton:
                    if j1 < num_joints and j2 < num_joints:
                        ax.plot([joints[j1, 0], joints[j2, 0]], 
                                [joints[j1, 1], joints[j2, 1]], 
                                [joints[j1, 2], joints[j2, 2]], c='red', linewidth=1)

            # Render to buffer
            canvas.draw()
            buf = canvas.buffer_rgba()
            img = np.asarray(buf)[:, :, :3] # RGBA to RGB
            
            # Convert to torch tensor [H, W, 3] and normalize to 0-1
            img_tensor = torch.from_numpy(img).float() / 255.0
            images.append(img_tensor)
            
            pbar.update(1)
            
        plt.close(fig)
        
        # Stack into [F, H, W, 3]
        return (torch.stack(images),)

NODE_CLASS_MAPPINGS = {
    "HYMotionPreview3D": HYMotionPreview3D,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYMotionPreview3D": "HY-Motion 3D Preview Animation",
}
