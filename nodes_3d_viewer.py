import os
import json
import torch
import numpy as np
import folder_paths
from .nodes_modular import HYMotionData

class HYMotion3DViewer:
    """
    Serializes motion data for the custom 3D frontend viewer.
    Supports up to 5 motion inputs for side-by-side comparison.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_0": ("HYMOTION_DATA", {
                    "tooltip": "Primary motion data input from HYMotionSampler"
                }),
            },
            "optional": {
                "motion_1": ("HYMOTION_DATA", {
                    "tooltip": "Optional second motion for side-by-side comparison"
                }),
                "motion_2": ("HYMOTION_DATA", {
                    "tooltip": "Optional third motion for comparison"
                }),
                "motion_3": ("HYMOTION_DATA", {
                    "tooltip": "Optional fourth motion for comparison"
                }),
                "motion_4": ("HYMOTION_DATA", {
                    "tooltip": "Optional fifth motion for comparison"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("viewer_data",)
    FUNCTION = "serialize_data"
    CATEGORY = "HY-Motion/view"
    OUTPUT_NODE = True

    def serialize_data(self, motion_0, motion_1=None, motion_2=None, motion_3=None, motion_4=None):
        input_motions = [motion_0, motion_1, motion_2, motion_3, motion_4]
        serialized_motions = []
        
        spacing = 1.5 # 1.5 meters between skeletons
        total_samples = 0
        max_samples = 10 # Limit to 10 total skeletons for performance

        for i, m in enumerate(input_motions):
            if m is None:
                continue
            
            output_dict = m.output_dict
            batch_size = m.batch_size
            
            for b in range(batch_size):
                if total_samples >= max_samples:
                    print(f"[HY-Motion] Warning: Max samples ({max_samples}) reached. Skipping others.")
                    break
                    
                # Check availability of required data
                if "transl" not in output_dict:
                    continue

                # Extract keypoints (T, J, 3)
                keypoints = None
                if "keypoints3d" in output_dict:
                    keypoints = output_dict["keypoints3d"][b].cpu().numpy().tolist()
                
                # Extract translation (T, 3)
                transl = output_dict["transl"][b].cpu().numpy().tolist()
                
                # Distinct colors for each sample to avoid confusion
                sample_colors = [0x3366ff, 0xff6633, 0x22ff88, 0xff2244, 0xaa22ff, 0xffaa22]
                
                serialized_motions.append({
                    "id": f"input_{i}_batch_{b}",
                    "label": f"Input {i} (Sample {b})",
                    "posX": total_samples * spacing,
                    "color": sample_colors[total_samples % len(sample_colors)],
                    "text": m.text,
                    "duration": m.duration,
                    "keypoints": keypoints,
                    "transl": transl,
                })
                total_samples += 1

        # Return as JSON string
        result = json.dumps(serialized_motions)
        return {"ui": {"motions": result}, "result": (result,)}

class HYMotion3DModelLoader:
    """
    Standalone viewer for 3D models (FBX, GLB, GLTF, OBJ).
    Scans both input and output directories.
    """
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        output_dir = folder_paths.get_output_directory()
        
        files = []
        extensions = [".fbx", ".glb", ".gltf", ".obj"]
        
        def scan_dir(base_path, prefix=""):
            if not os.path.exists(base_path): return
            for root, _, filenames in os.walk(base_path):
                for f in filenames:
                    if any(f.lower().endswith(ext) for ext in extensions):
                        rel = os.path.relpath(os.path.join(root, f), base_path)
                        files.append(f"{prefix}{rel}")

        scan_dir(input_dir, "input/")
        scan_dir(output_dir, "output/")
        
        if not files: files = ["none"]
        else: files = sorted(files)

        return {
            "required": {
                "model_path": (files, {
                    "default": files[0],
                    "tooltip": "Select a 3D model to view immediately"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_url",)
    FUNCTION = "load_model"
    CATEGORY = "HY-Motion/view"
    OUTPUT_NODE = True

    def load_model(self, model_path):
        if model_path == "none":
            return (None,)
        
        ext = os.path.splitext(model_path)[1].lower()[1:]
        return {"ui": {"model_url": model_path, "format": ext}, "result": (model_path,)}

class HYMotionFBXPlayer:
    """
    (Legacy) Specific player for FBX files in the output directory.
    Useful for existing workflows or dedicated FBX playback.
    """
    @classmethod
    def INPUT_TYPES(s):
        output_dir = folder_paths.get_output_directory()
        fbx_files = []
        if os.path.exists(output_dir):
            for root, _, files in os.walk(output_dir):
                for f in files:
                    if f.lower().endswith(".fbx"):
                        rel = os.path.relpath(os.path.join(root, f), output_dir)
                        fbx_files.append(rel)
        fbx_files = ["none"] + sorted(fbx_files)

        return {
            "required": {
                "fbx_name": (fbx_files, {"default": fbx_files[0]}),
            },
            "optional": {
                "fbx_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fbx_path",)
    FUNCTION = "play_fbx"
    CATEGORY = "HY-Motion/view"
    OUTPUT_NODE = True

    def play_fbx(self, fbx_name, fbx_path=""):
        selected = fbx_path.strip() if fbx_path and fbx_path.strip() else fbx_name
        if selected == "none": return (None,)
        # Map to "output/" prefix for consistency with the generic loader's logic
        full_path = f"output/{selected}" if not selected.startswith("output/") else selected
        return {"ui": {"fbx_url": full_path}, "result": (full_path,)}

NODE_CLASS_MAPPINGS = {
    "HYMotion3DViewer": HYMotion3DViewer,
    "HYMotionFBXPlayer": HYMotionFBXPlayer,
    "HYMotion3DModelLoader": HYMotion3DModelLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYMotion3DViewer": "HY-Motion 3D Multi-Viewer",
    "HYMotionFBXPlayer": "HY-Motion 3D FBX Player (Legacy)",
    "HYMotion3DModelLoader": "HY-Motion 3D Model Loader"
}
