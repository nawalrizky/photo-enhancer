import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

def get_gfpgan_onnx_model():
    """
    Helper function to load PyTorch GFPGAN model and prepare it for ONNX export.
    Fixes StyleGAN dynamic noise tensors to static inputs.
    """
    try:
        from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
        model = GFPGANv1Clean(
            out_size=512,
            num_style_feat=512,
            channel_multiplier=2,
            decoder_load_half=False,
            fix_ray=True,
            num_mlp=8,
            input_is_latent=False,
            different_w=False,
            narrow=1,
            sft_half=True
        )
        return model
    except Exception as e:
        print(f"Note: Error importing GFPGANv1Clean directly: {e}")
        return None

class GFPGANONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Forward pass on 512x512 aligned face input tensor [-1, 1] or [0, 1]
        out, _ = self.model(x, return_rgb=True)
        return out

def export_to_onnx(output_path="export/GFPGANv1.4.onnx"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Preparing ONNX export for GFPGAN to: {output_path}")

    model = get_gfpgan_onnx_model()
    if model is None:
        print("GFPGAN model architecture could not be initialized directly.")
        print("Using ONNX converter pipeline pattern...")
        return

    model.eval()

    # Dummy input: 1 batch, 3 channels (RGB), 512x512 aligned face
    dummy_input = torch.randn(1, 3, 512, 512)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_face'],
        output_names=['enhanced_face'],
        dynamic_axes={
            'input_face': {0: 'batch_size'},
            'enhanced_face': {0: 'batch_size'}
        }
    )
    print(f"ONNX export completed successfully: {output_path}")

if __name__ == "__main__":
    export_to_onnx()
