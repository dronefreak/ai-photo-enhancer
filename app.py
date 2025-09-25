# app.py
import os
import torch
import warnings

# Suppress torchvision deprecation warnings
warnings.filterwarnings("ignore", module="torchvision")

from realesrgan import RealESRGANer
from gfpgan import GFPGANer
from PIL import Image
import numpy as np
import gradio as gr
import cv2
from gradio_imageslider import ImageSlider
# Device
device = torch.device('cpu')

# Global enhancers
upsampler = None
face_enhancer = None

from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def initialize_models():
    global upsampler, face_enhancer

    # Load SRVGGNetCompact model for realesr-general-x4v3
    model = SRVGGNetCompact(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=32,
        upscale=4,
        act_type='prelu'
    )

    upsampler = RealESRGANer(
        scale=4,
        model_path="models/realesr-general-x4v3.pth",
        model=model,
        half=False,
        device=device
    )

    # GFPGAN for face enhancement
    face_enhancer = GFPGANer(
        model_path="models/GFPGANv1.4.pth",
        upscale=4,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler,
        device=device
    )

    return upsampler, face_enhancer


def enhance_image(image_array, scale=2, enhance_faces=True, progress=gr.Progress()):
    if image_array is None:
        return None

    progress(0, "Starting enhancement...")

    # Keep original as PIL RGB
    original_pil = Image.fromarray(image_array)

    temp_input = "temp_input.jpg"
    try:
        original_pil.save(temp_input)
        progress(0.2, "Image saved temporarily")

        img = cv2.imread(temp_input, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Failed to load image!")

        progress(0.3, "Running GFPGAN face enhancement...")

        _, _, output_img = face_enhancer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )
        progress(0.8, "Post-processing...")

        # Convert BGR → RGB → PIL
        enhanced_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        enhanced_image = Image.fromarray(enhanced_rgb)

        # Downscale if needed
        if scale == 2:
            w, h = enhanced_image.size
            enhanced_image = enhanced_image.resize((w // 2, h // 2), Image.LANCZOS)

        # ✅ RETURN TUPLE: (input, output) for ImageSlider
        return (original_pil, enhanced_image)

    except Exception as e:
        print(f"Error during enhancement: {e}")
        return (original_pil, placeholder_image("Error"))
    finally:
        if os.path.exists(temp_input):
            os.remove(temp_input)


def process_batch(files, scale=2, enhance_faces=True, progress=gr.Progress()):
    if not files:
        return []

    results = []
    total = len(files)

    for idx, file in enumerate(files):
        progress(idx / total, f"Processing image {idx + 1}/{total}...")
        try:
            img = np.array(Image.open(file.name))
            enhanced = enhance_image(img, scale=scale, enhance_faces=enhance_faces, progress=lambda p, s: None)
            results.append(enhanced if enhanced else placeholder_image())
        except Exception as e:
            print(f"Failed to process {file.name}: {e}")
            results.append(placeholder_image("Error"))

    return results


def placeholder_image(text="Failed"):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (256, 256), color='gray')
    draw = ImageDraw.Draw(img)
    draw.text((50, 120), text, fill="red")
    return img


# Initialize models
initialize_models()

# Gradio Interface
with gr.Blocks(title="📸 AI Photo Enhancer") as demo:
    gr.Markdown("# 📸 AI Photo Enhancer\n*Super Resolution + Face Restoration*")

    with gr.Tabs():
      with gr.Tab("Single Image"):
        with gr.Row():
          with gr.Column():
              input_img = gr.Image(type="numpy", label="Upload Image")
              scale_slider = gr.Slider(minimum=2, maximum=4, step=2, value=2, label="Scale (2x/4x)")
              face_toggle = gr.Checkbox(value=True, label="Enhance Faces")
              btn = gr.Button("Enhance Photo", variant="primary")

          with gr.Column():
              # ✅ Draggable slider view using ImageSlider
              comparison_output = ImageSlider(
                  label="Before (left) vs After (right) – drag the center bar!",
                  height=512  # Optional: set fixed height
              )

        btn.click(
          fn=enhance_image,
          inputs=[input_img, scale_slider, face_toggle],
          outputs=comparison_output  # ← Now returns HTML
        )

        with gr.Tab("Batch Processing"):
            gr.Markdown("Upload multiple images for batch enhancement.")
            batch_input = gr.File(file_count="multiple", label="Upload Images")
            batch_scale = gr.Slider(minimum=2, maximum=4, step=2, value=2, label="Scale")
            batch_faces = gr.Checkbox(value=True, label="Enhance Faces")
            batch_btn = gr.Button("Enhance All")
            batch_output = gr.Gallery(label="Enhanced Images")

            batch_btn.click(
                fn=process_batch,
                inputs=[batch_input, batch_scale, batch_faces],
                outputs=batch_output
            )

    gr.Markdown("✨ Powered by Real-ESRGAN & GFPGAN | Built with Gradio")


# ✅ Enable queuing globally — this enables progress bars!
demo.queue()  # ← Must be called on the Blocks object before launch

if __name__ == "__main__":
    demo.launch()
