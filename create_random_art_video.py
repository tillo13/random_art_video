import json
import os
import time
import subprocess
from PIL import Image, ImageEnhance
import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import numpy as np
import random
from datetime import datetime

# GLOBAL VARIABLES
MOVIE_LENGTH = 3  # Movie duration in seconds
FPS = 24  # Frames per second
OUTPUT_FOLDER = "generated_frames"
PROMPT = "A highly detailed psychedelic portrait of a person"
MODEL_NAME = "digiplay/Photon_v1"

# Frame Generation Parameters
INITIAL_IMAGE = None  # Initial image for the first frame, None will create a white image
STRENGTH = 0.8  # Lower strength to keep changes minimal
GUIDANCE_SCALE = 7.5  # Higher to maintain details
NUM_INFERENCE_STEPS = 3  # Increased for higher detail and coherence
INITIAL_SEED = 42  # Seed for randomness

# Generate a timestamped output video filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_VIDEO = f"{timestamp}_output_art_movie.mp4"

def apply_perturbations(image):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1 + random.uniform(-0.02, 0.02))
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1 + random.uniform(-0.02, 0.02))
    return image

def blend_images(prev_img, curr_img, alpha=0.05):
    prev_array = np.array(prev_img)
    curr_array = np.array(curr_img)
    blended_array = (alpha * curr_array + (1 - alpha) * prev_array).astype(np.uint8)
    return Image.fromarray(blended_array)

def generate_frame(pipe, prompt, prev_image, seed, strength, guidance_scale, num_inference_steps):
    torch.manual_seed(seed)
    if prev_image is None:
        init_image = Image.new('RGB', (512, 512), color='white') if INITIAL_IMAGE is None else Image.open(INITIAL_IMAGE)
        transformed_image = pipe(prompt=prompt,
                                 init_image=init_image,
                                 strength=strength,
                                 guidance_scale=guidance_scale,
                                 num_inference_steps=num_inference_steps).images[0]
    else:
        perturbed_image = apply_perturbations(prev_image)
        transformed_image = pipe(prompt=prompt,
                                 init_image=perturbed_image,
                                 strength=strength,
                                 guidance_scale=guidance_scale,
                                 num_inference_steps=num_inference_steps).images[0]
        transformed_image = blend_images(prev_image, transformed_image, alpha=0.05)
    return transformed_image

def generate_frames(output_folder, duration_seconds, fps, prompt, model_name):
    os.makedirs(output_folder, exist_ok=True)
    total_frames = int(duration_seconds * fps)

    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(model_name)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # Try using CUDA if available
    if torch.cuda.is_available():
        pipe.to("cuda")
        print("Using CUDA for computation.")
    else:
        pipe.to("cpu")
        print("CUDA is not available. Using CPU for computation.")

    prev_image = None
    strength = STRENGTH
    guidance_scale = GUIDANCE_SCALE
    num_inference_steps = NUM_INFERENCE_STEPS
    seed = INITIAL_SEED

    frame_count = 0
    start_time = time.time()
    for i in range(total_frames):
        transformed_image = generate_frame(pipe, prompt, prev_image, seed, strength, guidance_scale, num_inference_steps)
        prev_image = transformed_image
        frame_path = os.path.join(output_folder, f"frame_{frame_count:03d}.png")
        transformed_image.save(frame_path)
        frame_count += 1
        seed += 1  # Change the seed for each frame to introduce slight variations

        # Calculate and print the estimated time remaining
        elapsed_time = time.time() - start_time
        average_time_per_frame = elapsed_time / frame_count
        remaining_frames = total_frames - frame_count
        estimated_time_remaining = average_time_per_frame * remaining_frames
        print(f"Generated frame {frame_count}/{total_frames} - Estimated time remaining: {estimated_time_remaining:.2f} seconds")

    # Call ffmpeg to create the video
    create_video_from_frames(output_folder, fps, OUTPUT_VIDEO)

def create_video_from_frames(output_folder, framerate, output_video):
    ffmpeg_command = [
        "ffmpeg",
        "-framerate", str(framerate),
        "-i", os.path.join(output_folder, "frame_%03d.png"),
        "-c:v", "libx264",
        "-r", "30",
        "-pix_fmt", "yuv420p",
        output_video
    ]

    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"Video saved as {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e}")

def main():
    generate_frames(OUTPUT_FOLDER, MOVIE_LENGTH, FPS, PROMPT, MODEL_NAME)

if __name__ == "__main__":
    main()