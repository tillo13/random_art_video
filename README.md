# Art Video Generator

This repository contains a script to generate a psychedelic art video using the Stable Diffusion model. The script creates a series of frames and compiles them into a video file. It is designed to be highly configurable through global variables.

## Requirements

- Python 3.7+
- Required Python packages:
  - `torch`
  - `diffusers`
  - `PIL`
  - `numpy`

You can install the required packages using pip:

```bash
pip install torch diffusers pillow numpy
```

- **FFmpeg** (for creating the video)

## Running the Script

To run the script, simply execute the `art_video_generator.py` file. The script will generate frames and compile them into a video file.

```bash
python art_video_generator.py
```

## Configuration

You can configure the script by changing the global variables at the top of `art_video_generator.py`. The available variables and their default values are:

```python
# MOVIE SETTINGS
MOVIE_LENGTH = 30  # Movie duration in seconds
FPS = 24  # Frames per second

# OUTPUT SETTINGS
OUTPUT_FOLDER = "generated_frames"
OUTPUT_VIDEO = "output_art_video.mp4"  # Will be prefixed with the current timestamp

# GENERATION SETTINGS
PROMPT = "A highly detailed psychedelic portrait of a person"
MODEL_NAME = "digiplay/Photon_v1"
INITIAL_IMAGE = None  # Initial image for the first frame, None will create a white image
STRENGTH = 0.8  # Lower strength to keep changes minimal
GUIDANCE_SCALE = 7.5  # Higher to maintain details
NUM_INFERENCE_STEPS = 3  # Increased for higher detail and coherence
INITIAL_SEED = 42  # Seed for randomness
```

By changing these variables, you can modify the length and frame rate of the video, the output folder and filename, and the specific parameters used by the Stable Diffusion model to generate each frame.

## Usage Notes

- The script attempts to use CUDA for computations but will fall back to CPU if CUDA is not available.
- Each run of the script generates a uniquely named output video file, prefixed with the current timestamp (YYYYMMDD_HHMMSS).

## Example

Here's an example usage of the script:

```python
# Set custom values for the global variables here if desired
MOVIE_LENGTH = 60  # 60 seconds
FPS = 30  # 30 frames per second
PROMPT = "A serene landscape with mountains and a river"
MODEL_NAME = "digiplay/Photon_v2"
```

By modifying the global variables in this way, you can customize the behavior of the script to generate different types of videos.