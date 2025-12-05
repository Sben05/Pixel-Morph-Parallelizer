# Parallel Pixel Morphing

**Author:** Shreeniket Bendre  
**Date:** December 2, 2025  

## Overview

This project generates a morphing GIF by rearranging pixels from a source image into a target image using a one-to-one assignment (a bijection). The expensive part is computing a high-quality assignment between many source “particles” and target pixels while enforcing the bijection constraint (no source pixel can be used twice).

This work was inspired by the following short; watching it is strongly recommended for intuition:
https://youtube.com/shorts/MeFi68a2pP8?si=xkXPHo609b_GO8Yz

High-level pipeline:

1. Preprocess  
   - Center-crop both images to a square and resize to `--size` (bicubic).
2. Sample (optional)  
   - Optionally subsample to `--n` particles for speed (or use all pixels if `--n 0`).
3. Prioritize targets  
   - Compute an importance score for each target pixel using luma / center bias / edge strength (or a weighted combination).
4. Greedy matching  
   - Assign target pixels in descending importance, choosing the best available source among a small candidate set (kNN-style).
5. Parallelization  
   - Sort both source and target sets by hue and split into `--jobs` hue segments; each worker runs the same greedy logic on its segment, followed by a final merge/repair step to enforce a global bijection.
6. Render  
   - Animate particles from start to end positions, splat to pixels, and save as a GIF.

## Repository Contents

Main files:

- `pixel_morph_greedy_rgb.py`  
  Sequential baseline (single-process greedy matching).

- `pixel_morph_greedy_rgb_parallel.py`  
  Parallel hue-segmented algorithm with merge/repair to enforce global bijection.

- `pixel_morph_visualizer.py`  
  Streamlit “Pixel Morph Lab” explorer for experimenting with parameters and running demos.

- `pixel_morph_class_demo.py`  
  Pygame class/demo visualization.

Example images/outputs in this repository:

- `bread.jpg`, `Obama.jpg`  
- `morph.gif`

## Requirements

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
