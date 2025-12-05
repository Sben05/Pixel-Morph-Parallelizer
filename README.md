# Parallel Pixel Morphing (CS 358)

**Author:** Shreeniket Bendre
**Course:** CS 358 (Parallel Computing)
**Date:** December 2, 2025

---

## Table of contents

* [1. What this project is](#1-what-this-project-is)
* [2. What’s in this repository](#2-whats-in-this-repository)
* [3. Quick start](#3-quick-start)
* [4. How to run (CLI)](#4-how-to-run-cli)
* [5. How to run (Streamlit UI)](#5-how-to-run-streamlit-ui)
* [6. The core problem: bijective pixel assignment](#6-the-core-problem-bijective-pixel-assignment)
* [7. Pipeline overview](#7-pipeline-overview)
* [8. Target importance (why ordering matters)](#8-target-importance-why-ordering-matters)
* [9. Sequential algorithm (greedy + candidate restriction)](#9-sequential-algorithm-greedy--candidate-restriction)
* [10. Parallel algorithm (hue segmentation)](#10-parallel-algorithm-hue-segmentation)
* [11. Merge/repair step (enforcing global bijection)](#11-mergerepair-step-enforcing-global-bijection)
* [12. Rendering the morph](#12-rendering-the-morph)
* [13. Performance intuition](#13-performance-intuition)
* [14. Parameter guide](#14-parameter-guide)
* [15. Inspiration link](#15-inspiration-link)

---

## 1. What this project is

This project generates a morphing GIF by *rearranging* pixels from a **source** image into a **target** image.
Unlike a crossfade (which blends colors), this is a particle-style morph:

* Each source pixel (or sampled pixel) becomes a particle.
* Each particle travels from its original location to a destination location.
* The final frame resembles the target image because particles end at target positions.

The difficult part is computing a **good assignment** between source particles and target pixels under a strict constraint:

**Bijection constraint (one-to-one mapping):**

* Each target position must get exactly one source particle.
* No source particle can be used twice.

That constraint is what makes the problem nontrivial and what complicates parallelization.

---

## 2. What’s in this repository

Files currently present (core):

* `pixel_morph_greedy_rgb.py`
  Sequential baseline greedy matching.

* `pixel_morph_greedy_rgb_parallel.py`
  Parallel hue-segmented greedy matching with merge/repair to enforce a global bijection.

* `pixel_morph_visualizer.py`
  Streamlit “Pixel Morph Lab” explorer.

* `pixel_morph_class_demo.py`
  Pygame demo/visualization.

Example assets in the repo:

* `bread.jpg`, `Obama.jpg`
* `morph.gif`

You can use any `.jpg/.jpeg/.png` as input.

---

## 3. Quick start

1. Create a virtual environment and install requirements.
2. Run either:

   * The sequential script for a baseline morph
   * The parallel script for the hue-segmented morph
   * The Streamlit UI to experiment interactively

Suggested `requirements.txt`:

```
numpy
pillow
matplotlib
streamlit
pygame
```

Install:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. How to run (CLI)

All commands assume you are in the directory containing the scripts and the images.

### 4.1 Sequential baseline

```bash
python pixel_morph_greedy_rgb.py \
  --src bread.jpg --tgt Obama.jpg \
  --size 256 --n 25000 \
  --frames 200 --fps 10 \
  --importance combo --w_luma 0.45 --w_center 0.45 --w_edge 0.10 \
  --k_init 64 --k_max 2048 \
  --bg black --blur 0.0 \
  --out seq.gif
```

### 4.2 Parallel hue-segmented version

```bash
python pixel_morph_greedy_rgb_parallel.py \
  --src bread.jpg --tgt Obama.jpg \
  --size 256 --n 25000 \
  --frames 200 --fps 10 \
  --jobs 4 \
  --importance combo --w_luma 0.45 --w_center 0.45 --w_edge 0.10 \
  --k_init 64 --k_max 2048 \
  --bg black --blur 0.0 \
  --out par.gif
```

---

## 5. How to run (Streamlit UI)

```bash
streamlit run pixel_morph_visualizer.py
```

The UI is a parameter lab: you can try different `size`, `n`, importance weights, candidate settings, and (for the parallel version) `jobs`.

---

## 6. The core problem: bijective pixel assignment

Let:

* `N` = number of source particles (pixels used from the source)
* `N` = number of target positions (pixels used from the target)

We want a mapping:

* `f(target_position) -> source_particle`

subject to:

* Each target gets one source
* Each source used at most once

If we ignore the one-to-one constraint, each target can independently pick the closest color match.
But with bijection, assignments interact: using a “perfect” source for one target prevents using it elsewhere.

A globally optimal solution resembles a large assignment problem, which becomes computationally expensive at image scale. This project uses a greedy strategy plus heuristics that make the runtime practical while still producing visually strong results.

---

## 7. Pipeline overview

Both the sequential and parallel versions follow the same overall structure:

1. **Preprocess**

   * Center-crop both images to a square
   * Resize to `size x size` (bicubic)
2. **(Optional) Sample**

   * If `--n > 0`, sample `n` particles from the full pixel grid
   * If `--n 0`, use all pixels
3. **Score target importance**

   * Compute an importance score for each target pixel
   * Sort targets from most important to least
4. **Greedy matching**

   * Assign targets in that order
   * Each target chooses the best available source among a restricted candidate set
5. **Parallelization (parallel version only)**

   * Partition sources/targets into hue segments across `--jobs` workers
   * Worker-level greedy assignments
   * Merge/repair to enforce a single global bijection
6. **Render**

   * Animate particles from start to assigned end positions
   * Save frames as a GIF

---

## 8. Target importance (why ordering matters)

Greedy algorithms depend heavily on processing order. If you assign random background pixels first, you can waste good sources early and leave important structure under-resourced later.

This project prioritizes target pixels using a score built from three signals:

* **Luma**: brightness structure helps preserve contrast and recognizability.
* **Center bias**: many images have salient content near the center.
* **Edge strength**: edges represent structure (boundaries, outlines, facial features).

For `--importance combo`, the score is a weighted sum:

* `score = w_luma * luma + w_center * center + w_edge * edge`

Targets are sorted by descending `score`. This concentrates matching effort on high-impact pixels early.

---

## 9. Sequential algorithm (greedy + candidate restriction)

### 9.1 Why naive greedy is too slow

Naively, for each target you could scan all available sources to find the closest match:

* `N` targets × `N` sources = `O(N^2)` comparisons

At pixel scale, that quickly becomes too expensive.

### 9.2 Candidate restriction (kNN-style)

Instead of searching all sources, the algorithm searches a smaller candidate set:

* Start with `k = --k_init`
* For the current target, consider only those `k` candidates
* If all candidates are already used, increase `k` and retry
* Stop expanding at `--k_max`

This behaves like a kNN-style approximation: “search a neighborhood of likely matches” rather than “search everything.”

### 9.3 What “best match” means

The matching metric is primarily color-driven (RGB-based in the baseline), plus whatever additional structure is embedded by:

* the target importance ordering
* the candidate restriction logic
* the downstream rendering (which rewards coherent assignment patterns visually)

Even when the mapping is approximate, the morph can look very good because:

* early match decisions focus on salient target pixels
* multiple good-enough matches exist for many pixels
* strict bijection keeps the particle field well-distributed

---

## 10. Parallel algorithm (hue segmentation)

Parallelizing greedy bijection matching is hard because different workers can:

* attempt to use the same source particle
* produce partial maps that do not combine cleanly

A naive parallel split (e.g., split by spatial blocks) tends to create many collisions and poor visual coherence.

This project uses **hue segmentation**:

1. Convert pixel colors to a hue-based representation.
2. Sort source particles by hue.
3. Sort target pixels by hue.
4. Split both sorted lists into `--jobs` contiguous hue segments.
5. Run greedy matching in each segment in parallel.

Why hue is effective:

* Pixels with similar hue are more likely to be good mutual matches.
* Segmenting by hue reduces contention across workers.
* Each worker solves a subproblem that is closer to “color-local,” producing more stable partial assignments.

---

## 11. Merge/repair step (enforcing global bijection)

Even with hue segmentation, conflicts can still occur:

* Two segments can attempt to use the same source particle.
* Some targets may remain unassigned after segment-level matching.
* Some sources may remain unused.

The parallel version therefore includes a merge/repair stage that:

* merges worker outputs into one mapping
* resolves duplicates (source collisions)
* fills unassigned targets using remaining available sources
* guarantees the final mapping is a valid global bijection

This step is essential: correctness is not “optional.” The global one-to-one constraint must hold.

---

## 12. Rendering the morph

Once the assignment is computed:

* Each particle has a start position (source pixel position).
* Each particle has an end position (assigned target pixel position).

For each frame:

* Interpolate particle positions from start to end over `--frames`.
* Draw (“splat”) particles into a frame buffer.
* Apply background and any configured post effects.
* Save frames and compile into a GIF at `--fps`.

Rendering controls:

* `--frames` controls smoothness.
* `--fps` controls playback speed.
* `--bg` sets background.
* `--blur` can soften artifacts depending on settings.

---

## 13. Performance intuition

This project is fast enough to run at practical scales because:

* It avoids full all-pairs matching by restricting candidates (`k_init` to `k_max`).
* It processes targets in an order that increases perceived quality early.
* The parallel version partitions work by hue to reduce cross-worker interference.
* Merge/repair enforces bijection without requiring a full global optimal assignment solve.

The sequential baseline is included as:

* a correctness reference
* a performance baseline
* a quality comparison point

---

## 14. Parameter guide

### Core scaling parameters

* `--size`
  Output resolution. Larger values increase quality but increase compute cost.

* `--n`
  Number of particles.

  * `--n 0` uses all pixels.
  * Smaller values speed up matching and rendering.

### Quality / structure parameters

* `--importance` and weights (`--w_luma`, `--w_center`, `--w_edge`)
  Controls which target pixels get matched early.

### Candidate search parameters

* `--k_init`
  Starting candidate count per target.

* `--k_max`
  Maximum candidate count used when a target cannot find an available source in smaller sets.

### Parallel parameter

* `--jobs`
  Number of hue segments / parallel workers (parallel script only).
  Higher values increase parallelism but can increase merge/repair overhead.

### Animation parameters

* `--frames`
  More frames = smoother motion.

* `--fps`
  Playback speed.

---

## 15. Inspiration link

[https://youtube.com/shorts/MeFi68a2pP8?si=xkXPHo609b_GO8Yz](https://youtube.com/shorts/MeFi68a2pP8?si=xkXPHo609b_GO8Yz)
