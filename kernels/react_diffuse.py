# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Gray-Scott Reaction-Diffusion on P300C Blackhole hardware.

Solves the two-species Gray-Scott PDE each frame:

  ∂U/∂t = Du·∇²U  -  U·V²  +  f·(1 − U)
  ∂V/∂t = Dv·∇²V  +  U·V²  −  (f+k)·V

where ∇²X is the 2-D discrete Laplacian, approximated by four averaging passes
using the existing smooth_height_map (eltwise-add) kernel:

  lap(X) ≈ avg(X, roll(X,+1,ew))   (east-west neighbour)
           + avg(X, roll(X,-1,ew))  (west-east neighbour)
           + avg(X, roll(X,+1,ns))  (north-south neighbour)
           + avg(X, roll(X,-1,ns))  (south-north neighbour)
           − 2·X                    (centre correction)

This makes 8 TT kernel calls per frame (4 per species), all on the same
smooth_height_map kernel — zero additional kernel code needed.

Gray-Scott parameters control the pattern morphology:
  Coral / branching  : f=0.0545  k=0.062
  Mitosis / spots    : f=0.0367  k=0.0649
  Fingerprints       : f=0.037   k=0.06
  Worms / solitons   : f=0.078   k=0.061
  Mazes              : f=0.029   k=0.057

The output V field (activator) is colorized to RGB and saved as PNG frames.
Run in a loop to produce live art as the FreeCiv game progresses.

Usage (standalone):
  source ~/code/tt-lang/build/env/activate
  python kernels/react_diffuse.py --frames 200 --preset coral --out /tmp/tt_art

Usage (server integration):
  from react_diffuse import ReactDiffuseModel
  model = ReactDiffuseModel(device)
  model.step(n=4)
  model.save_frame("/tmp/tt_art/frame_042.png")
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import ttnn

from height_map_smooth import smooth_height_map, zeros_like_on_device

# ── Constants ──────────────────────────────────────────────────────────────────

GRID = 256          # grid size — 256×256 → 64 Tensix cores, ~0.8ms per frame
DT   = 1.0          # time step (Grey-Scott dimensionless units)
STEPS_PER_FRAME = 8 # PDE steps between saved frames (more → faster evolution)

# Preset parameter library
PRESETS = {
    'coral':       {'f': 0.0545, 'k': 0.062,   'Du': 0.16, 'Dv': 0.08},
    'mitosis':     {'f': 0.0367, 'k': 0.0649,  'Du': 0.16, 'Dv': 0.08},
    'fingerprint': {'f': 0.037,  'k': 0.060,   'Du': 0.16, 'Dv': 0.08},
    'worms':       {'f': 0.078,  'k': 0.061,   'Du': 0.16, 'Dv': 0.08},
    'maze':        {'f': 0.029,  'k': 0.057,   'Du': 0.16, 'Dv': 0.08},
    'spirals':     {'f': 0.018,  'k': 0.051,   'Du': 0.16, 'Dv': 0.08},
}

# Color palette for V-field visualization (black → teal → white)
# Matches Tenstorrent brand palette
_PALETTE = [
    (0.0,  (15,  42,  53)),    # deep blue-gray  (V ≈ 0, background)
    (0.25, (26,  60,  71)),    # blue-gray        (low V)
    (0.5,  (79, 209, 197)),    # teal             (mid V)
    (0.75, (129,230, 217)),    # light teal       (high V)
    (1.0,  (232,240, 242)),    # near-white       (V ≈ 1, peak activator)
]


def _lerp_palette(v: float) -> tuple[int, int, int]:
    """Map V ∈ [0,1] to RGB via the Tenstorrent palette."""
    v = max(0.0, min(1.0, v))
    for i in range(len(_PALETTE) - 1):
        t0, c0 = _PALETTE[i]
        t1, c1 = _PALETTE[i + 1]
        if t0 <= v <= t1:
            w = (v - t0) / (t1 - t0)
            return tuple(int(c0[j] + w * (c1[j] - c0[j])) for j in range(3))
    return _PALETTE[-1][1]


# ── ReactDiffuseModel ──────────────────────────────────────────────────────────

class ReactDiffuseModel:
    """
    Stateful Gray-Scott simulation on TT hardware.

    Initialize once, then call step() each frame.  The U and V fields
    persist in CPU memory between frames; TT hardware handles the
    Laplacian and reaction terms.

    The Laplacian is approximated by four smooth_height_map passes:
      lap(X) ≈ avg_ew(X) + avg_we(X) + avg_ns(X) + avg_sn(X) − 2X
    where avg_ew(X) = 0.5*X + 0.5*roll(X,+1,dims=1)  (east-west avg)
    and the − 2X correction keeps the centre weight = 0 in the stencil.
    """

    def __init__(self, device, preset: str = 'coral', grid: int = GRID):
        self.device = device
        self.N      = grid
        p = PRESETS.get(preset, PRESETS['coral'])
        self.f  = p['f']
        self.k  = p['k']
        self.Du = p['Du']
        self.Dv = p['Dv']
        self.frame_idx = 0
        self._init_fields()

    def _init_fields(self):
        """Seed U=1 everywhere, V=0 except a small random-ish patch in centre."""
        N = self.N
        self.U = torch.ones(N, N, dtype=torch.float32)
        self.V = torch.zeros(N, N, dtype=torch.float32)

        # Seed multiple small square patches of (U≈0.5, V≈0.25) for richer start
        import random
        rng = random.Random(42)
        for _ in range(6):
            cx = rng.randint(N // 4, 3 * N // 4)
            cy = rng.randint(N // 4, 3 * N // 4)
            r  = rng.randint(4, 12)
            self.U[cy-r:cy+r, cx-r:cx+r] = 0.50
            self.V[cy-r:cy+r, cx-r:cx+r] = 0.25

    def _laplacian_tt(self, X: torch.Tensor) -> torch.Tensor:
        """
        Approximate 2-D Laplacian of X using smooth_height_map on TT hardware.

        The 5-point discrete Laplacian stencil:
          ∇²X[i,j] ≈ X[i,j-1] + X[i,j+1] + X[i-1,j] + X[i+1,j] − 4X[i,j]

        Rewritten as four neighbour averages minus the centre:
          ∇²X ≈ (avg_ew + avg_we + avg_ns + avg_sn) − 2X

        Each avg = smooth_height_map(X*0.5, roll(X)*0.5)
        Four smooth_height_map calls per field → 8 kernel calls for U+V.
        """
        N   = self.N
        dev = self.device
        Xh  = X.to(torch.bfloat16)     # bfloat16 for TT hardware

        lap = torch.zeros(N, N, dtype=torch.float32)

        for (shift, dim) in [(+1, 1), (-1, 1), (+1, 0), (-1, 0)]:
            a = (Xh * 0.5).to(torch.bfloat16)
            b = (torch.roll(Xh, shift, dims=dim) * 0.5).to(torch.bfloat16)
            a_t = ttnn.from_torch(a, dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=dev)
            b_t = ttnn.from_torch(b, dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=dev)
            out_t = zeros_like_on_device(a, dev)
            smooth_height_map(a_t, b_t, out_t)
            lap += ttnn.to_torch(out_t).float()

        # Centre correction: four averages each contribute 0.5*X,
        # total 2X — subtract to recover Laplacian stencil sum
        lap -= 2.0 * X
        return lap

    def step(self, n: int = STEPS_PER_FRAME) -> float:
        """
        Advance the PDE n steps on TT hardware.
        Returns total kernel time in milliseconds.
        """
        U, V = self.U, self.V
        f, k, Du, Dv, dt = self.f, self.k, self.Du, self.Dv, DT
        total_ms = 0.0

        for _ in range(n):
            t0 = time.perf_counter()

            lap_U = self._laplacian_tt(U)
            lap_V = self._laplacian_tt(V)

            UV2 = U * V * V          # reaction term (shared)

            U_new = U + dt * (Du * lap_U - UV2 + f * (1.0 - U))
            V_new = V + dt * (Dv * lap_V + UV2 - (f + k) * V)

            # Clamp to [0, 1] — Gray-Scott concentrations are bounded
            self.U = U_new.clamp(0.0, 1.0)
            self.V = V_new.clamp(0.0, 1.0)
            U, V   = self.U, self.V

            total_ms += (time.perf_counter() - t0) * 1000

        return total_ms

    def to_rgb(self) -> torch.Tensor:
        """
        Colorize V field → (N, N, 3) uint8 RGB tensor using Tenstorrent palette.
        """
        N = self.N
        V_np = self.V.numpy()
        rgb  = torch.zeros(N, N, 3, dtype=torch.uint8)
        for y in range(N):
            for x in range(N):
                r, g, b = _lerp_palette(float(V_np[y, x]))
                rgb[y, x, 0] = r
                rgb[y, x, 1] = g
                rgb[y, x, 2] = b
        return rgb

    def save_frame(self, path: str) -> None:
        """Save current V field as a colorized PNG."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow required: pip install Pillow")

        rgb = self.to_rgb()
        img = Image.fromarray(rgb.numpy(), mode="RGB")
        # Scale up for visibility (nearest-neighbour, no blur)
        img = img.resize((512, 512), Image.NEAREST)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        self.frame_idx += 1


# ── Standalone runner ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gray-Scott reaction-diffusion on P300C Blackhole hardware"
    )
    parser.add_argument("--preset",  default="coral",
                        choices=list(PRESETS),
                        help="Pattern preset (default: coral)")
    parser.add_argument("--frames",  type=int, default=120,
                        help="Number of PNG frames to generate (default: 120)")
    parser.add_argument("--steps",   type=int, default=STEPS_PER_FRAME,
                        help="PDE steps between frames (default: 8)")
    parser.add_argument("--grid",    type=int, default=GRID,
                        help="Grid size — must be tile-aligned ×32 (default: 256)")
    parser.add_argument("--out",     default="/tmp/tt_art",
                        help="Output directory for PNG frames")
    parser.add_argument("--all-presets", action="store_true",
                        help="Run all presets in sequence, 30 frames each")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Gray-Scott Reaction-Diffusion — TT-Lang on P300C Blackhole")
    print(f"Grid: {args.grid}×{args.grid}  "
          f"Preset: {args.preset}  "
          f"Frames: {args.frames}  "
          f"Steps/frame: {args.steps}")
    print(f"Output: {out_dir}/")
    print("=" * 64)
    print()

    device = ttnn.open_device(device_id=0)
    try:
        # Warm up smooth_height_map kernel for this grid size
        print("Warming up TT kernels...")
        model_wu = ReactDiffuseModel(device, preset=args.preset, grid=args.grid)
        model_wu.step(n=1)
        print("  Kernels compiled and cached.")
        print()

        presets_to_run = (list(PRESETS.keys()) if args.all_presets
                          else [args.preset])

        frame_idx = 0
        for preset_name in presets_to_run:
            model = ReactDiffuseModel(device, preset=preset_name, grid=args.grid)
            n_frames = 30 if args.all_presets else args.frames
            params   = PRESETS[preset_name]

            print(f"Preset: {preset_name:<12}  "
                  f"f={params['f']}  k={params['k']}")
            print(f"{'Frame':>6}  {'Step ms':>8}  {'fps':>6}  path")
            print("-" * 50)

            for i in range(n_frames):
                ms  = model.step(n=args.steps)
                fps = 1000.0 / ms if ms > 0 else 0

                fname = out_dir / f"frame_{frame_idx:05d}.png"
                model.save_frame(str(fname))
                frame_idx += 1

                if i % 10 == 0 or i == n_frames - 1:
                    print(f"  {i:4d}  {ms:8.2f}ms  {fps:6.1f}  {fname.name}")

            print()

        print(f"Done. {frame_idx} frames saved to {out_dir}/")
        print()
        print("To view as live slideshow:")
        print(f"  feh --slideshow-delay 0.05 --zoom fill {out_dir}/")
        print("  # or: mpv --no-audio --fps=20 mf://{}/frame_*.png".format(out_dir))

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
