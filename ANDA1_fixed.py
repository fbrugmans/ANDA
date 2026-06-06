"""
ANDA1.py — first exploratory active-nematic simulation on a spherical surface.

Purpose
-------
This script reproduces the first qualitative simulation used in ANDA:
a near-uniform nematic orientation field on a closed curved surface evolves
towards defect-containing configurations.

How to run in Jupyter Lab
-------------------------
Option A:
    %run ANDA1_fixed.py

Option B:
    Open this file in Jupyter Lab and run the cells / code directly.

The script both saves and displays the generated figure.
"""

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Parameters
# -----------------------------
alpha = 1.0
beta = 1.0
kappa = 0.3
D0 = 0.5
dt = 0.01
steps = 2000
epsilon = 1e-8

N = 600
k_neighbors = 6
random_seed = 7

output_figure = "ANDA1_dynamic_defects.png"


# -----------------------------
# Generate quasi-uniform sphere points
# Fibonacci sphere sampling
# -----------------------------
def fibonacci_sphere(samples: int) -> np.ndarray:
    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(samples):
        y = 1.0 - (i / float(samples - 1)) * 2.0
        radius = np.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])

    return np.asarray(points, dtype=float)


points = fibonacci_sphere(N)


# -----------------------------
# Neighbourhood definition
# -----------------------------
def compute_neighbors(points: np.ndarray, k: int = 6) -> list[np.ndarray]:
    neighbours = []

    for i in range(len(points)):
        distances = np.linalg.norm(points - points[i], axis=1)
        idx = np.argsort(distances)[1 : k + 1]
        neighbours.append(idx)

    return neighbours


neighbors = compute_neighbors(points, k=k_neighbors)


# -----------------------------
# Initial field: near-uniform orientation plus small perturbation
# -----------------------------
rng = np.random.default_rng(random_seed)

# Nematic order parameter: psi = A exp(2 i phi)
# Starting near uniform means phi is close to zero everywhere.
phi0 = 0.1 * rng.standard_normal(N)
psi = np.exp(2j * phi0)

# Spatially varying elastic alignment strength.
# Here this is a simple pole-to-equator gradient used only for qualitative demonstration.
D = D0 * (1.0 - points[:, 1] ** 2)


# -----------------------------
# Graph Laplacian on Fibonacci sphere
# -----------------------------
def laplacian(field: np.ndarray) -> np.ndarray:
    lap = np.zeros_like(field, dtype=complex)

    for i in range(N):
        lap[i] = np.mean(field[neighbors[i]] - field[i])

    return lap


# -----------------------------
# Time integration: explicit Euler with phase normalisation
# -----------------------------
for _ in range(steps):
    lap = laplacian(psi)

    F = (
        alpha * psi
        - beta * np.abs(psi) ** 2 * psi
        + D * lap
        + 1j * kappa * lap
    )

    psi = psi + dt * F

    # Phase-normalised exploratory implementation:
    # this suppresses amplitude dynamics and keeps the simulation focused on orientation.
    psi = psi / (np.abs(psi) + epsilon)


# -----------------------------
# Defect proxy and plotting
# -----------------------------
phase = np.angle(psi) / 2.0
amplitude = np.abs(psi)

# A simple defect proxy: points with largest local phase variation.
local_variation = np.zeros(N)
for i in range(N):
    # Nematic phase differences should respect pi-periodicity.
    diff = phase[neighbors[i]] - phase[i]
    diff = (diff + np.pi / 2.0) % np.pi - np.pi / 2.0
    local_variation[i] = np.mean(np.abs(diff))

defect_idx = np.argsort(local_variation)[-12:]


fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    points[:, 0],
    points[:, 1],
    points[:, 2],
    c=phase,
    s=18,
)

ax.scatter(
    points[defect_idx, 0],
    points[defect_idx, 1],
    points[defect_idx, 2],
    marker="o",
    s=90,
    facecolors="none",
    linewidths=1.5,
)

ax.set_title("ANDA1: phase-normalised active-nematic field on a sphere")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=22, azim=35)

cbar = fig.colorbar(scatter, ax=ax, shrink=0.75)
cbar.set_label("nematic orientation angle φ")

plt.tight_layout()
plt.savefig(output_figure, dpi=200, bbox_inches="tight")

print(f"Saved figure to: {output_figure}")
print(f"Detected high-variation proxy points: {len(defect_idx)}")
print("For Jupyter Lab: if the plot is not shown automatically, run plt.show() in the next cell.")

plt.show()
