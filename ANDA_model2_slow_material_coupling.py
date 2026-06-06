"""
ANDA_model2_slow_material_coupling.py

Jupyter-Lab-ready simulation for the proposed extended ANDA model:

Model 2 — Slow material-field coupling and mechanochemical lock-in
-----------------------------------------------------------------
This simulation is deliberately generic. It does NOT model scalp, skin disease,
or any specific tissue. It tests whether a fast orientational field can generate
defects/spirals and whether a slower scalar material field can accumulate near
orientational frustration, progressively reducing dynamical accessibility.

Run in Jupyter Lab:
    %run ANDA_model2_slow_material_coupling.py

Outputs:
    anda_model2_outputs/
        01_final_orientation_and_material.png
        02_timeseries_metrics.png
        03_snapshots_material_lockin.png
        04_summary_card.png
        model2_summary.json
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Output directory
# -----------------------------
outdir = Path("anda_model2_outputs")
outdir.mkdir(exist_ok=True)


# -----------------------------
# Parameters
# -----------------------------
N = 650
k_neighbors = 6
random_seed = 13

# Fast orientational field psi = A exp(2 i phi)
alpha = 1.0          # linear growth/order formation
beta = 1.0           # nonlinear saturation
kappa = 0.25         # chiral/rotational phase term
D0 = 0.45            # baseline orientational coupling
dt = 0.01
steps = 3000
epsilon = 1e-8

# Slow scalar material field h
# h represents generic material accumulation/stiffening/reduced accessibility.
tau_h = 0.08         # slow timescale multiplier: smaller = slower h dynamics
prod_rate = 2.2      # production from orientational frustration
decay_rate = 0.25    # turnover/relaxation
Dh = 0.025           # material-field smoothing
threshold = 0.018    # frustration threshold before accumulation
power = 1.4          # nonlinear selectivity of accumulation

# Coupling: h reduces effective orientational mobility/accessibility
lock_strength = 5.0

# Sampling for saved snapshots
snapshot_steps = [200, 800, 1600, 2999]


# -----------------------------
# Geometry: quasi-uniform sphere
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


def compute_neighbors(points: np.ndarray, k: int = 6) -> list[np.ndarray]:
    neighbors = []
    for i in range(len(points)):
        distances = np.linalg.norm(points - points[i], axis=1)
        idx = np.argsort(distances)[1 : k + 1]
        neighbors.append(idx)
    return neighbors


neighbors = compute_neighbors(points, k=k_neighbors)


# -----------------------------
# Discrete operators
# -----------------------------
def laplacian(field: np.ndarray) -> np.ndarray:
    """Nearest-neighbour graph Laplacian on the spherical point cloud."""
    out = np.zeros_like(field)
    for i in range(N):
        out[i] = np.mean(field[neighbors[i]] - field[i])
    return out


def nematic_phase_difference(phi_neighbors: np.ndarray, phi_i: float) -> np.ndarray:
    """Pi-periodic angular differences for nematic orientation."""
    diff = phi_neighbors - phi_i
    return (diff + np.pi / 2.0) % np.pi - np.pi / 2.0


def orientational_frustration(psi: np.ndarray) -> np.ndarray:
    """
    Local orientational frustration proxy.

    High values indicate strong local phase variation in the nematic field.
    This is used as a generic defect/frustration source for the slow material field.
    """
    phi = np.angle(psi) / 2.0
    fr = np.zeros(N)
    for i in range(N):
        diff = nematic_phase_difference(phi[neighbors[i]], phi[i])
        fr[i] = np.mean(diff * diff)
    return fr


def detect_defect_proxy(frustration: np.ndarray, n=14) -> np.ndarray:
    """Indices of highest-frustration locations; simple proxy, not topological charge."""
    return np.argsort(frustration)[-n:]


# -----------------------------
# Initial fields
# -----------------------------
rng = np.random.default_rng(random_seed)

# Near-uniform nematic orientation with small perturbations.
phi0 = 0.12 * rng.standard_normal(N)
psi = np.exp(2j * phi0)

# Slow scalar material field initially near zero.
h = np.zeros(N)

# Generic spatial gradient in orientational coupling:
# strongest at poles, weakest near equator.
# This is optional and only serves to break perfect rotational symmetry.
D_base = D0 * (0.35 + 0.65 * points[:, 1] ** 2)


# -----------------------------
# Time evolution
# -----------------------------
history = []
snapshots = {}

for step in range(steps):
    # h-dependent mobility/accessibility:
    # h does not remove order; it suppresses dynamic expression by reducing local coupling.
    mobility = 1.0 / (1.0 + lock_strength * h)
    D_eff = D_base * mobility

    lap_psi = laplacian(psi)

    # Fast orientational dynamics, phase-normalised for this exploratory model.
    F = (
        alpha * psi
        - beta * np.abs(psi) ** 2 * psi
        + D_eff * lap_psi
        + 1j * kappa * lap_psi
    )
    psi = psi + dt * F
    psi = psi / (np.abs(psi) + epsilon)

    # Slow material-field dynamics driven by orientational frustration.
    fr = orientational_frustration(psi)
    source = prod_rate * np.maximum(fr - threshold, 0.0) ** power * (1.0 - h)
    h = h + dt * tau_h * (
        source
        - decay_rate * h
        + Dh * laplacian(h)
    )
    h = np.clip(h, 0.0, 1.0)

    if step % 50 == 0 or step == steps - 1:
        history.append(
            {
                "step": step,
                "mean_h": float(np.mean(h)),
                "max_h": float(np.max(h)),
                "mean_frustration": float(np.mean(fr)),
                "max_frustration": float(np.max(fr)),
                "mean_mobility": float(np.mean(mobility)),
                "min_mobility": float(np.min(mobility)),
            }
        )

    if step in snapshot_steps:
        snapshots[step] = {
            "psi": psi.copy(),
            "h": h.copy(),
            "frustration": fr.copy(),
        }


# -----------------------------
# Final diagnostics
# -----------------------------
fr_final = orientational_frustration(psi)
defect_idx = detect_defect_proxy(fr_final, n=14)
phase = np.angle(psi) / 2.0


summary = {
    "parameters": {
        "N": N,
        "k_neighbors": k_neighbors,
        "steps": steps,
        "dt": dt,
        "tau_h": tau_h,
        "prod_rate": prod_rate,
        "decay_rate": decay_rate,
        "Dh": Dh,
        "threshold": threshold,
        "power": power,
        "lock_strength": lock_strength,
        "D0": D0,
        "kappa": kappa,
        "random_seed": random_seed,
    },
    "final_metrics": history[-1],
    "interpretation": [
        "The orientational field evolves on a fast timescale.",
        "The scalar material field h evolves more slowly.",
        "h accumulates preferentially where orientational frustration exceeds a threshold.",
        "Increasing h reduces local dynamic accessibility rather than deleting the underlying orientation field.",
        "This provides a minimal model for defect-mediated material memory or mechanochemical lock-in."
    ],
}

(outdir / "model2_summary.json").write_text(json.dumps(summary, indent=2))


# -----------------------------
# Visualisation helpers
# -----------------------------
def sphere_scatter(values, title, ax, cmap=None, vmin=None, vmax=None, mark_defects=False):
    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=values,
        s=13,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    if mark_defects:
        ax.scatter(
            points[defect_idx, 0],
            points[defect_idx, 1],
            points[defect_idx, 2],
            s=90,
            marker="o",
            facecolors="none",
            linewidths=1.4,
        )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=35)
    return sc


# Figure 1: final orientation and material field
fig = plt.figure(figsize=(13, 6))

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
sc1 = sphere_scatter(
    phase,
    "Final nematic orientation field",
    ax1,
    mark_defects=True,
)
cbar1 = fig.colorbar(sc1, ax=ax1, shrink=0.75)
cbar1.set_label("orientation angle φ")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
sc2 = sphere_scatter(
    h,
    "Slow material field h",
    ax2,
    vmin=0.0,
    vmax=max(0.01, float(np.max(h))),
    mark_defects=True,
)
cbar2 = fig.colorbar(sc2, ax=ax2, shrink=0.75)
cbar2.set_label("material accumulation / lock-in h")

fig.suptitle("Model 2: slow material-field coupling and mechanochemical lock-in")
plt.tight_layout()
plt.savefig(outdir / "01_final_orientation_and_material.png", dpi=220, bbox_inches="tight")
plt.show()


# Figure 2: time-series metrics
steps_hist = [r["step"] for r in history]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(steps_hist, [r["mean_h"] for r in history], label="mean h")
ax.plot(steps_hist, [r["max_h"] for r in history], label="max h")
ax.plot(steps_hist, [r["mean_frustration"] for r in history], label="mean frustration")
ax.plot(steps_hist, [r["max_frustration"] for r in history], label="max frustration")
ax.plot(steps_hist, [r["mean_mobility"] for r in history], label="mean mobility")
ax.set_title("Timescale separation: fast orientation, slow material response")
ax.set_xlabel("timestep")
ax.set_ylabel("dimensionless diagnostic")
ax.grid(True, alpha=0.3)
ax.legend(ncol=2)
plt.tight_layout()
plt.savefig(outdir / "02_timeseries_metrics.png", dpi=220, bbox_inches="tight")
plt.show()


# Figure 3: snapshots of material accumulation over time
available_steps = [s for s in snapshot_steps if s in snapshots]
fig = plt.figure(figsize=(4 * len(available_steps), 4.8))

hmax = max(float(np.max(snapshots[s]["h"])) for s in available_steps)
hmax = max(hmax, 0.01)

for idx, s in enumerate(available_steps, start=1):
    ax = fig.add_subplot(1, len(available_steps), idx, projection="3d")
    sc = sphere_scatter(
        snapshots[s]["h"],
        f"h at step {s}",
        ax,
        vmin=0.0,
        vmax=hmax,
        mark_defects=False,
    )

cbar = fig.colorbar(sc, ax=fig.axes, shrink=0.75, pad=0.04)
cbar.set_label("h")
fig.suptitle("Progressive slow material-field accumulation")
plt.tight_layout()
plt.savefig(outdir / "03_snapshots_material_lockin.png", dpi=220, bbox_inches="tight")
plt.show()


# Figure 4: compact summary card
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.axis("off")
text = (
    "ANDA Model 2 — slow material-field coupling\n\n"
    f"Final mean h:           {history[-1]['mean_h']:.4f}\n"
    f"Final max h:            {history[-1]['max_h']:.4f}\n"
    f"Final mean frustration: {history[-1]['mean_frustration']:.4f}\n"
    f"Final max frustration:  {history[-1]['max_frustration']:.4f}\n"
    f"Final mean mobility:    {history[-1]['mean_mobility']:.4f}\n"
    f"Final min mobility:     {history[-1]['min_mobility']:.4f}\n\n"
    "Interpretation:\n"
    "Fast orientational dynamics create defects/frustration.\n"
    "A slower scalar material field accumulates near high-frustration regions.\n"
    "Accumulation reduces local dynamic accessibility, producing lock-in.\n"
    "This is a generic tissue-like mechanism, not a model of a specific tissue."
)
ax.text(0.02, 0.98, text, va="top", ha="left", family="monospace", fontsize=10.5)
plt.tight_layout()
plt.savefig(outdir / "04_summary_card.png", dpi=220, bbox_inches="tight")
plt.show()


print(json.dumps(summary, indent=2))
print(f"\nSaved outputs in: {outdir.resolve()}")
