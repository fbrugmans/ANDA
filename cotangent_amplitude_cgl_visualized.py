"""
cotangent_amplitude_cgl_visualized.py

Robustness test for ANDA:
- amplitude-resolving tangent-field extension;
- no per-timestep amplitude normalisation;
- nonlinear saturation retained;
- semi-implicit diffusion;
- cotangent-weighted Laplace-Beltrami approximation;
- visualisation of the output values.

Run in Jupyter Lab:
    %run cotangent_amplitude_cgl_visualized.py

The script saves and displays:
    1. amplitude-resolving sphere plots for uniform and weak-gradient cases;
    2. amplitude saturation history;
    3. defect-centre displacement summary.
"""

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


outdir = Path("bof_cotangent_cgl_visualized")
outdir.mkdir(exist_ok=True)


# -----------------------------
# Geometry and discrete operators
# -----------------------------
def fibonacci_sphere(n):
    i = np.arange(n)
    phi = (1 + np.sqrt(5)) / 2
    z = 1 - 2 * (i + 0.5) / n
    theta = 2 * np.pi * i / phi
    r = np.sqrt(np.maximum(0, 1 - z * z))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])


def cotangent(a, b, c):
    u = b - a
    v = c - a
    cross = np.linalg.norm(np.cross(u, v))
    if cross < 1e-14:
        return 0.0
    return np.dot(u, v) / cross


def build_cot_laplacian(vertices, faces):
    n = len(vertices)
    weights = {}
    areas = np.zeros(n)
    negative_weights = 0

    for tri in faces:
        i, j, k = tri
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        area = 0.5 * np.linalg.norm(np.cross(vj - vi, vk - vi))

        for idx in tri:
            areas[idx] += area / 3.0

        cot_i = cotangent(vi, vj, vk)  # opposite edge jk
        cot_j = cotangent(vj, vk, vi)  # opposite edge ki
        cot_k = cotangent(vk, vi, vj)  # opposite edge ij

        for a, b, cot in [(j, k, cot_i), (k, i, cot_j), (i, j, cot_k)]:
            if a > b:
                a, b = b, a
            weights[(a, b)] = weights.get((a, b), 0.0) + 0.5 * cot

    rows, cols, data = [], [], []

    for (i, j), w in weights.items():
        # Fibonacci-convex-hull triangulations are not guaranteed intrinsic Delaunay.
        # Clamping negative cotangent weights avoids non-physical anti-diffusion
        # in this minimal robustness test.
        if w < 0:
            negative_weights += 1
            w = 0.0

        wi = w / max(areas[i], 1e-14)
        wj = w / max(areas[j], 1e-14)

        # L f_i = sum_j w_ij/A_i * (f_j - f_i)
        rows += [i, i, j, j]
        cols += [j, i, i, j]
        data += [wi, -wi, wj, -wj]

    L = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    mesh_info = {
        "vertices": n,
        "faces": len(faces),
        "edges": len(weights),
        "negative_cotangent_weights_clamped": negative_weights,
    }
    return L, areas, mesh_info


def orient_faces_outward(vertices, faces):
    faces = faces.copy()
    for idx, tri in enumerate(faces):
        a, b, c = vertices[tri]
        if np.dot(np.cross(b - a, c - a), (a + b + c) / 3.0) < 0:
            faces[idx] = [tri[0], tri[2], tri[1]]
    return faces


def tangent_project(normals, X):
    return X - np.sum(X * normals, axis=1)[:, None] * normals


def rotate90(normals, X):
    # Local 90-degree rotation in the tangent plane.
    return np.cross(normals, X)


# -----------------------------
# Simulation
# -----------------------------
def simulate_amplitude_cgl(
    n_vertices=900,
    steps=1600,
    dt=0.02,
    D=0.025,
    kappa=0.06,
    alpha0=1.0,
    beta=1.0,
    gradient=0.0,
    seed=11,
):
    vertices = fibonacci_sphere(n_vertices)
    hull = ConvexHull(vertices)
    faces = orient_faces_outward(vertices, hull.simplices)

    L, areas, mesh_info = build_cot_laplacian(vertices, faces)
    A = sp.eye(n_vertices, format="csr") - dt * D * L
    solve = spla.factorized(A.tocsc())

    rng = np.random.default_rng(seed)

    constant_vector = np.array([0.25, 0.10, 1.0])
    constant_vector = constant_vector / np.linalg.norm(constant_vector)

    field = tangent_project(vertices, np.tile(constant_vector, (n_vertices, 1)))
    field += 0.04 * tangent_project(vertices, rng.normal(size=(n_vertices, 3)))

    alpha = alpha0 + gradient * vertices[:, 0]
    alpha = np.maximum(alpha, 0.15)

    history = []

    for step in range(steps):
        amp2 = np.sum(field * field, axis=1)

        lap = np.column_stack([L.dot(field[:, d]) for d in range(3)])
        lap = tangent_project(vertices, lap)

        nonlinear_and_chiral = (
            alpha[:, None] * field
            - beta * amp2[:, None] * field
            + kappa * rotate90(vertices, lap)
        )

        rhs = field + dt * nonlinear_and_chiral

        new_field = np.empty_like(field)
        for d in range(3):
            new_field[:, d] = solve(rhs[:, d])

        field = tangent_project(vertices, new_field)

        if step % 100 == 0 or step == steps - 1:
            amp = np.linalg.norm(field, axis=1)
            history.append(
                {
                    "step": step,
                    "min_amp": float(amp.min()),
                    "mean_amp": float(amp.mean()),
                    "max_amp": float(amp.max()),
                }
            )

    return vertices, faces, field, history, mesh_info


def estimate_defect_centres(vertices, field, n_low=32):
    amp = np.linalg.norm(field, axis=1)
    idx = np.argsort(amp)[:n_low]
    points = vertices[idx]

    # Initialise centres from the two most distant low-amplitude points.
    dist = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
    a, b = np.unravel_index(np.argmax(dist), dist.shape)
    centres = np.array([points[a], points[b]])

    for _ in range(30):
        dist_to_centres = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        labels = dist_to_centres.argmin(axis=1)

        for j in range(2):
            if np.any(labels == j):
                centres[j] = points[labels == j].mean(axis=0)
                centres[j] /= np.linalg.norm(centres[j])

    centres = centres[np.argsort(centres[:, 2])]
    return centres, idx


def angular_separation_degrees(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


# -----------------------------
# Visualisations
# -----------------------------
def plot_comparison_spheres(vertices0, field0, centres0, vertices1, field1, centres1, outpath):
    amp0 = np.linalg.norm(field0, axis=1)
    amp1 = np.linalg.norm(field1, axis=1)

    vmin = min(amp0.min(), amp1.min())
    vmax = max(amp0.max(), amp1.max())

    fig = plt.figure(figsize=(12, 6))

    for idx, (vertices, amp, centres, title) in enumerate(
        [
            (vertices0, amp0, centres0, "Uniform case"),
            (vertices1, amp1, centres1, "Weak-gradient case"),
        ],
        start=1,
    ):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        sc = ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=amp, s=10, vmin=vmin, vmax=vmax)
        ax.scatter(centres[:, 0], centres[:, 1], centres[:, 2], s=110, marker="x", linewidths=2)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=35)

    cbar = fig.colorbar(sc, ax=fig.axes, shrink=0.75, pad=0.08)
    cbar.set_label("field amplitude |u|")

    fig.suptitle("Amplitude-resolving tangent-field robustness test", y=0.98)
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.show()


def plot_amplitude_history(hist0, hist1, outpath):
    fig, ax = plt.subplots(figsize=(8, 5))

    for history, label in [(hist0, "uniform"), (hist1, "weak gradient")]:
        steps = [h["step"] for h in history]
        ax.plot(steps, [h["min_amp"] for h in history], linestyle="--", label=f"{label}: min")
        ax.plot(steps, [h["mean_amp"] for h in history], label=f"{label}: mean")
        ax.plot(steps, [h["max_amp"] for h in history], linestyle=":", label=f"{label}: max")

    ax.set_title("Amplitude saturation without per-timestep normalisation")
    ax.set_xlabel("timestep")
    ax.set_ylabel("amplitude |u|")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.show()


def plot_defect_displacement(centres0, centres1, outpath):
    labels = ["defect 1", "defect 2"]
    x = np.arange(len(labels))
    width = 0.35

    radial_shift = np.linalg.norm(centres1 - centres0, axis=1)
    x_shift = centres1[:, 0] - centres0[:, 0]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, radial_shift, width, label="3D displacement")
    ax.bar(x + width / 2, x_shift, width, label="x-shift")

    ax.set_title("Defect-centre shift under weak imposed gradient")
    ax.set_xlabel("detected low-amplitude defect core")
    ax.set_ylabel("shift on unit sphere")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.show()


def plot_summary_card(summary, outpath):
    # A compact figure with the key numbers for inclusion in a manuscript supplement.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")

    text = (
        "Robustness-test summary\n\n"
        f"Uniform case: min / mean / max amplitude = "
        f"{summary['uniform_final']['min_amp']:.3f} / "
        f"{summary['uniform_final']['mean_amp']:.3f} / "
        f"{summary['uniform_final']['max_amp']:.3f}\n"
        f"Weak-gradient case: min / mean / max amplitude = "
        f"{summary['gradient_final']['min_amp']:.3f} / "
        f"{summary['gradient_final']['mean_amp']:.3f} / "
        f"{summary['gradient_final']['max_amp']:.3f}\n\n"
        f"Uniform defect-axis separation: {summary['uniform_defect_axis_separation_deg']:.1f}°\n"
        f"Gradient defect-axis separation: {summary['gradient_defect_axis_separation_deg']:.1f}°\n"
        f"Mean defect-core displacement under gradient: {summary['mean_defect_displacement']:.3f}\n\n"
        "Interpretation: low-amplitude defect cores persist after amplitude-resolving\n"
        "nonlinear saturation, semi-implicit diffusion, tangent-plane projection,\n"
        "and cotangent-weighted Laplace-Beltrami discretisation."
    )

    ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=11, family="monospace")

    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.show()


# -----------------------------
# Main run
# -----------------------------
if __name__ == "__main__":
    v0, f0, field0, hist0, mesh0 = simulate_amplitude_cgl(gradient=0.0, seed=11)
    c0, idx0 = estimate_defect_centres(v0, field0)

    v1, f1, field1, hist1, mesh1 = simulate_amplitude_cgl(gradient=0.10, seed=11)
    c1, idx1 = estimate_defect_centres(v1, field1)

    displacement = np.linalg.norm(c1 - c0, axis=1)

    summary = {
        "uniform_final": hist0[-1],
        "gradient_final": hist1[-1],
        "uniform_defect_centres_xyz": c0.tolist(),
        "gradient_defect_centres_xyz": c1.tolist(),
        "uniform_defect_axis_separation_deg": angular_separation_degrees(c0[0], c0[1]),
        "gradient_defect_axis_separation_deg": angular_separation_degrees(c1[0], c1[1]),
        "defect_core_displacements": displacement.tolist(),
        "mean_defect_displacement": float(displacement.mean()),
        "uniform_mesh_info": mesh0,
        "gradient_mesh_info": mesh1,
        "notes": [
            "Amplitude was not normalised after timesteps.",
            "Nonlinear saturation alpha*u - beta*|u|^2*u was retained.",
            "Diffusion was treated semi-implicitly.",
            "The Laplace-Beltrami operator was approximated with cotangent weights on a triangulated spherical mesh.",
            "The tangent field was represented extrinsically in R3 and projected back onto the tangent plane after each step.",
            "The chiral/imaginary Laplacian contribution was represented by local 90-degree tangent-plane rotation of the Laplacian term."
        ]
    }

    (outdir / "run_summary_visualized.json").write_text(json.dumps(summary, indent=2))

    plot_comparison_spheres(
        v0, field0, c0,
        v1, field1, c1,
        outdir / "01_comparison_spheres.png",
    )

    plot_amplitude_history(
        hist0,
        hist1,
        outdir / "02_amplitude_history.png",
    )

    plot_defect_displacement(
        c0,
        c1,
        outdir / "03_defect_displacement.png",
    )

    plot_summary_card(
        summary,
        outdir / "04_summary_card.png",
    )

    print(json.dumps(summary, indent=2))
    print(f"\nSaved outputs in: {outdir.resolve()}")
