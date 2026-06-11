#!/usr/bin/env python3
"""
01_anti_crown_stress_field.py

Conceptual coupled-defect field simulation.

This script extends the basic scalp-orientation field model by adding a
convergent rotational defect, termed here an "anti-crown". The model is
geometrical and physical only. It is not a diagnostic or therapeutic model.

Field components:
- crown: divergent rotational defect
- forehead point: weaker secondary divergent defect
- anti-crown: convergent rotational defect

The background image shows a simple directional-gradient stress proxy.
The streamlines show the normalized orientation field.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def defect_field(
    X: np.ndarray,
    Y: np.ndarray,
    position: tuple[float, float],
    strength: float = 1.0,
    rotational: float = 0.0,
    convergent: bool = False,
    regularization: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a combined radial + rotational vector field."""

    px, py = position
    dx = X - px
    dy = Y - py

    r2 = dx**2 + dy**2 + regularization
    sign = -1.0 if convergent else 1.0

    U = sign * strength * dx / r2
    V = sign * strength * dy / r2

    U += rotational * (-dy) / r2
    V += rotational * dx / r2

    return U, V


def compute_directional_stress_proxy(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute a simple directional-gradient stress proxy.

    This is not a material stress tensor. It is a qualitative scalar proxy
    for rapid changes in vector-field direction.
    """

    dU_dy, dU_dx = np.gradient(U)
    dV_dy, dV_dx = np.gradient(V)

    return np.sqrt(dU_dx**2 + dU_dy**2 + dV_dx**2 + dV_dy**2)


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    nx, ny = 90, 90
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)

    crown = (-0.15, 0.25)
    forehead = (0.0, -0.75)
    anti_crown = (0.45, 0.10)

    U_crown, V_crown = defect_field(
        X, Y, crown, strength=1.0, rotational=0.4, convergent=False
    )

    U_forehead, V_forehead = defect_field(
        X, Y, forehead, strength=0.5, rotational=-0.2, convergent=False
    )

    U_anti, V_anti = defect_field(
        X, Y, anti_crown, strength=1.2, rotational=0.8, convergent=True
    )

    U = U_crown + U_forehead + U_anti
    V = V_crown + V_forehead + V_anti

    magnitude = np.sqrt(U**2 + V**2) + 1e-12
    U = U / magnitude
    V = V / magnitude

    stress_proxy = compute_directional_stress_proxy(U, V)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(stress_proxy, extent=[-1, 1, -1, 1], origin="lower", alpha=0.85)
    ax.streamplot(x, y, U, V, density=1.7, linewidth=0.8, arrowsize=0.8)

    ax.plot(*crown, "wo", markersize=9)
    ax.text(crown[0] + 0.03, crown[1] + 0.03, "Crown", color="white")

    ax.plot(*forehead, "co", markersize=8)
    ax.text(forehead[0] + 0.03, forehead[1] + 0.03, "Forehead point", color="cyan")

    ax.plot(*anti_crown, "ro", markersize=9)
    ax.text(anti_crown[0] + 0.03, anti_crown[1] + 0.03, "Anti-crown", color="red")

    ax.set_title("Conceptual Stress Field with Anti-Crown")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()

    output_path = output_dir / "anti_crown_stress_field.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
