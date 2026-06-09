# Replacing the heuristic structural proxy in FPP-JAX-Optimizer

## What is wrong with the current fem.py

The current `evaluate_structural_response` computes:

```python
stress_index = (pressure / 70.0) * geometry_factor / denominator
```

`geometry_factor` is a product of three hand-tuned Gaussians (boss ring, curvature
indicator, transition bump). `denominator` is a linear function of ply count.
Neither term has a physical derivation. The consequence: the optimizer has no
gradient signal that distinguishes patch fiber angle from another, so all patches
converge to the same angle (~72°, the helical reference). A real stress field
cares about fiber orientation through Qbar rotation — this one does not.

The replacement is a three-step calculation:
1. Compute the analytical membrane stress resultants N_φ(θ) and N_θ(θ) for the
   oblate ellipsoidal dome under internal pressure (no FEA needed).
2. Assemble the local laminate A-matrix at every dome grid point from the helical
   plies and the optimizer-controlled FPP patches.
3. Invert A to get midplane strains, transform to ply material coordinates, and
   compute the Tsai-Wu failure index.

Everything is differentiable through JAX. Gradients flow back to patch center,
angle, size, ply count, and the transition boundary through the A-matrix.

---

## Part 1 — Analytical membrane stress resultants

### Geometry

The dome is an oblate ellipsoid parameterized by polar angle θ (from the boss) and
azimuthal angle φ. This matches `geometry.py` exactly:

```
r(θ) = a·sin θ          equatorial radius
z(θ) = c·cos θ          axial coordinate
a = major_radius_m      equatorial semi-axis
c = minor_radius_m      polar semi-axis
```

The metric function and sine of the meridian inclination angle α (angle between
meridian tangent and the equatorial plane):

```
f(θ) = sqrt(a²cos²θ + c²sin²θ)     arc-length rate: ds/dθ = f
sin α = c·sin θ / f(θ)              inclination of tangent
```

### Derivation of N_φ and N_θ

Equilibrium of a polar cap from the boss to angle θ in the z-direction:

```
N_φ · 2πr · sin α  =  p · π · r²

N_φ  =  p · r / (2 sin α)
      =  p · a·sin θ / (2 · c·sin θ / f)
      =  p · a · f(θ) / (2c)
```

From the Laplace equilibrium equation N_φ/R₁ + N_θ/R₂ = p with principal radii of
curvature:

```
R₁  =  f(θ)³ / (a·c)          meridional radius of curvature
R₂  =  a · f(θ) / c            circumferential radius of curvature
```

Substituting:

```
N_θ  =  (p − N_φ/R₁) · R₂
      =  p · a · (2f² − a²) / (2c · f)
```

where f = f(θ) = sqrt(a²cos²θ + c²sin²θ).

Verification (sphere a = c = R): f = R, N_φ = pR/2, N_θ = p·R·(2R² − R²)/(2R²) = pR/2. ✓

For axisymmetric loading N_φθ = 0 everywhere.

### Code to add to the new fem.py

```python
def membrane_stress_resultants(
    dome: DomeState,
    material: MaterialConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns (N_phi, N_theta) in N/m at every dome grid point.
    N_phi  = meridional stress resultant
    N_theta = circumferential stress resultant
    Both have shape (n_theta, n_phi).
    """
    a = dome.config.major_radius_m
    c = dome.config.minor_radius_m
    p = material.pressure_mpa * 1.0e6  # Pa

    theta = dome.theta_grid
    f = jnp.sqrt(a**2 * jnp.cos(theta)**2 + c**2 * jnp.sin(theta)**2)

    N_phi   = p * a * f / (2.0 * c)
    N_theta = p * a * (2.0 * f**2 - a**2) / (2.0 * c * f)
    return N_phi, N_theta
```

---

## Part 2 — Classical laminate theory in JAX

### Ply stiffness matrix Q (material coordinates)

For a unidirectional ply with fibre in the 1-direction:

```
ν21 = ν12 · E2 / E1                    (reciprocal relation)
denom = 1 − ν12 · ν21

Q11 = E1 / denom
Q22 = E2 / denom
Q12 = ν12 · E2 / denom
Q66 = G12
Q16 = Q26 = Q61 = Q62 = 0             (orthotropic in material frame)
```

In 3-component Voigt notation [11, 22, 12] the Q matrix is:

```
Q = [[Q11, Q12,   0 ],
     [Q12, Q22,   0 ],
     [  0,   0, Q66]]
```

### Transformed ply stiffness Qbar (global coordinates)

The global frame has x aligned with the meridional direction, y with the
circumferential direction. For a ply whose fibre makes angle θ_ply with the
meridional direction (x-axis):

```
c = cos(θ_ply),   s = sin(θ_ply)
m = c²,           n = s²,   p = c·s

Q̄11 = Q11·m² + 2(Q12 + 2Q66)·mn + Q22·n²
Q̄22 = Q11·n² + 2(Q12 + 2Q66)·mn + Q22·m²
Q̄12 = (Q11 + Q22 − 4Q66)·mn + Q12·(m² + n²)
Q̄66 = (Q11 + Q22 − 2Q12 − 2Q66)·mn + Q66·(m² + n²)
Q̄16 = (Q11 − Q12 − 2Q66)·c³s − (Q22 − Q12 − 2Q66)·cs³
Q̄26 = (Q11 − Q12 − 2Q66)·cs³ − (Q22 − Q12 − 2Q66)·c³s
```

(indices run over [x, y, xy] in Voigt notation)

For a ±θ pair (helical winding), Q̄16 and Q̄26 cancel and the combined Qbar is
orthotropic.

### Code for Q and Qbar

```python
def ply_Q_matrix(material: MaterialConfig) -> jnp.ndarray:
    """Returns 3×3 Q matrix in material coordinates."""
    E1   = material.E1_gpa * 1.0e9
    E2   = material.E2_gpa * 1.0e9
    G12  = material.G12_gpa * 1.0e9
    nu12 = material.nu12
    nu21 = nu12 * E2 / E1
    d    = 1.0 - nu12 * nu21
    Q11  = E1 / d
    Q22  = E2 / d
    Q12  = nu12 * E2 / d
    Q66  = G12
    return jnp.array([[Q11, Q12, 0.0],
                      [Q12, Q22, 0.0],
                      [0.0, 0.0, Q66]])


def qbar_matrix(Q: jnp.ndarray, angle_rad: jnp.ndarray) -> jnp.ndarray:
    """
    Transforms Q from material to global (meridional-circumferential) frame.
    angle_rad: fibre angle from the meridional direction (scalar or broadcastable).
    Returns Qbar with shape (*angle_rad.shape, 3, 3).
    """
    c  = jnp.cos(angle_rad)
    s  = jnp.sin(angle_rad)
    m  = c * c
    n  = s * s
    mn = c * s

    Q11, Q12, Q22, Q66 = Q[0,0], Q[0,1], Q[1,1], Q[2,2]

    Qb11 = Q11*m**2 + 2*(Q12 + 2*Q66)*m*n + Q22*n**2
    Qb22 = Q11*n**2 + 2*(Q12 + 2*Q66)*m*n + Q22*m**2
    Qb12 = (Q11 + Q22 - 4*Q66)*m*n + Q12*(m**2 + n**2)
    Qb66 = (Q11 + Q22 - 2*Q12 - 2*Q66)*m*n + Q66*(m**2 + n**2)
    Qb16 = (Q11 - Q12 - 2*Q66)*c**3*s - (Q22 - Q12 - 2*Q66)*c*s**3
    Qb26 = (Q11 - Q12 - 2*Q66)*c*s**3 - (Q22 - Q12 - 2*Q66)*c**3*s

    Qbar = jnp.stack([
        jnp.stack([Qb11, Qb12, Qb16], axis=-1),
        jnp.stack([Qb12, Qb22, Qb26], axis=-1),
        jnp.stack([Qb16, Qb26, Qb66], axis=-1),
    ], axis=-2)
    return Qbar
```

---

## Part 3 — A-matrix assembly from the patch layout

At every (θ, φ) grid point the total laminate consists of:

- Helical layer: baseline_helical_plies/2 plies at +α AND baseline_helical_plies/2
  plies at −α (both multiplied by the helical_mask from `mapping.py`).  The +/−
  pair makes the helical contribution orthotropic (Q̄16 = Q̄26 = 0 combined).
- FPP patches: each patch i contributes mask_i(θ,φ) × plies_i × Qbar(angle_i).

```python
def assemble_a_matrix_field(
    dome: DomeState,
    layout: dict[str, jnp.ndarray],
    thickness_state: dict[str, jnp.ndarray],
    Q: jnp.ndarray,
    material: MaterialConfig,
) -> jnp.ndarray:
    """
    Returns the extensional stiffness matrix A at every dome grid point.
    Shape: (n_theta, n_phi, 3, 3).  Units: N/m.
    """
    t = material.ply_thickness_mm * 1.0e-3        # m per ply

    # ── helical contribution (±α pair, orthotropic) ──────────────────────────
    alpha = jnp.array(math.radians(material.helical_reference_angle_deg))
    Qbar_pos = qbar_matrix(Q, +alpha)      # (3, 3)
    Qbar_neg = qbar_matrix(Q, -alpha)      # (3, 3)
    Qbar_helical = Qbar_pos + Qbar_neg     # (3, 3) — symmetric, A16=A26=0

    helical_plies_field = thickness_state["baseline_plies_map"]    # (n_theta, n_phi)
    # half the total helical plies are at +α, half at −α; combined = full count
    A_helical = (helical_plies_field * t)[..., None, None] * Qbar_helical  # (n_theta, n_phi, 3, 3)

    # ── FPP patch contributions ───────────────────────────────────────────────
    masks    = thickness_state["masks"]           # (n_patches, n_theta, n_phi)
    plies    = layout["plies"]                    # (n_patches,)
    angles   = layout["angle_rad"]                # (n_patches,)

    # Qbar per patch: (n_patches, 3, 3)
    Qbar_patches = qbar_matrix(Q, angles)

    # Effective thickness contribution per patch per grid point: (n_patches, n_theta, n_phi)
    patch_t = masks * plies[:, None, None] * t

    # Sum over patches: (n_theta, n_phi, 3, 3)
    # patch_t[i, :, :] * Qbar_patches[i, :, :] — broadcast and sum
    A_patches = jnp.einsum("ijk,iab->jkab", patch_t, Qbar_patches)

    # ── minimum laminate guard (prevents singular A-matrix) ──────────────────
    # Add a thin quasi-isotropic floor: 0 contribution to mechanics but
    # avoids A^{-1} blowing up at unpatched regions far from the boss.
    A_floor = jnp.eye(3) * (Q[0,0] * t * 0.5)   # (3, 3) — isotropic floor

    A_field = A_helical + A_patches + A_floor
    return A_field
```

---

## Part 4 — Midplane strains and Tsai-Wu failure index

### Midplane strains from inverting A

```python
def midplane_strains(
    A_field: jnp.ndarray,          # (n_theta, n_phi, 3, 3)
    N_phi: jnp.ndarray,            # (n_theta, n_phi)
    N_theta: jnp.ndarray,          # (n_theta, n_phi)
) -> jnp.ndarray:
    """
    Returns midplane strains (eps_x, eps_y, gamma_xy) at every grid point.
    Shape: (n_theta, n_phi, 3).
    """
    N_vec = jnp.stack([N_phi, N_theta, jnp.zeros_like(N_phi)], axis=-1)  # (n_theta, n_phi, 3)
    # Batched solve: C_field @ N_vec where C_field = A_field^{-1}
    eps = jnp.linalg.solve(A_field, N_vec)      # (n_theta, n_phi, 3)
    return eps
```

`jnp.linalg.solve` supports batched shapes when the last two dimensions are the
matrix and the second-to-last dimension is the right-hand side. This is
differentiable through JAX's VJP rules.

### Ply strain in material coordinates

For ply at angle θ_ply from the meridional direction:

```
ε1  = εx·c² + εy·s² + γxy·s·c
ε2  = εx·s² + εy·c² − γxy·s·c
γ12 = −2εx·s·c + 2εy·s·c + γxy·(c²−s²)
```

### Tsai-Wu failure index

Material-frame ply stresses σ1, σ2, τ12 from σ = Q @ [ε1, ε2, γ12]:

```
σ1  = Q11·ε1 + Q12·ε2
σ2  = Q12·ε1 + Q22·ε2
τ12 = Q66·γ12
```

Tsai-Wu coefficients:

```
F1  = 1/Xt − 1/Xc
F2  = 1/Yt − 1/Yc
F11 = 1/(Xt·Xc)
F22 = 1/(Yt·Yc)
F66 = 1/S²
F12 = −1/(2·sqrt(Xt·Xc·Yt·Yc))      (Tsai-Hahn interaction)

FI  = F1·σ1 + F2·σ2 + F11·σ1² + F22·σ2² + F66·τ12² + 2·F12·σ1·σ2
```

FI ≥ 1 means failure. The optimizer minimizes FI to drive the layout away from
failure.

---

## Part 5 — Complete code changes

### 5.1  config.py — add these fields to MaterialConfig

```python
@dataclass(frozen=True)
class MaterialConfig:
    # existing fields unchanged ...
    pressure_mpa: float = 70.0
    ply_thickness_mm: float = 0.125
    liner_thickness_mm: float = 0.60
    baseline_helical_plies: float = 4.0
    areal_density_kg_per_m2_per_ply: float = 0.145
    allowable_shear_strain: float = 0.015
    allowable_distortion: float = 0.015
    max_thickness_gradient_mm_per_m: float = 8.0
    helical_reference_angle_deg: float = 55.0
    helical_cost_per_kg: float = 110.0
    fpp_cost_per_kg: float = 185.0

    # ── NEW: ply mechanics (T700/epoxy) ──────────────────────────────────────
    E1_gpa:  float = 140.0     # fibre-direction modulus
    E2_gpa:  float = 10.0      # transverse modulus
    G12_gpa: float = 5.0       # in-plane shear modulus
    nu12:    float = 0.30      # major Poisson's ratio

    # ── NEW: ply strengths (MPa) ──────────────────────────────────────────────
    Xt_mpa:  float = 2200.0    # fibre tension
    Xc_mpa:  float = 1500.0    # fibre compression
    Yt_mpa:  float = 60.0      # transverse tension
    Yc_mpa:  float = 150.0     # transverse compression
    S_mpa:   float = 90.0      # in-plane shear
```

These are standard T700/epoxy room-temperature dry values. They are consistent
with MIL-HDBK-17-3 and Toray T700G data sheets. If the actual prepreg is known,
replace them.

### 5.2  fem.py — complete replacement

Delete the current file content and replace with the following.

```python
from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ..config import MaterialConfig
from ..core.geometry import DomeState


# ──────────────────────────────────────────────────────────────────────────────
# Classical Laminate Theory primitives
# ──────────────────────────────────────────────────────────────────────────────

def ply_Q_matrix(material: MaterialConfig) -> jnp.ndarray:
    """3×3 ply stiffness matrix Q in material (1-2) coordinates. Units: Pa."""
    E1   = material.E1_gpa  * 1.0e9
    E2   = material.E2_gpa  * 1.0e9
    G12  = material.G12_gpa * 1.0e9
    nu12 = material.nu12
    nu21 = nu12 * E2 / E1
    d    = 1.0 - nu12 * nu21
    Q11  = E1  / d
    Q22  = E2  / d
    Q12  = nu12 * E2 / d
    Q66  = G12
    return jnp.array([[Q11, Q12, 0.0],
                      [Q12, Q22, 0.0],
                      [0.0, 0.0, Q66]])


def qbar_matrix(Q: jnp.ndarray, angle_rad: jnp.ndarray) -> jnp.ndarray:
    """
    Transforms Q from material frame to global (meridional-x, circum-y) frame.
    angle_rad: fibre angle from the meridional direction.
                Scalar or any shape — returned Qbar has shape (*angle.shape, 3, 3).
    """
    c  = jnp.cos(angle_rad)
    s  = jnp.sin(angle_rad)
    m  = c * c
    n  = s * s
    Q11, Q12, Q22, Q66 = Q[0,0], Q[0,1], Q[1,1], Q[2,2]

    Qb11 = Q11*m**2 + 2*(Q12 + 2*Q66)*m*n       + Q22*n**2
    Qb22 = Q11*n**2 + 2*(Q12 + 2*Q66)*m*n       + Q22*m**2
    Qb12 = (Q11 + Q22 - 4*Q66)*m*n              + Q12*(m**2 + n**2)
    Qb66 = (Q11 + Q22 - 2*Q12 - 2*Q66)*m*n      + Q66*(m**2 + n**2)
    Qb16 = (Q11 - Q12 - 2*Q66)*c**3*s           - (Q22 - Q12 - 2*Q66)*c*s**3
    Qb26 = (Q11 - Q12 - 2*Q66)*c*s**3           - (Q22 - Q12 - 2*Q66)*c**3*s

    return jnp.stack([
        jnp.stack([Qb11, Qb12, Qb16], axis=-1),
        jnp.stack([Qb12, Qb22, Qb26], axis=-1),
        jnp.stack([Qb16, Qb26, Qb66], axis=-1),
    ], axis=-2)


# ──────────────────────────────────────────────────────────────────────────────
# Membrane stress resultants (analytical thin-wall solution)
# ──────────────────────────────────────────────────────────────────────────────

def membrane_stress_resultants(
    dome: DomeState,
    material: MaterialConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Closed-form membrane stress resultants for an oblate ellipsoidal dome
    under uniform internal pressure p.

    Derivation: equilibrium of a polar cap of the closed vessel:
        N_phi  = p · a · f(θ) / (2c)
        N_theta = p · a · (2f² − a²) / (2c · f)
    where f(θ) = sqrt(a²cos²θ + c²sin²θ)

    Returns (N_phi, N_theta), shape (n_theta, n_phi), units N/m.
    N_phi_theta = 0 everywhere (axisymmetric loading).
    """
    a = dome.config.major_radius_m
    c = dome.config.minor_radius_m
    p = material.pressure_mpa * 1.0e6          # Pa

    theta = dome.theta_grid
    f     = jnp.sqrt(a**2 * jnp.cos(theta)**2 + c**2 * jnp.sin(theta)**2)

    N_phi   = p * a * f / (2.0 * c)
    N_theta = p * a * (2.0 * f**2 - a**2) / (2.0 * c * f)
    return N_phi, N_theta


# ──────────────────────────────────────────────────────────────────────────────
# A-matrix assembly
# ──────────────────────────────────────────────────────────────────────────────

def assemble_a_matrix_field(
    dome: DomeState,
    layout: dict[str, jnp.ndarray],
    thickness_state: dict[str, jnp.ndarray],
    Q: jnp.ndarray,
    material: MaterialConfig,
) -> jnp.ndarray:
    """
    Extensional stiffness matrix A = Σ_k Q̄_k · t_k at every dome grid point.
    Contributions:
        • Helical ± pair  → orthotropic, field-weighted by helical_mask
        • FPP patches     → each patch at its optimized angle, weighted by soft mask
        • Isotropic floor → minimum guard against near-singular A in unpatched zones
    Returns shape (n_theta, n_phi, 3, 3), units N/m.
    """
    t = material.ply_thickness_mm * 1.0e-3      # m

    # ── helical layer (±α, orthotropic combined) ──────────────────────────────
    alpha         = jnp.array(math.radians(material.helical_reference_angle_deg))
    Qbar_helical  = qbar_matrix(Q, +alpha) + qbar_matrix(Q, -alpha)    # (3, 3)
    helical_t     = thickness_state["baseline_plies_map"] * t           # (n_theta, n_phi)
    A_helical     = helical_t[..., None, None] * Qbar_helical           # (n_theta, n_phi, 3, 3)

    # ── FPP patches ──────────────────────────────────────────────────────────
    # masks  : (n_patches, n_theta, n_phi)
    # plies  : (n_patches,)
    # angles : (n_patches,)
    masks        = thickness_state["masks"]
    patch_t      = masks * layout["plies"][:, None, None] * t           # (n_patches, n_theta, n_phi)
    Qbar_patches = qbar_matrix(Q, layout["angle_rad"])                  # (n_patches, 3, 3)
    A_patches    = jnp.einsum("ijk,iab->jkab", patch_t, Qbar_patches)  # (n_theta, n_phi, 3, 3)

    # ── isotropic floor (0.5 ply equivalent, avoids singular A) ──────────────
    A_floor = jnp.eye(3) * (Q[0, 0] * t * 0.5)                         # (3, 3)

    return A_helical + A_patches + A_floor


# ──────────────────────────────────────────────────────────────────────────────
# Failure index
# ──────────────────────────────────────────────────────────────────────────────

def tsai_wu_field(
    A_field: jnp.ndarray,          # (n_theta, n_phi, 3, 3)  N/m
    N_phi:   jnp.ndarray,          # (n_theta, n_phi)         N/m
    N_theta: jnp.ndarray,          # (n_theta, n_phi)         N/m
    Q: jnp.ndarray,                # (3, 3)                   Pa
    layout: dict[str, jnp.ndarray],
    material: MaterialConfig,
) -> jnp.ndarray:
    """
    Computes the Tsai-Wu failure index at every grid point for each patch
    and the helical layer, then returns the maximum over all layers.

    FI = F1·σ1 + F2·σ2 + F11·σ1² + F22·σ2² + F66·τ12² + 2·F12·σ1·σ2
    Failure when FI ≥ 1.

    Returns FI_max field, shape (n_theta, n_phi).
    """
    # Tsai-Wu coefficients (Pa units)
    Xt = material.Xt_mpa * 1.0e6
    Xc = material.Xc_mpa * 1.0e6
    Yt = material.Yt_mpa * 1.0e6
    Yc = material.Yc_mpa * 1.0e6
    S  = material.S_mpa  * 1.0e6
    F1  = 1.0/Xt - 1.0/Xc
    F2  = 1.0/Yt - 1.0/Yc
    F11 = 1.0/(Xt * Xc)
    F22 = 1.0/(Yt * Yc)
    F66 = 1.0/(S  * S)
    F12 = -0.5 / jnp.sqrt(Xt * Xc * Yt * Yc)

    # Midplane strains [eps_x, eps_y, gamma_xy] — shape (n_theta, n_phi, 3)
    N_vec = jnp.stack([N_phi, N_theta, jnp.zeros_like(N_phi)], axis=-1)
    eps   = jnp.linalg.solve(A_field, N_vec)    # (n_theta, n_phi, 3)

    Q11, Q12, Q22, Q66 = Q[0,0], Q[0,1], Q[1,1], Q[2,2]

    def ply_tsai_wu(angle_rad):
        """Evaluate Tsai-Wu FI at all grid points for one ply angle."""
        c  = jnp.cos(angle_rad)
        s  = jnp.sin(angle_rad)
        ex, ey, gxy = eps[..., 0], eps[..., 1], eps[..., 2]

        # Strain in material coordinates
        e1   = ex*c**2  + ey*s**2  + gxy*s*c
        e2   = ex*s**2  + ey*c**2  - gxy*s*c
        g12  = -2*ex*s*c + 2*ey*s*c + gxy*(c**2 - s**2)

        # Ply stresses (material frame)
        s1   = Q11*e1 + Q12*e2
        s2   = Q12*e1 + Q22*e2
        t12  = Q66*g12

        FI = (F1*s1 + F2*s2
              + F11*s1**2 + F22*s2**2 + F66*t12**2
              + 2.0*F12*s1*s2)
        return FI

    # FI for each FPP patch: vmap over patch angles (n_patches,) → (n_patches, n_theta, n_phi)
    fi_patches = jax.vmap(ply_tsai_wu)(layout["angle_rad"])

    # FI for helical +α and −α plies (scalar angles)
    alpha = jnp.array(math.radians(material.helical_reference_angle_deg))
    fi_hel_pos = ply_tsai_wu(+alpha)[None, ...]    # (1, n_theta, n_phi)
    fi_hel_neg = ply_tsai_wu(-alpha)[None, ...]

    fi_all = jnp.concatenate([fi_patches, fi_hel_pos, fi_hel_neg], axis=0)  # (n_layers, n_theta, n_phi)
    fi_max = jnp.max(fi_all, axis=0)               # (n_theta, n_phi)
    return fi_max


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation function (drop-in replacement for old evaluate_structural_response)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_structural_response(
    dome: DomeState,
    layout: dict[str, jnp.ndarray],
    thickness_state: dict[str, jnp.ndarray],
    material: MaterialConfig | None = None,
) -> dict[str, jnp.ndarray]:
    """
    Drop-in replacement for the old heuristic proxy.
    Returns the same dictionary keys consumed by loss.py and generate_readme_assets.py.
    """
    material = material or MaterialConfig()

    Q        = ply_Q_matrix(material)
    N_phi, N_theta = membrane_stress_resultants(dome, material)
    A_field  = assemble_a_matrix_field(dome, layout, thickness_state, Q, material)
    fi_field = tsai_wu_field(A_field, N_phi, N_theta, Q, layout, material)

    # structural_loss: area-weighted mean + peak penalty (same shape as before)
    mean_fi  = jnp.sum(dome.area_weights * fi_field**2) / jnp.maximum(dome.total_area_m2, 1.0e-8)
    peak_fi  = 0.35 * jnp.max(fi_field)**2
    structural_loss = mean_fi + peak_fi

    # Transition height (unchanged from before)
    transition_height_m = dome.config.minor_radius_m * jnp.cos(layout["transition_theta"])

    # Expose N fields for diagnostics / visualisation
    N_vec = jnp.stack([N_phi, N_theta, jnp.zeros_like(N_phi)], axis=-1)
    eps   = jnp.linalg.solve(A_field, N_vec)

    return {
        # Keys consumed by loss.py — names preserved
        "stress_index":          fi_field,          # Tsai-Wu FI (was heuristic index)
        "structural_loss":       structural_loss,
        "peak_stress_index":     jnp.max(fi_field),
        "mean_stress_index":     jnp.sum(dome.area_weights * fi_field) / jnp.maximum(dome.total_area_m2, 1.0e-8),
        "transition_height_m":   transition_height_m,
        # New keys available for visualisation
        "tsai_wu_field":         fi_field,
        "N_phi_field":           N_phi,             # N/m
        "N_theta_field":         N_theta,            # N/m
        "eps_field":             eps,                # (n_theta, n_phi, 3)
        "A_field":               A_field,            # (n_theta, n_phi, 3, 3)
        # Retain for backward compat with existing asset generator
        "preferred_angle_map":   jnp.zeros_like(fi_field),
        "patch_angle_field":     jnp.zeros_like(fi_field),
        "effective_total_plies": thickness_state["total_plies"],
    }
```

### 5.3  loss.py — three-line change only

In `evaluate_layout`, the metrics dict references `peak_stress_index` and
`mean_stress_index` by those exact keys. The new `fem.py` returns the same keys,
so loss.py is a **zero-change file** for the core optimization loop.

One optional improvement: rename the history entry for clarity:

```python
# In optimize_patch_layout, inside the history append block:
# Change:
"peak_stress_index": _scalarize(aux, "metrics.peak_stress_index"),
# To the same thing — no change required, the key name is preserved.
```

The only mandatory change is in `OptimizationConfig`: the `stress_weight` default
of `1.0` was tuned for the heuristic proxy which returned values around 1–5.  The
Tsai-Wu index for a 70 MPa vessel with T700/epoxy will be in the range 0.05–0.8
(not failed but loaded). Scale `stress_weight` to `5.0` to compensate:

```python
@dataclass(frozen=True)
class OptimizationConfig:
    # ... existing fields ...
    stress_weight:     float = 5.0     # was 1.0 — Tsai-Wu values are smaller
```

---

## Part 6 — Visualization changes

### What to show

The new `stress_index` field returned by `fem.py` is the Tsai-Wu failure index.
It naturally concentrates near the boss (θ ≈ θ_open) because N_phi peaks there and
the helical-only laminate is not optimally oriented at that location. The
optimized patches will redistribute this concentration — the visualization should
make that directly visible.

Two changes to `patch_dome_viz.py`:

**Change 1: Color the dome by Tsai-Wu failure index instead of total plies**

The existing code in `draw_dome_with_patches` uses:
```python
stress = np.asarray(result["optimized"]["structure"]["stress_index"])
```

This already reads `stress_index`, which is now the Tsai-Wu field. The only
change is the colormap and colorbar label:

```python
# In draw_dome_with_patches, change:
norm_s = mcolors.Normalize(vmin=stress.min(), vmax=stress.max())
fc     = cm.plasma(norm_s(stress))
# To:
norm_s = mcolors.Normalize(vmin=0.0, vmax=min(1.0, float(stress.max()) * 1.1))
fc     = cm.RdYlGn_r(norm_s(stress))   # green=safe, red=near failure
```

And in the colorbar:
```python
# Change:
cbar = fig.colorbar(..., label="Pressure-response proxy (dimensionless)")
# To:
cbar = fig.colorbar(..., label="Tsai-Wu failure index  (1 = failure)")
```

**Change 2: Add principal membrane stress direction lines on the dome surface**

After drawing the wireframe and before drawing patches, add short line segments at
every 4th grid point showing the N_phi direction (meridional, radially inward) and
N_theta direction (circumferential). This is the ARTIST STUDIO–style stress
resultant overlay:

```python
# Add this block in draw_dome_with_patches, after the wireframe call:

N_phi_np  = np.asarray(result["optimized"]["structure"]["N_phi_field"])
N_theta_np = np.asarray(result["optimized"]["structure"]["N_theta_field"])
xyz_np    = np.asarray(dome.xyz)
stride    = 5   # plot every 5th point

# Meridional principal stress (N_phi) arrows — pointing along theta-tangent
t_theta_np = np.asarray(dome.tangent_theta)
t_phi_np   = np.asarray(dome.tangent_phi)

n_t, n_p, _ = xyz_np.shape
for it in range(0, n_t, stride):
    for ip in range(0, n_p, stride):
        pt     = xyz_np[it, ip]
        N_phi_val   = N_phi_np[it, ip]
        N_theta_val = N_theta_np[it, ip]
        t_mer = t_theta_np[it, ip]
        t_cir = t_phi_np[it, ip]

        norm_mer = t_mer / (np.linalg.norm(t_mer) + 1e-8)
        norm_cir = t_cir / (np.linalg.norm(t_cir) + 1e-8)

        # Scale arrow by stress magnitude; cap for display
        scale = 2.0e-7  # m / (N/m)
        arrow_mer = norm_mer * float(N_phi_val)   * scale
        arrow_cir = norm_cir * float(N_theta_val) * scale

        ax.quiver(*pt, *arrow_mer, color="#88ccff", linewidth=0.5,
                  arrow_length_ratio=0.3, alpha=0.45, zorder=3)
        ax.quiver(*pt, *arrow_cir, color="#ffcc44", linewidth=0.5,
                  arrow_length_ratio=0.3, alpha=0.45, zorder=3)
```

**Change 3: Generate a separate comparison figure**

Add a new function `make_comparison_figure` that shows the Tsai-Wu field side by
side before and after optimization, coloured on the same scale. This is the figure
that maps directly onto Sikoutris's CAMX design exploration approach — it shows
exactly what the optimizer is reducing:

```python
def make_comparison_figure(result: dict, output_path: Path) -> None:
    """Side-by-side Tsai-Wu field: baseline (helical only) vs optimised hybrid."""
    dome     = result["dome"]
    baseline = result["baseline"]
    optimized = result["optimized"]

    fi_base = np.asarray(baseline["structure"]["stress_index"])
    fi_opt  = np.asarray(optimized["structure"]["stress_index"])

    theta_deg = np.rad2deg(np.asarray(dome.theta))
    extent = [0.0, 360.0, float(theta_deg[0]), float(theta_deg[-1])]
    vmax   = max(float(fi_base.max()), float(fi_opt.max()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    fig.patch.set_facecolor("#0e1117")

    for ax, fi, title in [(ax1, fi_base, "Helical-only baseline"),
                           (ax2, fi_opt,  "Optimised FPP hybrid")]:
        ax.set_facecolor("#0e1117")
        im = ax.imshow(fi, origin="lower", aspect="auto", extent=extent,
                       cmap="RdYlGn_r", vmin=0.0, vmax=vmax)
        ax.set_title(title, color="#f0f0f0", fontsize=12, weight="bold")
        ax.set_xlabel("φ (deg)", color="#aaaaaa")
        ax.set_ylabel("θ (deg)", color="#aaaaaa")
        ax.tick_params(colors="#aaaaaa")
        ax.text(0.02, 0.97, f"peak FI = {fi.max():.3f}",
                transform=ax.transAxes, color="white", fontsize=10,
                va="top",
                bbox=dict(facecolor="black", alpha=0.4, edgecolor="none"))

    fig.suptitle("Tsai-Wu failure index  (green=safe, red=near failure, 1=predicted failure)",
                 color="#f0f0f0", fontsize=13)
    sm = plt.cm.ScalarMappable(plt.Normalize(0.0, vmax), "RdYlGn_r")
    fig.colorbar(sm, ax=[ax1, ax2], fraction=0.025, label="FI (dimensionless)")

    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="#0e1117")
    plt.close(fig)
    print(f"Saved → {output_path}")
```

Call it in `main()`:
```python
make_comparison_figure(result, REPO_ROOT / "assets" / "tsai_wu_comparison.png")
```

---

## Part 7 — What to expect after the fix

**Before (heuristic proxy):** all four patches converge to ~72° fiber angle. The
structural proxy has no gradient signal through Qbar rotation, so the optimizer
cannot distinguish an optimal angle from a suboptimal one by location.

**After (CLT + Tsai-Wu):**
- Near the boss (low θ), N_phi is the dominant resultant. The preferred fiber
  angle to carry N_phi efficiently is close to 0° (meridional). Patches placed
  near the boss should be pushed toward lower angles than 55°.
- At the transition zone, N_theta increases relative to N_phi. Patches here should
  be pushed toward higher circumferential angles.
- The four patches will diverge in angle because the gradient of the Tsai-Wu
  index with respect to angle_rad is now non-zero and location-dependent.
- The mass term will increase more sharply because the optimizer now has a real
  stress target to meet rather than an alignment heuristic — it will add more
  material near the boss.

**Numerical check before committing:**

Run with 4 patches and 90 steps, print the final angles:
```python
result = optimize_patch_layout()
angles = [math.degrees(float(a)) for a in result["layout"]["angle_rad"]]
print("Final fiber angles (deg):", angles)
```

If angles are not meaningfully spread (e.g., all within 2° of each other), check:
1. `stress_weight` in OptimizationConfig — raise it if the structural term is
   being dominated by the kinematic and thickness penalties.
2. The `A_floor` value in `assemble_a_matrix_field` — if too large, it dilutes the
   patch contribution to the A-matrix gradient.
3. The `learning_rate` in OptimizationConfig — the new loss landscape has steeper
   gradients near the boss; 0.01 may be more stable than the default 0.03.

---

## Summary checklist

- [ ] Add E1, E2, G12, nu12, Xt, Xc, Yt, Yc, S fields to `MaterialConfig` in `config.py`
- [ ] Replace `fem.py` with the new version above
- [ ] Change `stress_weight` from `1.0` to `5.0` in `OptimizationConfig`
- [ ] In `patch_dome_viz.py`: change colormap from `plasma` to `RdYlGn_r`, update colorbar label
- [ ] In `patch_dome_viz.py`: add principal stress direction arrows (optional but recommended)
- [ ] Add `make_comparison_figure` function and call it from `main()`
- [ ] Run smoke test: `PYTHONPATH=src python -c "from fpp_jax_optimizer import optimize_patch_layout; r=optimize_patch_layout(); print(r['optimized']['metrics']['peak_stress_index'])"`
- [ ] Verify final patch angles are meaningfully differentiated (not all the same)

