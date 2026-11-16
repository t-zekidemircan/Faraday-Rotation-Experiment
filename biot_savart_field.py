"""
Biot-Savart magnetic field of the Faraday-experiment solenoid
=============================================================

Current design: one air-core solenoid, 279 mm winding, 30 mm bore,
1.1 mm enamelled copper wire, 8 layers x 225 turns = 1800 turns,
centred in a 300 mm olive-oil path, 30 V / 6 A supply.

What the script does
--------------------
1. General Biot-Savart integral for any parametric wire r(t) (scipy quad_vec):
       B(r) = mu0 I / (4 pi) * integral[ dl x (r - l) / |r - l|^3 ]
2. Exact on-axis field of one circular loop (closed form of the same integral).
3. Builds the 8-layer winding (each layer a helix, traverse direction
   alternating) and computes
     - B_z on the axis by summing the exact loop formula over all 1800 turns,
     - the same field at a few points by numerically integrating the full
       helical wire (cross-check of the loop approximation),
     - the field integral over the oil and the expected Faraday rotation.
4. Saves the on-axis profile to bz_profile.csv and bz_profile.png.

Requires numpy, scipy, matplotlib.
"""

import math
import time
import numpy as np
from scipy.integrate import quad_vec

MU0 = 4e-7 * math.pi

# ----------------------------------------------------------------------------
# Design parameters (change here)
# ----------------------------------------------------------------------------
BORE_MM = 30.0             # inner diameter of the printed barrel
WALL_MM = 2.4              # barrel wall
WIRE_MM = 1.1              # copper diameter
ENAMEL_EXTRA_MM = 0.08     # insulated diameter = copper + this
AXIAL_K = 1.05             # winding pitch = insulated diameter * AXIAL_K
TURNS_PER_LAYER = 225
LAYERS = 8
OIL_PATH_MM = 300.0        # optical path in oil, coil centred in it
CURRENT_A = 5.71           # 30 V / R(60 C)
VERDET_DEG_PER_TM = 316.0  # olive oil at 520 nm (Drude fit to Carr et al.)


# ----------------------------------------------------------------------------
# 1) General Biot-Savart integral for a parametric curve
# ----------------------------------------------------------------------------
def biot_savart_parametric_curve(curve, dcurve_dt, t0, t1, r_obs, current_A):
    """Numerical Biot-Savart field (T) at r_obs (m) of a wire r(t), t in [t0, t1]."""
    r_obs = np.asarray(r_obs, dtype=float)

    def integrand(t):
        l = curve(t)
        dl = dcurve_dt(t)
        sep = r_obs - l
        norm = np.linalg.norm(sep)
        return np.cross(dl, sep) / norm**3

    integral = quad_vec(integrand, t0, t1, epsabs=1e-12, epsrel=1e-9)[0]
    return MU0 * current_A / (4.0 * math.pi) * integral


def circular_loop_curve(radius_m, z0_m=0.0):
    curve = lambda t: np.array([radius_m * np.cos(t), radius_m * np.sin(t), z0_m])
    dcurve = lambda t: np.array([-radius_m * np.sin(t), radius_m * np.cos(t), 0.0])
    return curve, dcurve


def helix_curve(radius_m, z_start_m, pitch_m, direction=+1):
    """Helix; t = 2*pi*k after k turns. direction = +1 or -1 (axial traverse)."""
    curve = lambda t: np.array([radius_m * np.cos(t), radius_m * np.sin(t),
                                z_start_m + direction * pitch_m * t / (2 * math.pi)])
    dcurve = lambda t: np.array([-radius_m * np.sin(t), radius_m * np.cos(t),
                                 direction * pitch_m / (2 * math.pi)])
    return curve, dcurve


# ----------------------------------------------------------------------------
# 2) Exact on-axis field of one loop and its integral along z
# ----------------------------------------------------------------------------
def bz_loop_axis(z_m, radius_m, z0_m, current_A):
    u = np.asarray(z_m) - z0_m
    return MU0 * current_A * radius_m**2 / (2.0 * (radius_m**2 + u**2) ** 1.5)


def int_bz_loop_axis(z1_m, z2_m, radius_m, z0_m, current_A):
    u1, u2 = z1_m - z0_m, z2_m - z0_m
    return MU0 * current_A / 2.0 * (u2 / math.hypot(radius_m, u2) - u1 / math.hypot(radius_m, u1))


# ----------------------------------------------------------------------------
# 3) Winding geometry
# ----------------------------------------------------------------------------
def winding():
    """Radius (m), axial centre (m) of every turn, plus layer data for the helices."""
    d_ins = (WIRE_MM + ENAMEL_EXTRA_MM) * 1e-3
    pitch = d_ins * AXIAL_K
    r0 = (BORE_MM / 2 + WALL_MM) * 1e-3 + d_ins / 2
    length = TURNS_PER_LAYER * pitch
    radii, zs, layers = [], [], []
    for k in range(LAYERS):
        r = r0 + k * d_ins
        z = -length / 2 + pitch / 2 + np.arange(TURNS_PER_LAYER) * pitch
        radii.append(np.full(TURNS_PER_LAYER, r))
        zs.append(z)
        direction = +1 if k % 2 == 0 else -1
        z_start = -length / 2 if direction > 0 else +length / 2
        layers.append((r, z_start, pitch, direction))
    return np.concatenate(radii), np.concatenate(zs), layers, length, pitch


def bz_coil_axis(z_m, radii, zs, current_A):
    z_m = np.atleast_1d(z_m)
    return np.array([np.sum(bz_loop_axis(z, radii, zs, current_A)) for z in z_m])


def bz_coil_helical_numeric(z_m, layers, current_A):
    """Full helical-wire Biot-Savart at (0, 0, z_m), integrated turn by turn."""
    B = np.zeros(3)
    for (r, z_start, pitch, direction) in layers:
        curve, dcurve = helix_curve(r, z_start, pitch, direction)
        for n in range(TURNS_PER_LAYER):
            B += biot_savart_parametric_curve(curve, dcurve, 2 * math.pi * n, 2 * math.pi * (n + 1),
                                              [0.0, 0.0, z_m], current_A)
    return B


# ----------------------------------------------------------------------------
# 4) Run
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # (a) single-loop check: numerical integral vs closed form
    c, dc = circular_loop_curve(0.025, 0.004)
    b_num = biot_savart_parametric_curve(c, dc, 0, 2 * math.pi, [0, 0, 0.037], 3.2)
    b_ex = bz_loop_axis(0.037, 0.025, 0.004, 3.2)
    print(f"single loop: quad_vec Bz = {b_num[2]:.6e} T, exact = {b_ex:.6e} T, rel. error = {abs(b_num[2]/b_ex-1):.1e}")

    # (b) the coil
    radii, zs, layers, length, pitch = winding()
    N = radii.size
    w = OIL_PATH_MM / 2 * 1e-3
    print(f"\ncoil: N = {N}, winding length = {length*1e3:.1f} mm, pitch = {pitch*1e3:.3f} mm, "
          f"layer radii {radii.min()*1e3:.2f}-{radii.max()*1e3:.2f} mm")

    z = np.linspace(-0.25, 0.25, 1001)
    bz = bz_coil_axis(z, radii, zs, CURRENT_A)
    int_bz = sum(int_bz_loop_axis(-w, w, r, z0, CURRENT_A) for r, z0 in zip(radii, zs))
    eta = int_bz / (MU0 * N * CURRENT_A)
    theta = VERDET_DEG_PER_TM * int_bz
    print(f"I = {CURRENT_A} A:  B(0) = {bz_coil_axis(0.0, radii, zs, CURRENT_A)[0]*1e3:.2f} mT, "
          f"B(oil end) = {bz_coil_axis(w, radii, zs, CURRENT_A)[0]*1e3:.2f} mT")
    print(f"int Bz dz over {OIL_PATH_MM:.0f} mm of oil = {int_bz*1e3:.3f} mT m  (eta = {eta:.4f})")
    print(f"expected Faraday rotation = {theta:.3f} deg  ({theta/CURRENT_A:.3f} deg/A)")

    # (c) loop approximation vs full helical wire at a few points
    print("\nloop sum vs full helical Biot-Savart (numerical, may take ~1 min):")
    t0 = time.time()
    for zp in (0.0, 0.10, w):
        b_loop = bz_coil_axis(zp, radii, zs, CURRENT_A)[0]
        b_hel = bz_coil_helical_numeric(zp, layers, CURRENT_A)
        print(f"  z = {zp*100:5.1f} cm: loops {b_loop*1e3:.4f} mT, helix {b_hel[2]*1e3:.4f} mT, "
              f"diff {abs(b_hel[2]/b_loop-1)*100:.3f} %, |B_perp| on axis {math.hypot(b_hel[0], b_hel[1])*1e6:.2f} uT")
    print(f"  ({time.time()-t0:.0f} s)")

    # (d) save profile
    np.savetxt("bz_profile.csv", np.c_[z * 1e3, bz * 1e3], delimiter=",", header="z_mm,Bz_mT", comments="")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(z * 100, bz * 1e3)
        ax.axvspan(-w * 100, w * 100, alpha=0.15, label="oil")
        ax.set_xlabel("z (cm)")
        ax.set_ylabel("Bz on axis (mT)")
        ax.legend()
        fig.tight_layout()
        fig.savefig("bz_profile.png", dpi=150)
        print("\nsaved bz_profile.csv and bz_profile.png")
    except ImportError:
        print("\nsaved bz_profile.csv (matplotlib not installed, no plot)")
