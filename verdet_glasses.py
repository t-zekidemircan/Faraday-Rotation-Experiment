"""
Verdet constants of optical glasses from the Becquerel formula
==============================================================

    V_B(lambda) = (e / 2 m_e c) * lambda * |dn/dlambda|

dn/dlambda is obtained from the Schott Sellmeier equation
    n^2 - 1 = B1 l^2/(l^2 - C1) + B2 l^2/(l^2 - C2) + B3 l^2/(l^2 - C3)     (l in um)
    dn/dl   = -(l/n) [ B1 C1/(l^2-C1)^2 + B2 C2/(l^2-C2)^2 + B3 C3/(l^2-C3)^2 ]

These are classical estimates; real glasses have V = gamma * V_B with gamma < 1
(magneto-optic anomaly).  Olive oil is given by a Drude-type fit to published
data (Carr et al. 2020): V = 7.9e7 / (lambda^2 - 142^2) deg/(T m), lambda in nm.

The script also gives the expected rotation of each sample when it is placed
in the centre of our solenoid (1976 turns, 332 mm, 8 layers of 1.2 mm wire, 6 A).
"""
import math
import numpy as np

E_CHARGE, M_E, C_LIGHT = 1.602176634e-19, 9.1093837139e-31, 299792458.0
MU0 = 4e-7 * math.pi

# Schott Sellmeier coefficients (lambda in um): B1, B2, B3, C1, C2, C3
SELLMEIER = {
    "N-F2":    (1.39757037, 0.159201403, 1.2686543,   0.00995906143, 0.0546931752, 119.248346),
    "N-FK5":   (0.844309338, 0.344147824, 0.910790213, 0.00475111955, 0.0149814849, 97.8600293),
    "LLF1":    (1.2164012,  0.13366454,  0.883399468, 0.00857807248, 0.0420143003, 107.59306),
    "N-ZK7A":  (1.07509891, 0.168895044, 0.860503983, 0.00676601657, 0.0230642817, 89.0498778),
    "N-BK7":   (1.03961212, 0.231792344, 1.01046945,  0.00600069867, 0.0200179144, 103.560653),
    "N-BAF51": (1.51503623, 0.153621958, 1.15427909,  0.00942734715, 0.04308265,   124.889868),
    "N-PK52A": (1.029607,   0.1880506,   0.736488165, 0.00516800155, 0.0166658798, 138.964129),
    "N-SSK5":  (1.59222659, 0.103520774, 1.05174016,  0.00920284626, 0.0423530072, 106.927374),
}
# sample lengths we can obtain (mm); N-F2 not available
LENGTH_MM = {"N-FK5": 140, "LLF1": 200, "N-ZK7A": 200, "N-BK7": 107,
             "N-BAF51": 60, "N-PK52A": 70, "N-SSK5": 160}
ORDERED = ("N-BAF51", "N-SSK5", "LLF1")


def n_and_dndl(glass, lam_nm):
    B1, B2, B3, C1, C2, C3 = SELLMEIER[glass]
    l = lam_nm * 1e-3
    l2 = l * l
    n = math.sqrt(1 + B1 * l2 / (l2 - C1) + B2 * l2 / (l2 - C2) + B3 * l2 / (l2 - C3))
    dndl = -(l / n) * (B1 * C1 / (l2 - C1) ** 2 + B2 * C2 / (l2 - C2) ** 2 + B3 * C3 / (l2 - C3) ** 2)
    return n, dndl * 1e6                       # dn/dlambda in 1/m


def verdet_becquerel(glass, lam_nm):
    """Return (n, V in rad/(T m), V in deg/(T m))."""
    n, dndl = n_and_dndl(glass, lam_nm)
    V = E_CHARGE / (2 * M_E * C_LIGHT) * lam_nm * 1e-9 * abs(dndl)
    return n, V, math.degrees(V)


def verdet_olive_oil(lam_nm):
    return 7.9e7 / (lam_nm ** 2 - 142.0 ** 2)   # deg/(T m)


def field_integral_centred(length_m, N=1976, I=6.0, wire_mm=1.2, layers=8, tpl=247):
    """int Bz dz (T m) over a sample of given length centred in our solenoid."""
    d_ins = (wire_mm + 0.08) * 1e-3
    L = tpl * d_ins * 1.05
    radii = 17.4e-3 + (np.arange(layers) + 0.5) * d_ins
    w = length_m / 2
    eta = np.mean([(math.hypot(w + L / 2, a) - math.hypot(w - L / 2, a)) / L for a in radii])
    return MU0 * N * I * eta


if __name__ == "__main__":
    lam = 520.0
    V_oil = verdet_olive_oil(lam)
    V_ref = verdet_becquerel("N-F2", lam)[2]
    print(f"lambda = {lam:.0f} nm,  olive oil (Drude fit): V = {V_oil:.0f} deg/(T m),  N-F2: V = {V_ref:.0f} deg/(T m)\n")
    print(f"{'glass':8s} {'n':>8s} {'V (rad/Tm)':>11s} {'V (deg/Tm)':>11s} {'/oil':>6s} {'length':>7s} {'theta (deg)':>11s}")
    for g in SELLMEIER:
        n, Vr, Vd = verdet_becquerel(g, lam)
        Lmm = LENGTH_MM.get(g)
        th = f"{Vd * field_integral_centred(Lmm / 1000):11.2f}" if Lmm else f"{'-':>11s}"
        mark = "  <- ordered" if g in ORDERED else ""
        print(f"{g:8s} {n:8.5f} {Vr:11.2f} {Vd:11.1f} {Vd / V_oil:6.2f} {str(Lmm or '-'):>7s} {th}{mark}")
    print(f"{'oil':8s} {'':8s} {math.radians(V_oil):11.2f} {V_oil:11.1f} {1:6.2f} {340:>7d} {V_oil * field_integral_centred(0.340):11.2f}")
