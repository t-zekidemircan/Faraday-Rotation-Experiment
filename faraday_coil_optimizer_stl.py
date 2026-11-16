"""
Faraday experiment - single-coil optimizer for the printed (STL) bobbin
======================================================================

The printed parts fix the coil geometry:
  barrel (two halves, lap joint) : bore 30 mm, wall 2.4 mm  -> winding starts at r = 17.4 mm
  flanges                        : OD 64 mm, 4 mm thick; coil 340 mm long including flanges
                                   -> winding length (between flange inner faces) = 332 mm
  oil tank                       : 35 cm aquarium; oil path = tank inner length (set OIL_PATH_MM)
So the only free choices are the wire diameter and the number of turns
(how many layers, and whether the last layer is full).

Goal: maximise the Faraday rotation in the oil
    theta = V * integral_{-w}^{+w} Bz(z) dz          (w = half the oil path)
with a 30 V / 6 A supply.

Physics model
-------------
* Each turn = a circular current loop at its own radius r_i and axial position z_i.
  On the axis the Biot-Savart integral of one loop has the closed form
      Bz(z) = mu0 I r^2 / (2 (r^2 + (z - z_i)^2)^(3/2))
  and its integral over the oil is
      int Bz dz = (mu0 I / 2) [ u/sqrt(r^2+u^2) ]_{u=-w-z_i}^{u=w-z_i}
  Summing over all turns gives K = int Bz dz per ampere (T m / A).
* Wire length = sum over turns of sqrt((2 pi r)^2 + pitch^2).
* Copper resistance at 60 C (warm coil): R = rho20 (1 + alpha (T - 20)) * length / area.
* Current = min(6 A, 30 V / R): the supply limits either the current or the voltage.
* theta = V_oil * K * I, with V_oil = 316 deg/(T m) at 520 nm.

Constraints
-----------
* outer diameter of the winding <= flange OD - 2 * FLANGE_LIP_MM (wire must not reach the flange rim)
* copper mass <= MAX_CU_KG (cost / weight), current density <= MAX_J (heating)
"""

import math
import numpy as np

MU0 = 4e-7 * math.pi
RHO20, ALPHA_CU, DENS_CU = 1.724e-8, 0.00393, 8960.0       # copper

# ---------------- fixed by the printed parts (STL) ----------------
BORE_MM, WALL_MM = 30.0, 2.4
WINDING_LENGTH_MM = 332.0      # 340 mm total - 2 x 4 mm flanges
FLANGE_OD_MM = 64.0
OIL_PATH_MM = 340.0            # tank inner length; measure and correct

# ---------------- wire and winding assumptions ----------------
ENAMEL_EXTRA_MM = 0.08      # insulated d = copper d + 0.08 mm (grade-2 enamel); check the datasheet
AXIAL_K = 1.05              # real pitch is ~5 % larger than the insulated diameter (hand winding)
FLANGE_LIP_MM = 2.0         # keep 2 mm of flange above the last layer

# ---------------- supply and limits ----------------
V_MAX, I_MAX = 30.0, 6.0
T_HOT_C = 60.0
MAX_CU_KG = 3.5
MAX_J_A_MM2 = 8.0
VERDET = 316.0              # deg/(T m), olive oil, 520 nm
OIL_HEAT_CAP_J_K = 2.39 * 0.91 * 1970     # 2.4 L of oil
WIRES_MM = (0.8, 0.9, 1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5)


def winding(wire_mm, n_turns):
    """Radius and axial position (m) of every turn, filled layer by layer."""
    d_ins = (wire_mm + ENAMEL_EXTRA_MM) * 1e-3
    pitch = d_ins * AXIAL_K
    tpl = int(math.floor(WINDING_LENGTH_MM * 1e-3 / pitch + 1e-9))    # turns per layer
    r0 = (BORE_MM / 2 + WALL_MM) * 1e-3 + d_ins / 2
    z_full = -tpl * pitch / 2 + pitch / 2 + np.arange(tpl) * pitch
    r, z, left, k = [], [], n_turns, 0
    while left > 0:
        n = min(tpl, left)
        r.append(np.full(n, r0 + k * d_ins))
        z.append(z_full[:n] if k % 2 == 0 else z_full[::-1][:n])        # layers wound back and forth
        left -= n
        k += 1
    return np.concatenate(r), np.concatenate(z), tpl, k, d_ins, pitch


def evaluate(wire_mm, n_turns):
    r, z, tpl, layers, d_ins, pitch = winding(wire_mm, n_turns)
    w = OIL_PATH_MM / 2 * 1e-3
    u2, u1 = w - z, -w - z
    K = MU0 / 2 * np.sum(u2 / np.hypot(r, u2) - u1 / np.hypot(r, u1))   # T m per A

    length = np.sum(np.sqrt((2 * math.pi * r) ** 2 + pitch ** 2))
    area = math.pi * (wire_mm * 1e-3 / 2) ** 2
    R_hot = RHO20 * (1 + ALPHA_CU * (T_HOT_C - 20)) * length / area
    R_cold = RHO20 * length / area
    I = min(I_MAX, V_MAX / R_hot)
    P = I * I * R_hot
    od = 2 * (r.max() + d_ins / 2) * 1e3
    return dict(wire=wire_mm, N=n_turns, tpl=tpl, layers=layers, full=(n_turns % tpl == 0),
                OD=od, length_m=length, Cu_kg=length * area * DENS_CU,
                R20=R_cold, R60=R_hot, I=I, V=I * R_hot, P=P, J=I / (area * 1e6),
                limit="current" if I >= I_MAX - 1e-9 else "voltage",
                K=K, eta=K / (MU0 * n_turns), intB=K * I, theta=VERDET * K * I,
                heat_K_min=P / (OIL_HEAT_CAP_J_K + length * area * DENS_CU * 385) * 60)


def feasible(d):
    return (d["OD"] <= FLANGE_OD_MM - 2 * FLANGE_LIP_MM and d["Cu_kg"] <= MAX_CU_KG and d["J"] <= MAX_J_A_MM2)


def choose_balanced(designs, frac=0.90):
    """Same rule as the original optimizer: among designs within `frac` of the
    best rotation, take the one with the least copper (cheapest, lightest)."""
    tmax = max(d["theta"] for d in designs)
    near = [d for d in designs if d["theta"] >= frac * tmax]
    return min(near, key=lambda d: d["Cu_kg"])


def search():
    rows = []
    for wire in WIRES_MM:
        tpl = winding(wire, 1)[2]
        for n in range(tpl, 20 * tpl + 1):           # at least one full layer
            d = evaluate(wire, n)
            if d["OD"] > FLANGE_OD_MM:                # cannot fit at all
                break
            rows.append(d)
    return rows


if __name__ == "__main__":
    rows = search()
    ok = [d for d in rows if feasible(d)]
    full = [d for d in ok if d["full"]]
    hdr = f"{'wire':>5} {'N':>5} {'lay':>3} {'OD':>6} {'Cu kg':>6} {'R60':>6} {'I':>5} {'V':>5} {'P':>5} {'limit':>8} {'eta':>6} {'intB':>6} {'theta':>6} {'K/min':>5}"
    fmt = lambda d: (f"{d['wire']:5.2f} {d['N']:5d} {d['layers']:3d} {d['OD']:6.1f} {d['Cu_kg']:6.2f} {d['R60']:6.2f} "
                     f"{d['I']:5.2f} {d['V']:5.1f} {d['P']:5.0f} {d['limit']:>8} {d['eta']:6.3f} {d['intB']*1e3:6.2f} "
                     f"{d['theta']:6.2f} {d['heat_K_min']:5.2f}")
    print(f"STL limits: bore {BORE_MM} mm, winding length {WINDING_LENGTH_MM} mm, max winding OD "
          f"{FLANGE_OD_MM - 2*FLANGE_LIP_MM} mm, oil path {OIL_PATH_MM} mm, supply {V_MAX} V / {I_MAX} A\n")
    print("Best FULL-LAYER design for each wire diameter:")
    print(hdr)
    for wire in WIRES_MM:
        c = [d for d in full if d["wire"] == wire]
        if c:
            print(fmt(max(c, key=lambda d: d["theta"])))
    print("\nBest design for each wire if the last layer may be partial:")
    print(hdr)
    for wire in WIRES_MM:
        c = [d for d in ok if d["wire"] == wire]
        if c:
            print(fmt(max(c, key=lambda d: d["theta"])))
    b = max(full, key=lambda d: d["theta"])
    c = choose_balanced(full)
    for tag, d in (("Maximum rotation", b), ("Chosen (>=90% of max, least copper)", c)):
        print(f"\n{tag}: {d['wire']} mm wire, {d['layers']} layers x {d['tpl']} = {d['N']} turns, "
              f"OD {d['OD']:.1f} mm, Cu {d['Cu_kg']:.2f} kg ({d['length_m']:.0f} m), "
              f"I = {d['I']:.2f} A ({d['limit']}-limited), P = {d['P']:.0f} W, theta = {d['theta']:.2f} deg")
