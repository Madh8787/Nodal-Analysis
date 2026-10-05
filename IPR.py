# =====================================================
# Petroleum Production Engineering Project
# IPR & VLP Comparison for Two Wells
# Case: Well A Producing Below Bubble Point
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# -------------------------------
# Step 1: Define Well Data
# -------------------------------

wells = {
    "Well A": {
        "Pr": 3000,    # psia
        "Pb": 2500,    # psia (Bubble Point Pressure)
        "J": 2.5,      # STB/day/psi
        "depth": 7000, # ft
        "rho": 52,     # lb/ft3
        "friction_grad": 0.05 # psi/ft
    },
    "Well B": {
        "Pr": 2800,
        "Pb": 2400,
        "J": 1.8,
        "depth": 8500,
        "rho": 50,
        "friction_grad": 0.07
    }
}

results = {}

# -------------------------------
# Step 2: IPR + VLP Calculations
# -------------------------------

for well, data in wells.items():
    Pr, Pb, J = data["Pr"], data["Pb"], data["J"]
    depth, rho, fg = data["depth"], data["rho"], data["friction_grad"]

    # Flowing pressure range
    Pwf = np.linspace(0, Pr, 50)

    # IPR calculation
    if well == "Well A":  # Below bubble point → Vogel equation
        Q = J * Pr * (1 - 0.2*(Pwf/Pr) - 0.8*(Pwf/Pr)**2)
        ipr_label = "Vogel IPR (Below Pb)"
    else:  # Well B → Above bubble point → Darcy linear IPR
        Q = J * (Pr - Pwf)
        ipr_label = "Darcy IPR (Above Pb)"

    # VLP calculation
    hydro_grad = rho/144
    total_grad = hydro_grad + fg
    Pwf_vlp = Pr - (total_grad * depth)

    Q_max = max(Q) * 0.9
    Q_vlp = np.linspace(0, Q_max, 50)
    Pwf_vlp_curve = np.linspace(Pwf_vlp, Pr, 50)

    # Operating point (intersection)
    f_ipr = interp1d(Q, Pwf, bounds_error=False, fill_value="extrapolate")
    f_vlp = interp1d(Q_vlp, Pwf_vlp_curve, bounds_error=False, fill_value="extrapolate")
    Q_common = np.linspace(0, min(max(Q), max(Q_vlp)), 100)
    diff = abs(f_ipr(Q_common) - f_vlp(Q_common))
    Q_op = Q_common[np.argmin(diff)]
    Pwf_op = f_ipr(Q_op)

    # Save results
    results[well] = {
        "Q": Q, "Pwf": Pwf,
        "Q_vlp": Q_vlp, "Pwf_vlp": Pwf_vlp_curve,
        "Q_op": Q_op, "Pwf_op": Pwf_op,
        "ipr_label": ipr_label
    }

# -------------------------------
# Step 3: Plot Separate Graphs
# -------------------------------

for well, res in results.items():
    plt.figure(figsize=(8,6))
    plt.plot(res["Q"], res["Pwf"], label=f"{well} {res['ipr_label']}", lw=2)
    plt.plot(res["Q_vlp"], res["Pwf_vlp"], "r", lw=2, alpha=0.7, label=f"{well} VLP")
    plt.scatter(res["Q_op"], res["Pwf_op"], marker="o", s=80, color="black", label="Operating Point")

    plt.gca().invert_yaxis()
    plt.xlabel("Flowrate (STB/day)")
    plt.ylabel("Bottomhole Flowing Pressure (psi)")
    plt.title(f"IPR–VLP Curve for {well}")
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------------
# Step 4: Export to CSV
# -------------------------------

all_data = []
for well, res in results.items():
    df = pd.DataFrame({
        "Q": res["Q"],
        "Pwf": res["Pwf"],
        "Q_vlp": np.pad(res["Q_vlp"], (0, len(res["Q"])-len(res["Q_vlp"])), constant_values=np.nan),
        "Pwf_vlp": np.pad(res["Pwf_vlp"], (0, len(res["Pwf"])-len(res["Pwf_vlp"])), constant_values=np.nan),
        "Well": well
    })
    all_data.append(df)

df_export = pd.concat(all_data, ignore_index=True)
df_export.to_csv("Two_Well_IPR_VLP.csv", index=False)
print("✅ Results exported to Two_Well_IPR_VLP.csv")

# Print operating points
for well, res in results.items():
    print(f"{well}: Q = {res['Q_op']:.1f} STB/day, Pwf = {res['Pwf_op']:.1f} psi")
