import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="IS 10262:2019 Concrete Mix Design | IIT Mandi CE203",
    page_icon="🏗️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
.main-title { text-align: center; font-size: 28px; font-weight: 800; color: #1E3A8A; margin-bottom: 0px;}
.sub-title { text-align: center; font-size: 16px; color: #4B5563; margin-bottom: 2px;}
.group-title { text-align: center; font-size: 14px; font-weight: 600; color: #2563EB; margin-bottom: 15px;}
.metric-card { background: #f8fafc; border: 1px solid #e2e8f0; padding: 12px; border-radius: 8px; text-align: center;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">INDIAN INSTITUTE OF TECHNOLOGY MANDI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Civil Engineering Department | <b>Course: Civil Engineering Materials CE203</b></div>', unsafe_allow_html=True)
st.markdown('<div class="group-title">Group 5 | Concrete Mix Design Calculator (IS 10262:2019 & IS 456:2000)</div>', unsafe_allow_html=True)
st.divider()

# =========================================================
# STANDARD DATA (IS 10262 & IS 456)
# =========================================================
# Table 2: Assumed Standard Deviation S (Clause 4.2.1.3)
S_VALUES = {
    "M10": 3.5, "M15": 3.5, "M20": 4.0, "M25": 4.0,
    "M30": 5.0, "M35": 5.0, "M40": 5.0, "M45": 5.0,
    "M50": 5.0, "M55": 6.0, "M60": 6.0
}

# Table 1: Value of X (Clause 4.2)
X_VALUES = {
    "M10": 5.0, "M15": 5.0, "M20": 5.5, "M25": 5.5,
    "M30": 6.5, "M35": 6.5, "M40": 6.5, "M45": 6.5,
    "M50": 6.5, "M55": 6.5, "M60": 6.5
}

# Table 4: Water Content per m3 for 50mm slump
WATER_50 = {10: 208, 20: 186, 40: 165}

# Table 3: Approximate Entrapped Air Content
AIR = {10: 0.015, 20: 0.010, 40: 0.008}

# Table 5: CA volume fraction for W/C = 0.50
CA_TABLE = {
    10: {"Zone I": 0.48, "Zone II": 0.50, "Zone III": 0.52, "Zone IV": 0.54},
    20: {"Zone I": 0.60, "Zone II": 0.62, "Zone III": 0.64, "Zone IV": 0.66},
    40: {"Zone I": 0.69, "Zone II": 0.71, "Zone III": 0.72, "Zone IV": 0.73}
}

# IS 456 Table 5: Durability Requirements (Reinforced Concrete)
EXPOSURE = {
    "Mild": {"min_cement": 300, "max_wc": 0.55, "min_grade": 20},
    "Moderate": {"min_cement": 300, "max_wc": 0.50, "min_grade": 25},
    "Severe": {"min_cement": 320, "max_wc": 0.45, "min_grade": 30},
    "Very Severe": {"min_cement": 340, "max_wc": 0.45, "min_grade": 35},
    "Extreme": {"min_cement": 360, "max_wc": 0.40, "min_grade": 40}
}

# =========================================================
# SIDEBAR INPUTS
# =========================================================
st.sidebar.header("🔧 Mix Specifications")

grade = st.sidebar.selectbox("Concrete Grade", list(S_VALUES.keys()), index=6) # Default M40
fck = int(grade.replace("M", ""))

exposure = st.sidebar.selectbox("Exposure Condition (IS 456)", list(EXPOSURE.keys()), index=2) # Default Severe
max_agg = st.sidebar.selectbox("Nominal Max Aggregate Size (mm)", [10, 20, 40], index=1)
zone = st.sidebar.selectbox("Fine Aggregate Zone", ["Zone I", "Zone II", "Zone III", "Zone IV"], index=1)
slump = st.sidebar.slider("Workability / Slump (mm)", min_value=25, max_value=175, value=75, step=25)

st.sidebar.subheader("Curve Selection (Fig. 1)")
curve_type = st.sidebar.selectbox(
    "Reference Curve",
    ["Curve 1 (OPC 33 / Low)", "Curve 2 (OPC 43 / PPC Default)", "Curve 3 (OPC 53 / High)"],
    index=1
)

st.sidebar.subheader("Chemical Admixture")
use_sp = st.sidebar.checkbox("Use Superplasticizer", value=True)
if use_sp:
    sp_dosage = st.sidebar.number_input("Dosage (% by wt of cement)", 0.1, 3.0, 1.0, 0.1)
    water_reduction = st.sidebar.number_input("Water Reduction (%)", 0.0, 40.0, 23.0, 1.0)
    sg_sp = st.sidebar.number_input("Admixture Specific Gravity", 0.90, 1.40, 1.145, 0.005)
else:
    sp_dosage = 0.0
    water_reduction = 0.0
    sg_sp = 1.145

st.sidebar.subheader("Material Specific Gravity")
sg_cement = st.sidebar.number_input("Cement (PPC)", 2.50, 3.20, 2.88, 0.01)
sg_ca = st.sidebar.number_input("Coarse Aggregate (SSD)", 2.40, 3.00, 2.74, 0.01)
sg_fa = st.sidebar.number_input("Fine Aggregate (SSD)", 2.40, 3.00, 2.65, 0.01)

st.sidebar.subheader("Moisture & Absorption (Field Correction)")
ca_absorption = st.sidebar.number_input("CA Water Absorption (%)", 0.0, 5.0, 0.5, 0.1)
fa_absorption = st.sidebar.number_input("FA Water Absorption (%)", 0.0, 5.0, 1.0, 0.1)

# =========================================================
# CALCULATIONS
# =========================================================
# 1. Target Mean Strength (Clause 4.2)
S = S_VALUES[grade]
X = X_VALUES[grade]
t1 = fck + 1.65 * S
t2 = fck + X
f_target = max(t1, t2)

# 2. Water-Cement Ratio from Fig. 1 Interpolation
# Curve functions mapping target strength to free w/c ratio
def get_wc_from_curve(f_target, curve_choice):
    # Parametric equations closely fitting IS 10262 Fig. 1 curves
    if "Curve 1" in curve_choice:
        wc = 0.25 + (60 - f_target) * (0.40 / 48)
    elif "Curve 2" in curve_choice:
        # Curve 2 passes through ~48.25 MPa at w/c = 0.36
        wc = 0.36 + (48.25 - f_target) * 0.0105
    else:
        # Curve 3 (OPC 53)
        wc = 0.40 + (50 - f_target) * 0.009
    return float(np.clip(wc, 0.28, 0.65))

calculated_wc = get_wc_from_curve(f_target, curve_type)
max_wc_durability = EXPOSURE[exposure]["max_wc"]
min_cement_durability = EXPOSURE[exposure]["min_cement"]

# Selection of governing w/c
adopted_wc = min(calculated_wc, max_wc_durability)

# 3. Water Content (Clause 5.3)
base_water = WATER_50[max_agg]
if slump > 50:
    water_slump_adjusted = base_water + (base_water * 0.03 * ((slump - 50) / 25))
else:
    water_slump_adjusted = base_water

final_water = water_slump_adjusted * (1 - (water_reduction / 100.0))

# 4. Cement Content (Clause 5.4)
cement_from_wc = final_water / adopted_wc
cement = max(cement_from_wc, min_cement_durability)
actual_wc = final_water / cement

# Admixture mass
sp_mass = cement * (sp_dosage / 100.0)

# 5. Coarse Aggregate Fraction (Clause 5.5)
base_ca_ratio = CA_TABLE[max_agg][zone]
# Correction: for every ±0.05 change in w/c from 0.50, change CA volume fraction by ∓0.01
wc_diff = 0.50 - actual_wc
ca_correction = (wc_diff / 0.05) * 0.01
ca_fraction = float(np.clip(base_ca_ratio + ca_correction, 0.35, 0.85))
fa_fraction = 1.0 - ca_fraction

# 6. Absolute Volume Method (Clause 5.6)
v_air = AIR[max_agg]
v_cement = cement / (sg_cement * 1000.0)
v_water = final_water / 1000.0
v_sp = sp_mass / (sg_sp * 1000.0) if use_sp else 0.0

v_total_agg = 1.0 - (v_air + v_cement + v_water + v_sp)
v_ca = v_total_agg * ca_fraction
v_fa = v_total_agg * fa_fraction

# SSD Masses
ca_mass_ssd = v_ca * sg_ca * 1000.0
fa_mass_ssd = v_fa * sg_fa * 1000.0

# 7. Field Adjustments for Dry Aggregates (CE203 Annex A-11)
ca_dry = ca_mass_ssd / (1.0 + (ca_absorption / 100.0))
fa_dry = fa_mass_ssd / (1.0 + (fa_absorption / 100.0))
extra_water = (ca_mass_ssd - ca_dry) + (fa_mass_ssd - fa_dry)
field_water = final_water + extra_water

# =========================================================
# UI LAYOUT & DASHBOARD
# =========================================================
top_col1, top_col2, top_col3, top_col4 = st.columns(4)
top_col1.metric("Design Target Strength", f"{f_target:.2f} MPa")
top_col2.metric("Adopted W/C Ratio", f"{actual_wc:.3f}", delta=f"{actual_wc - max_wc_durability:.3f} vs code limit", delta_color="inverse")
top_col3.metric("Cement Content", f"{cement:.1f} kg/m³", delta=f"{cement - min_cement_durability:.1f} kg excess")
top_col4.metric("Water Content", f"{final_water:.1f} kg/m³")

st.write("")

# Layout: Left for Calculation Details, Right for Graphs
left_pane, right_pane = st.columns([1.1, 1.2])

with left_pane:
    st.subheader("📋 Step-by-Step Mix Proportioning")
    
    with st.expander("Step 1: Target Mean Strength (Clause 4.2)", expanded=True):
        st.write(f"• $f'_{{ck}} = f_{{ck}} + 1.65S = {fck} + 1.65 \\times {S} = {t1:.2f}\\text{{ MPa}}$")
        st.write(f"• $f'_{{ck}} = f_{{ck}} + X = {fck} + {X} = {t2:.2f}\\text{{ MPa}}$")
        st.success(f"**Target Mean Strength = {f_target:.2f} MPa**")

    with st.expander("Step 2: Durability & W/C Ratio (IS 456 Table 5)", expanded=True):
        st.write(f"• Exposure: **{exposure}**")
        st.write(f"• Max Free W/C Limit = **{max_wc_durability:.2f}** | Minimum Cement = **{min_cement_durability} kg/m³**")
        st.write(f"• Curve-derived W/C = **{calculated_wc:.3f}**")
        st.info(f"**Adopted W/C Ratio = {adopted_wc:.3f}** (governed by {'Strength Curve' if adopted_wc < max_wc_durability else 'Durability Limit'})")

    with st.expander("Step 3: Water Content & Admixture Correction"):
        st.write(f"• Base water for {max_agg}mm aggregate at 50mm slump: **{base_water} kg/m³**")
        st.write(f"• Adjusted for {slump}mm slump: **{water_slump_adjusted:.2f} kg/m³**")
        if use_sp:
            st.write(f"• Admixture reduction ({water_reduction}%): **-{water_slump_adjusted * (water_reduction/100):.2f} kg**")
        st.write(f"• **Final Water = {final_water:.2f} kg/m³**")

    with st.expander("Step 4: Absolute Volume Proportions"):
        st.write(f"• Air Volume: **{v_air:.3f} m³**")
        st.write(f"• Cement Volume: **{v_cement:.4f} m³** | Water Volume: **{v_water:.4f} m³**")
        st.write(f"• Total Aggregate Volume: **{v_total_agg:.4f} m³**")
        st.write(f"• CA Volume Fraction (corrected): **{ca_fraction:.3f}** | FA: **{fa_fraction:.3f}**")

with right_pane:
    st.subheader("📈 Reference Graphs (IS 10262 & CE203)")

    # 1. Figure 1 Curve Generation
    fig, ax = plt.subplots(figsize=(6, 3.4))
    wc_range = np.linspace(0.25, 0.65, 100)
    
    # Representative IS 10262 curves
    c1 = 60 - (wc_range - 0.25) * 115
    c2 = 72 - (wc_range - 0.25) * 118
    c3 = 85 - (wc_range - 0.25) * 122

    ax.plot(wc_range, c1, label="Curve 1 (OPC 33)", color="#9ca3af", linestyle="--")
    ax.plot(wc_range, c2, label="Curve 2 (OPC 43 / PPC)", color="#2563eb", linewidth=2)
    ax.plot(wc_range, c3, label="Curve 3 (OPC 53)", color="#475569", linestyle=":")
    
    # Mark design point
    ax.scatter([actual_wc], [f_target], color="red", zorder=5, s=60, label=f"Design Point ({actual_wc:.2f}, {f_target:.1f} MPa)")
    ax.axvline(actual_wc, color="red", linestyle="--", alpha=0.5)
    ax.axhline(f_target, color="red", linestyle="--", alpha=0.5)

    ax.set_xlim(0.25, 0.65)
    ax.set_ylim(10, 80)
    ax.set_xlabel("Free Water-Cement Ratio", fontsize=9)
    ax.set_ylabel("28-day Compressive Strength (N/mm²)", fontsize=9)
    ax.set_title("IS 10262:2019 Fig. 1: Strength vs W/C", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper right")
    st.pyplot(fig)

    # 2. Material Quantities Bar Chart
    fig2, ax2 = plt.subplots(figsize=(6, 2.5))
    materials = ["Cement", "Water", "Fine Agg (SSD)", "Coarse Agg (SSD)"]
    masses = [cement, final_water, fa_mass_ssd, ca_mass_ssd]
    colors = ["#475569", "#38bdf8", "#fbbf24", "#3b82f6"]

    bars = ax2.bar(materials, masses, color=colors, width=0.55)
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 15, f"{int(yval)}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2.set_ylim(0, max(masses) * 1.25)
    ax2.set_ylabel("kg/m³", fontsize=9)
    ax2.set_title("Mix Quantities per Unit Volume (SSD Condition)", fontsize=10, fontweight="bold")
    ax2.grid(axis='y', linestyle=":", alpha=0.6)
    st.pyplot(fig2)

# =========================================================
# FINAL OUTPUT TABLES & COMPARISONS
# =========================================================
st.divider()
st.subheader("⚖️ Mix Proportions Summary & Dry Batch Adjustments")

col_table1, col_table2 = st.columns(2)

with col_table1:
    st.markdown("#### Initial Mix (SSD Condition)")
    ssd_df = pd.DataFrame({
        "Material": ["Cement", "Water", "Fine Aggregate (SSD)", "Coarse Aggregate (SSD)", "Chemical Admixture"],
        "Mass (kg/m³)": [f"{cement:.2f}", f"{final_water:.2f}", f"{fa_mass_ssd:.2f}", f"{ca_mass_ssd:.2f}", f"{sp_mass:.2f}"],
        "Ratio (by mass)": ["1.00", f"{actual_wc:.2f}", f"{fa_mass_ssd/cement:.2f}", f"{ca_mass_ssd/cement:.2f}", f"{sp_mass/cement:.3f}"]
    })
    st.table(ssd_df)

with col_table2:
    st.markdown("#### Actual Field Batching (Dry Condition Adjusted)")
    dry_df = pd.DataFrame({
        "Material": ["Cement", "Water to be Added", "Dry Fine Aggregate", "Dry Coarse Aggregate", "Chemical Admixture"],
        "Corrected Mass (kg/m³)": [f"{cement:.2f}", f"{field_water:.2f}", f"{fa_dry:.2f}", f"{ca_dry:.2f}", f"{sp_mass:.2f}"],
        "Absorption Correction": ["Nil", f"+{extra_water:.1f} kg absorbed", f"-{fa_mass_ssd - fa_dry:.1f} kg dry mass", f"-{ca_mass_ssd - ca_dry:.1f} kg dry mass", "Nil"]
    })
    st.table(dry_df)

# CSV Export
csv_data = pd.concat([ssd_df, dry_df], axis=1).to_csv(index=False)
st.download_button("📥 Download Proportioning Report", data=csv_data, file_name="IS_10262_Mix_Report.csv", mime="text/csv")