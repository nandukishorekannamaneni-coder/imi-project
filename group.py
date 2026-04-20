"""
Synthetic Dataset Generator for Thermosetting Polymer Biodegradability Prediction
==================================================================================
Generates 720 samples × 100 features suitable for:
  - Binary classification (biodegradable vs. non-biodegradable)
  - Inverse design tasks

Author: Generated for ML polymer research
"""

import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings("ignore")

# ── RDKit import (graceful fallback) ──────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
    print("RDKit found – will extract descriptors from SMILES.")
except ImportError:
    RDKIT_AVAILABLE = False
    print("RDKit not found – all descriptors will be simulated.")

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# =============================================================================
# 1.  Polymer Library
#     Each entry: (name, SMILES, base_biodegradability_score, polymer_type)
#     base_biodegradability_score in [0,1]; higher → more likely biodegradable
# =============================================================================
POLYMER_LIBRARY = [
    # ── Epoxy resins (low biodegradability) ───────────────────────────────────
    ("Bisphenol-A Epoxy",
     "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1",
     0.15, "epoxy"),
    ("Bisphenol-F Epoxy",
     "C(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1",
     0.18, "epoxy"),
    ("Novolac Epoxy",
     "C(c1ccc(OCC2CO2)cc1)c1cc(OCC2CO2)ccc1OCC1CO1",
     0.12, "epoxy"),
    ("Cycloaliphatic Epoxy",
     "C1CC2CC1CC2OCC1CO1",
     0.20, "epoxy"),
    ("Aliphatic Epoxy",
     "C(COC1CO1)OCC1CO1",
     0.30, "epoxy"),

    # ── Phenolic resins (very low biodegradability) ───────────────────────────
    ("Phenol-Formaldehyde Resin",
     "Oc1ccccc1Cc1cccc(Cc2cccc(O)c2)c1O",
     0.08, "phenolic"),
    ("Resol Resin",
     "Oc1ccc(CO)cc1CO",
     0.10, "phenolic"),
    ("Novolac Phenolic",
     "Oc1ccccc1Cc1ccccc1O",
     0.09, "phenolic"),
    ("Cresol Novolac",
     "Cc1cc(Cc2cc(C)c(O)cc2C)c(C)c(O)c1C",
     0.07, "phenolic"),

    # ── Polyurethanes (moderate biodegradability) ─────────────────────────────
    ("MDI-Based Polyurethane",
     "O=C(Nc1ccc(Cc2ccc(NC(=O)OCCO)cc2)cc1)OCCO",
     0.35, "polyurethane"),
    ("TDI-Based Polyurethane",
     "O=C(Nc1ccc(C)c(N)c1)OCCO",
     0.30, "polyurethane"),
    ("Aliphatic Polyurethane",
     "O=C(NCCCCCCNC(=O)OCCCCO)OCCCCO",
     0.45, "polyurethane"),
    ("Polyester Polyurethane",
     "O=C(NCCO)OCC(=O)OCCO",
     0.55, "polyurethane"),
    ("Waterborne Polyurethane",
     "O=C(Nc1ccc(Cc2ccc(NC(=O)OCC(O)CO)cc2)cc1)OCC(O)CO",
     0.40, "polyurethane"),

    # ── Unsaturated polyesters (moderate-high biodegradability) ───────────────
    ("Orthophthalic Polyester",
     "O=C(OCCO)c1ccccc1C(=O)OCCO",
     0.50, "polyester"),
    ("Isophthalic Polyester",
     "O=C(OCCO)c1cccc(C(=O)OCCO)c1",
     0.48, "polyester"),
    ("Maleic Anhydride Polyester",
     "O=C(OCCO)/C=C/C(=O)OCCO",
     0.60, "polyester"),
    ("Bio-Based Polyester",
     "O=C(OCCCCO)CCCCC(=O)OCCCCO",
     0.72, "polyester"),
    ("Fumaric Polyester",
     "O=C(OCCO)/C=C/C(=O)OCCO",
     0.62, "polyester"),

    # ── Melamine-formaldehyde (very low biodegradability) ─────────────────────
    ("Melamine-Formaldehyde",
     "Nc1nc(N)nc(N)n1",
     0.06, "melamine"),
    ("Melamine Cyanurate",
     "Nc1nc(N)nc(N)n1.O=C1NC(=O)NC(=O)N1",
     0.05, "melamine"),
    ("Urea-Formaldehyde",
     "NC(=O)N",
     0.15, "melamine"),

    # ── Additional bio-friendly thermosets ────────────────────────────────────
    ("Polylactic Acid Thermoset",
     "CC(OC(=O)C(C)OC(=O)C(C)O)C(=O)O",
     0.80, "biobased"),
    ("Furan Resin",
     "C1=COC(C=C1)c1ccco1",
     0.65, "biobased"),
    ("Epoxidized Soybean Oil",
     "CCCCCCCC1OC1CCCCCCCC(=O)OCC",
     0.70, "biobased"),
    ("Lignin-Based Epoxy",
     "COc1cc(CC2CO2)ccc1OCC1CO1",
     0.55, "biobased"),
    ("Tung-Oil Based Resin",
     "OCC(/C=C/C=C/C=C/CCCCCCCCC(=O)O)O",
     0.68, "biobased"),
]

# =============================================================================
# 2.  RDKit descriptor extraction (or simulation)
# =============================================================================

def extract_rdkit_descriptors(smiles: str) -> dict:
    """Extract MW, LogP, TPSA, ring counts etc. from SMILES via RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Fallback with neutral values if SMILES parsing fails
        mol = Chem.MolFromSmiles("C")

    return {
        "MW_rdkit":          Descriptors.MolWt(mol),
        "LogP_rdkit":        Descriptors.MolLogP(mol),
        "TPSA_rdkit":        Descriptors.TPSA(mol),
        "HBD_rdkit":         rdMolDescriptors.CalcNumHBD(mol),
        "HBA_rdkit":         rdMolDescriptors.CalcNumHBA(mol),
        "RotBonds_rdkit":    rdMolDescriptors.CalcNumRotatableBonds(mol),
        "AromaticRings_rdkit": rdMolDescriptors.CalcNumAromaticRings(mol),
        "AliphRings_rdkit":  rdMolDescriptors.CalcNumAliphaticRings(mol),
        "NumRings_rdkit":    rdMolDescriptors.CalcNumRings(mol),
        "FractionCSP3_rdkit": rdMolDescriptors.CalcFractionCSP3(mol),
        "NumHeavyAtoms_rdkit": mol.GetNumHeavyAtoms(),
        "NumAtoms_rdkit":    mol.GetNumAtoms(),
    }


def simulate_rdkit_descriptors(smiles: str, bio_score: float, rng: np.random.Generator) -> dict:
    """Simulate RDKit-like descriptors when RDKit is unavailable."""
    # Use string properties as cheap proxies
    n_aromatic = smiles.count("c") + smiles.count("n")
    n_oxygen   = smiles.count("O") + smiles.count("o")
    n_nitrogen = smiles.count("N") + smiles.count("n")
    length     = len(smiles)

    mw   = 150 + length * 2.5 + rng.normal(0, 20)
    logp = 2.0 - bio_score * 3 + rng.normal(0, 0.5)

    return {
        "MW_rdkit":           np.clip(mw, 100, 2000),
        "LogP_rdkit":         np.clip(logp, -3, 8),
        "TPSA_rdkit":         np.clip(n_oxygen * 10 + n_nitrogen * 8 + rng.normal(0, 5), 0, 300),
        "HBD_rdkit":          max(0, int(n_oxygen * 0.4 + rng.normal(0, 0.5))),
        "HBA_rdkit":          max(0, int(n_oxygen * 0.8 + rng.normal(0, 1))),
        "RotBonds_rdkit":     max(0, int(length * 0.05 + rng.normal(0, 1))),
        "AromaticRings_rdkit": max(0, int(n_aromatic * 0.08)),
        "AliphRings_rdkit":   max(0, int(rng.normal(1, 0.5))),
        "NumRings_rdkit":     max(0, int(n_aromatic * 0.08 + rng.normal(1, 0.5))),
        "FractionCSP3_rdkit": np.clip(0.8 - n_aromatic * 0.02 + rng.normal(0, 0.05), 0, 1),
        "NumHeavyAtoms_rdkit": max(5, int(length * 0.45)),
        "NumAtoms_rdkit":     max(5, int(length * 0.55)),
    }


# =============================================================================
# 3.  Functional-group count helpers (SMARTS-based or string-based)
# =============================================================================

# SMARTS patterns for key functional groups
FG_SMARTS = {
    "n_epoxy":    "[C;R1]1[O;R1][C;R1]1",          # 3-membered epoxide ring
    "n_ester":    "[#6]C(=O)O[#6]",
    "n_amide":    "[NX3][CX3](=O)",
    "n_ether":    "[OD2]([#6])[#6]",
    "n_hydroxyl": "[OX2H]",
    "n_amine":    "[NX3;H2,H1][#6]",
    "n_carbonyl": "[CX3]=[OX1]",
    "n_urethane": "[NX3][CX3](=O)[OX2]",
    "n_triazine": "c1ncncn1",
}

def count_fg_rdkit(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: 0 for k in FG_SMARTS}
    counts = {}
    for name, smarts in FG_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        counts[name] = len(mol.GetSubstructMatches(patt)) if patt else 0
    return counts


def count_fg_string(smiles: str) -> dict:
    """String-based proxy when RDKit unavailable."""
    return {
        "n_epoxy":    smiles.count("C1CO1") + smiles.count("C2CO2"),
        "n_ester":    smiles.count("C(=O)O") - smiles.count("NC(=O)O"),
        "n_amide":    smiles.count("C(=O)N") + smiles.count("NC(=O)"),
        "n_ether":    smiles.count("[O]") + smiles.count("COC"),
        "n_hydroxyl": smiles.count("O)") + smiles.count("(O"),
        "n_amine":    smiles.count("N") - smiles.count("NC(=O)"),
        "n_carbonyl": smiles.count("C=O") + smiles.count("(=O)"),
        "n_urethane": smiles.count("NC(=O)O"),
        "n_triazine": smiles.count("nc(n)"),
    }


# =============================================================================
# 4.  Physicochemical & engineering property generators
#     All values driven by bio_score + polymer_type + noise for realism
# =============================================================================

def gen_physicochemical(bio_score: float, ptype: str, rng: np.random.Generator) -> dict:
    """Generate physicochemical properties with scientifically grounded ranges."""

    # ── Molecular weight (repeat unit, g/mol) ─────────────────────────────────
    mw_base = {"epoxy": 400, "phenolic": 300, "polyurethane": 500,
                "polyester": 350, "melamine": 250, "biobased": 300}.get(ptype, 350)
    mw = np.clip(rng.normal(mw_base, 80), 150, 1200)

    # ── Density (g/cm³) ───────────────────────────────────────────────────────
    density_base = {"epoxy": 1.2, "phenolic": 1.3, "polyurethane": 1.1,
                    "polyester": 1.2, "melamine": 1.5, "biobased": 1.1}.get(ptype, 1.2)
    density = np.clip(rng.normal(density_base, 0.08), 0.9, 1.8)

    # ── Glass transition temperature Tg (°C) ──────────────────────────────────
    tg_base = {"epoxy": 120, "phenolic": 180, "polyurethane": 80,
               "polyester": 100, "melamine": 200, "biobased": 60}.get(ptype, 100)
    # Higher crosslink → higher Tg; more bio-friendly tends toward lower Tg
    tg = np.clip(rng.normal(tg_base - bio_score * 40, 20), -50, 350)

    # ── Thermal stability (onset degradation, °C) ─────────────────────────────
    thermal_base = {"epoxy": 300, "phenolic": 380, "polyurethane": 250,
                    "polyester": 280, "melamine": 350, "biobased": 230}.get(ptype, 280)
    thermal_stability = np.clip(rng.normal(thermal_base - bio_score * 50, 25), 150, 500)

    # ── Crosslink density (mol/m³) ─────────────────────────────────────────────
    # Key driver: high crosslink → non-biodegradable
    xlink_base = {"epoxy": 800, "phenolic": 1200, "polyurethane": 400,
                  "polyester": 300, "melamine": 1500, "biobased": 200}.get(ptype, 600)
    crosslink_density = np.clip(
        rng.normal(xlink_base * (1 - bio_score * 0.6), xlink_base * 0.15), 50, 2000)

    # ── LogP (hydrophobicity) ─────────────────────────────────────────────────
    logp_base = {"epoxy": 2.5, "phenolic": 2.0, "polyurethane": 1.5,
                 "polyester": 1.0, "melamine": -0.5, "biobased": 0.5}.get(ptype, 1.5)
    logp = np.clip(rng.normal(logp_base - bio_score * 1.5, 0.6), -3, 8)

    # ── Hildebrand solubility parameter (MPa^0.5) ─────────────────────────────
    solub_param = np.clip(rng.normal(20 + bio_score * 3, 2), 15, 35)

    return {
        "molecular_weight":   mw,
        "density":            density,
        "Tg":                 tg,
        "thermal_stability":  thermal_stability,
        "crosslink_density":  crosslink_density,
        "LogP":               logp,
        "solubility_parameter": solub_param,
    }


def gen_structural(fg_counts: dict, bio_score: float,
                   ptype: str, rng: np.random.Generator) -> dict:
    """Structural descriptors beyond functional group counts."""
    ar_rings  = fg_counts.get("n_epoxy", 0)   # placeholder reuse
    n_aromatic = {"epoxy": 2, "phenolic": 3, "polyurethane": 1,
                  "polyester": 1, "melamine": 0, "biobased": 0}.get(ptype, 1)
    n_aromatic = max(0, int(rng.normal(n_aromatic, 0.5)))

    degree_branching = np.clip(rng.normal(0.3 - bio_score * 0.2, 0.1), 0, 1)
    chain_length     = np.clip(rng.normal(50 + bio_score * 30, 10), 10, 200)
    repeat_unit_mass = np.clip(rng.normal(200 + bio_score * 50, 40), 80, 600)

    return {
        "n_aromatic_rings": n_aromatic,
        "degree_branching": degree_branching,
        "chain_length":     chain_length,
        "repeat_unit_mass": repeat_unit_mass,
    }


def gen_mechanical(bio_score: float, ptype: str, rng: np.random.Generator) -> dict:
    """Mechanical properties (MPa)."""
    ts_base = {"epoxy": 70, "phenolic": 60, "polyurethane": 40,
               "polyester": 55, "melamine": 50, "biobased": 35}.get(ptype, 55)
    em_base = {"epoxy": 3000, "phenolic": 4000, "polyurethane": 1500,
               "polyester": 2500, "melamine": 5000, "biobased": 1200}.get(ptype, 2500)

    tensile_strength = np.clip(rng.normal(ts_base - bio_score * 15, 10), 10, 150)
    elastic_modulus  = np.clip(rng.normal(em_base - bio_score * 500, 400), 200, 8000)
    elongation_break = np.clip(rng.normal(3 + bio_score * 8, 2), 0.5, 50)
    hardness_shore   = np.clip(rng.normal(80 - bio_score * 20, 8), 20, 100)
    flexural_strength= np.clip(rng.normal(ts_base * 1.3 - bio_score * 15, 12), 10, 200)

    return {
        "tensile_strength": tensile_strength,
        "elastic_modulus":  elastic_modulus,
        "elongation_at_break": elongation_break,
        "hardness_shore_D": hardness_shore,
        "flexural_strength": flexural_strength,
    }


def gen_processing(bio_score: float, ptype: str, rng: np.random.Generator) -> dict:
    """Processing / cure parameters."""
    ct_base = {"epoxy": 150, "phenolic": 170, "polyurethane": 80,
               "polyester": 130, "melamine": 160, "biobased": 100}.get(ptype, 130)
    cure_temp = np.clip(rng.normal(ct_base, 20), 50, 250)
    cure_time = np.clip(rng.normal(60 - bio_score * 20, 15), 5, 300)  # minutes
    post_cure_temp = np.clip(cure_temp + rng.normal(20, 10), 60, 280)
    viscosity = np.clip(rng.normal(5000 - bio_score * 2000, 1000), 100, 20000)  # mPa·s
    gel_time  = np.clip(rng.normal(30 - bio_score * 10, 8), 2, 120)  # minutes

    return {
        "cure_temperature":  cure_temp,
        "cure_time":         cure_time,
        "post_cure_temp":    post_cure_temp,
        "viscosity_mPas":    viscosity,
        "gel_time_min":      gel_time,
    }


def gen_environmental(bio_score: float, ptype: str, rng: np.random.Generator) -> dict:
    """Environmental & degradation features."""
    # Degradation rate (% mass loss per year) – key correlation with label
    deg_rate = np.clip(rng.normal(bio_score * 25, 5), 0, 80)

    # Water absorption (% by weight) – higher for polar / biodegradable polymers
    water_abs = np.clip(rng.normal(bio_score * 4 + 0.5, 1), 0, 15)

    # Oxygen permeability (cm³·mm/m²·day·atm)
    o2_perm   = np.clip(rng.normal(bio_score * 50 + 5, 10), 0.5, 200)

    # UV stability index [0–1], biodegradable tend to be less UV stable
    uv_stability = np.clip(rng.normal(0.8 - bio_score * 0.4, 0.1), 0, 1)

    # Biodegradation half-life (days)
    halflife  = np.clip(rng.normal(5000 - bio_score * 4500, 500), 30, 10000)

    # Chemical resistance score [0–10]
    chem_res  = np.clip(rng.normal(8 - bio_score * 4, 1), 0, 10)

    # pH of aqueous extract
    ph_extract = np.clip(rng.normal(7 + bio_score * 0.5 - 0.5, 0.5), 4, 11)

    return {
        "degradation_rate":    deg_rate,
        "water_absorption":    water_abs,
        "O2_permeability":     o2_perm,
        "UV_stability":        uv_stability,
        "biodeg_halflife_days": halflife,
        "chemical_resistance": chem_res,
        "pH_aqueous_extract":  ph_extract,
    }


def gen_additional_descriptors(bio_score: float, ptype: str,
                                rng: np.random.Generator, n: int = 57) -> dict:
    """
    Generate remaining descriptors to reach 100 total features.
    These are plausible polymer science descriptors with noise.
    """
    extras = {}

    # Dielectric & electrical
    extras["dielectric_constant"]   = np.clip(rng.normal(4 - bio_score * 1.5, 0.5), 2, 10)
    extras["dielectric_loss"]       = np.clip(rng.normal(0.02 + bio_score * 0.01, 0.005), 0.001, 0.2)
    extras["surface_resistivity"]   = np.clip(rng.normal(1e14 - bio_score * 5e13, 1e13), 1e10, 1e16)

    # Thermal
    extras["thermal_conductivity"]  = np.clip(rng.normal(0.25 - bio_score * 0.05, 0.05), 0.1, 0.8)
    extras["CTE_ppm_K"]             = np.clip(rng.normal(60 + bio_score * 20, 10), 20, 200)
    extras["heat_deflection_temp"]  = np.clip(rng.normal(120 - bio_score * 40, 20), 30, 300)
    extras["specific_heat"]         = np.clip(rng.normal(1.2 + bio_score * 0.3, 0.1), 0.8, 2.5)

    # Optical
    extras["refractive_index"]      = np.clip(rng.normal(1.55 - bio_score * 0.05, 0.02), 1.3, 1.8)
    extras["transmittance_pct"]     = np.clip(rng.normal(60 + bio_score * 20, 15), 0, 100)
    extras["haze_pct"]              = np.clip(rng.normal(10 - bio_score * 5, 3), 0, 60)

    # Rheological / flow
    extras["gel_content_pct"]       = np.clip(rng.normal(90 - bio_score * 20, 8), 30, 100)
    extras["swelling_ratio"]        = np.clip(rng.normal(1.05 + bio_score * 0.3, 0.05), 1.0, 3.0)
    extras["sol_fraction_pct"]      = np.clip(rng.normal(5 + bio_score * 15, 3), 0, 60)

    # Surface energy
    extras["surface_energy_mNm"]    = np.clip(rng.normal(40 + bio_score * 10, 5), 20, 70)
    extras["contact_angle_water"]   = np.clip(rng.normal(80 - bio_score * 35, 8), 5, 120)

    # Impact & fracture
    extras["impact_strength_kJm2"]  = np.clip(rng.normal(10 + bio_score * 15, 4), 0.5, 60)
    extras["fracture_toughness"]    = np.clip(rng.normal(0.8 + bio_score * 0.5, 0.2), 0.2, 5)
    extras["fatigue_limit_MPa"]     = np.clip(rng.normal(25 + bio_score * 10, 5), 5, 80)

    # Chemical composition
    extras["carbon_content_pct"]    = np.clip(rng.normal(65 - bio_score * 10, 5), 30, 80)
    extras["oxygen_content_pct"]    = np.clip(rng.normal(15 + bio_score * 15, 3), 5, 50)
    extras["nitrogen_content_pct"]  = np.clip(rng.normal(3 + bio_score * 2, 1), 0, 20)
    extras["hydrogen_content_pct"]  = np.clip(rng.normal(7 + bio_score * 2, 1), 3, 15)

    # Polymerization
    extras["conversion_degree"]     = np.clip(rng.normal(0.95 - bio_score * 0.1, 0.03), 0.5, 1.0)
    extras["monomer_ratio"]         = np.clip(rng.normal(1 + bio_score * 0.5, 0.2), 0.5, 3.0)
    extras["initiator_conc_pct"]    = np.clip(rng.normal(1 + bio_score * 0.5, 0.3), 0.1, 5)
    extras["catalyst_loading_phr"]  = np.clip(rng.normal(3 - bio_score, 0.5), 0.5, 10)

    # Aging / durability
    extras["moisture_absorption_rate"] = np.clip(rng.normal(bio_score * 0.5, 0.1), 0, 2)
    extras["creep_compliance"]      = np.clip(rng.normal(0.1 + bio_score * 0.15, 0.03), 0.01, 0.8)
    extras["stress_relaxation_modulus"] = np.clip(rng.normal(2000 - bio_score * 800, 300), 200, 6000)
    extras["retention_strength_pct"]= np.clip(rng.normal(90 - bio_score * 10, 5), 40, 100)

    # Formulation
    extras["filler_content_vol_pct"]= np.clip(rng.normal(20 - bio_score * 10, 5), 0, 60)
    extras["fiber_vol_fraction"]    = np.clip(rng.normal(0.3 - bio_score * 0.1, 0.05), 0, 0.7)
    extras["void_content_pct"]      = np.clip(rng.normal(1 + bio_score * 0.5, 0.5), 0, 10)
    extras["aspect_ratio_filler"]   = np.clip(rng.normal(10 - bio_score * 3, 2), 1, 50)

    # Biodegradation-support descriptors
    extras["ester_bond_density"]    = np.clip(rng.normal(bio_score * 3, 0.5), 0, 8)
    extras["hydrolysis_rate_const"] = np.clip(rng.normal(bio_score * 0.05, 0.01), 0, 0.2)
    extras["enzymatic_accessibility"] = np.clip(rng.normal(bio_score * 0.8, 0.1), 0, 1)
    extras["crystallinity_pct"]     = np.clip(rng.normal(5 + bio_score * 20, 5), 0, 60)

    # Miscellaneous
    extras["shrinkage_pct"]         = np.clip(rng.normal(2 + bio_score * 1, 0.5), 0.2, 10)
    extras["residual_stress_MPa"]   = np.clip(rng.normal(30 - bio_score * 10, 5), 0, 80)
    extras["pot_life_min"]          = np.clip(rng.normal(40 + bio_score * 20, 10), 5, 240)
    extras["mixing_ratio"]          = np.clip(rng.normal(2 - bio_score * 0.5, 0.3), 0.5, 5)
    extras["VOC_content_g_L"]       = np.clip(rng.normal(50 - bio_score * 40, 10), 0, 200)
    extras["flash_point_C"]         = np.clip(rng.normal(60 + bio_score * 20, 10), 20, 200)
    extras["ash_content_pct"]       = np.clip(rng.normal(3 - bio_score, 1), 0, 15)
    extras["limiting_oxygen_index"] = np.clip(rng.normal(25 - bio_score * 5, 3), 15, 60)
    extras["smoke_density"]         = np.clip(rng.normal(400 - bio_score * 200, 60), 10, 800)
    extras["toxicity_index"]        = np.clip(rng.normal(0.6 - bio_score * 0.4, 0.1), 0, 1)
    extras["recyclability_score"]   = np.clip(rng.normal(bio_score * 0.7 + 0.1, 0.1), 0, 1)
    extras["LCA_score"]             = np.clip(rng.normal(bio_score * 80 + 10, 10), 0, 100)
    extras["carbon_footprint_kgCO2"] = np.clip(rng.normal(5 - bio_score * 3, 1), 0.5, 20)
    extras["price_USD_per_kg"]      = np.clip(rng.normal(8 - bio_score * 4, 2), 0.5, 50)
    extras["market_maturity"]       = np.clip(rng.normal(0.6 + bio_score * 0.2, 0.1), 0, 1)

    return extras


# =============================================================================
# 5.  Label generation with calibrated threshold
# =============================================================================

def generate_label(bio_score: float, fg: dict, rng: np.random.Generator) -> int:
    """
    Convert continuous bio_score + functional-group signals into binary label.
    Ester groups push toward 1; crosslink proxies (epoxy, triazine) push toward 0.
    """
    score = bio_score
    score += fg.get("n_ester", 0) * 0.06    # ester → biodegradable
    score += fg.get("n_urethane", 0) * 0.03
    score -= fg.get("n_epoxy", 0) * 0.04    # epoxy → resistant
    score -= fg.get("n_triazine", 0) * 0.10 # triazine (melamine) → very resistant
    score -= fg.get("n_amide", 0) * 0.02

    # Logistic noise for realistic class overlap near boundary
    prob = 1 / (1 + np.exp(-10 * (score - 0.45)))
    return int(rng.random() < prob)


# =============================================================================
# 6.  Main dataset assembly
# =============================================================================

def build_dataset(n_samples: int = 720, normalize: bool = False,
                  random_seed: int = SEED) -> pd.DataFrame:

    rng = np.random.default_rng(random_seed)
    records = []

    for i in range(n_samples):
        # ── Sample a polymer entry ────────────────────────────────────────────
        polymer = random.choice(POLYMER_LIBRARY)
        name, smiles, base_bio, ptype = polymer

        # Add small perturbation to bio score so same polymer can appear varied
        bio_score = float(np.clip(base_bio + rng.normal(0, 0.08), 0.0, 1.0))

        # ── SMILES-based descriptors ──────────────────────────────────────────
        if RDKIT_AVAILABLE:
            rdkit_desc = extract_rdkit_descriptors(smiles)
            fg_counts  = count_fg_rdkit(smiles)
        else:
            rdkit_desc = simulate_rdkit_descriptors(smiles, bio_score, rng)
            fg_counts  = count_fg_string(smiles)

        # ── Property blocks ───────────────────────────────────────────────────
        physico   = gen_physicochemical(bio_score, ptype, rng)
        structural = gen_structural(fg_counts, bio_score, ptype, rng)
        mechanical = gen_mechanical(bio_score, ptype, rng)
        processing = gen_processing(bio_score, ptype, rng)
        environ    = gen_environmental(bio_score, ptype, rng)
        additional = gen_additional_descriptors(bio_score, ptype, rng)

        # ── Target label ──────────────────────────────────────────────────────
        label = generate_label(bio_score, fg_counts, rng)

        # ── Assemble row ──────────────────────────────────────────────────────
        row = {
            # Identifiers (not used as ML features)
            "Polymer_Name":  name,
            "SMILES":        smiles,
            "Polymer_Type":  ptype,
            # Continuous latent score (useful for regression / inverse design)
            "bio_score":     round(bio_score, 4),
            # Target
            "Biodegradability": label,
        }
        row.update(rdkit_desc)
        row.update(fg_counts)
        row.update(physico)
        row.update(structural)
        row.update(mechanical)
        row.update(processing)
        row.update(environ)
        row.update(additional)

        records.append(row)

    df = pd.DataFrame(records)

    # ── Verify feature count ──────────────────────────────────────────────────
    non_feature_cols = {"Polymer_Name", "SMILES", "Polymer_Type",
                        "bio_score", "Biodegradability"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    print(f"  Feature columns generated: {len(feature_cols)}")

    # ── Optional min-max normalisation of numeric features ───────────────────
    if normalize:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        print("  Features normalized to [0, 1].")

    return df


# =============================================================================
# 7.  Entry point
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  Thermoset Polymer Biodegradability Dataset Generator")
    print("=" * 60)

    # Build dataset
    print("\n[1] Building dataset …")
    df = build_dataset(n_samples=720, normalize=False, random_seed=SEED)

    # Summary statistics
    print(f"\n[2] Dataset shape : {df.shape}")
    print(f"    Total columns  : {len(df.columns)}")
    print(f"    Class balance  :")
    vc = df["Biodegradability"].value_counts()
    print(f"      0 (non-biodegradable) : {vc.get(0, 0)}")
    print(f"      1 (biodegradable)     : {vc.get(1, 0)}")
    print(f"\n    Polymer type distribution:")
    print(df["Polymer_Type"].value_counts().to_string(header=False))

    # Preview numeric stats
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    print(f"\n[3] Numeric feature preview (first 10):")
    print(df[numeric_cols[:10]].describe().round(2).to_string())

    # Correlation check (top correlates with label)
    print(f"\n[4] Top-10 features correlated with Biodegradability:")
    corrs = (df[numeric_cols]
             .corr()["Biodegradability"]
             .drop("Biodegradability")
             .abs()
             .sort_values(ascending=False)
             .head(10))
    print(corrs.round(3).to_string())

    # Save CSV
    out_path = "thermoset_polymer_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[5] Dataset saved → '{out_path}'")
    print("=" * 60)
    print("Done.")
