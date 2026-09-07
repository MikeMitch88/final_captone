"""Synthetic Incident-Grounded Dataset Generator for HAKIKI-KPC Phase 1.

Generates realistic synthetic consignment records representing:
  - Scenario A: MT Paloma Quality Discrepancy (substandard PMS at handover)
  - Scenario B: Corridor Dumping & Geofence Deviation (route deviation, dwell time, shrinkage)
  - Scenario C: Normal Operations (compliant transit/domestic consignments)

Output: SQLite database with 10,000+ normal + ~300 labeled anomaly events.
"""

import numpy as np
import pandas as pd
import sqlite3
import os
from datetime import datetime, timedelta
from faker import Faker
from typing import Tuple, Dict, Any

fake = Faker()
np.random.seed(42)

# ---------------------------------------------------------------------------
# Constants & Ranges (Kenya downstream corridor: Kipevu Mombasa → Nairobi/Nakuru/Eldoret/Kisumu)
# ---------------------------------------------------------------------------

DEPOT_IDS = ["KOT2", "NBI", "NAK", "ELD", "KIS"]
OMCS = ["Kenya Power", "TotalEnergies", "Shell", "Cleopatra", "Ondeo"]
SOURCES = ["Kipevu Oil Terminal II", "Mombasa Road Depot", "Nairobi Fuel Hub"]
DEST_TYPES = ["Domestic", "Transit EAC"]
PRODUCT_TYPES = ["PMS", "AGO", "DPK"]

# Declared density ranges by product type (kg/m³)
DECLARED_DENSITY_RANGES = {
    "PMS": (720, 775),
    "AGO": (820, 900),
    "DPK": (775, 830),
}

# Normal volumetric shrinkage threshold
NORMAL_SHRINKAGE_TOLERANCE = 0.3  # percent
CORRIDOR_SHRINKAGE_CUTOFF = 0.5  # percent (anomaly threshold)

# Quality flags
ANOMALY_DENSITY_LOW = 720.0
ANOMALY_DENSITY_HIGH = 775.0
ANOMALY_RON_THRESHOLD = 91


# ---------------------------------------------------------------------------
# Helper: generate base timestamp range (last 30 days)
# ---------------------------------------------------------------------------

def generate_base_timestamps(n: int, start_days_ago: int = 30) -> pd.Series:
    """Generate n random timestamps within the last *start_days_ago* days."""
    end = datetime.now()
    start = end - timedelta(days=start_days_ago)
    return pd.Series(
        [fake.date_time_between(start_date=start, end_date=end) for _ in range(n)]
    )


# ---------------------------------------------------------------------------
# Scenario C: Normal Operations
# ---------------------------------------------------------------------------

def generate_normal_consignments(n: int = 8000) -> Dict[str, pd.DataFrame]:
    """Generate normal compliant consignments (Scenario C)."""

    manifest_ids = [f"MKP_{fake.random_number(digits=6):06d}" for _ in range(n)]
    timestamps = generate_base_timestamps(n)

    # Consignment ID linked to manifest (used for telemetry, meters, lab quality)
    consignment_ids = [f"CONS_{i+1:06d}" for i in range(n)]

    # Product type: mix of PMS, AGO, DPK
    product_choices = np.random.choice(PRODUCT_TYPES, size=n, p=[0.5, 0.3, 0.2])
    omc_choices = np.random.choice(OMCS, size=n)

    # Declared volume: 20,000 – 50,000 litres typical for road tanker
    declared_volumes = np.random.normal(loc=35000, scale=5000, size=n).astype(int)

    # Destination type correlated with product but not strictly
    dest_choices = np.random.choice(DEST_TYPES, size=n, p=[0.6, 0.4])

    # Bond status
    bond_choices = np.random.choice(
        ["Active", "Pending", "Discharged"], size=n, p=[0.7, 0.2, 0.1]
    )

    # Manifest product declared density at 15°C (kg/m³) ± 2 around true type mean
    declared_densities = np.zeros(n)
    for i, pt in enumerate(product_choices):
        lo, hi = DECLARED_DENSITY_RANGES[pt]
        declared_densities[i] = np.random.uniform(lo + 5, hi - 5)

    # OMC-declared product mapping
    manifest_dfs = pd.DataFrame(
        {
            "manifest_id": manifest_ids,
            "omc": omc_choices,
            "product_type": product_choices,
            "declared_volume_litres": declared_volumes,
            "source": np.random.choice(SOURCES, size=n),
            "destination_type": dest_choices,
            "bond_status": bond_choices,
            "declared_density_kg_per_m3": declared_densities,
            "timestamp": timestamps,
            "consignment_id": consignment_ids,  # link manifest to consignment
        }
    )

    # RECTS telemetry: minimal deviations, geofence OK, no tamper
    gps_lat = np.random.uniform(-4.0, 4.0, size=n)  # broad Kenya corridor band
    gps_lon = np.random.uniform(34.0, 41.0)
    geofence_status = np.random.choice(
        ["OK", "Out-of-Corridor", "Route-Deviation"], size=n, p=[0.92, 0.05, 0.03]
    )
    e_seal_tamper = np.random.choice([0, 1], size=n, p=[0.98, 0.02])
    dwell_times = np.random.randint(5, 48, size=n)  # hours typical

    rects_dfs = pd.DataFrame(
        {
            "consignment_id": consignment_ids,
            "vehicle_seal_id": [f"SEAL_{fake.uuid4()[:8].upper()}" for _ in range(n)],
            "timestamp": timestamps,
            "gps_latitude": gps_lat,
            "gps_longitude": gps_lon,
            "geofence_status": geofence_status,
            "e_seal_tamper_flag": e_seal_tamper,
            "dwell_time_minutes": dwell_times,
        }
    )

    # KPC depot meters: volume tolerance ≤ ±0.3%
    # meter_in ≈ declared_volume (within 0.3%), meter_out slightly less
    in_volumes = declared_volumes
    # slight shrinkage even in normal ops
    shrinkage_pct = np.random.uniform(-0.3, 0.3, size=n)
    out_volumes = (in_volumes * (1 + shrinkage_pct / 100)).astype(int)

    # Temperature: 15–40°C range
    temperatures = np.random.uniform(15, 40, size=n).round(1)

    # Observed density: close to declared (within normal measurement variance)
    observed_densities = declared_densities + np.random.normal(0, 1.5, size=n)

    meter_dfs = pd.DataFrame(
        {
            "meter_transaction_id": [f"MTRX_{i+1:06d}" for i in range(n)],
            "consignment_id": consignment_ids,
            "depot_id": np.random.choice(DEPOT_IDS, size=n),
            "meter_in_volume": in_volumes,
            "meter_out_volume": out_volumes,
            "temperature": temperatures,
            "observed_density_kg_per_m3": np.round(observed_densities, 2),
        }
    )

    # Depot lab quality: passing specs, density within declared ±3
    lab_densities = observed_densities + np.random.normal(0, 0.8, size=n)
    rons = np.random.randint(91, 98, size=n)  # PMS RON typically 91–96
    flash_points = np.random.uniform(35, 65, size=n).round(1)  # PMS flash point
    sulfur = np.random.uniform(10, 50, size=n).round(2)  # ppm sulfur

    lab_dfs = pd.DataFrame(
        {
            "lab_test_id": [f"LABT_{i+1:06d}" for i in range(n)],
            "consignment_id": consignment_ids,
            "depot_id": np.random.choice(DEPOT_IDS, size=n),
            "sample_timestamp": timestamps,
            "density_at_15c": np.round(np.maximum(lab_densities, 700), 2),
            "research_octane_number": rons,
            "flash_point": np.round(flash_points, 1),
            "sulfur_content_ppm": sulfur,
            "pass_quality_flag": 1,  # all pass for normal ops
        }
    )

    return {
        "kra_manifests": manifest_dfs,
        "rects_telemetry": rects_dfs,
        "kpc_depot_meters": meter_dfs,
        "depot_lab_quality": lab_dfs,
    }


# ---------------------------------------------------------------------------
# Scenario A: MT Paloma Quality Discrepancy
# ---------------------------------------------------------------------------

def generate_quality_anomalies(n: int = 150) -> Dict[str, pd.DataFrame]:
    """Generate Scenario A: Substandard PMS quality at handover."""

    manifest_ids = [f"MKP_QUAL_{i:04d}" for i in range(n)]
    timestamps = generate_base_timestamps(n)

    # Consignment ID linked to manifest
    consignment_ids = [f"CONS_QUAL_{i:04d}" for i in range(n)]

    # All scenario A are PMS with bad quality
    product_types = ["PMS"] * n
    omc_choices = np.random.choice(OMCS, size=n)

    # Declared volume looks normal (20k–50k) but quality is substandard
    declared_volumes = np.random.normal(loc=35000, scale=4000, size=n).astype(int)

    # Destination: mainly Domestic
    dest_choices = np.random.choice(["Domestic"], size=n, p=[1.0])

    bond_choices = np.random.choice(["Active", "Pending"], size=n, p=[0.6, 0.4])

    # Declared density is within normal range but LAB density is out of spec
    # Declared: 730–770 (looks fine on paper), but lab finds < 720 or > 775
    declared_densities = np.random.uniform(730, 770, size=n)

    # Lab density: deliberately skewed
    # ~70% below 720, ~20% above 775, ~10% borderline
    lab_densities = np.zeros(n)
    for i in range(n):
        r = np.random.random()
        if r < 0.7:
            # Sub-low: 690–719
            lab_densities[i] = np.random.uniform(690, 719.9)
        elif r < 0.9:
            # Sub-high: 775.1–790
            lab_densities[i] = np.random.uniform(775.1, 790)
        else:
            # Borderline but flagged
            lab_densities[i] = np.random.uniform(720, 725)

    # RON deliberately < 91
    rons = np.random.randint(85, 91, size=n)

    # Flash point too low (substandard PMS)
    flash_points = np.random.uniform(30, 45, size=n).round(1)

    # Sulfur content elevated
    sulfur = np.random.uniform(50, 200, size=n).round(2)

    manifest_dfs = pd.DataFrame(
        {
            "manifest_id": manifest_ids,
            "omc": omc_choices,
            "product_type": product_types,
            "declared_volume_litres": declared_volumes,
            "source": np.random.choice(SOURCES, size=n),
            "destination_type": dest_choices,
            "bond_status": bond_choices,
            "declared_density_kg_per_m3": np.round(declared_densities, 2),
            "timestamp": timestamps,
            "consignment_id": consignment_ids,
        }
    )

# RECTS telemetry: geofence mostly OK but some route deviation
# eSeal tamper_flag higher than normal
    gps_lat = np.random.uniform(-4.0, 4.0, size=n)
    gps_lon = np.random.uniform(34.0, 41.0)
    geofence_status = np.random.choice(
        ["OK", "Out-of-Corridor", "Route-Deviation"], size=n, p=[0.7, 0.2, 0.1]
    )
    e_seal_tamper = np.random.choice([0, 1], size=n, p=[0.85, 0.15])
    dwell_times = np.random.randint(5, 72, size=n)  # some prolonged dwell

    rects_dfs = pd.DataFrame(
        {
            "consignment_id": consignment_ids,
            "vehicle_seal_id": [f"SEAL_QUAL_{i:04d}" for i in range(n)],
            "timestamp": timestamps,
            "gps_latitude": gps_lat,
            "gps_longitude": gps_lon,
            "geofence_status": geofence_status,
            "e_seal_tamper_flag": e_seal_tamper,
            "dwell_time_minutes": dwell_times,
        }
    )

    # Depot meters: volumes match (normal shrinkage) but quality is bad
    in_volumes = declared_volumes
    out_volumes = in_volumes  # minimal shrinkage in quality case
    temperatures = np.random.uniform(15, 40, size=n).round(1)
    observed_densities = lab_densities  # lab density = observed density

    meter_dfs = pd.DataFrame(
        {
            "meter_transaction_id": [f"MTRX_QUAL_{i:04d}" for i in range(n)],
            "consignment_id": consignment_ids,
            "depot_id": np.random.choice(DEPOT_IDS, size=n),
            "meter_in_volume": in_volumes,
            "meter_out_volume": out_volumes,
            "temperature": temperatures,
            "observed_density_kg_per_m3": np.round(observed_densities, 2),
        }
    )

    # Lab quality: pass_flag = 0 due to density/RON/flash point failures
    lab_dfs = pd.DataFrame(
        {
            "lab_test_id": [f"LABT_QUAL_{i:04d}" for i in range(n)],
            "consignment_id": consignment_ids,
            "depot_id": np.random.choice(DEPOT_IDS, size=n),
            "sample_timestamp": timestamps,
            "density_at_15c": np.round(lab_densities, 2),
            "research_octane_number": rons,
            "flash_point": np.round(flash_points, 1),
            "sulfur_content_ppm": sulfur,
            "pass_quality_flag": 0,  # all fail for substandard PMS
        }
    )

    return {
        "kra_manifests": manifest_dfs,
        "rects_telemetry": rects_dfs,
        "kpc_depot_meters": meter_dfs,
        "depot_lab_quality": lab_dfs,
    }


# ---------------------------------------------------------------------------
# Scenario B: Corridor Dumping & Geofence Deviation
# ---------------------------------------------------------------------------

def generate_corridor_anomalies(n: int = 150) -> Dict[str, pd.DataFrame]:
    """Generate Scenario B: Export transit with route deviation, dwell time, shrinkage."""

    manifest_ids = [f"MKP_CORR_{i:04d}" for i in range(n)]
    timestamps = generate_base_timestamps(n)

    # Consignment ID linked to manifest
    consignment_ids = [f"CONS_CORR_{i:04d}" for i in range(n)]

    product_types = np.random.choice(["PMS", "AGO", "DPK"], size=n, p=[0.4, 0.3, 0.3])
    omc_choices = np.random.choice(OMCS, size=n)

    # Export transit cargo: declared volumes inflated (shrinkage will be discovered later)
    declared_volumes = np.random.normal(loc=42000, scale=6000, size=n).astype(int)

    # Transit EAC destination
    dest_choices = np.random.choice(["Transit EAC"], size=n, p=[1.0])

    bond_choices = np.random.choice(["Active", "Pending", "Discharged"], size=n, p=[0.5, 0.3, 0.2])

    # Declared density normal
    declared_densities = np.zeros(n)
    for i, pt in enumerate(product_types):
        lo, hi = DECLARED_DENSITY_RANGES[pt]
        declared_densities[i] = np.random.uniform(lo + 10, hi - 10)

    manifest_dfs = pd.DataFrame(
        {
            "manifest_id": manifest_ids,
            "omc": omc_choices,
            "consignment_id": consignment_ids,
            "product_type": product_types,
            "declared_volume_litres": declared_volumes,
            "source": np.random.choice(SOURCES, size=n),
            "destination_type": dest_choices,
            "bond_status": bond_choices,
            "declared_density_kg_per_m3": np.round(declared_densities, 2),
            "timestamp": timestamps,
        }
    )

    # RECTS telemetry: significant route deviation, GPS far from corridor
    consignment_ids = [f"CONS_CORR_{i:04d}" for i in range(n)]
    # GPS deliberately shifted away optimal corridor route
    gps_lat = np.random.uniform(-6.0, -2.0, size=n)  # far south/offshore-ish
    gps_lon = np.random.uniform(30.0, 35.0)  # west of Kenya
    geofence_status = np.random.choice(
        ["OK", "Out-of-Corridor", "Route-Deviation"], size=n, p=[0.1, 0.3, 0.6]
    )
    e_seal_tamper = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    # Prolonged dwell times: 48–360 hours (2 weeks) unauthorized
    dwell_times = np.random.randint(48, 360, size=n)

    rects_dfs = pd.DataFrame(
        {
            "consignment_id": consignment_ids,
            "vehicle_seal_id": [f"SEAL_CORR_{i:04d}" for i in range(n)],
            "timestamp": timestamps,
            "gps_latitude": gps_lat,
            "gps_longitude": gps_lon,
            "geofence_status": geofence_status,
            "e_seal_tamper_flag": e_seal_tamper,
            "dwell_time_minutes": dwell_times,
        }
    )

    # KPC depot meters: volumetric shrinkage > 0.5% loss
    # manifest volume is inflated; depot meter_in records what was actually received
    # shrinkage calculated as (V_manifest - V_depot_meter) / V_manifest * 100
    # We create: 0.5% – 5.0% shrinkage
    shrinkage_pct = np.random.uniform(0.5, 5.0, size=n)
    in_volumes = declared_volumes
    out_volumes = (in_volumes * (1 - shrinkage_pct / 100)).astype(int)

    temperatures = np.random.uniform(15, 45, size=n).round(1)

    # Observed density may differ from declared due to product loss/condensation
    observed_densities = declared_densities + np.random.normal(0, 2, size=n)

    meter_dfs = pd.DataFrame(
        {
            "meter_transaction_id": [f"MTRX_CORR_{i:04d}" for i in range(n)],
            "consignment_id": consignment_ids,
            "depot_id": np.random.choice(["NAK", "ELD", "KIS"], size=n),  # inland depots
            "meter_in_volume": in_volumes,
            "meter_out_volume": out_volumes,
            "temperature": temperatures,
            "observed_density_kg_per_m3": np.round(observed_densities, 2),
        }
    )

    # Lab quality: some pass, some flagged based on density deviation
    lab_densities = observed_densities + np.random.normal(0, 1, size=n)
    rons = np.random.randint(88, 96, size=n)
    flash_points = np.random.uniform(38, 68, size=n).round(1)
    sulfur = np.random.uniform(10, 80, size=n).round(2)

    # Lower pass rate for corridor anomalies (some quality issues mixed in)
    pass_probs = np.clip(1 - shrinkage_pct / 10, 0.4, 1.0)
    pass_quality = np.random.choice([0, 1], size=n, p=[0.6, 0.4])

    lab_dfs = pd.DataFrame(
        {
            "lab_test_id": [f"LABT_CORR_{i:04d}" for i in range(n)],
            "consignment_id": consignment_ids,
            "depot_id": np.random.choice(["NAK", "ELD", "KIS"], size=n),
            "sample_timestamp": timestamps,
            "density_at_15c": np.round(lab_densities, 2),
            "research_octane_number": rons,
            "flash_point": np.round(flash_points, 1),
            "sulfur_content_ppm": sulfur,
            "pass_quality_flag": pass_quality,
        }
    )

    return {
        "kra_manifests": manifest_dfs,
        "rects_telemetry": rects_dfs,
        "kpc_depot_meters": meter_dfs,
        "depot_lab_quality": lab_dfs,
    }


# ---------------------------------------------------------------------------
# Main generator: combine all scenarios
# ---------------------------------------------------------------------------

def generate_custody_dataset(
    normal_n: int = 8000,
    quality_n: int = 150,
    corridor_n: int = 150,
    output_path: str = "/home/mike_mitch/Desktop/EMTech/final_capstone/data/custody_data.db",
) -> str:
    """Generate the full synthetic dataset and persist to SQLite."""

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Generate each scenario
    normal_df = generate_normal_consignments(normal_n)
    quality_df = generate_quality_anomalies(quality_n)
    corridor_df = generate_corridor_anomalies(corridor_n)

    # --- kra_manifests ---
    kra_manifests = pd.concat(
        [normal_df["kra_manifests"], quality_df["kra_manifests"], corridor_df["kra_manifests"]],
        ignore_index=True,
    )

    # --- rects_telemetry ---
    rects_telemetry = pd.concat(
        [normal_df["rects_telemetry"], quality_df["rects_telemetry"], corridor_df["rects_telemetry"]],
        ignore_index=True,
    )

    # --- kpc_depot_meters ---
    kpc_depot_meters = pd.concat(
        [normal_df["kpc_depot_meters"], quality_df["kpc_depot_meters"], corridor_df["kpc_depot_meters"]],
        ignore_index=True,
    )

    # --- depot_lab_quality ---
    depot_lab_quality = pd.concat(
        [normal_df["depot_lab_quality"], quality_df["depot_lab_quality"], corridor_df["depot_lab_quality"]],
        ignore_index=True,
    )

    # --- reconciled_custody_events (computed) ---
    reconciled = compute_reconciliation(
        kra_manifests, rects_telemetry, kpc_depot_meters, depot_lab_quality
    )

    # Write to SQLite
    conn = sqlite3.connect(output_path)
    kra_manifests.to_sql("kra_manifests", conn, if_exists="replace", index=False)
    rects_telemetry.to_sql("rects_telemetry", conn, if_exists="replace", index=False)
    kpc_depot_meters.to_sql("kpc_depot_meters", conn, if_exists="replace", index=False)
    depot_lab_quality.to_sql("depot_lab_quality", conn, if_exists="replace", index=False)
    reconciled.to_sql("reconciled_custody_events", conn, if_exists="replace", index=False)

    # Apply indexes
    from_index_sql = open("/home/mike_mitch/Desktop/EMTech/final_capstone/src/schema.sql").read()
    for statement in from_index_sql.split(";"):
        statement = statement.strip()
        if statement.upper().startswith("CREATE INDEX") or statement.upper().startswith("CREATE TABLE IF NOT EXISTS"):
            conn.execute(statement)
    conn.commit()
    conn.close()

    print(f"Dataset generated: {output_path}")
    print(f"  - kra_manifests: {len(kra_manifests)} rows")
    print(f"  - rects_telemetry: {len(rects_telemetry)} rows")
    print(f"  - kpc_depot_meters: {len(kpc_depot_meters)} rows")
    print(f"  - depot_lab_quality: {len(depot_lab_quality)} rows")
    print(f"  - reconciled_custody_events: {len(reconciled)} rows")

    return output_path


# ---------------------------------------------------------------------------
# Custody Reconciliation & Discrepancy Engine
# ---------------------------------------------------------------------------

def compute_reconciliation(
    manifests: pd.DataFrame,
    telemetry: pd.DataFrame,
    depot_meters: pd.DataFrame,
    lab_quality: pd.DataFrame,
) -> pd.DataFrame:
    """Compute custody reconciliation discrepancies and anomaly index."""

    # Merge manifest + telemetry + depot meters + lab quality by consignment_id
    merged = manifests.merge(
        telemetry, on="consignment_id", how="inner", suffixes=("_manifest", "_telemetry")
    ).merge(
        depot_meters, on="consignment_id", how="inner"
    ).merge(
        lab_quality, on="consignment_id", how="inner"
    )

    # Handle duplicate depot_id columns (depot_meters and depot_lab_quality both have depot_id)
    depot_id_cols = [c for c in merged.columns if c.startswith("depot_id")]
    if len(depot_id_cols) > 1:
        # Keep depot_id from depot_meters (primary source for volumetric calculations)
        # Rename the remaining one to depot_id for consistency
        merged = merged.rename(columns={depot_id_cols[0]: "depot_id_std", depot_id_cols[1]: "depot_id_lab"})
        merged = merged.drop(columns="depot_id_lab")
        merged = merged.rename(columns={"depot_id_std": "depot_id"})
    elif len(depot_id_cols) == 1:
        # Single depot_id column - ensure it's named depot_id
        if depot_id_cols[0] != "depot_id":
            merged = merged.rename(columns={depot_id_cols[0]: "depot_id"})

    # --- Volumetric Shrinkage (%)
    merged["volumetric_shrinkage_pct"] = (
        (merged["declared_volume_litres"] - merged["meter_in_volume"])
        / merged["declared_volume_litres"]
        * 100
    )

    # --- Density Deviation (%)
    merged["density_deviation_pct"] = (
        np.abs(lab_quality["density_at_15c"] - merged["declared_density_kg_per_m3"])
        / merged["declared_density_kg_per_m3"]
        * 100
    )

    # --- Custody Handover Anomaly Index (composite z-score weighted)
    # Components: seal tamper (0/1), geofence deviation (0/1/2), quality variance
    seal_tamper = merged["e_seal_tamper_flag"].astype(float)
    geofence_dev = merged["geofence_status"].map({"OK": 0, "Out-of-Corridor": 1, "Route-Deviation": 2}).astype(float)
    quality_var = merged["density_deviation_pct"] / merged["density_deviation_pct"].abs().max()  # normalized

    # Weighted sum: seal 0.4 + geofence 0.4 + quality 0.2
    anomaly_index_raw = 0.4 * seal_tamper + 0.4 * geofence_dev + 0.2 * quality_var

    # Scale to 0–100
    anomaly_index = np.clip(anomaly_index_raw * 100, 0, 100).round(2)

    # --- Anomaly Risk Flag ---
    conditions = [
        (anomaly_index >= 70),
        (anomaly_index >= 40),
    ]
    choices = ["High", "Medium"]
    # Default Low where none match
    anomaly_risk_flag = np.select(conditions, choices, default="Low")

    # Adjust risk based on explicit quality failures
    # If pass_quality_flag == 0, bump at least to Medium
    quality_failure = merged["pass_quality_flag"] == 0
    anomaly_risk_flag = np.where(
        quality_failure,
        np.where(anomaly_risk_flag == "Low", "Medium", anomaly_risk_flag),
        anomaly_risk_flag,
    )

    # Cap High if density deviation is within normal AND no seal/tamper
    normal_quality = merged["density_deviation_pct"] < 5.0
    anomaly_risk_flag = np.where(
        normal_quality & (anomaly_risk_flag == "High") & (seal_tamper == 0) & (geofence_dev == 0),
        "Medium",
        anomaly_risk_flag,
    )

    reconciled_df = pd.DataFrame(
        {
            "event_id": [f"REC_{merged.index[i]:06d}" for i in range(len(merged))],
            "manifest_id": merged["manifest_id"],
            "consignment_id": merged["consignment_id"],
            "depot_id": merged["depot_id"],
            "volumetric_shrinkage_pct": np.round(merged["volumetric_shrinkage_pct"], 2),
            "density_deviation_pct": np.round(merged["density_deviation_pct"], 2),
            "custody_handover_anomaly_index": anomaly_index,
            "anomaly_risk_flag": anomaly_risk_flag,
            "calculated_at": datetime.now(),
        }
    )

    return reconciled_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate HAKIKI-KPC synthetic custody dataset."
    )
    parser.add_argument(
        "--normal", type=int, default=8000, help="Number of normal consignments (Scenario C)"
    )
    parser.add_argument(
        "--quality", type=int, default=150, help="Number of quality anomalies (Scenario A)"
    )
    parser.add_argument(
        "--corridor", type=int, default=150, help="Number of corridor anomalies (Scenario B)"
    )
    parser.add_argument(
        "--output", type=str, default="data/custody_data.db", help="Output SQLite path"
    )
    args = parser.parse_args()

    path = generate_custody_dataset(
        normal_n=args.normal, quality_n=args.quality, corridor_n=args.corridor, output_path=args.output
    )
    print(f"\nDataset written to: {path}")
    print("\nSample reconciliation record:")
    conn = sqlite3.connect(path)
    sample = pd.read_sql("SELECT * FROM reconciled_custody_events LIMIT 5", conn)
    print(sample.to_string(index=False))
    conn.close()


if __name__ == "__main__":
    import argparse
    main()