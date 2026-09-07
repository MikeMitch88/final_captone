"""Feature engineering for fuel custody risk scoring."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


STANDARD_PMS_DENSITY_KG_PER_M3 = 745.0
FEATURE_COLUMNS = [
    "volumetric_discrepancy",
    "density_drift",
    "risk_telemetry_score",
    "custody_gap_flag",
]


class CustodyFeatureEngineer:
    """Join custody source tables and calculate model-ready features."""

    def __init__(self, pms_nominal_density: float = STANDARD_PMS_DENSITY_KG_PER_M3):
        self.pms_nominal_density = pms_nominal_density

    def load_sqlite(self, db_path: str | Path) -> dict[str, pd.DataFrame]:
        """Load the source tables used by the feature pipeline."""
        tables = [
            "kra_manifests",
            "rects_telemetry",
            "kpc_depot_meters",
            "depot_lab_quality",
        ]
        with sqlite3.connect(str(db_path)) as connection:
            return {
                table: pd.read_sql_query(f"SELECT * FROM {table}", connection)
                for table in tables
            }

    def build_features(self, tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Return one feature record per consignment.

        ``meter_out_volume`` is the delivered volume because the generator
        records diversion shrinkage between meter-in and meter-out. Missing
        source records are retained so the custody gap feature can identify them.
        """
        required = {
            "kra_manifests",
            "rects_telemetry",
            "kpc_depot_meters",
            "depot_lab_quality",
        }
        missing = required.difference(tables)
        if missing:
            raise ValueError(f"Missing source tables: {sorted(missing)}")

        manifests = tables["kra_manifests"].copy()
        telemetry = tables["rects_telemetry"].copy()
        meters = tables["kpc_depot_meters"].copy()
        quality = tables["depot_lab_quality"].copy()

        # Older schema.sql files omit this generated column. The feature layer
        # can still run by using the PMS nominal density as the declaration.
        if "declared_density_kg_per_m3" not in manifests:
            manifests["declared_density_kg_per_m3"] = np.where(
                manifests["product_type"].eq("PMS"),
                self.pms_nominal_density,
                np.nan,
            )

        frame = manifests.merge(telemetry, on="consignment_id", how="left", suffixes=("_manifest", "_telemetry"))
        frame = frame.merge(meters, on="consignment_id", how="left", suffixes=("", "_meter"))
        frame = frame.merge(quality, on="consignment_id", how="left", suffixes=("", "_quality"))

        declared = pd.to_numeric(frame["declared_volume_litres"], errors="coerce")
        delivered = pd.to_numeric(frame["meter_out_volume"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            frame["volumetric_discrepancy"] = ((declared - delivered) / declared).replace(
                [np.inf, -np.inf], np.nan
            )

        product = frame["product_type"].fillna("")
        observed_density = pd.to_numeric(frame["density_at_15c"], errors="coerce")
        declared_density = pd.to_numeric(frame["declared_density_kg_per_m3"], errors="coerce")
        nominal_density = np.where(
            product.eq("PMS"), self.pms_nominal_density, declared_density
        )
        frame["density_drift"] = (observed_density - nominal_density).abs()

        dwell = pd.to_numeric(frame["dwell_time_minutes"], errors="coerce").fillna(0)
        dwell_component = np.clip(dwell / 240.0, 0, 1)
        route_component = frame["geofence_status"].map(
            {"OK": 0.0, "Out-of-Corridor": 0.5, "Route-Deviation": 1.0}
        ).fillna(1.0)
        seal_component = pd.to_numeric(
            frame["e_seal_tamper_flag"], errors="coerce"
        ).fillna(1.0).clip(0, 1)
        frame["risk_telemetry_score"] = (
            100 * (0.4 * dwell_component + 0.35 * route_component + 0.25 * seal_component)
        ).round(2)

        quality_missing = frame[["lab_test_id", "density_at_15c", "pass_quality_flag"]].isna().any(axis=1)
        quality_anomalous = ~frame["pass_quality_flag"].isin([0, 1]) | frame["pass_quality_flag"].eq(0)
        frame["custody_gap_flag"] = (quality_missing | quality_anomalous).astype(int)

        frame["volumetric_discrepancy"] = frame["volumetric_discrepancy"].fillna(0.0)
        frame["density_drift"] = frame["density_drift"].fillna(0.0)
        frame["risk_telemetry_score"] = frame["risk_telemetry_score"].fillna(100.0)

        # Historical labels are explicit quality failures or known custody
        # diversion signals. The volumetric label remains separate for metrics.
        frame["volumetric_anomaly_label"] = (
            frame["volumetric_discrepancy"].abs() >= 0.005
        ).astype(int)
        frame["anomaly_label"] = (
            frame["volumetric_anomaly_label"].eq(1)
            | frame["density_drift"].ge(25.0)
            | frame["risk_telemetry_score"].ge(40.0)
            | frame["custody_gap_flag"].eq(1)
        ).astype(int)
        frame["Risk_Score"] = np.nan
        frame["High_Risk_Alert"] = False
        return frame

    def build_from_sqlite(self, db_path: str | Path) -> pd.DataFrame:
        return self.build_features(self.load_sqlite(db_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate custody model features")
    parser.add_argument("--db", default="data/custody_data.db")
    parser.add_argument("--output", default="data/custody_features.parquet")
    args = parser.parse_args()
    features = CustodyFeatureEngineer().build_from_sqlite(args.db)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)
    print(f"Wrote {len(features):,} feature records to {args.output}")


if __name__ == "__main__":
    main()