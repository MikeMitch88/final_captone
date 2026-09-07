from __future__ import annotations

from datetime import datetime, timedelta
import random
from typing import Any

import pandas as pd


RISK_LEVELS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
RISK_THRESHOLDS = {
    "LOW": (0, 24),
    "MODERATE": (25, 49),
    "HIGH": (50, 74),
    "CRITICAL": (75, 100),
}
PRODUCT_VALUES = {"AGO": 42.0, "PMS": 48.0, "DPK": 46.0}
TOLERANCE_PERCENT = 0.5


def risk_level_for_score(score: float | int) -> str:
    score = float(score)
    if 75 <= score <= 100:
        return "CRITICAL"
    if 50 <= score <= 74:
        return "HIGH"
    if 25 <= score <= 49:
        return "MODERATE"
    return "LOW"


def reconciliation_status(variance_pct: float) -> str:
    magnitude = abs(float(variance_pct))
    if magnitude <= 0.5:
        return "PASS"
    if magnitude <= 1.5:
        return "WARNING"
    return "FAIL"


def calculate_risk_score(consignment: dict[str, Any]) -> int:
    score = 0
    variance_pct = abs(float(consignment.get("variance_pct", 0)))

    if variance_pct > 1.5:
        score += 25
    elif variance_pct > 0.5:
        score += 10

    if consignment.get("quality_status") in {"MISSING", "FAILED", "EXPIRED"}:
        score += 20
    elif consignment.get("quality_status") == "PENDING":
        score += 8

    if consignment.get("route_changed"):
        score += 15
    if int(consignment.get("manifest_amendments", 0)) >= 3:
        score += 15
    if consignment.get("custody_status") in {"EXCEPTION", "ABNORMAL"}:
        score += 15
    if float(consignment.get("stock_variance_pct", 0)) > 0.8:
        score += 10
    if consignment.get("other_anomaly"):
        score += 10

    return min(100, score)


def build_demo_consignments(count: int = 120) -> list[dict[str, Any]]:
    random.seed(42)
    origins = [
        "Mombasa Terminal",
        "Kipevu Oil Terminal",
        "Nairobi Storage",
        "Eldoret Bulk Depot",
        "Kisumu Import Terminal",
    ]
    destinations = [
        "Nairobi Depot",
        "Kisumu Depot",
        "Nakuru Depot",
        "Eldoret Depot",
        "Mombasa Customer",
        "Naivasha Depot",
    ]
    products = ["AGO", "PMS", "DPK"]
    quality_statuses = ["PASSED", "PASSED", "PASSED", "PENDING", "FAILED", "MISSING", "EXPIRED"]
    custody_statuses = ["OK", "OK", "OK", "EXCEPTION", "ABNORMAL"]

    records: list[dict[str, Any]] = []
    for idx in range(count):
        product = random.choice(products)
        declared = random.randint(700000, 1500000)
        variance_pct = round(random.uniform(-5.0, 2.0), 2)
        actual = int(round(declared * (1 + variance_pct / 100)))
        variance = actual - declared
        route_changed = random.random() < 0.12
        manifest_amendments = random.randint(0, 5)
        quality_status = random.choice(quality_statuses)
        custody_status = random.choice(custody_statuses)
        stock_variance_pct = round(random.uniform(-1.2, 1.2), 2)
        other_anomaly = random.random() < 0.1

        if idx % 13 == 0:
            quality_status = "MISSING"
            variance_pct = -2.8
            actual = int(round(declared * (1 - 0.028)))
            route_changed = True
            manifest_amendments = 4
            custody_status = "EXCEPTION"
            stock_variance_pct = -1.7
            other_anomaly = True
        elif idx % 17 == 0:
            quality_status = "FAILED"
            variance_pct = -1.1
            actual = int(round(declared * (1 - 0.011)))
            route_changed = False
            manifest_amendments = 2
            custody_status = "ABNORMAL"
        elif idx % 19 == 0:
            quality_status = "PASSED"
            variance_pct = 0.3
            actual = int(round(declared * (1 + 0.003)))
            manifest_amendments = 0
            custody_status = "OK"
            route_changed = False

        record = {
            "consignment_id": f"KPC-IMP-2026-{idx + 1:05d}",
            "manifest_id": f"KPC-MF-{idx + 1:05d}",
            "product": product,
            "origin": random.choice(origins),
            "destination": random.choice(destinations),
            "declared_volume": declared,
            "actual_volume": actual,
            "variance": variance,
            "variance_pct": variance_pct,
            "quality_status": quality_status,
            "custody_status": custody_status,
            "manifest_amendments": manifest_amendments,
            "route_changed": route_changed,
            "stock_variance_pct": stock_variance_pct,
            "other_anomaly": other_anomaly,
            "risk_score": 0,
            "risk_level": "LOW",
            "financial_exposure": 0.0,
            "investigation_status": random.choice(["New", "Assigned", "Under Review", "Escalated"]),
            "timestamp": datetime.now() - timedelta(days=random.randint(1, 32), hours=random.randint(1, 23)),
            "value": declared * PRODUCT_VALUES[product] / 1000,
        }

        record["risk_score"] = calculate_risk_score(record)
        record["risk_level"] = risk_level_for_score(record["risk_score"])
        record["financial_exposure"] = abs(record["variance"]) * PRODUCT_VALUES[product] * 0.7
        records.append(record)

    return records


def build_dashboard_data() -> dict[str, Any]:
    consignments = pd.DataFrame(build_demo_consignments(120))
    consignments["risk_level"] = consignments["risk_score"].apply(risk_level_for_score)
    consignments["reconciliation_status"] = consignments["variance_pct"].apply(reconciliation_status)
    consignments["financial_exposure"] = (
        (consignments["declared_volume"] - consignments["actual_volume"]).abs() * consignments["product"].map(PRODUCT_VALUES) * 0.7
    )

    priority_queue = consignments.sort_values(["risk_score", "financial_exposure"], ascending=[False, False]).head(12)
    quality_records = []
    for _, row in consignments.iterrows():
        quality_records.append(
            {
                "consignment": row["consignment_id"],
                "product": row["product"],
                "sample_date": row["timestamp"].strftime("%Y-%m-%d"),
                "test_status": row["quality_status"],
                "certificate": "MISSING" if row["quality_status"] == "MISSING" else "VALID",
                "key_parameter": "Density@15C",
                "specification": "715-770 kg/m³",
                "result": f"{random.uniform(710, 780):.1f}",
                "risk": row["risk_level"],
            }
        )
    quality_df = pd.DataFrame(quality_records)

    reconciliation_df = pd.DataFrame(
        [
            {
                "Consignment": row["consignment_id"],
                "Declared Volume": row["declared_volume"],
                "Movement Volume": int(round(row["declared_volume"] * (1 + row["variance_pct"] / 100))),
                "Received Volume": row["actual_volume"],
                "Stock Volume": int(round(row["actual_volume"] * (1 + row["stock_variance_pct"] / 100))),
                "Variance": row["variance"],
                "Tolerance": "±5,000",
                "Result": row["reconciliation_status"],
                "Risk": row["risk_level"],
            }
            for _, row in consignments.iterrows()
        ]
    )

    risk_drivers = pd.DataFrame(
        [
            {"Risk Driver": "Volume mismatch", "Number of Cases": int((consignments["variance_pct"].abs() > 1.5).sum()), "Exposure": float(consignments.loc[consignments["variance_pct"].abs() > 1.5, "financial_exposure"].sum()), "Severity": "Critical"},
            {"Risk Driver": "Missing quality certificate", "Number of Cases": int((consignments["quality_status"] == "MISSING").sum()), "Exposure": float(consignments.loc[consignments["quality_status"] == "MISSING", "financial_exposure"].sum()), "Severity": "High"},
            {"Risk Driver": "Repeated manifest amendment", "Number of Cases": int((consignments["manifest_amendments"] >= 3).sum()), "Exposure": float(consignments.loc[consignments["manifest_amendments"] >= 3, "financial_exposure"].sum()), "Severity": "High"},
            {"Risk Driver": "Unexpected destination", "Number of Cases": int(consignments["route_changed"].sum()), "Exposure": float(consignments.loc[consignments["route_changed"], "financial_exposure"].sum()), "Severity": "Critical"},
            {"Risk Driver": "Product reclassification", "Number of Cases": 11, "Exposure": 7800000.0, "Severity": "Moderate"},
            {"Risk Driver": "Abnormal custody transfer", "Number of Cases": int((consignments["custody_status"] == "ABNORMAL").sum()), "Exposure": float(consignments.loc[consignments["custody_status"] == "ABNORMAL", "financial_exposure"].sum()), "Severity": "High"},
        ]
    )

    alerts = []
    for _, row in priority_queue.head(8).iterrows():
        alerts.append(
            {
                "Alert ID": f"ALERT-{row['consignment_id'][-5:]}",
                "Severity": row["risk_level"],
                "Timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
                "Consignment": row["consignment_id"],
                "Type": "Volume discrepancy" if row["variance_pct"] < -1 else "Quality anomaly",
                "Description": f"Consignment {row['consignment_id']} has a {row['variance_pct']:.2f}% variance with risk score {row['risk_score']}",
                "Risk Score": row["risk_score"],
                "Status": "Open",
                "Assigned Investigator": "J. Mwangi",
            }
        )
    alerts_df = pd.DataFrame(alerts)

    investigations = []
    for _, row in priority_queue.head(6).iterrows():
        investigations.append(
            {
                "id": f"INV-{row['consignment_id'][-5:]}",
                "consignment_id": row["consignment_id"],
                "status": row["investigation_status"],
                "priority": row["risk_level"],
                "assigned_to": "A. Otieno",
                "reason": "Volume variance and missing quality certification",
                "financial_exposure": row["financial_exposure"],
            }
        )
    investigations_df = pd.DataFrame(investigations)

    overview = {
        "total_consignments": int(len(consignments)),
        "reconciled_consignments": int((consignments["reconciliation_status"] == "PASS").sum()),
        "open_exceptions": int((consignments["reconciliation_status"] != "PASS").sum()),
        "high_critical_risk": int(((consignments["risk_level"] == "HIGH") | (consignments["risk_level"] == "CRITICAL")).sum()),
        "quality_exceptions": int((consignments["quality_status"].isin(["MISSING", "FAILED", "EXPIRED"])).sum()),
        "volume_discrepancy": f"{consignments['variance'].sum():,.0f} L",
        "estimated_financial_exposure": f"KES {consignments['financial_exposure'].sum():,.0f}",
        "investigations_open": int((investigations_df["status"] != "Closed").sum()),
    }

    risk_distribution = consignments["risk_level"].value_counts().reindex(RISK_LEVELS, fill_value=0)

    custody_chain = [
        {
            "node": "Mombasa Terminal",
            "volume": 1000000,
            "status": "Manifest created",
            "risk": "LOW",
        },
        {
            "node": "Storage Facility A",
            "volume": 998700,
            "status": "Received",
            "risk": "MODERATE",
        },
        {
            "node": "Pipeline Segment 4",
            "volume": 997900,
            "status": "Dispatched",
            "risk": "HIGH",
        },
        {
            "node": "Nairobi Depot",
            "volume": 972000,
            "status": "Received",
            "risk": "CRITICAL",
        },
    ]

    risk_trends = pd.DataFrame(
        {
            "Week": ["W1", "W2", "W3", "W4"],
            "Risk Score": [48, 52, 61, 69],
            "Volume Discrepancy": [8.4, 9.1, 9.8, 11.4],
        }
    )

    return {
        "consignments": consignments,
        "priority_queue": priority_queue,
        "risk_distribution": risk_distribution,
        "risk_drivers": risk_drivers,
        "overview": overview,
        "quality_records": quality_df,
        "reconciliation_records": reconciliation_df,
        "alerts": alerts_df,
        "investigations": investigations_df,
        "custody_chain": custody_chain,
        "risk_trends": risk_trends,
    }
