"""KPC Integrity Intelligence — Enterprise Dashboard (HAKIKI-KPC).

Enterprise UI/UX refactor of the Streamlit application with an integrated
AI Intelligence Layer (Investigation Assistant).

Key behaviours
--------------
* Blue/white enterprise styling injected from ``styles.css`` (no dark mode).
* Strict severity coding: red = Critical/Fail, orange = High/Warning,
  green = Low/Pass. Indigo/purple accents are used *only* for AI insights.
* Actionable-exception-first layout: a top row of severity-coded KPI cards.
* Real reconciled custody data (``data/custody_pipeline.db``) when available,
  with a graceful synthetic-demo fallback so the showcase never breaks.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ai_assistant import InvestigationAssistant, analyze_event
from dashboard_data import build_dashboard_data, PRODUCT_VALUES

# ---------------------------------------------------------------------------
# Paths & theme constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent  # final_capstone/
SRC_DIR = Path(__file__).resolve().parent  # src/
DB_CANDIDATES = [
    SRC_DIR / "data" / "custody_pipeline.db",
    REPO_ROOT / "data" / "custody_pipeline.db",
    SRC_DIR / "data" / "custody_data.db",
    REPO_ROOT / "data" / "custody_data.db",
]
STYLES_PATH = Path(__file__).resolve().parent / "styles.css"

SEVERITY_CSS = {
    "HIGH": ("critical", "#d92d20", "#fee4e2"),
    "CRITICAL": ("critical", "#d92d20", "#fee4e2"),
    "MEDIUM": ("high", "#b54708", "#fef0c7"),
    "MODERATE": ("high", "#b54708", "#fef0c7"),
    "LOW": ("low", "#067647", "#d1fadf"),
}

SAMPLE_PRICE_KES_PER_L = {"PMS": 48.0, "AGO": 42.0, "DPK": 46.0}


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

def load_css() -> None:
    """Inject the enterprise stylesheet via st.markdown."""
    if STYLES_PATH.exists():
        st.markdown(f"<style>{STYLES_PATH.read_text()}</style>", unsafe_allow_html=True)


def render_topbar() -> None:
    """Render the enterprise top navigation band."""
    st.markdown(
        """
        <div class="kpc-topbar">
            <div class="kpc-title">🛢️ KPC Integrity Intelligence</div>
            <div class="kpc-badge">HAKIKI · KPC x KRA</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    label: str, value: str, delta: str, tone: str = "", ai: bool = False
) -> None:
    """Render an elevated enterprise KPI metric card."""
    tone_cls = {"critical": "kpi-critical", "high": "kpi-high", "low": "kpi-pass"}.get(
        tone, ""
    )
    ai_cls = " kpi-ai" if ai else ""
    delta_color = "#d92d20" if tone == "critical" else "#12b76a"
    st.markdown(
        f"""
        <div class="kpc-kpi {tone_cls}{ai_cls}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta" style="color:{delta_color};">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chip(text: str) -> None:
    """Render a severity chip with the strict enterprise color coding."""
    cls = {
        "critical": "chip-critical",
        "high": "chip-high",
        "low": "chip-low",
    }.get(SEVERITY_CSS.get(text.upper(), ("", "", ""))[0], "chip-low")
    st.markdown(f'<span class="kpc-chip {cls}">{text}</span>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data layer (real DB with synthetic fallback)
# ---------------------------------------------------------------------------

def _normalise_risk(raw: str) -> str:
    """Map engine risk labels to the 3-grade UI taxonomy."""
    upper = str(raw or "").upper()
    if upper in {"HIGH", "CRITICAL"}:
        return "HIGH"
    if upper in {"MEDIUM", "MODERATE"}:
        return "MEDIUM"
    return "LOW"


def _estimate_exposure_kcs(df: pd.DataFrame) -> pd.Series:
    """Estimate financial exposure from volumetric shrinkage."""
    litre_loss = df["declared_volume_litres"] * df["volumetric_shrinkage_pct"].fillna(0) / 100
    price = df["product_type"].map(SAMPLE_PRICE_KES_PER_L).fillna(45.0)
    return (litre_loss * price * 0.7).clip(lower=0)


def _load_from_db(db_path: str | Path) -> pd.DataFrame:
    """Build the unified events frame from the reconciled SQLite dataset."""
    query = """
        SELECT
            r.event_id, r.manifest_id, r.consignment_id, r.depot_id,
            r.volumetric_shrinkage_pct, r.density_deviation_pct,
            r.custody_handover_anomaly_index, r.anomaly_risk_flag, r.calculated_at,
            m.omc, m.product_type, m.declared_volume_litres, m.source,
            m.destination_type, m.bond_status, m.timestamp,
            q.density_at_15c, q.research_octane_number, q.flash_point,
            q.sulfur_content_ppm, q.pass_quality_flag,
            t.geofence_status, t.e_seal_tamper_flag, t.dwell_time_minutes,
            t.vehicle_seal_id
        FROM reconciled_custody_events r
        LEFT JOIN kra_manifests        m ON m.manifest_id   = r.manifest_id
        LEFT JOIN depot_lab_quality    q ON q.consignment_id = r.consignment_id
        LEFT JOIN rects_telemetry      t ON t.consignment_id = r.consignment_id
    """
    frame = pd.read_sql_query(query, sqlite3.connect(str(db_path)))
    return frame


@st.cache_data(show_spinner=False)
def load_events() -> tuple[pd.DataFrame, str]:
    """Load unified consignment events; fall back to synthetic demo data."""
    for db_path in DB_CANDIDATES:
        if not db_path.exists():
            continue
        try:
            frame = _load_from_db(db_path)
            if frame.empty:
                continue
            frame["risk_flag"] = frame["anomaly_risk_flag"].map(_normalise_risk)
            frame["financial_exposure_kcs"] = _estimate_exposure_kcs(frame)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
            frame["calculated_at"] = pd.to_datetime(frame["calculated_at"], errors="coerce")
            return frame, f"reconciled SQLite dataset ({db_path.name})"
        except Exception:
            continue

    # -- Synthetic demo fallback: derive the same unified schema ------------
    demo = build_dashboard_data()
    consignments = demo["consignments"].copy()
    consignments["risk_flag"] = consignments["risk_level"].map(_normalise_risk)
    consignments["product_type"] = consignments["product"]
    consignments["declared_volume_litres"] = consignments["declared_volume"]
    consignments["volumetric_shrinkage_pct"] = consignments["variance_pct"].clip(upper=0).abs()
    consignments["density_deviation_pct"] = 0.0
    consignments["custody_handover_anomaly_index"] = consignments["risk_score"]
    consignments["anomaly_risk_flag"] = consignments["risk_level"]
    consignments["source"] = consignments["origin"]
    consignments["destination_type"] = consignments["destination"]
    consignments["omc"] = "—"
    consignments["financial_exposure_kcs"] = consignments["financial_exposure"]
    frame = consignments.rename(columns={
        "manifest_id": "manifest_id",
        "geofence_status": "geofence_status",
        "e_seal_tamper_flag": "e_seal_tamper_flag",
        "dwell_time_minutes": "dwell_time_minutes",
        "density_at_15c": "density_at_15c",
        "research_octane_number": "research_octane_number",
        "flash_point": "flash_point",
        "sulfur_content_ppm": "sulfur_content_ppm",
        "pass_quality_flag": "pass_quality_flag",
    })
    for col in ("geofence_status", "e_seal_tamper_flag", "dwell_time_minutes",
                "density_at_15c", "research_octane_number", "flash_point",
                "sulfur_content_ppm", "pass_quality_flag", "depot_id"):
        if col not in frame:
            frame[col] = None
    frame["depot_id"] = frame["depot_id"].fillna(frame["destination_type"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame, "synthetic demo dataset"


# ---------------------------------------------------------------------------
# Table styling (strict severity coding)
# ---------------------------------------------------------------------------

def style_risk_table(df: pd.DataFrame, risk_col: str = "risk_flag") -> pd.io.formats.style.Styler:
    """Return a Styler that paints the risk column with severity colors."""
    def color_risk(value: str) -> str:
        cls, fg, bg = SEVERITY_CSS.get(str(value).upper(), SEVERITY_CSS["LOW"])
        return f"background-color: {bg}; color: {fg}; font-weight: 700;"

    styler = df.style.map(color_risk, subset=[risk_col])

    if "custody_handover_anomaly_index" in df.columns:
        styler = styler.bar(
            subset=["custody_handover_anomaly_index"],
            color="#d0ddf5",
            align="mid",
        )
    return styler


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

def page_overview(events: pd.DataFrame) -> None:
    st.subheader("Command Overview")
    st.caption("RECONCILE → DETECT ANOMALIES → SCORE RISK → INVESTIGATE → TAKE ACTION")

    total = len(events)
    high = int((events["risk_flag"] == "HIGH").sum())
    medium = int((events["risk_flag"] == "MEDIUM").sum())
    low = total - high - medium
    avg_shrink = float(events["volumetric_shrinkage_pct"].fillna(0).mean())
    avg_index = float(events["custody_handover_anomaly_index"].fillna(0).mean())
    quality_fails = int((events.get("pass_quality_flag") == 0).sum()) if "pass_quality_flag" in events else 0
    exposure = float(events["financial_exposure_kcs"].sum())

    st.markdown(
        f"Data source: <code>{st.session_state.get('data_source', '')}</code>",
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    with cols[0]:
        render_kpi_card("Total Consignments", f"{total:,}", "monitored", tone="low")
    with cols[1]:
        render_kpi_card("High / Critical Anomalies", f"{high:,}", f"{medium:,} medium",
                        tone="critical" if high else "low")
    with cols[2]:
        render_kpi_card("Avg Volumetric Shrinkage", f"{avg_shrink:.2f}%",
                        f"index {avg_index:.1f}/100", tone="high" if avg_shrink > 0.5 else "low")
    with cols[3]:
        render_kpi_card("Est. Financial Exposure", f"KES {exposure:,.0f}",
                        f"{quality_fails:,} quality fails", tone="high")

    left, right = st.columns([1.4, 1.0])
    with left:
        st.subheader("Risk Distribution")
        dist = events["risk_flag"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0)
        st.bar_chart(dist.rename("Consignments"), horizontal=True)
    with right:
        st.subheader("Top Risk Drivers")
        drivers = risk_drivers(events).head(6)
        st.dataframe(
            style_risk_table(drivers.rename(columns={"risk_flag": "Severity"}), "Severity"),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Exception Queue (Priority)")
    queue_cols = [
        "consignment_id", "product_type", "omc", "source", "destination_type",
        "declared_volume_litres", "volumetric_shrinkage_pct", "density_deviation_pct",
        "custody_handover_anomaly_index", "risk_flag",
    ]
    queue = events.sort_values(
        ["custody_handover_anomaly_index", "financial_exposure_kcs"], ascending=False
    ).head(15)[queue_cols]
    queue = queue.rename(columns={
        "consignment_id": "Consignment ID",
        "product_type": "Product",
        "omc": "OMC",
        "source": "Source",
        "destination_type": "Destination",
        "declared_volume_litres": "Declared Vol (L)",
        "volumetric_shrinkage_pct": "Shrinkage %",
        "density_deviation_pct": "Density Dev %",
        "custody_handover_anomaly_index": "Anomaly Index",
        "risk_flag": "Severity",
    })
    st.dataframe(
        style_risk_table(queue, "Severity"),
        hide_index=True,
        use_container_width=True,
    )


def risk_drivers(events: pd.DataFrame) -> pd.DataFrame:
    """Summarize top anomaly drivers from the unified events frame."""
    rows = []
    if "volumetric_shrinkage_pct" in events:
        mask = events["volumetric_shrinkage_pct"].fillna(0) > 0.5
        rows.append({
            "Risk Driver": "Volumetric shrinkage > 0.5%",
            "Number of Cases": int(mask.sum()),
            "Exposure (KES)": float(events.loc[mask, "financial_exposure_kcs"].sum()),
            "risk_flag": "HIGH" if mask.sum() else "LOW",
        })
    if "density_deviation_pct" in events:
        mask = events["density_deviation_pct"].fillna(0) > 3.0
        rows.append({
            "Risk Driver": "Density deviation > 3%",
            "Number of Cases": int(mask.sum()),
            "Exposure (KES)": float(events.loc[mask, "financial_exposure_kcs"].sum()),
            "risk_flag": "HIGH" if mask.sum() else "LOW",
        })
    if "e_seal_tamper_flag" in events:
        mask = events["e_seal_tamper_flag"].fillna(0).astype(int).eq(1)
        rows.append({
            "Risk Driver": "eSeal tamper flags",
            "Number of Cases": int(mask.sum()),
            "Exposure (KES)": float(events.loc[mask, "financial_exposure_kcs"].sum()),
            "risk_flag": "HIGH" if mask.sum() else "LOW",
        })
    if "geofence_status" in events:
        mask = events["geofence_status"].isin(["Route-Deviation", "Out-of-Corridor"])
        rows.append({
            "Risk Driver": "Route deviation / out of corridor",
            "Number of Cases": int(mask.sum()),
            "Exposure (KES)": float(events.loc[mask, "financial_exposure_kcs"].sum()),
            "risk_flag": "HIGH" if mask.sum() else "LOW",
        })
    if "pass_quality_flag" in events:
        mask = events["pass_quality_flag"].fillna(1).astype(int).eq(0)
        rows.append({
            "Risk Driver": "Failed laboratory quality",
            "Number of Cases": int(mask.sum()),
            "Exposure (KES)": float(events.loc[mask, "financial_exposure_kcs"].sum()),
            "risk_flag": "HIGH" if mask.sum() else "LOW",
        })
    frame = pd.DataFrame(rows)
    return frame if not frame.empty else pd.DataFrame(
        columns=["Risk Driver", "Number of Cases", "Exposure (KES)", "risk_flag"]
    )


def page_consignments(events: pd.DataFrame) -> None:
    st.subheader("Consignment Register")
    depot_filter = st.selectbox("Filter by depot", ["All"] + sorted(events["depot_id"].dropna().unique().tolist()))
    product_filter = st.selectbox("Filter by product", ["All"] + sorted(events["product_type"].dropna().unique().tolist()))
    risk_filter = st.selectbox("Filter by risk", ["All", "HIGH", "MEDIUM", "LOW"])

    view = events.copy()
    if depot_filter != "All":
        view = view[view["depot_id"] == depot_filter]
    if product_filter != "All":
        view = view[view["product_type"] == product_filter]
    if risk_filter != "All":
        view = view[view["risk_flag"] == risk_filter]

    # Surface the most anomalous records first and keep the render fast.
    view = view.sort_values("custody_handover_anomaly_index", ascending=False).head(300)

    display_cols = [
        "consignment_id", "depot_id", "product_type", "source", "destination_type",
        "declared_volume_litres", "volumetric_shrinkage_pct", "density_deviation_pct",
        "custody_handover_anomaly_index", "risk_flag",
    ]
    view = view[display_cols].rename(columns={
        "consignment_id": "Consignment ID",
        "depot_id": "Depot",
        "product_type": "Product",
        "source": "Source",
        "destination_type": "Destination",
        "declared_volume_litres": "Declared Vol (L)",
        "volumetric_shrinkage_pct": "Shrinkage %",
        "density_deviation_pct": "Density Dev %",
        "custody_handover_anomaly_index": "Anomaly Index",
        "risk_flag": "Severity",
    })
    st.dataframe(
        style_risk_table(view, "Severity"),
        hide_index=True,
        use_container_width=True,
    )


def page_quality(events: pd.DataFrame) -> None:
    st.subheader("Quality Intelligence")
    q = events.copy()
    if "density_at_15c" not in q or q["density_at_15c"].isna().all():
        st.info("No laboratory quality records available for the current data source.")
        return

    tested = int(q["density_at_15c"].notna().sum())
    passed = int((q["pass_quality_flag"].fillna(1).astype(int).eq(1)).sum())
    failed = int((q["pass_quality_flag"].fillna(1).astype(int).eq(0)).sum())
    avg_density = float(q["density_at_15c"].fillna(0).mean())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Samples Tested", f"{tested:,}", "", tone="low")
    with c2:
        render_kpi_card("Quality Passed", f"{passed:,}", f"{passed / max(tested, 1):.0%}", tone="low")
    with c3:
        render_kpi_card("Quality Failed", f"{failed:,}", "requires review", tone="critical" if failed else "low")
    with c4:
        render_kpi_card("Avg Density@15C", f"{avg_density:.1f} kg/m³", "declaration basis", tone="high")

    qcols = ["consignment_id", "product_type", "depot_id", "density_at_15c",
             "research_octane_number", "flash_point", "sulfur_content_ppm", "pass_quality_flag"]
    qview = q[~q["density_at_15c"].isna()].sort_values(
        "custody_handover_anomaly_index", ascending=False
    ).head(30)[qcols].rename(columns={
        "consignment_id": "Consignment ID",
        "product_type": "Product",
        "depot_id": "Depot",
        "density_at_15c": "Density@15C",
        "research_octane_number": "RON",
        "flash_point": "Flash Point",
        "sulfur_content_ppm": "Sulfur (ppm)",
        "pass_quality_flag": "Pass Flag",
    })
    st.dataframe(qview, hide_index=True, use_container_width=True)


def page_custody_chain() -> None:
    st.subheader("Custody Chain")
    st.caption("Origin → Storage → Pipeline → Depot → Destination")
    chain = [
        ("Kipevu Oil Terminal II", "KOT2 export / import handling", "LOW"),
        ("Mombasa Bulk Storage", "Manifest creation & bond control", "LOW"),
        ("Cross-Country Pipeline", "Metered dispatch KOT2 → NBI", "MEDIUM"),
        ("Nairobi Depot (NBI)", "Receipt metering & quality gate", "HIGH"),
        ("Inland Depots (NAK / ELD / KIS)", "Delivery & custody handover", "HIGH"),
    ]
    for idx, (node, note, risk) in enumerate(chain):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{node}**")
            st.caption(note)
        with col2:
            render_chip(risk)
        if idx < len(chain) - 1:
            st.markdown("<div style='text-align:center; color:#1d5fb0;'>&#8595;</div>",
                        unsafe_allow_html=True)


def page_investigation(events: pd.DataFrame) -> None:
    st.subheader("Investigation Assistant")
    st.markdown(
        """
        <div class="kpc-ai-panel">
            <div class="ai-tag">AI Intelligence Layer</div>
            <h4>Targeted analytical support for KPC / KRA investigators</h4>
            <p style="margin:0;">Generated insights are evidence-grounded recommendations
            for human review. They are not allegations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    flagged = events[events["risk_flag"] == "HIGH"].copy()
    if flagged.empty:
        flagged = events.nlargest(10, "custody_handover_anomaly_index", keep="first").copy()

    if flagged.empty:
        st.info("No flagged consignments are available for AI analysis.")
        return

    options = flagged.sort_values(
        "custody_handover_anomaly_index", ascending=False
    ).head(15)
    selection = st.selectbox(
        "Select a High/Critical consignment",
        options["consignment_id"].tolist(),
        index=0,
    )
    event = options[options["consignment_id"] == selection].iloc[0].to_dict()

    st.markdown(
        "**Evidence snapshot** — " + _evidence_line(event),
    )

    if st.button("Analyze with AI", type="primary"):
        with st.spinner("Running Investigation Assistant…"):
            assistant = InvestigationAssistant()
            result = analyze_event(event, assistant)

        mode_tag = "Groq (live)" if result.mode == "groq" else "Rule-based (offline)"
        st.markdown(
            f"""
            <div class="kpc-ai-panel">
                <div class="ai-tag">AI Summary · {mode_tag}</div>
                <p style="margin:0.3rem 0 0;">{result.summary}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Recommended investigation questions")
        for index, question in enumerate(result.questions, start=1):
            st.markdown(f"- {question}")
        if result.mode != "groq" and not os.environ.get("GROQ_API_KEY"):
            st.caption(
                "Tip: set GROQ_API_KEY to enable live LLM insights. The "
                "deterministic mode above is guaranteed in offline showcases."
            )


def _evidence_line(event: dict[str, Any]) -> str:
    parts = []
    shrink = float(event.get("volumetric_shrinkage_pct", 0) or 0)
    density = float(event.get("density_deviation_pct", 0) or 0)
    index = float(event.get("custody_handover_anomaly_index", 0) or 0)
    seal = int(event.get("e_seal_tamper_flag", 0) or 0)
    geofence = event.get("geofence_status", "—")
    parts.append(f"Product {event.get('product_type', '—')}")
    parts.append(f"declared {event.get('declared_volume_litres', '—'):,} L" if isinstance(
        event.get("declared_volume_litres"), (int, float)) else "declared volume N/A")
    parts.append(f"shrinkage {shrink:.2f}%")
    parts.append(f"density deviation {density:.2f}%")
    parts.append(f"anomaly index {index:.1f}")
    if seal:
        parts.append("eSeal tampered")
    if geofence not in {"OK", "—", None}:
        parts.append(f"geofence {geofence}")
    return "; ".join(parts)


def page_system_status(data_source: str) -> None:
    st.subheader("Data / System Status")
    st.markdown("- Data source: " + data_source)
    st.markdown("- AI layer: LangChain + Groq (automatic offline fallback)")
    st.markdown("- Risk engine: custody reconciliation + anomaly scoring")
    st.markdown("- Guardrails: responsible language, evidence-only reasoning")
    st.markdown("- Role-based access control: ready for extension")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="KPC Integrity Intelligence",
        page_icon="🛢️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_css()
    render_topbar()

    events, data_source = load_events()
    st.session_state["data_source"] = data_source

    with st.sidebar:
        st.markdown("## Navigation")
        st.caption("KPC Integrity Intelligence")
        st.markdown("##### Demo data")
        if st.button("Reset / Reload Data"):
            st.cache_data.clear()
            st.rerun()
        nav = st.radio(
            "Go to",
            ["Overview", "Consignments", "Quality Intelligence", "Custody Chain",
             "Investigation Assistant", "Data / System Status"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption(
            "Built on the HAKIKI-KPC reconciled custody dataset. "
            "Findings are recommendations for human review."
        )

    if nav == "Overview":
        page_overview(events)
    elif nav == "Consignments":
        page_consignments(events)
    elif nav == "Quality Intelligence":
        page_quality(events)
    elif nav == "Custody Chain":
        page_custody_chain()
    elif nav == "Data / System Status":
        page_system_status(data_source)
    else:
        page_investigation(events)

    st.markdown("---")
    st.caption(
        "This is an explainable prototype intended for investigation and "
        "early-warning review — not autonomous fraud accusation."
    )


if __name__ == "__main__":
    main()