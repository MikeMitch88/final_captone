"""KPC Integrity Intelligence dashboard for fuel quality and custody-chain monitoring."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard_data import build_dashboard_data


st.set_page_config(page_title="KPC Integrity Intelligence", page_icon="🛢️", layout="wide")


def render_kpi(title: str, value: str, delta: str, positive: bool = True) -> None:
    st.markdown(
        f"""
        <div style="padding: 0.9rem 1rem; border: 1px solid #dfe3e8; border-radius: 10px; background: #f8fafc; margin-bottom: 0.5rem;">
            <div style="font-size: 0.8rem; color: #475467;">{title}</div>
            <div style="font-size: 1.8rem; font-weight: 700; margin-top: 0.25rem; color: #101828;">{value}</div>
            <div style="font-size: 0.75rem; margin-top: 0.25rem; color: {'#067647' if positive else '#b42318'};">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def get_dashboard() -> dict:
    return build_dashboard_data()


def main() -> None:
    data = get_dashboard()
    overview = data["overview"]
    consignments = data["consignments"].copy()
    priority_queue = data["priority_queue"].copy()
    risk_distribution = data["risk_distribution"].copy()
    risk_drivers = data["risk_drivers"].copy()
    quality_records = data["quality_records"].copy()
    reconciliation_records = data["reconciliation_records"].copy()
    alerts = data["alerts"].copy()
    investigations = data["investigations"].copy()
    custody_chain = data["custody_chain"]
    trend_data = data["risk_trends"].copy()

    st.markdown(
        """
        <style>
            .block-container { padding-top: 1rem; }
            div[data-testid="stSidebar"] { background: #f8fafc; }
            .stTabs [role="tablist"] { gap: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## KPC Integrity Intelligence")
        st.caption("Fuel Quality & Custody-Chain Risk Monitoring")
        st.markdown("#### DEMO DATA")
        if st.button("Reset Demo Data"):
            st.cache_data.clear()
            st.rerun()
        nav = st.radio(
            "Navigation",
            [
                "Overview",
                "Risk Monitor",
                "Consignments",
                "Custody Chain",
                "Quality Intelligence",
                "Reconciliation",
                "Investigations",
                "Alerts",
                "Analytics",
                "Data / System Status",
            ],
            index=0,
        )

    st.title("KPC Integrity Intelligence")
    st.caption("RECONCILE → DETECT ANOMALIES → SCORE RISK → INVESTIGATE → TAKE ACTION")

    if nav == "Overview":
        kpi_data = [
            ("Total Consignments Monitored", str(overview["total_consignments"]), "+3.1% vs previous period"),
            ("Reconciled Consignments", str(overview["reconciled_consignments"]), "+4.7% vs previous month"),
            ("Open Exceptions", str(overview["open_exceptions"]), "-2.3% from last week"),
            ("High/Critical Risk Consignments", str(overview["high_critical_risk"]), "+18% vs previous period"),
            ("Quality Exceptions", str(overview["quality_exceptions"]), "+6 cases this month"),
            ("Volume Discrepancy", overview["volume_discrepancy"], "+9.6% variance trend"),
            ("Estimated Financial Exposure", overview["estimated_financial_exposure"], "+7.4% this month"),
            ("Investigations Open", str(overview["investigations_open"]), "5 escalated this week"),
        ]

        cols = st.columns(4)
        for i, (title, value, delta) in enumerate(kpi_data):
            with cols[i % 4]:
                render_kpi(title, value, delta, positive=("+" in delta or "High/Critical" in title or "Estimated" in title))

        risk_col, driver_col = st.columns([1.5, 1.5])
        with risk_col:
            st.subheader("Risk Distribution")
            st.bar_chart(risk_distribution.rename("Consignments"), horizontal=True)
        with driver_col:
            st.subheader("Top Risk Drivers")
            st.dataframe(risk_drivers, hide_index=True, use_container_width=True)

        st.subheader("Priority Investigation Queue")
        queue = priority_queue[[
            "consignment_id",
            "product",
            "origin",
            "destination",
            "declared_volume",
            "actual_volume",
            "variance",
            "quality_status",
            "custody_status",
            "risk_score",
            "financial_exposure",
            "risk_level",
            "investigation_status",
        ]].copy()
        queue = queue.rename(columns={
            "consignment_id": "Consignment ID",
            "product": "Product",
            "origin": "Origin",
            "destination": "Destination",
            "declared_volume": "Declared Volume",
            "actual_volume": "Actual Volume",
            "variance": "Variance",
            "quality_status": "Quality Status",
            "custody_status": "Custody Status",
            "risk_score": "Risk Score",
            "financial_exposure": "Financial Exposure",
            "risk_level": "Risk",
            "investigation_status": "Investigation Status",
        })
        st.dataframe(queue, use_container_width=True, hide_index=True)

    elif nav == "Risk Monitor":
        st.subheader("Risk Overview")
        risk_distribution = risk_distribution.rename("Consignments")
        st.bar_chart(risk_distribution, horizontal=True)
        st.dataframe(risk_drivers, hide_index=True, use_container_width=True)

    elif nav == "Consignments":
        st.subheader("Consignment Records")
        search = st.text_input("Search by consignment or product")
        filtered = consignments.copy()
        if search:
            filtered = filtered[
                filtered["consignment_id"].str.contains(search, case=False, na=False)
                | filtered["product"].str.contains(search, case=False, na=False)
            ]
        st.dataframe(filtered[[
            "consignment_id",
            "product",
            "origin",
            "destination",
            "declared_volume",
            "actual_volume",
            "variance_pct",
            "quality_status",
            "custody_status",
            "risk_score",
            "risk_level",
            "financial_exposure",
        ]].rename(columns={
            "consignment_id": "Consignment ID",
            "product": "Product",
            "origin": "Origin",
            "destination": "Destination",
            "declared_volume": "Declared Volume",
            "actual_volume": "Actual Volume",
            "variance_pct": "Variance %",
            "quality_status": "Quality Status",
            "custody_status": "Custody Status",
            "risk_score": "Risk Score",
            "risk_level": "Risk",
            "financial_exposure": "Financial Exposure",
        }), hide_index=True, use_container_width=True)

    elif nav == "Custody Chain":
        st.subheader("Custody-Chain Visualization")
        st.caption("Origin → Storage → Pipeline → Depot → Destination")
        for idx, node in enumerate(custody_chain):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### {node['node']}")
                st.write(f"Volume: {node['volume']:,} L")
                st.write(f"Status: {node['status']}")
                st.write(f"Risk: {node['risk']}")
            with col2:
                if idx < len(custody_chain) - 1:
                    st.markdown("<div style='text-align:center; font-size:2rem;'>↓</div>", unsafe_allow_html=True)
        if st.checkbox("Highlight discrepancy path"):
            st.warning("Discrepancy emerges at Nairobi Depot with a 2.8% variance and custody anomaly.")

    elif nav == "Quality Intelligence":
        st.subheader("Quality Intelligence")
        qcols = st.columns(6)
        quality_values = [
            ("Total consignments tested", str(len(quality_records)), "+2.1%"),
            ("Quality records verified", str(len(quality_records)), "+8 this week"),
            ("Missing quality records", str((quality_records["test_status"] == "MISSING").sum()), "Needs review"),
            ("Failed tests", str((quality_records["test_status"] == "FAILED").sum()), "2 escalated"),
            ("Pending tests", str((quality_records["test_status"] == "PENDING").sum()), "2 awaiting lab"),
            ("Expired certificates", str((quality_records["test_status"] == "EXPIRED").sum()), "1 requires action"),
        ]
        for col, (title, value, delta) in zip(qcols, quality_values):
            with col:
                st.metric(title, value, delta)
        st.dataframe(quality_records, hide_index=True, use_container_width=True)

    elif nav == "Reconciliation":
        st.subheader("Reconciliation Workspace")
        reconc_cols = [
            "Consignment",
            "Declared Volume",
            "Movement Volume",
            "Received Volume",
            "Stock Volume",
            "Variance",
            "Tolerance",
            "Result",
            "Risk",
        ]
        st.dataframe(reconciliation_records[reconc_cols], hide_index=True, use_container_width=True)

    elif nav == "Investigations":
        st.subheader("Investigation Management")
        selected_inv = st.selectbox("Choose investigation", investigations["consignment_id"].tolist())
        case = investigations[investigations["consignment_id"] == selected_inv].iloc[0]
        priority_map = {"LOW": 20, "MODERATE": 45, "HIGH": 70, "CRITICAL": 92}
        progress_value = priority_map.get(str(case["priority"]).upper(), 55)
        st.write(f"**Why was this transaction flagged?** {case['reason']}")
        st.write(f"**Financial exposure:** KES {case['financial_exposure']:,.0f}")
        st.write(f"**Recommended next action:** Validate the physical stock measurements, review the manifest amendment history, and confirm the quality documentation before releasing the shipment.")
        st.progress(progress_value / 100, text=f"Investigation priority: {case['priority']}")
        st.text_area("Investigation notes", value="Evidence supports escalation due to volume variance, missing quality certificate, and abnormal custody transfer timing.")

    elif nav == "Alerts":
        st.subheader("Alert Center")
        st.dataframe(alerts, hide_index=True, use_container_width=True)
        st.button("Acknowledge selected alert")
        st.button("Escalate alert")

    elif nav == "Analytics":
        st.subheader("Analytics")
        st.line_chart(trend_data.set_index("Week"), x=None)
        st.bar_chart(
            consignments.groupby("product")["financial_exposure"].sum().rename("Exposure by Product"),
            horizontal=False,
        )

    elif nav == "Data / System Status":
        st.subheader("System Status")
        st.markdown("- Demo Mode: Enabled")
        st.markdown("- Synthetic data: Active")
        st.markdown("- Risk scoring model: Explainable rule-based engine")
        st.markdown("- Data coverage: 120 consignments, 50 manifests, 100+ movement records, 80 quality records, 30 alerts")
        st.markdown("- Role-based access control: ready for extension")

    st.markdown("---")
    st.caption("This is an explainable prototype intended for investigation and early-warning review, not autonomous fraud accusation.")


if __name__ == "__main__":
    main()