"""Streamlit investigation dashboard for custody risk alerts."""

from pathlib import Path

import pandas as pd
import streamlit as st

from feature_engineering import CustodyFeatureEngineer
from risk_model import CustodyRiskModel


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "custody_data.db"
MODEL_PATH = ROOT / "data" / "custody_risk_model.pkl"


@st.cache_data(show_spinner=False)
def load_features(db_path: str) -> pd.DataFrame:
    return CustodyFeatureEngineer().build_from_sqlite(db_path)


@st.cache_resource(show_spinner=False)
def load_model(db_path: str, model_path: str) -> CustodyRiskModel:
    if Path(model_path).exists():
        return CustodyRiskModel.load(model_path)
    features = load_features(db_path)
    model = CustodyRiskModel().fit(features)
    model.save(model_path)
    return model


def main() -> None:
    st.set_page_config(
        page_title="Custody risk operations",
        page_icon=":material/local_gas_station:",
        layout="wide",
    )
    st.title("Custody risk operations")
    st.caption("Network view for volumetric shrinkage, quality drift, and transit exceptions")

    try:
        features = load_features(str(DB_PATH))
        model = load_model(str(DB_PATH), str(MODEL_PATH))
        scored = model.predict_scores(features)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    scored = scored.copy()
    scored["Destination"] = scored["destination_type"].replace({"Transit EAC": "Export"})
    scored["Severity"] = scored["Risk_Score"].map(
        lambda score: "Critical" if score >= 80 else "High" if score >= 50 else "Medium" if score >= 25 else "Low"
    )

    total_declared = scored["declared_volume_litres"].sum()
    shrinkage = (scored["volumetric_discrepancy"].clip(lower=0) * scored["declared_volume_litres"]).sum()
    active_alerts = int(scored["High_Risk_Alert"].sum())
    critical_alerts = int(scored["Severity"].eq("Critical").sum())
    with st.container(horizontal=True):
        st.metric("Network shrinkage", f"{shrinkage:,.0f} L", border=True)
        st.metric("Processed volume", f"{total_declared:,.0f} L", border=True)
        st.metric("Active high-risk alerts", f"{active_alerts:,}", border=True)
        st.metric("Critical alerts", f"{critical_alerts:,}", border=True)

    with st.sidebar:
        st.header("Investigation filters")
        destination = st.pills(
            "Destination", ["Domestic", "Export"], selection_mode="multi", default=["Domestic", "Export"]
        )
        severity = st.pills(
            "Risk severity", ["Critical", "High", "Medium", "Low"], selection_mode="multi", default=["Critical", "High"]
        )
        product = st.multiselect("Product", sorted(scored["product_type"].dropna().unique()), default=sorted(scored["product_type"].dropna().unique()))
        depot = st.multiselect("Depot", sorted(scored["depot_id"].dropna().unique()), default=sorted(scored["depot_id"].dropna().unique()))
        minimum_risk = st.slider("Minimum risk score", 0, 100, 0)

    view = scored[
        scored["Destination"].isin(destination)
        & scored["Severity"].isin(severity)
        & scored["product_type"].isin(product)
        & scored["depot_id"].isin(depot)
        & scored["Risk_Score"].ge(minimum_risk)
    ]
    view = view.sort_values("Risk_Score", ascending=False)

    chart_one, chart_two = st.columns(2)
    with chart_one:
        with st.container(border=True):
            st.subheader("Risk profile")
            risk_counts = scored["Severity"].value_counts().reindex(["Critical", "High", "Medium", "Low"], fill_value=0)
            st.bar_chart(risk_counts.rename("Consignments"), horizontal=True)
    with chart_two:
        with st.container(border=True):
            st.subheader("Shrinkage by destination")
            shrinkage_by_destination = (
                scored.assign(shrinkage_litres=scored["volumetric_discrepancy"].clip(lower=0) * scored["declared_volume_litres"])
                .groupby("Destination", as_index=True)["shrinkage_litres"]
                .sum()
                .rename("Litres")
            )
            st.bar_chart(shrinkage_by_destination)

    st.subheader(f"Alert queue · {len(view):,} matching consignments")
    display_columns = [
        "consignment_id",
        "manifest_id",
        "Destination",
        "product_type",
        "depot_id",
        "Risk_Score",
        "Severity",
        "volumetric_discrepancy",
        "density_drift",
        "risk_telemetry_score",
        "custody_gap_flag",
    ]
    st.dataframe(
        view[display_columns],
        column_config={
            "Risk_Score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.1f"),
            "volumetric_discrepancy": st.column_config.NumberColumn("Volume discrepancy", format="%.2f%%"),
            "density_drift": st.column_config.NumberColumn("Density drift", format="%.1f kg/m³"),
            "risk_telemetry_score": st.column_config.NumberColumn("Telemetry score", format="%.1f"),
        },
        width="stretch",
        hide_index=True,
    )

    if not view.empty:
        st.subheader("Consignment investigation")
        selected_id = st.selectbox("Select a consignment", view["consignment_id"].tolist())
        selected = view.loc[view["consignment_id"].eq(selected_id)].iloc[0]
        detail_columns = st.columns(4)
        detail_columns[0].metric("Risk score", f"{selected['Risk_Score']:.1f}/100")
        detail_columns[1].metric("Volume discrepancy", f"{selected['volumetric_discrepancy']:.2%}")
        detail_columns[2].metric("Density drift", f"{selected['density_drift']:.1f} kg/m³")
        detail_columns[3].metric("Telemetry score", f"{selected['risk_telemetry_score']:.1f}/100")
        st.caption(
            f"{selected['consignment_id']} · {selected['product_type']} · {selected['Destination']} · depot {selected['depot_id']} · {selected['Severity']}"
        )
        st.download_button(
            "Download filtered alerts",
            view[display_columns].to_csv(index=False).encode("utf-8"),
            file_name="custody_alerts.csv",
            mime="text/csv",
            icon=":material/download:",
        )


if __name__ == "__main__":
    main()