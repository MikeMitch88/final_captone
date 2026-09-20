# AGENTS.md — HAKIKI-KPC (Kenya Pipeline / KRA Integrity Intelligence)

Fuel custody-chain integrity platform: detect transit diversion, quality
adulteration (MT Paloma pattern), and unauthorized export dumping across
Kenya's downstream corridor (KOT2 → Nairobi → Nakuru → Eldoret → Kisumu).

## Commands

```bash
# Synthetic dataset (Phase 1)
python src/generate_custody_data.py --normal 8000 --quality 150 --corridor 150
# Validation + reconciliation pipeline -> Parquet + SQLite
python src/ingestion_pipeline.py --db data/custody_data.db \
                                 --parquet data/custody_reconciled.parquet \
                                 --target_db data/custody_pipeline.db
# ML risk scoring (XGBoost)
python src/risk_model.py --db data/custody_data.db --model data/custody_risk_model.pkl
# Enterprise dashboard (the demo entrypoint)
streamlit run src/app.py
```

Run the generator + pipeline first to build `data/custody_*.db`; the app
auto-loads the reconciled dataset and falls back to synthetic demo data if the
DBs are missing.

## Architecture (data flow)

```text
generate_custody_data.py ──> data/custody_data.db (raw)
ingestion_pipeline.py ─────> data/custody_pipeline.db + custody_reconciled.parquet
feature_engineering.py ────> ML features (from raw DB)
risk_model.py ─────────────> data/custody_risk_model.pkl (XGBoost)
app.py ────────────────────> Streamlit enterprise dashboard + AI assistant
```

- Core tables (Postgres/SQLite DDL in `src/schema.sql`):
  `kra_manifests`, `rects_telemetry`, `kpc_depot_meters`,
  `depot_lab_quality`, `reconciled_custody_events`.
- Keys: manifests key on `manifest_id`; telemetry/meters/quality join on
  `consignment_id`. `reconciled_custody_events` has BOTH and joins on either.
- Reconciliation metrics already in DB: `volumetric_shrinkage_pct`,
  `density_deviation_pct`, `custody_handover_anomaly_index`, `anomaly_risk_flag`.
- `kra_manifests` may lack `declared_density_kg_per_m3` in old DBs —
  `feature_engineering.py` falls back to PMS nominal density 745 kg/m³.

## Quirks / gotchas (verified)

- **DB locations:** files live in `data/` (root). Older builds had them under
  `src/data/`. `app.py`'s `DB_CANDIDATES` checks both.
- **join keys differ:** `generate_custody_data.py` writes `consignment_id`
  into `kra_manifests` too; `schema.sql` omits it. Merging manifests with
  telemetry/meters/quality must be on `consignment_id`, and manifests↔events on
  `manifest_id`.
- **duplicate `depot_id`:** `depot_lab_quality` and `kpc_depot_meters` both have
  `depot_id`. A 4-way merge creates `depot_id_x`/`depot_id_y` — the
  reconciliation engine rewrites them to a single `depot_id`.
- **risk taxonomies differ:** engine uses Low/Medium/High; legacy demo data uses
  LOW/MODERATE/HIGH/CRITICAL. UI normalizes to 3 grades in `app._normalise_risk`.
- **idempotent:** both pipeline scripts use `if_exists="replace"`; safe to re-run.
- **fixed seed** `42` in generator for reproducible demo data.
- Streamlit `AppTest` first run is slow (matplotlib font-cache build); use
  `default_timeout=120` in tests.

## AI Intelligence Layer (`src/ai_assistant.py`)

- LangChain + `langchain-groq` (`ChatGroq`, model `llama-3.3-70b-versatile`).
- Guardrails are enforced in the system prompt AND scrubbed in code
  (`_apply_guardrail_language`): "high-risk anomaly detected" — NEVER "fraud";
  reason only from provided `<event_context>`; outputs are always framed as
  "recommendations for human review".
- **Offline fallback:** if `GROQ_API_KEY` unset or package missing, the
  `InvestigationAssistant` transparently uses deterministic rule-based
  generation so the demo never breaks. Check `.mode` ("groq" vs "offline-rules").
- LLM question parsing tolerates numbered lists: `Q1: …`, `1. …`, leading
  whitespace; else falls back to `_split_questions`.

## Enterprise UI (`src/app.py` + `src/styles.css`)

- Blue/white enterprise theme, NO dark mode. Indigo/purple accents ONLY on AI
  panels (`kpc-ai-panel`). Severity chips: red=Critical/High-warning-tier,
  orange=High, green=Low (classes `chip-critical/high/low`).
- Hide deploy/toolbar via CSS; KPI cards are HTML `kpc-kpi` blocks (not
  `st.metric`) with dropshadow. Large tables are capped (e.g. 300 rows) before
  styling to keep renders fast.

## Notes

- Requires: `pandas numpy faker pyarrow xgboost streamlit langchain langchain-groq`.
- This is a demo/prototype: explanations are recommendations for human review,
  not fraud determinations.