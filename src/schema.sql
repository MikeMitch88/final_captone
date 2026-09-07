CREATE TABLE IF NOT EXISTS kra_manifests (
    manifest_id TEXT PRIMARY KEY,
    omc TEXT NOT NULL,
    product_type TEXT NOT NULL CHECK (product_type IN ('PMS', 'AGO', 'DPK')),
    declared_volume_litres REAL NOT NULL,
    source TEXT NOT NULL,
    destination_type TEXT NOT NULL CHECK (destination_type IN ('Domestic', 'Transit EAC')),
    bond_status TEXT NOT NULL DEFAULT 'Unknown',
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rects_telemetry (
    consignment_id TEXT PRIMARY KEY,
    vehicle_seal_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    gps_latitude REAL NOT NULL,
    gps_longitude REAL NOT NULL,
    geofence_status TEXT NOT NULL CHECK (geofence_status IN ('OK', 'Out-of-Corridor', 'Route-Deviation')),
    e_seal_tamper_flag INTEGER NOT NULL CHECK (e_seal_tamper_flag IN (0, 1)),
    dwell_time_minutes INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS kpc_depot_meters (
    meter_transaction_id TEXT PRIMARY KEY,
    consignment_id TEXT NOT NULL,
    depot_id TEXT NOT NULL,
    meter_in_volume REAL NOT NULL,
    meter_out_volume REAL NOT NULL,
    temperature REAL NOT NULL,
    observed_density_kg_per_m3 REAL NOT NULL,
    FOREIGN KEY (consignment_id) REFERENCES rects_telemetry (consignment_id)
);

CREATE TABLE IF NOT EXISTS depot_lab_quality (
    lab_test_id TEXT PRIMARY KEY,
    consignment_id TEXT NOT NULL,
    depot_id TEXT NOT NULL,
    sample_timestamp TIMESTAMP NOT NULL,
    density_at_15c REAL NOT NULL,
    research_octane_number INTEGER NOT NULL,
    flash_point REAL NOT NULL,
    sulfur_content_ppm REAL NOT NULL,
    pass_quality_flag INTEGER NOT NULL CHECK (pass_quality_flag IN (0, 1)),
    FOREIGN KEY (consignment_id) REFERENCES rects_telemetry (consignment_id)
);

CREATE TABLE IF NOT EXISTS reconciled_custody_events (
    event_id TEXT PRIMARY KEY,
    manifest_id TEXT NOT NULL,
    consignment_id TEXT NOT NULL,
    depot_id TEXT NOT NULL,
    volumetric_shrinkage_pct REAL NOT NULL,
    density_deviation_pct REAL NOT NULL,
    custody_handover_anomaly_index REAL NOT NULL,
    anomaly_risk_flag TEXT NOT NULL CHECK (anomaly_risk_flag IN ('Low', 'Medium', 'High')),
    calculated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (manifest_id) REFERENCES kra_manifests (manifest_id),
    FOREIGN KEY (consignment_id) REFERENCES rects_telemetry (consignment_id),
    FOREIGN KEY (depot_id) REFERENCES kpc_depot_meters (depot_id)
);

-- Indexes for query performance
CREATE INDEX IF NOT EXISTS idx_kra_manifests_product_type ON kra_manifests (product_type);
CREATE INDEX IF NOT EXISTS idx_kra_manifests_destination ON kra_manifests (destination_type);
CREATE INDEX IF NOT EXISTS idx_kra_manifests_omc ON kra_manifests (omc);
CREATE INDEX IF NOT EXISTS idx_rects_telemetry_geofence ON rects_telemetry (geofence_status);
CREATE INDEX IF NOT EXISTS idx_rects_telemetry_tamper ON rects_telemetry (e_seal_tamper_flag);
CREATE INDEX IF NOT EXISTS idx_depot_meters_depot ON kpc_depot_meters (depot_id);
CREATE INDEX IF NOT EXISTS idx_depot_lab_depot ON depot_lab_quality (depot_id);
CREATE INDEX IF NOT EXISTS idx_custody_events_manifest ON reconciled_custody_events (manifest_id);
CREATE INDEX IF NOT EXISTS idx_custody_events_consignment ON reconciled_custody_events (consignment_id);
CREATE INDEX IF NOT EXISTS idx_custody_events_risk ON reconciled_custody_events (anomaly_risk_flag);