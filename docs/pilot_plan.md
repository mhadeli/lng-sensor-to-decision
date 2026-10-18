# Proposed plant pilot

The local implementation is complete as a demonstration. The following steps require real plant data and stakeholder agreement; they are not deployed integrations.

| Stage | Owner / partners | Deliverable and decision gate |
|---|---|---|
| Define decision | Reliability engineer and operations lead | Agree on actionable incident labels, warning horizon, minimum lead time, missed-incident tolerance and maximum review workload before selecting a model. |
| Data readiness | Data engineer and data scientist | Map historian tags, equipment IDs, units, timestamps, availability times, operating modes and maintenance records. Agree on freshness and quarantine limits. |
| Offline pilot | Data scientist and reliability engineer | Freeze candidates and evaluate on later periods and separate equipment; meet incident and workload requirements with documented uncertainty. |
| Shadow operation | Operations owner | Review recommendations alongside existing processes, record usefulness and missed events, and assess enough incidents and operating modes to support a decision. |
| Release and support | Operations owner, IT and data science | Approve versioned model/threshold changes, monitoring, ownership and a tested rollback path. Otherwise retain the baseline or stop the candidate. |

Retain approved alerts and the established planning process throughout the pilot. Model outputs never change control setpoints. Dependencies include historian access, label review, maintenance/schedule history and engineering time; duration and staffing follow from those dependencies rather than an invented guaranteed timeline.

For production forecasting, replace terminal exports with daily train-level LNG rundown in consistent plant units. Add availability, operating mode, feed-gas quantity/quality, ambient conditions and planned maintenance as known at the forecast cutoff. Cargo schedules belong in shipment/inventory planning. Validate against the current planning method before recommending a change.

## Local-to-cloud architecture mapping

| Implemented locally | Comparable role in the job description's platform |
|---|---|
| Python runner and per-stage logs | Data Factory orchestration |
| Source files, hashes and Parquet layers | Data Lake ingestion and storage |
| DuckDB SQL + Python preparation | Databricks / PySpark transformations |
| Saved models and registry JSON | Managed model tracking and registry |
| Historical replay, quality and score logs | Streaming or scheduled inference and monitoring |
| Tests, GitHub Actions and a static dashboard | DevOps, delivery and stakeholder reporting |

These are functional counterparts, not claims of Azure or Spark execution. A future cloud implementation should port the transformations and contracts rather than simply rename local folders.

## Handoff / mentoring exercise

Ask another analyst to trace one machine's raw records to its feature row, verify a future-information test, reproduce a replay batch, and explain why a forecast cannot use an unpublished month. Review their proposed change through tests and the model card. This is a reusable teaching artifact, not a claim that mentoring has already occurred.
