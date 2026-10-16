# What the data is for

Venture Global uses plant data to keep liquefaction trains producing as much LNG as they safely can. The public website does not describe this. The purpose is written in their job postings.

Calcasieu Pass is the operating plant. Plaquemines and CP2 are the next ones named in the postings.

Three labels are used below:

- **Confirmed:** stated in a Venture Global job posting.
- **Inferred:** a reasonable reading of those postings, not a named system.
- **Proposed:** a design for investigation, not a claim about a deployed model.

Public postings do not name trained models, target columns, or production error rates.

## What data science is meant to improve

The senior data scientist posting names two outcomes: higher LNG production efficiency, and lower safety and downtime risk. Production postings say the same thing in plant language: more LNG rundown, with the trains available and reliable. "Drive optimization" in the data scientist posting is that business outcome. The methods it names are forecasting and classification.

| Problem | What better looks like | Evidence |
|---|---|---|
| Production efficiency | More LNG from the trains that are already running | Confirmed |
| Downtime and reliability | Fewer hours a train is offline or producing less than it should | Confirmed |
| Safety | A train stays inside its allowed operating range | Confirmed |
| Planning | A production forecast that matches what the plant delivers | Confirmed |
| The next plant | Screens, tags, and lessons from Calcasieu Pass reused at Plaquemines, CP2, and later plants | Confirmed |

## Data sources

PI is the historian. Calculated tags and alerts are checks built from the raw readings. The lake is where reporting and models are built. The lake is not a separate physical measurement.

| Source | What it contains | Evidence |
|---|---|---|
| Plant sensors, through OSIsoft PI | Flows, pressures, temperatures, levels, and equipment status | Confirmed |
| PI calculated tags, alerts, and screens | Derived values and warnings | Confirmed |
| Flow computers | Custody-transfer flow, including Flow-X configurations checked against PI | Confirmed |
| SCADA | Pipeline and control-room signals. Liquefaction trains are also described with a DCS and PLCs | Confirmed as systems; the train-control split is inferred |
| Field and meter checks | Calibration and comparisons used to find measurement error and volume imbalance | Confirmed |
| Gas supply, maintenance, and the LNG ship schedule | Inputs to the daily production plan | Confirmed as inputs; the system that stores them is unknown |
| Azure Data Lake, Data Factory, and Databricks | The copy of operational data, cleaned for reports and models. Includes Data Lake Gen2, infrastructure as code, and CI/CD | Confirmed as the platform |
| Streaming and IoT | Live plant readings. Listed as a skill and a platform goal | Requirement confirmed; which feeds are live is unknown |

Equipment-status history, alarm acknowledgements, and ambient conditions are inferred. They are not named as integrated datasets.

## How they use the data

1. Sensors write to PI as the train runs.
2. PI screens, calculated tags, and alerts show when a reading leaves the allowed range.
3. Flow-computer and field measurements are checked against PI and against accounting volumes.
4. PI tags are copied into Azure Data Lake. Data Factory and Databricks clean them for reporting and models.
5. Data scientists forecast production and classify risk. Machine learning engineers run those models, including on streaming data when that path is used.
6. Production engineers combine the readings, the models, and the day's schedule to set how much LNG each train should make.
7. What worked at Calcasieu Pass is carried to the next plant: PI screens, tags, and operating lessons.

Physics-based models estimate expected performance from mass and energy balances and known limits. Machine-learning models learn from history. Postings name both. A hybrid, where a physics model sets the expected value and a learned model explains the gap, is a proposal, not a confirmed design.

## Working architecture

This is a reading of the postings, not an internal diagram. Azure, Data Factory, Data Lake Gen2, Databricks, machine learning, dashboards, and CI/CD are confirmed. The exact ingest schedule, storage zones, model registry, and batch-versus-online split are unknown.

```text
PHYSICAL LNG FACILITY
    |
    +-- Process instruments and equipment measurements
    +-- Flow meters and custody-transfer measurement systems
    +-- Pipeline SCADA, and the plant DCS / PLC signals
    +-- Equipment and train status
    |
    v
OPERATIONAL DATA SYSTEMS
    |
    +-- OSIsoft PI historian
    +-- Flow computers and measurement systems
    +-- Control-room monitoring and operating limits
    +-- Gas supply, maintenance, and LNG ship schedule
    |
    v
DATA INGESTION AND INTEGRATION
    |
    +-- Azure Data Factory
    +-- Batch ingestion and scheduled pipelines
    +-- Streaming / IoT ingestion, if implemented
    |
    v
AZURE DATA PLATFORM
    |
    +-- Azure Data Lake Gen2
    +-- Databricks / Spark
    +-- Raw and standardized datasets
    +-- Curated analytical datasets
    +-- Data warehouse or serving layer, if implemented
    |
    +------------------------------+
    |                              |
    v                              v
BUSINESS INTELLIGENCE          DATA SCIENCE / ML
    |                              |
    +-- Reports                    +-- Feature engineering
    +-- Dashboards                 +-- Training datasets
    +-- Production KPIs            +-- Predictive models
    +-- Reconciliation             +-- Classification models
                                   +-- Model evaluation
                                   +-- Model registry, if used
                                           |
                                           v
                                  DEPLOYMENT / SERVING
                                           |
                                  +--------+--------+
                                  |                 |
                                  v                 v
                             Batch scoring    Online scoring
                                  |                 |
                                  +--------+--------+
                                           |
                                           v
                              Predictions / alerts / reports
                                           |
                                           v
                              Production engineering review
                                           |
                                           v
                             Authorized operating decisions
```

**Facility.** Instruments on each train measure flow, pressure, temperature, level, and equipment state. Custody-transfer meters and flow computers (Flow-X is named) measure the gas and LNG that change hands. Pipeline SCADA is confirmed. The liquefaction trains are also described with a DCS and PLCs.

**Operational systems.** PI stores the historian, calculated tags, alerts, and screens. The control room compares live readings with the operating envelope. Gas supply, maintenance, and the ship schedule enter the daily plan. Which system stores those schedules is unknown.

**Ingestion.** Data Factory is the confirmed integration tool. Batch pipelines are the straightforward reading of that role. Streaming or IoT ingestion is a documented requirement, not a confirmed live feed.

**Platform.** Data Lake Gen2 holds the copy. Databricks and Spark standardize it and build the tables used for reports and models. A separate warehouse or serving layer is possible and not confirmed. Infrastructure as code and CI/CD are part of the data-engineer posting.

**Two consumers.** Business intelligence uses the curated tables for reports, dashboards, production KPIs, and measurement reconciliation. Data science uses the same tables for features, training sets, forecasts, and classifiers. A model registry is a normal part of that path and is not named in the postings.

**Serving.** Batch scoring fits daily forecasts and performance reviews. Online scoring fits a live alert. The senior data scientist posting asks for streaming experience. It does not say every model scores online. Predictions go to production engineers, who still set targets and decide what to change. An unconstrained model changing setpoints is not described.

### Proposed table layers

These names are a design, not Venture Global's published conventions.

| Layer | Contents | Operations |
|---|---|---|
| Bronze / raw | Source extracts and minimally altered measurements | Keep original values, source timestamps, identifiers, and ingestion metadata |
| Silver / standardized | Validated and time-aligned records | Unit conversion, duplicate handling, quality flags, train and equipment identity |
| Gold / curated | Train-level KPIs, reconciled production, and feature tables | Aggregation, joins to schedules, business rules |
| Model outputs | Predictions, scores, and model versions | Scoring, storage, and review by production engineering |

## Candidate models

These are not confirmed deployed models. The postings do not name algorithms or target columns.

| Use | Kind of model | What it would predict |
|---|---|---|
| Production forecast | Regression or time series | LNG rate or daily rundown over a stated horizon |
| Expected versus actual | Physics model, then a residual model | Gap between measured and expected production |
| Downtime risk | Classification | Whether an unplanned outage starts within a stated window |
| Operating-envelope check | Approved limits, with a classifier only as support | Whether the train is outside the limit for its current mode |
| Measurement check | Rules or anomaly detection | Disagreement between meter, flow computer, and PI |
| Next action | Constrained optimization | A feasible production change inside approved limits |

Before treating any of these as real, the target, horizon, features, baseline, owner, and whether the result is advice or an approved control action still have to be confirmed.

## Terms

**The plant**

- **LNG rundown** is the liquefied gas a train actually produces.
- **Liquefaction train** is one production unit. Calcasieu Pass has 18 of them.
- **Operating envelope** is the allowed range for temperature, pressure, and flow.
- **Availability** is the share of time a train can run. **Reliability** is how rarely it fails while running.
- **Custody transfer** is the official measurement when gas or LNG changes hands. A **flow computer** (Flow-X is named) calculates that volume.
- **Mass and energy balance** checks that what went in, what was produced, and what was lost all add up.
- **LAUF** is lost and unaccounted-for gas, the gap found by that balance.
- **Integrated Production Excellence** is the team that sets daily targets and watches the trains.

**Where the data lives**

- **PI tag** is one sensor's time series in the OSIsoft PI historian.
- **Calculated tag** is a value PI derives from other tags. An **alert** or **event frame** fires when a condition is met.
- **PI Vision** is the screen operators and engineers look at.
- **SCADA** is the pipeline and control-room system. **DCS** and **PLC** are the plant control systems on the trains.
- **Data Lake Gen2** is the Azure copy of those readings. **Data Factory** moves data in. **Databricks** and **PySpark** clean it and train models.
- **Unity Catalog** is Databricks' catalog of tables, permissions, and lineage.
- **Streaming / IoT** means using the readings as they arrive, not only in a nightly batch.
- **Bronze, silver, gold** are the raw, cleaned, and business-ready table layers. Those names are a common design, not names Venture Global has published.

**The modeling work**

- **Forecasting** predicts a number ahead of time, such as LNG rate. **Classification** predicts a category, such as downtime risk.
- **Feature** is a model input computed from raw tags, such as the last hour's average flow. A **feature store** keeps one definition of that input for both training and live scoring.
- **Horizon** is how far ahead the prediction looks.
- **Label** is the known outcome used to train the model.
- **Residual** is actual production minus what a physics model expected.
- **Batch scoring** runs the model on a schedule. **Online scoring** runs it as each new reading arrives.
- **Baseline** is the simple comparison a new model has to beat, such as "same as the last hour."
- **Drift** means the live data no longer looks like the data the model was trained on.

## References

- Plants, as the company describes them publicly: [venturegloballng.com](http://venturegloballng.com)
- Forecasting and classification for efficiency and downtime risk: [Senior Data Scientist](https://venturegloballng.wd108.myworkdayjobs.com/en-US/External_Careers/job/Senior-Data-Scientist_JR_1200)
- Daily targets, live limits, and physics plus data models: [Senior Engineer, Production](https://venturegloballng.wd108.myworkdayjobs.com/en-US/External_Careers/job/Senior-Engineer--Production_JR573)
- Gas supply, maintenance, and cargo schedule in the daily plan: [Advisor, Control Room Optimizing](https://job-boards.greenhouse.io/venturegloballng/jobs/7529121002)
- PI screens, alerts, and a corporate view as each plant comes online: [Administrator, OSI PI](https://venturegloballng.wd108.myworkdayjobs.com/en-US/External_Careers/job/Administrator--OSI-PI_JR_1010)
- Flow computers, SCADA, PI checks, and custody-transfer reconciliation: [Senior Specialist, Production Measurement](https://venturegloballng.wd108.myworkdayjobs.com/en-US/External_Careers/job/Senior-Specialist--Production-Measurement_JR547)
- Azure ingestion, Databricks, infrastructure as code, and CI/CD: [Data Engineer](https://venturegloballng.wd108.myworkdayjobs.com/en-US/External_Careers/job/Data-Engineer_JR_1054)
- Running models in production, including streaming: [Engineer, Machine Learning](https://builtin.com/job/engineer-machine-learning/11329653)
