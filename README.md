# GNN_based_mule_account_detector
**Privacy-Preserving Cross-Bank Mule Account Detection using Graph Neural Networks**

MuleGuard AI is a graph-based fraud detection framework designed to detect mule accounts across multiple banks in a privacy-preserving, explainable, and scalable manner.

---

## Key Features

- **Multi-Source Data Ingestion**  
  Ingests events from core banking transactions, onboarding/KYC data, payment logs, POS swipes, digital channel telemetry, call-centre metadata, and behavioral biometrics.

- **Graph Construction**  
  Builds a heterogeneous graph with nodes (accounts, customers, devices, merchants, sessions) and weighted edges (transactions, logins, device use) with temporal decay.

- **Graph Neural Network Modeling**  
  Learns embeddings that capture fan-in/out, suspicious clusters, and transaction cycles to detect coordinated mule activity.

- **Risk Scoring and Decision Engine**  
  Combines GNN scores, time-series anomaly detection, and rule-based risk uplifts into a single fused score and maps it to actions: ALLOW, STEP-UP, HOLD, or BLOCK.

- **Federated Learning**  
  Enables cross-bank model training without sharing raw data, ensuring compliance with regulatory and privacy requirements.

- **Explainability**  
  Produces human-readable evidence paths and top contributing features for each alert to support analyst review and audit trails.

- **Configuration Driven**  
  All thresholds, fusion weights, and edge-weight parameters are externalized in `muleguard_config.csv` for easy tuning.

---

## Architecture Overview

```mermaid
graph TD
    A[Streaming Events] --> B[Feature Engineering]
    B --> C[Identity Resolution]
    C --> D[Graph Construction]
    D --> E[GNN and Anomaly Models]
    E --> F[Score Fusion and Risk Scoring]
    F --> G[Decision Engine]
    G -->|Escalated Cases| H[Analyst Review and Case Management]
    H --> I[Feedback and Model Retraining]
    F -.->|Secure Updates| J[Federated Learning Aggregator]
    J -.-> E
