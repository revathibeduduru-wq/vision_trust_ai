# TrustLens System Architecture — CV & Dataset Integrity Engine (SIH26228)

## 1. System Context & Layered Design

```
+---------------------------------------------------------------------------------+
|                                TrustLens Dashboard (M5)                         |
|   (Dataset Integrity Panel | Evidence Explorer | Pipeline Visualizer | Metrics) |
+---------------------------------------------------------------------------------+
                                         |
                                         v HTTP (REST / JSON)
+---------------------------------------------------------------------------------+
|                       FastAPI Routing & Service Orchestrator                    |
|                POST /api/datasets/analyze  |  POST /api/datasets/analyze-image  |
+---------------------------------------------------------------------------------+
                                         |
         +-------------------------------+-------------------------------+
         |                               |                               |
         v                               v                               v
+------------------+           +--------------------+          +--------------------+
|  Dataset Loader  |           |  Image Forensics   |          | Duplicate Detector |
|  - Folder Engine |           |  - SHA-256 Hashing |          | - SHA-256 (Exact)  |
|  - CSV Ingestion |           |  - Perceptual Hash |          | - pHash Hamming    |
|  - YOLO Parser   |           |  - Blur/Laplacian  |          | - DSU Clustering   |
+------------------+           |  - Brightness/RMS  |          +--------------------+
         |                     |  - Corruption Check|                    |
         |                     +--------------------+                    |
         +-------------------------------+-------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                         Embedding Feature Extraction Engine                     |
|           - TorchVision ResNet-18 Penultimate Layer (512-dim Normalized)        |
|           - Offline Spatial Pyramid Pooling + 2D DCT Texture Extractor (512-dim)|
+---------------------------------------------------------------------------------+
                                         |
         +-------------------------------+-------------------------------+
         |                                                               |
         v                                                               v
+------------------------------------+          +-----------------------------------+
|    OOD & Anomaly Detector (M2)     |          |   Label Consistency Checker (M2)  |
| - Isolation Forest Ensembles       |          | - k-NN Feature Cosine Consensus   |
| - Centroid Cosine Distance Scoring |          | - Cluster Disagreement Ratio      |
| - Normalized [0.0 - 1.0] Anomalies |          | - Section 11 Non-Adversarial Tone |
| - Section 10 Conservative Tone     |          +-----------------------------------+
+------------------------------------+                                   |
         |                                                               |
         +-------------------------------+-------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                    Section 22 Evidence Object Transformer                       |
|           Transforms all findings into standardized canonical Evidence:         |
|  id | finding_type | what_happened | why_flagged | evidence | severity |        |
|            confidence | affected_asset | recommended_action | limitations       |
+---------------------------------------------------------------------------------+
                                         |
         +-------------------------------+-------------------------------+
         |                                                               |
         v                                                               v
+------------------------------------+          +-----------------------------------+
|     Risk & Trust Engine (M4)       |          |    Tamper-Evident Ledger (M1)     |
| - Consumes Evidence for 30%        |          | - Hashes canonical dataset        |
|   Dataset Integrity trust score    |          |   manifest into tamper-evident    |
| - Feeds 9-Attack Security Lab      |          |   audit block chain               |
+------------------------------------+          +-----------------------------------+
```

---

## 2. Mathematical Formulations

### A. Perceptual Image Sharpness (Laplacian Blur Score)
Given a grayscale image matrix $I(x, y)$, the 2D discrete Laplacian convolution $\nabla^2 I$ is computed using the second-derivative kernel:
$$\nabla^2 I = I * \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$
The blur score $S_{\text{blur}}$ is defined as the statistical variance of the Laplacian:
$$S_{\text{blur}} = \sigma^2(\nabla^2 I) = \frac{1}{N} \sum_{x, y} \left( \nabla^2 I(x, y) - \mu_{\nabla^2 I} \right)^2$$
High variance indicates sharp, high-frequency edge content; near-zero variance indicates defocus or blur.

### B. Perceptual Similarity & Duplicate Flooding Clustering
For two 64-bit DCT perceptual hash values $H_1, H_2$, the Hamming distance $d_H$ is:
$$d_H(H_1, H_2) = \sum_{b=0}^{63} \left( H_1[b] \oplus H_2[b] \right)$$
The normalized similarity percentage is:
$$\text{Sim}(H_1, H_2) = \left( 1 - \frac{d_H(H_1, H_2)}{64} \right) \times 100\%$$
A Disjoint Set Union (DSU) forest groups samples with $\text{Sim} \ge \tau$ (default 90%) into duplicate clusters, tagged with the mandatory indicator `"Potential duplicate flooding indicator"`.

### C. Out-Of-Distribution (OOD) Scoring
Let $\mathbf{e}_i \in \mathbb{R}^{512}$ be the $L_2$-normalized feature embedding for sample $i$, and $\mathbf{c} = \frac{1}{N}\sum_{j=1}^N \mathbf{e}_j$ be the dataset feature centroid.
1. Isolation Forest decision function $s_{\text{iso}}(\mathbf{e}_i)$ measures recursive tree partitioning depth.
2. Centroid cosine distance is $d_{\cos}(\mathbf{e}_i, \mathbf{c}) = 1 - \frac{\mathbf{e}_i \cdot \mathbf{c}}{\|\mathbf{e}_i\| \|\mathbf{c}\|}$.
3. The normalized composite OOD score $S_{\text{OOD}}(\mathbf{e}_i) \in [0, 1]$ is:
   $$S_{\text{OOD}}(\mathbf{e}_i) = 0.60 \cdot \widetilde{s}_{\text{iso}}(\mathbf{e}_i) + 0.40 \cdot \widetilde{d}_{\cos}(\mathbf{e}_i, \mathbf{c})$$

### D. Label Consistency via Neighborhood Consensus
For a sample $i$ with assigned label $y_i$, let $\mathcal{N}_k(i)$ be the set of its $k$-nearest neighbors in cosine embedding space.
The neighborhood consensus label $y^*_i$ and agreement ratio $R_i$ are:
$$y^*_i = \arg\max_{c} \sum_{j \in \mathcal{N}_k(i)} \mathbb{I}(y_j = c), \quad R_i = \frac{1}{k} \sum_{j \in \mathcal{N}_k(i)} \mathbb{I}(y_j = y^*_i)$$
If $y^*_i \neq y_i$ and $R_i \ge 0.60$, the sample is flagged with `"Potential label inconsistency"`.

---

## 3. Offline-First Guarantee & Resiliency
1. **Zero External Network Dependencies**: The engine never invokes external cloud APIs, inference servers, or downloads.
2. **Dual-Path Embedding Pipeline**:
   - Primary: Local TorchVision ResNet-18 penultimate representations when weights exist.
   - Secondary / Air-Gapped Fallback: Multi-scale Spatial Pyramid Pooling (1x1, 2x2, 4x4) + 2D DCT texture decomposition producing deterministic 512-dimensional vectors.
3. **Graceful Corruption Isolation**: Corrupt or truncated files are isolated and flagged without interrupting the analysis of healthy samples.
