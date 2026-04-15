# Pipeline Flow Chart

## Flow Summary

This project runs as a full end-to-end image clustering pipeline:

- input discovery and runtime setup
- CLIP embedding generation
- visual and optional item-feature extraction
- two-stage clustering with default and strict branches
- match-score generation
- output folder and JSON report writing

## Complete Mermaid Flow Diagram

```mermaid
flowchart TD
    subgraph S1["1. Setup and Input"]
        A([Start]) --> B[Parse CLI arguments]
        B --> C[Resolve input output cache and device]
        C --> D[Discover supported images]
        D --> E{Any supported images found?}
    end

    E -- No --> F([Exit with error])

    subgraph S2["2. Feature Extraction"]
        G[Load CLIP runtime]
        H[Embed images with CLIP]
        I[Extract visual features<br/>layout edge color]
        J{Strict same-corner plus same-items mode?}
        K[Extract CLIP prompt-signature item features]
        L[Skip item feature extraction]
        M[Align valid image paths across feature branches]
        N[Combine CLIP and visual features into hybrid embeddings]
    end

    E -- Yes --> G
    G --> H
    H --> I
    I --> J
    J -- Yes --> K --> M
    J -- No --> L --> M
    M --> N

    subgraph S3["3. Stage 1 Semantic Clustering"]
        O[Run HDBSCAN on CLIP embeddings]
        P{Any semantic clusters found?}
        Q[Fallback to one-stage HDBSCAN on hybrid embeddings]
        R[Loop through each semantic cluster]
    end

    N --> O
    O --> P
    P -- No --> Q
    P -- Yes --> R

    subgraph S4["4A. Shared Viewpoint Logic"]
        V1[Load cluster images]
        V2[Build layout vectors]
        V3[Build edge vectors]
        V4[Build opening profiles]
        V5[Build ORB descriptors]
        V6[Compute viewpoint similarity matrix<br/>0.35 layout + 0.25 edge + 0.25 opening + 0.15 ORB]
        V1 --> V2 --> V6
        V1 --> V3 --> V6
        V1 --> V4 --> V6
        V1 --> V5 --> V6
    end

    subgraph S5["4B. Default Refinement Path"]
        D1[Run local HDBSCAN on hybrid embeddings]
        D2{4-image cluster with clean split?}
        D3[Split into 2 pairs using opening profile plus ORB]
        D4{Large broad cluster of 5 or more images?}
        D5[Agglomerative split using viewpoint threshold]
        D6[Keep local group as-is]
        D7[Merge semantically similar subclusters back<br/>using CLIP centroid similarity]
        D8[Assign final labels for this semantic cluster]
        D1 --> D2
        D2 -- Yes --> D3 --> D7
        D2 -- No --> D4
        D4 -- Yes --> D5 --> D7
        D4 -- No --> D6 --> D7
        D7 --> D8
    end

    subgraph S6["4C. Strict Refinement Path"]
        T1{Exactly 4 images?}
        T2[Try 2-pair split using opening profile plus ORB]
        T3{Pair split succeeded?}
        T4[Create 2 strict pairs]
        T5[Compute semantic similarity matrix]
        T6[Compute item similarity matrix]
        T7[Apply hard gates<br/>view >= threshold<br/>item >= threshold<br/>semantic >= floor]
        T8[Build strict similarity<br/>0.50 viewpoint + 0.30 item + 0.20 semantic]
        T9[Complete-link agglomerative clustering]
        T10[Groups below min_cluster_size become noise]
        T11[Assign final labels for this semantic cluster]
        T1 -- Yes --> T2 --> T3
        T3 -- Yes --> T4 --> T11
        T3 -- No --> T5
        T1 -- No --> T5
        T5 --> T6 --> T7 --> T8 --> T9 --> T10 --> T11
    end

    R --> X{Strict mode for this run?}
    X -- No --> D1
    X -- Yes --> T1
    R --> V1
    V6 --> D5
    V6 --> T7

    D8 --> Z{More semantic clusters left?}
    T11 --> Z
    Z -- Yes --> R

    subgraph S7["5. Output and Reporting"]
        Y1[Reset output directory]
        Y2[Copy images into cluster_k folders and noise folder]
        Y3[Write clusters.json]
        Y4[Build pairwise match score payload]
        Y5{Strict mode scoring?}
        Y6[Score with viewpoint + item + semantic]
        Y7[Score with hybrid + viewpoint + semantic]
        Y8[Write match_scores.json]
        Y9[Print final cluster summary]
        Y10([End])
    end

    Q --> Y1
    Z -- No --> Y1
    Y1 --> Y2 --> Y3 --> Y4 --> Y5
    Y5 -- Yes --> Y6 --> Y8
    Y5 -- No --> Y7 --> Y8
    Y8 --> Y9 --> Y10
```

## Step-by-Step Explanation

### 1. Read inputs

The script reads the input directory and keeps only `.jpg`, `.jpeg`, and `.png` files.

### 2. Build semantic embeddings

Each valid image is encoded with CLIP into a normalized embedding. These embeddings are the broad semantic representation of the image.

### 3. Build visual features

Each image also gets hand-crafted visual features:

- coarse grayscale layout
- edge structure
- color histogram

These are used later to separate images that are semantically similar but visually taken from different corners.

### 4. Optional strict item features

If strict mode is enabled, the script also computes a CLIP prompt-signature distribution over a fixed prompt list. This acts as a soft description of visible room/items.

### 5. Align feature rows

If any image failed to load in one feature branch, the script intersects the valid path sets so that every remaining row refers to the same image across all matrices.

### 6. Build hybrid embeddings

The pipeline fuses CLIP embeddings and visual features into one normalized hybrid vector.

Default weighting:

```text
semantic 0.45
layout   0.35
edge     0.15
color    0.05
```

### 7. First-stage clustering

HDBSCAN runs on CLIP embeddings only. This gives broad scene-level groups.

If no stable semantic clusters are found, the pipeline falls back to one-stage HDBSCAN on hybrid embeddings.

### 8. Second-stage refinement

Each semantic cluster is refined using one of two paths.

#### Default path

- cluster hybrid embeddings inside the semantic cluster
- if a cluster is large and broad, split it again using a viewpoint similarity matrix
- if a 4-image cluster looks like two clean pairs, split it into two pairs
- if the split was too aggressive, merge subclusters back by CLIP centroid similarity

#### Strict path

- optionally split 4-image clusters into two pairs first
- otherwise build a strict pairwise similarity matrix
- only allow links when viewpoint, item, and semantic thresholds all pass
- cluster with complete-link agglomerative clustering
- small groups become noise

### 9. Viewpoint similarity details

The viewpoint similarity matrix is built from:

```text
0.35 * layout similarity
+ 0.25 * edge similarity
+ 0.25 * opening profile similarity
+ 0.15 * ORB similarity
```

This is the main score used to judge whether two images are from the same corner or close viewpoints.

### 10. Match score reporting

The pipeline writes `match_scores.json` for inspection.

In strict mode, the reported match score is:

```text
0.50 * viewpoint
+ 0.30 * item
+ 0.20 * semantic
```

In default mode, the reported match score is:

```text
0.55 * hybrid
+ 0.30 * viewpoint
+ 0.15 * semantic
```

### 11. Output writing

The output directory is recreated, then:

- each cluster is copied into its own folder
- noise images are copied into `noise/`
- `clusters.json` and `match_scores.json` are written

## How To Read The Output

- `clusters.json`
  Final cluster membership only.
- `match_scores.json`
  Diagnostic view showing why images were or were not considered strong matches.

## Practical Interpretation

- high semantic score but low viewpoint score means same scene family, different corner
- high viewpoint and high item score usually means a near-duplicate or same-corner match
- noise often means the image is a unique viewpoint, a detail shot, or a threshold miss
