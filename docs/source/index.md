# SPATIA

**Multimodal Model for Prediction and Generation of Spatial Cell Phenotypes**

[![Website](https://img.shields.io/badge/Website-SPATIA-4CAF50?logo=googlechrome&logoColor=white)](https://zitniklab.hms.harvard.edu/SPATIA/)
[![Paper](https://img.shields.io/badge/Paper-arXiv%202507.04704-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2507.04704)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-mims--harvard-FFD21E)](https://huggingface.co/mims-harvard)
[![Docs](https://img.shields.io/badge/Docs-readthedocs-blue?logo=readthedocs&logoColor=white)](https://mims-harvard.readthedocs.io/en/latest/)
[![Lab](https://img.shields.io/badge/Lab-Zitnik%20Lab%20%40%20Harvard-crimson)](https://zitniklab.hms.harvard.edu/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

---

Understanding how cellular morphology, gene expression, and spatial organization jointly shape tissue function is a central challenge in biology. **SPATIA** is a multi-scale model for spatial transcriptomics that:

- Learns cell-level embeddings by fusing image-derived morphological tokens and transcriptomic tokens via cross-attention
- Aggregates embeddings at niche and tissue levels with transformer modules to capture spatial context
- Generates cell morphology images conditioned on predicted state transitions using flow matching

```{image} ../../static/images/overview_view.png
:width: 600px
:align: center
```

---

## Pipeline Overview

SPATIA is organized as a three-stage pipeline:

| Stage | Component | Purpose |
|-------|-----------|---------|
| **Stage 1** | `gene_encoders/` | Representation learning (scPRINT or scGPT backbone) |
| **Stage 2** | `generative_tasks/data_pairing_for_FM/` | OT-based perturbation pair construction |
| **Stage 3** | `generative_tasks/spatia_flow/` | Flow-matching cell image generation |

Downstream evaluation (clustering, annotation, biomarker prediction) lives in `prediction_tasks/`.

---

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
```

```{toctree}
:maxdepth: 2
:caption: Data & Training

dataset
pretraining
```

```{toctree}
:maxdepth: 2
:caption: Downstream Tasks

prediction
generation
```

```{toctree}
:maxdepth: 1
:caption: Reference

citation
```
