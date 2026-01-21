# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI/MLOps pipeline workshop focused on synthetic data generation, multimodal large language model (M-LLM) evaluation, and network analysis. The project generates synthetic images of a social movement using Google Gemini API, then uses M-LLMs to generate labels and captions, followed by semantic evaluation and network structure analysis.

**Status**: Still in development for a workshop scheduled January 23, 2026.
**Funded by**: Jetstream2
**Platform**: Designed to run on CyVerse Discovery Environment using Jupyter Lab PyTorch GPU.

## Environment Setup

The project is designed to run in Jupyter Lab on CyVerse Discovery Environment:

1. Enroll in the workshop to access CyVerse Discovery Environment
2. Open Jupyter Lab PyTorch GPU from the Instant Launches Section
3. Clone this repository
4. Create a virtual environment (details to be added - see README.md:24-25)
5. Install requirements via `pip install -r requirements.txt` (requirements file to be created)

**Key API Requirement**: Google Gemini API key needed for synthetic image generation.

## Repository Structure

```
.
├── DataCollection/          # Synthetic image generation using Google Gemini
│   ├── Generate_images.ipynb
│   └── README.md
├── docs/                    # Documentation and management plans
│   ├── assets/             # Images and graphic assets
│   ├── Data_Management_Plan.md
│   ├── Governance_Operations.md
│   └── index.md
├── mkdocs.yml              # MkDocs configuration for documentation site
├── AUTHORS.md
├── LICENSE                 # GNU GPL v3
└── README.md
```

## Workflow Architecture

This pipeline follows three main stages:

### 1. Data Collection and Evaluation
**Location**: `DataCollection/`

Synthetic scenario generation using Google Gemini API based on three data sources:
- Atropia data (fictional country news from U.S. military training)
- World Bank Synthetic Data for an imaginary country
- Public social movement images as visual references

**Implementation**: Follow notebooks in `DataCollection/` folder to generate images and comments.

**Evaluation**:
- Spot check samples for quality assurance
- Evaluate image labels and descriptions based on contextual information
- Create downloadable dataset for downstream tasks

### 2. Model Development
**Planned location**: To be organized (see README.md:40)

**Image Label and Caption Generation**:
- Training and fine-tuning on data subset to track performance and optimize models
- Text processing: lemmatization, emoji translation, hashtag normalization, lowercasing
- Multilingual sentence transformers (Huggingface, LaBASE)

**Evaluation Metrics**:
- Classification metrics: Accuracy, Precision, Recall, F1-Score, Hamming Loss
- Confusion Matrix visualization
- Semantic similarity: BERTScore, Sentence Transformer Cosine Similarities

**Resources Required**:
- Hugging Face access
- LLM access (project uses VERDE)
- GPU access
- Libraries: torch, torchvision, transformers, sentence-transformers, PIL, requests, pydantic, opencv, langchain

### 3. Model Deployment
**Network Analysis**: Identify semantic similarities through structural graphs

**Analysis types**:
- Viz_weights + generated label_weights
- Viz_weights & generated label_weights + generated captions
- Viz_weights & generated label_weights + generated captions & original post comments

**Community Detection** (planned):
- Centrality measures
- ERGM algorithm
- Leiden algorithm

**Graph types**: Multipartite, Bipartite, potentially Multiplex graphs

## Documentation Site

The repository uses MkDocs with Material theme for documentation:

**Build and serve locally**:
```bash
mkdocs serve
```

**Build static site**:
```bash
mkdocs build
```

The navigation structure is defined in `mkdocs.yml` and includes integration with Jupyter notebooks via the `mkdocs-jupyter` plugin.

## Learning Objectives

This workshop teaches:
1. Understanding of AI/MLOps and pipeline creation (using Open Source CARE principles)
2. Introduction to Synthetic Data Generation
3. Introduction to Multimodal-Large Language Models
4. Understanding of Centrality Measures
5. Introduction to Human in the Loop Evaluation

## Important Context

- **License**: GNU GPL v3
- **Author**: Laura Dozal, PhD Candidate, College of Information, University of Arizona
- **Support**: Jetstream2, CyVerse, The Data Science Institute at the University of Arizona
- **Workshop Type**: MLOps and LLMOps educational workshop
- **Data Ethics**: Original social movement data not shared due to Instagram privacy policy; synthetic data used instead
- **FOSS Principles**: Project follows FAIR and CARE principles for research tools
