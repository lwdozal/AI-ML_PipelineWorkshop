# AI Workbench: Visual Network Narrative Dissertation
MLOPs and LLMOPs for Jetstream2 AI Worskhop \
December 2025

### Implementation

1. Download/clone the repository and save to your desired folder 
2. Create a new virtual environment


## [Data Collection and Evaluation](https://github.com/lwdozal/Dissertation_AI_Workbench/tree/main/data_collection)
- Download image dataset created by Google Gemini based on original data\
  --- Original data not shared because of Instagram privacy policy

### Data Exploration (Clustering)

## [Model Development](https://github.com/lwdozal/Dissertation_AI_Workbench/tree/main/Step1_Pattern_Detection)

### Image Label and Caption Generation
Training and fine-tuning on a subset of data to track performance, identify errors, and optimize models.\
Clean post comments i.e. lemmetize, translate emojies, rename hashtags, lowercase sentences, etc \
Multlingual sentence transformers
- Huggingface sentence transformer
- LaBASE (Language Agnostic BERT sentence encoder)

Evaluation Metrics: 
- Acuracy, Precision, Recall, F1-Score, Hamming Loss
- Confusion Matrix (Visuallization)
- BERTScore, Sentence Transformer Cosine Similarities

Ongoing monitoring of security and ethical risks 

### Resources
Hugging Face Access, Access to LLM (I used VERDE), GPU Access

<!-- Torch, Torchvision, \
transformers, sentence transformers,  \
PIL, Requests, pydantic, open-cv, os \
langchain core and openai, \ -->



## Model Deployment

### Identify Semantic similarities

Create Structural graph (content-based knowledge representation) 
- Viz_weights + generated label_weights
- Viz_weights & generated label_weights + generated captions
- Viz_weights & generated label_weights + generated captions & original post comments

Community Detection (Evaluation of Network Structure):
- Centraility Measures
- ERGM algorithm
- Leidan algorithm

Graph Analysis
- Multipartite, Bipartite
- Multiplex Graphs?
