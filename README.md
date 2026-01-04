# AI Workbench for Synthetic Data Generation, Generative M-LLM Comparison, and Network Building
*Funded by Jetstream2* \
**Still in Development** - *MLOPs and LLMOPs for Jetstream2 AI Worskhop* \
January 23, 2026

AI/MLOPs pipeline that runs synthetically generated image data through a Multimodal-LLM to generate labels and captions of each image. These labels and captions are semantically evaluated and then implemented into a network structure to understand different thematic representations and how they are grouped. 

The synthetically generated images follow the topic of a specific case study, a social movement. The process uses quantitative and Human-In-The-Loop evaluation to identify patterns within the network structure to summarize the overall narrative found within the collection of images. 

### Learning Objectives
1) Understanding of AI/MLOPs and pipeline creation (Using Open Source CARE principles)
2) Introduction to Synthetic Data Generation
3) Introduction to Multimodal-Large Language Models
4) Understanding of Centrality Measures
5) Introduction to Human in the Loop Evaluation
   
### Implementation

1. Download/clone the repository and save to your desired folder 
2. Create a new virtual environment
- add how to create a new environment
- add requirements doc

## [Data Collection and Evaluation](https://github.com/lwdozal/AI-ML_PipelineWorkshop/tree/main/DataCollection)
- Follow along to create image dataset using Google Gemini API
- OR Download the image dataset based on original data\
  --- Original data not shared because of Instagram privacy policy

## Data Exploration (Content Analysis)

## [Model Development](https://github.com/lwdozal/Dissertation_AI_Workbench/tree/main/Step1_Pattern_Detection)

### Image Label and Caption Generation
Training and fine-tuning on a subset of data to track performance, identify errors, and optimize models.\
Clean post comments i.e. lemmetize, translate emojies, rename hashtags, lowercase sentences, etc \
Multlingual sentence transformers
- Huggingface sentence transformer
- LaBASE (Language Agnostic BERT sentence encoder)

Ongoing monitoring of security and ethical risks 

### Generative AI Model Reivew
Evaluation Metrics: 
- Acuracy, Precision, Recall, F1-Score, Hamming Loss
- Confusion Matrix (Visuallization)
- BERTScore, Sentence Transformer Cosine Similarities

### Resources
Hugging Face Access, Access to LLM (I used VERDE), GPU Access

<!-- Torch, Torchvision, \
transformers, sentence transformers,  \
PIL, Requests, pydantic, open-cv, os \
langchain core and openai, \ -->


## Model Deployment

### Narrative Structure and Community Detection

Identify Semantic similarities by creating and evaluating structural graphs (content-based knowledge representation) 
- Viz_weights + generated label_weights
- Viz_weights & generated label_weights + generated captions
- Viz_weights & generated label_weights + generated captions & original post comments

<!-- 
Community Detection (Evaluation of Network Structure):
- Centraility Measures
- ERGM algorithm
- Leidan algorithm

Graph Analysis
- Multipartite, Bipartite
- Multiplex Graphs?
-->
