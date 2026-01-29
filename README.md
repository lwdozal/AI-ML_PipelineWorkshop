# AI Workbench for Synthetic Data Generation, Generative M-LLM Comparison, and Network Building
*Funded by Jetstream2* \
*MLOPs and LLMOPs for Jetstream2 AI Worskhop*, January 23, 2026 \
[DOI: 10.5281/zenodo.18407334
](https://doi.org/10.5281/zenodo.18407373)

This AI/MLOPs pipeline creates synthetically generated images creating a specific scenario for an open source text to image generation models. The images are based on a specific case study, a social movement. The process enables customizable scenarios to create the data. Then, the pipeline runs the synthetically generated image data through CLIP to create image feature embeddings for visualization and evaluation.

### Learning Objectives
1) Understanding of AI/MLOPs and pipeline creation (Using Open Source CARE principles)
2) Introduction to Synthetic Data Generation
3) Introduction to Multimodal-Large Language Models
4) Understanding of Centrality Measures
5) Introduction to Human in the Loop Evaluation
   
## Implementation

### Jan. 2026 - Workshop Setup

1. **Enroll in the Workshop** to access CyVerse Discovery Environment
   - Workshop enrollment: [CyVerse Workshop Registration](https://user.cyverse.org/workshops/208)
   - You will receive access credentials and instructions

2. **Launch Jupyter Lab PyTorch GPU**
   - Navigate to CyVerse Discovery Environment
   - Go to Instant Launches Section
   - Select "Jupyter Lab PyTorch GPU"
   - Wait for environment to initialize
  
### General Set-up 

3. **Clone the Repository**
   ```bash
   cd ~/data-store
   git clone https://github.com/lwdozal/AI-ML_PipelineWorkshop.git
   cd AI-ML_PipelineWorkshop
   ```

4. **Create Virtual Environment**
   ```bash
   # Using venv
   python -m venv venvname
   source venv/bin/activate
   ```
   ***Using conda (if available)***
   ```conda create -n aiml-workshop python=3.9
   conda activate aiml-workshop
   ```

5. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
6. **Navigate to Notebooks to get started!**

 `DataCollection/notebooks/`

   - 1) [Scenario Generation](https://github.com/lwdozal/AI-ML_PipelineWorkshop/blob/main/DataCollection/notebooks/01_scenario_generation.ipynb)
   - 2) [Synthetic (Image) Data Generation](https://github.com/lwdozal/AI-ML_PipelineWorkshop/blob/main/DataCollection/notebooks/02_generate_synth_images.ipynb)
   - 3) [Image Feature Clustering](https://github.com/lwdozal/AI-ML_PipelineWorkshop/blob/main/DataCollection/notebooks/03_cluster_img_embeds.ipynb)
   - 4) [DIY Problem Solving - Comparing M-LLM Generated Descriptions](https://github.com/lwdozal/AI-ML_PipelineWorkshop/blob/main/DataCollection/notebooks/04_compare_MLLMs.ipynb)






### Cite this work

APA:   Dozal, Laura W. (2026). lwdozal/AI-ML_PipelineWorkshop (Version v2). Zenodo. https://doi.org/10.5281/zenodo.18407373

IEEE:   [1]Laura W. Dozal, “lwdozal/AI-ML_PipelineWorkshop”. Zenodo, Jan. 28, 2026. doi: 10.5281/zenodo.18407373.


   
<!-- 6. **Configure API Keys Create .gitignore**
   ```bash
   cd DataCollection
   cp [config/.env.example] [config/.env] 
   # Edit config/.env and add your Google Gemini API key
   
   #create .gitigore file
   #mac, linux, git terminal
   touch .gitignore
   #windows
   echo > .gitignore

   # add DataCollection/config/.env to .gitignore to save your api keys -->

 <!-- 
 7. **Verify Setup**
   - Open Jupyter Lab
  - Run `01_setup_and_test.ipynb` to verify installation  -->




____________________________________________________________________________________
<!-- ## [Data Collection and Evaluation](https://github.com/lwdozal/AI-ML_PipelineWorkshop/tree/main/DataCollection) -->


<!-- Complete pipeline for generating synthetic social movement images with metadata. -->

<!-- ### What's Included -->

<!-- - **5 Jupyter Notebooks**: Step-by-step workflow from setup to QA
- **6 Python Modules**: Reusable components for data generation
- **Configuration System**: Customizable parameters via YAML
- **Rate Limiting**: Built-in API quota management
- **Checkpointing**: Resume interrupted generation runs
- **Quality Assurance**: Comprehensive validation and reporting -->

<!-- ### Quick Start

```bash
cd DataCollection/notebooks
jupyter lab
# Run notebooks in order: 01 → 02 → 03 → 04 → 05
``` -->

<!-- ### Output Dataset Structure

```
data/generated/
├── images/              # PNG images
├── captions/            # Descriptive captions (JSON)
├── labels/              # Semantic labels (JSON)
├── comments/            # Social media comments (JSON)
├── metadata/            # Full generation provenance (JSON)
├── all_captions.csv     # CSV export for analysis
├── all_labels.csv       # CSV export for analysis
└── all_comments.csv     # CSV export for analysis
``` -->

<!-- ### Cost Estimates

Based on Google Gemini API pricing (free tier available):
- 10 images: ~$0.01 USD
- 50 images: ~$0.06 USD
- 100 images: ~$0.12 USD
- 200 images: ~$0.24 USD

**Recommendation**: Start with 10-20 images for testing. -->


<!-- ## Data Exploration (Content Analysis)

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


<!-- ## Model Deployment

### Narrative Structure and Community Detection

Identify Semantic similarities by creating and evaluating structural graphs (content-based knowledge representation) 
- Viz_weights + generated label_weights
- Viz_weights & generated label_weights + generated captions
- Viz_weights & generated label_weights + generated captions & original post comments  -->


<!-- 
Community Detection (Evaluation of Network Structure):
- Centraility Measures
- ERGM algorithm
- Leidan algorithm

Graph Analysis
- Multipartite, Bipartite
- Multiplex Graphs?
-->
