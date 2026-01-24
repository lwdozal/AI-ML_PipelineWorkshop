# Synthetic Data Generation Pipeline

This directory contains the complete pipeline for generating synthetic social movement images using Google Gemini API, integrated with three contextual data sources.

**Documentation/Configuration guide:** [DataCollection/config/](./DataCollection/config/)

## Scenario Generation
[01_scenario_generation.ipynb](https://github.com/lwdozal/AI-ML_PipelineWorkshop/blob/main/DataCollection/notebooks/01_scenario_generation.ipynb) - This notebook combines three contextual data sources, use them as is or customize them as you see fit.

#### Data Sources
1. **Atropia Data**: Fictional country news from [U.S. military training scenarios](https://odin.tradoc.army.mil/DATE/Caucasus/Atropia): Use the data references provided, or create your own visual references within the [dataoader.py](https://github.com/lwdozal/AI-ML_PipelineWorkshop/blob/main/DataCollection/src/data_loader.py) script within the class AtropiaDataLoader.
2. **World Bank Synthetic Demographics**: [Imaginary country demographic data](https://microdata.worldbank.org/index.php/catalog/5906/study-description)
3. **Customizable Visual References**: Use the social media data provided, or create your own visual references within the [dataoader.py](https://github.com/lwdozal/AI-ML_PipelineWorkshop/blob/main/DataCollection/src/data_loader.py) script.
     - `DataCollection/src/dataloader.py` > class SocialMediaLoader
       
**Note**: Original social movement data not shared due to Instagram privacy policy. This pipeline generates synthetic alternatives.

## Synthetic (Image) Data Generation
[02_generate_synth_images.ipynb](https://github.com/lwdozal/AI-ML_PipelineWorkshop/blob/main/DataCollection/notebooks/02_generate_synth_images.ipynb) - Here we can finially generate our synthetic data. Two models are provided for different types of processing power, GPU and CPU.

#### Syntheic Data Generation Process
- Load configuration and source data
- Build prompts from combined data sources
- Generate images in batches with checkpoints
- Save images and metadata
- Review results


## Image Feature Clustering
[03_cluster_img_embeds.ipynb](https://github.com/lwdozal/AI-ML_PipelineWorkshop/blob/main/DataCollection/notebooks/03_cluster_img_embeds.ipynb) - Creates image embeddings using CLIP, clusters, and visualizes them to understand the types of images created.

#### Syntheic Data Review Process
- Load configuration and image data
- Create image embeddings
- Cluster and Evaluate image Embeddings
- Visualize embedding clusters


## DIY Problem Solving - Comparing M-LLM Generated Descriptions
[04_compare_MLLMs.ipynb](https://github.com/lwdozal/AI-ML_PipelineWorkshop/blob/main/DataCollection/notebooks/04_compare_MLLMs.ipynb) - Unpolished and unfinished script that attempts to compare the semantic output of three image understanding/reasoning M-LLMs.

#### Models attempted (you can bring in and use your own)
- "Qwen/Qwen2.5-VL-3B-Instruct"
- "microsoft/phi-4-multimodal-instruct"
- "meta-llama/Llama-4-Scout-17B-16E"

***Note*** These models were chosen based on their image and multilingual capabilities. Smaller models might be ollama, llava, BLIP-2 or llama3B vision - Find one depending on your image data.




