# Synthetic Data Generation Pipeline

This directory contains the complete pipeline for generating synthetic social movement images using Google Gemini API, integrated with three contextual data sources.

## Overview

The synthetic data generation pipeline creates realistic social movement images with associated metadata (captions, labels, comments) for AI/MLOps workshop training. The generated data combines:

1. **Atropia Data**: Fictional country news from [U.S. military training scenarios](https://odin.tradoc.army.mil/DATE/Caucasus/Atropia)
2. **World Bank Synthetic Demographics**: [Imaginary country demographic data](https://microdata.worldbank.org/index.php/catalog/5906/study-description)
3. **Social Movement Visual References**: Public social media imagery as visual guides

## Quick Start

### Prerequisites

- Python 3.8 or higher
- CyVerse Jupyter Lab PyTorch GPU environment (recommended)
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   cd /your/desired/location
   git clone https://github.com/your-repo/AI-ML_PipelineWorkshop.git
   cd AI-ML_PipelineWorkshop
   ```

2. **Create and activate virtual environment**
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # OR using conda
   conda create -n aiml-workshop python=3.9
   conda activate aiml-workshop
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API key**
   ```bash
   cd DataCollection
   cp config/.env.example config/.env
   # Edit config/.env and add your Google Gemini API key
   ```

5. **Run setup test**
   ```bash
   jupyter lab
   # Open and run notebooks/01_setup_and_test.ipynb
   ```

## Pipeline Architecture

```
DataCollection/
├── config/
│   ├── .env.example              # API key template
│   └── generation_config.yaml    # Generation parameters
├── src/
│   ├── config.py                 # Configuration management
│   ├── gemini_client.py          # API client with rate limiting
│   ├── data_loader.py            # Source data fetching
│   ├── prompt_builder.py         # Prompt construction
│   ├── output_handler.py         # Output management
│   └── validation.py             # Quality assurance
├── notebooks/
│   ├── 01_setup_and_test.ipynb          # Environment validation
│   ├── 02_prepare_source_data.ipynb     # Data preparation
│   ├── 03_generate_images.ipynb         # Image generation
│   ├── 04_generate_metadata.ipynb       # Metadata generation
│   └── 05_quality_assurance.ipynb       # QA workflow
├── data/
│   ├── raw/                      # Source data
│   ├── generated/                # Output images and metadata
│   └── qa/                       # QA reports
└── README.md                     # This file
```

## Workflow

### Step 1: Environment Setup and Testing

**Notebook**: `01_setup_and_test.ipynb`

- Validates Python packages
- Tests API authentication
- Generates test image
- Estimates costs for different batch sizes

**Run time**: 5-10 minutes

### Step 2: Prepare Source Data

**Notebook**: `02_prepare_source_data.ipynb`

- Fetches Atropia news samples (100 samples)
- Generates World Bank demographic profiles (50 profiles)
- Loads social movement visual references
- Demonstrates data combination

**Run time**: 2-5 minutes

### Step 3: Generate Synthetic Images

**Notebook**: `03_generate_images.ipynb`

- Builds prompts from combined data sources
- Generates images using Gemini API
- Implements rate limiting and retry logic
- Saves images with checkpoints for resume capability

**Configuration** (in `config/generation_config.yaml`):
- `num_images`: Number of images to generate (default: 50)
- `batch_size`: Images per batch (default: 10)
- `resolution`: "1K" or "4K" (default: "1K")
- `model`: Gemini model (default: "gemini-2.5-flash-image")

**Run time**: 10-60 minutes depending on image count

**Cost estimate**: ~$0.001 per image (1K resolution)

### Step 4: Generate Metadata

**Notebook**: `04_generate_metadata.ipynb`

- Generates descriptive captions for each image
- Creates semantic and categorical labels
- Generates social media-style comments
- Exports CSV summaries for analysis

**Run time**: 10-40 minutes depending on image count

### Step 5: Quality Assurance

**Notebook**: `05_quality_assurance.ipynb`

- Validates image integrity
- Checks metadata completeness
- Detects potential duplicates
- Analyzes label distribution
- Provides human review interface
- Generates comprehensive QA report

**Run time**: 5-15 minutes

## Configuration

### Adjusting Generation Parameters

Edit `config/generation_config.yaml` to customize:

```yaml
generation:
  num_images: 50              # How many images to generate
  batch_size: 10              # Batch size for checkpointing
  resolution: "1K"            # "1K" or "4K"
  model: "gemini-2.5-flash-image"

prompts:
  style: "realistic"          # realistic, artistic, documentary
  complexity: "medium"        # simple, medium, complex
  include_temporal_context: true
  include_demographics: true

metadata:
  num_comments_per_image: 5
  include_hashtags: true
  include_emojis: true

rate_limiting:
  requests_per_minute: 10     # Conservative for free tier
  requests_per_day: 1000
```

### API Rate Limits

Free tier limits (adjust in config if you have paid tier):
- 10 requests per minute (default setting)
- 1000 requests per day (default setting)

The pipeline includes automatic rate limiting and exponential backoff.

## Output Structure

Generated data is organized in `data/generated/`:

```
data/generated/
├── images/
│   ├── sm_20260121_0001.png
│   ├── sm_20260121_0002.png
│   └── ...
├── captions/
│   ├── sm_20260121_0001_caption.json
│   └── ...
├── labels/
│   ├── sm_20260121_0001_labels.json
│   └── ...
├── comments/
│   ├── sm_20260121_0001_comments.json
│   └── ...
├── metadata/
│   ├── sm_20260121_0001_metadata.json
│   └── ...
├── all_captions.csv          # CSV export of all captions
├── all_labels.csv            # CSV export of all labels
├── all_comments.csv          # CSV export of all comments
└── generation_log.json       # Complete generation log
```

### Metadata Format

Each image has associated JSON metadata with full provenance:

```json
{
  "image_id": "sm_20260121_0001",
  "generated_at": "2026-01-21T10:30:00",
  "prompt": "Full generation prompt...",
  "source_data": {
    "atropia": {"theme": "protest", "location": "..."},
    "demographics": {"age_group": "25-34", "occupation": "..."},
    "visual_reference": {"description": "..."}
  }
}
```

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
pip install -r requirements.txt
```

**2. API Key Error**
- Ensure `.env` file exists in `config/` directory
- Verify `GOOGLE_API_KEY` is set correctly
- Check API key has Gemini access enabled

**3. Rate Limit Exceeded**
- Reduce `requests_per_minute` in config
- Use smaller batch sizes
- Wait for rate limit reset

**4. Out of Memory**
- Reduce `batch_size`
- Close other applications
- Restart Jupyter kernel

**5. Checkpoint Resume Not Working**
- Check `data/checkpoints/generation_checkpoint.json` exists
- Set `RESUME = True` in notebook cell
- Verify checkpoint is not corrupted

### Getting Help

- Check CLAUDE.md for detailed project context
- Review notebook markdown cells for instructions
- Contact workshop instructors
- CyVerse support: https://cyverse.org/support

## Data Evaluation and Quality Assurance

The pipeline includes comprehensive QA:

1. **Automated Validation**
   - Image integrity checks
   - Metadata completeness verification
   - Duplicate detection using perceptual hashing
   - Label distribution analysis

2. **Human Review**
   - Random sampling interface
   - Side-by-side prompt/image comparison
   - Caption and label review
   - Comment quality assessment

3. **Statistical Analysis**
   - Label distribution by category
   - Caption length statistics
   - Word frequency analysis
   - Bias detection

4. **QA Reports**
   - Comprehensive validation report (JSON)
   - Quality assessment score
   - Recommendations for improvements
   - Saved in `data/qa/` directory

## Cost Estimation

Based on Gemini API pricing (as of January 2026):

| Images | Resolution | Image Gen | Metadata | Total (USD) |
|--------|-----------|-----------|----------|-------------|
| 10     | 1K        | $0.01     | $0.00    | ~$0.01      |
| 50     | 1K        | $0.05     | $0.01    | ~$0.06      |
| 100    | 1K        | $0.10     | $0.02    | ~$0.12      |
| 200    | 1K        | $0.20     | $0.04    | ~$0.24      |

**Recommendation**: Start with 10-20 images for testing, then scale up.

## Best Practices

1. **Start Small**: Test with 10 images before generating full dataset
2. **Review Costs**: Check cost estimates in notebook 01
3. **Use Checkpoints**: Enable checkpointing for long generation runs
4. **Monitor API Quota**: Track usage in Google Cloud Console
5. **Run QA**: Always run quality assurance before using data
6. **Document Changes**: Note any configuration changes in generation logs

## Next Steps

After completing this pipeline:

1. **Model Development**: Use generated dataset for training (see `../docs/model_development.md`)
2. **Semantic Analysis**: Evaluate label quality and semantic similarity
3. **Network Analysis**: Build structural graphs from metadata
4. **Dataset Packaging**: Prepare for sharing or publication

## License

GNU GPL v3 - See LICENSE file

## Citation

If you use this pipeline or generated data, please cite:

```
Laura Dozal. (2026). AI/MLOps Pipeline Workshop - Synthetic Data Generation.
College of Information, University of Arizona.
```

## Support

- Jetstream2: Computational resources
- CyVerse: Infrastructure and platform
- The Data Science Institute, University of Arizona

## Workshop Information

**Workshop**: AI/ML Pipeline - Synthetic Data Generation
**Date**: January 23, 2026
**Platform**: CyVerse Discovery Environment - Jupyter Lab PyTorch GPU
**Focus**: MLOps, LLMOps, FAIR and CARE principles
