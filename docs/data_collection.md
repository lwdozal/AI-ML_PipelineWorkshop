# Data Collection and Synthetic Data Generation

The first stage of our AI/ML pipeline focuses on generating synthetic image data using Google Gemini API. This approach allows us to create a realistic dataset while respecting privacy concerns.

## Overview

In this module, you'll learn how to:

- Generate synthetic images based on a fictional scenario
- Create descriptive prompts for image generation
- Evaluate synthetic data quality
- Prepare datasets for downstream ML tasks

## Why Synthetic Data?

This workshop uses synthetic data for several important reasons:

1. **Privacy Protection**: Original social movement data from Instagram cannot be shared due to privacy policies
2. **Ethical Research**: Synthetic data allows teaching without exposing real individuals
3. **Controlled Scenarios**: We can create specific scenarios for educational purposes
4. **Reproducibility**: Everyone generates similar datasets following the same process

## Data Sources

Our synthetic scenario combines three data sources:

### 1. Atropia Data
Fictional country news and events from U.S. military training exercises. Atropia is an imaginary country used for military simulations, providing realistic geopolitical scenarios.

### 2. World Bank Synthetic Data
Economic and demographic data for an imaginary country, providing context for social conditions and potential triggers for social movements.

### 3. Public Social Movement Images
Real public images from various social movements serve as visual references for the types of scenes, compositions, and elements to include in synthetic images.

## Workflow

### Step 1: Set Up Data Collection Environment

Navigate to the DataCollection folder:

```bash
cd /home/jovyan/data-store/AI-ML_PipelineWorkshop/DataCollection
```

Ensure your virtual environment is activated and Google Gemini API key is configured (see [Setup Guide](setup.md)).

### Step 2: Open the Generation Notebook

Open `Generate_images.ipynb` in Jupyter Lab:

1. In the file browser, navigate to `DataCollection/`
2. Double-click `Generate_images.ipynb`
3. Select your kernel (workshop_env)

### Step 3: Understand the Generation Process

The notebook guides you through:

#### A. Scenario Development
- Review the fictional Atropia scenario
- Understand the social movement context
- Identify key visual elements to generate

#### B. Prompt Engineering
Learn to craft effective prompts for image generation:

- **Specificity**: Detailed descriptions produce better results
- **Context**: Include setting, atmosphere, and mood
- **Visual elements**: Specify people, objects, actions, and composition
- **Style considerations**: Artistic style, lighting, perspective

Example prompt structure:
```
A [setting] with [people/objects] [performing action].
[Lighting/atmosphere]. [Composition details].
Style: [photorealistic/artistic style].
```

#### C. Image Generation
Use Google Gemini API to generate images:

```python
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Configure model
model = genai.GenerativeModel('gemini-pro-vision')

# Generate image
prompt = "Your detailed prompt here"
response = model.generate_content(prompt)

# Save image
with open('output_image.png', 'wb') as f:
    f.write(response.image)
```

#### D. Batch Generation
Generate multiple images with variation:

- Create prompt templates
- Add variation parameters
- Generate image sets
- Organize output files

### Step 4: Quality Assessment

Evaluate generated images using these criteria:

#### Visual Quality
- Resolution and clarity
- Realistic appearance
- Appropriate composition
- Consistent style

#### Content Relevance
- Matches scenario description
- Includes specified elements
- Appropriate context and setting
- Culturally appropriate

#### Dataset Balance
- Variety of scenes and scenarios
- Different perspectives and compositions
- Range of lighting conditions
- Diverse representation

### Step 5: Data Organization

Organize your generated dataset:

```
DataCollection/
├── generated_images/
│   ├── protest_scenes/
│   ├── gatherings/
│   ├── individual_portraits/
│   └── contextual_scenes/
├── metadata/
│   ├── image_prompts.json
│   ├── generation_parameters.json
│   └── quality_scores.csv
└── documentation/
    └── generation_log.md
```

## Evaluation Methods

### Spot Checking

Manually review a sample of generated images:

1. Select 10-15 random images
2. Evaluate against quality criteria
3. Identify common issues
4. Refine prompts if needed

### Automated Metrics

Use computational methods to assess quality:

```python
from PIL import Image
import numpy as np

def assess_image_quality(image_path):
    img = Image.open(image_path)

    # Resolution check
    width, height = img.size
    resolution_score = min(width * height / (1920 * 1080), 1.0)

    # Color distribution
    img_array = np.array(img)
    color_variance = np.std(img_array)

    # Brightness
    brightness = np.mean(img_array)

    return {
        'resolution_score': resolution_score,
        'color_variance': color_variance,
        'brightness': brightness
    }
```

### Human-in-the-Loop Evaluation

Involve human judgment for nuanced assessment:

- Rate image relevance (1-5 scale)
- Identify inappropriate content
- Assess cultural sensitivity
- Verify scenario alignment

## Best Practices

### Prompt Engineering Tips

1. **Be specific**: "A crowded urban plaza during sunset" vs "A place with people"
2. **Include context**: Add temporal, spatial, and emotional context
3. **Iterate**: Refine prompts based on results
4. **Document**: Keep track of successful prompts

### Data Management

1. **Version control**: Track prompt versions and parameters
2. **Metadata**: Record generation details for each image
3. **Quality logs**: Document evaluation results
4. **Backup**: Save generated data in multiple locations

### Ethical Considerations

1. **Avoid stereotypes**: Ensure diverse and respectful representations
2. **Cultural sensitivity**: Review images for cultural appropriateness
3. **Transparency**: Clearly label data as synthetic
4. **Purpose limitation**: Use data only for stated educational purposes

## Outputs

By the end of this module, you should have:

- [ ] A dataset of 50-100 synthetic images
- [ ] Metadata file with prompts and parameters
- [ ] Quality assessment scores
- [ ] Documentation of the generation process

## Next Steps

Once you've generated and evaluated your synthetic dataset:

1. Prepare data for model training
2. Proceed to [Model Development](model_development.md)
3. Begin generating labels and captions using M-LLMs

## Additional Resources

- [DataCollection README](../DataCollection/README.md)
- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Synthetic Data Best Practices](https://arxiv.org/abs/2205.01710)

## Troubleshooting

### API Rate Limits

If you hit rate limits:
- Add delays between requests (time.sleep(2))
- Batch requests appropriately
- Monitor your quota usage

### Poor Image Quality

If images don't meet quality standards:
- Refine prompts with more detail
- Adjust generation parameters
- Try different style specifications
- Review example prompts in the notebook

### Content Issues

If generated content is inappropriate:
- Add content filters to prompts
- Specify desired tone and atmosphere
- Review and reject problematic outputs
- Document issues for future prompt refinement

---

Questions? Check the [CyVerse Learning Center](https://learning.cyverse.org) or ask during the workshop.
