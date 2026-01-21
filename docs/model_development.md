# Model Development

In this module, you'll develop and evaluate Multimodal Large Language Models (M-LLMs) to generate labels and captions for synthetic images. This stage combines computer vision and natural language processing.

## Overview

This module covers:

- Using M-LLMs for image understanding
- Generating labels and captions
- Text preprocessing and normalization
- Model evaluation metrics
- Performance optimization

## What are Multimodal-LLMs?

Multimodal Large Language Models can process and understand multiple types of data:

- **Visual input**: Images, videos, diagrams
- **Text input**: Prompts, questions, descriptions
- **Combined output**: Text descriptions, classifications, captions

Examples: GPT-4 Vision, Google Gemini Pro Vision, LLaVA, CLIP

## Workflow

### Step 1: Prepare Your Data

Ensure you have:
- Generated synthetic images from [Data Collection](data_collection.md)
- Organized images in structured directories
- Created metadata files
- Split data into training/validation/test sets

### Step 2: Image Label Generation

#### Understanding Labels

Labels categorize images into predefined classes:
- Scene type (protest, gathering, march)
- Emotion (peaceful, tense, celebratory)
- Density (crowded, sparse, intimate)
- Time (day, night, dawn/dusk)

#### Using M-LLMs for Labeling

```python
import google.generativeai as genai
from PIL import Image
import os

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro-vision')

def generate_labels(image_path):
    # Load image
    img = Image.open(image_path)

    # Create prompt
    prompt = """
    Analyze this image and provide labels for:
    1. Scene type (e.g., protest, gathering, march)
    2. Emotional tone (e.g., peaceful, tense, celebratory)
    3. Crowd density (e.g., sparse, moderate, dense)
    4. Time of day (e.g., day, night, dusk)

    Provide only the labels, separated by commas.
    """

    # Generate labels
    response = model.generate_content([prompt, img])
    labels = response.text.strip().split(',')

    return [label.strip() for label in labels]

# Example usage
image_labels = generate_labels('path/to/image.jpg')
print(image_labels)
```

### Step 3: Caption Generation

#### Understanding Captions

Captions provide descriptive text about image content:
- Who: People and their actions
- What: Objects and events
- Where: Setting and location
- When: Time and temporal context
- How: Manner and atmosphere

#### Generating Descriptive Captions

```python
def generate_caption(image_path):
    img = Image.open(image_path)

    prompt = """
    Write a detailed caption for this image describing:
    - The scene and setting
    - People and their actions
    - The overall atmosphere
    - Any notable objects or elements

    Keep the caption concise (2-3 sentences) and objective.
    """

    response = model.generate_content([prompt, img])
    return response.text.strip()

# Example usage
caption = generate_caption('path/to/image.jpg')
print(caption)
```

### Step 4: Text Processing

Clean and normalize generated text for consistency.

#### Lemmatization

Convert words to their base form:

```python
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
nltk.download('wordnet')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    tokens = word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized)

# Example
text = "People are gathering and protesting"
lemmatized = lemmatize_text(text)  # "people be gather and protest"
```

#### Emoji Translation

Convert emojis to text descriptions:

```python
import emoji

def translate_emojis(text):
    return emoji.demojize(text, delimiters=(" ", " "))

# Example
text_with_emoji = "Great protest today! 🎉"
translated = translate_emojis(text_with_emoji)  # "Great protest today!  party_popper "
```

#### Hashtag Normalization

Process hashtags for analysis:

```python
import re

def normalize_hashtags(text):
    # Split camelCase hashtags
    def split_camel(hashtag):
        return re.sub('([a-z])([A-Z])', r'\1 \2', hashtag)

    # Find and process hashtags
    hashtags = re.findall(r'#\w+', text)
    for hashtag in hashtags:
        normalized = split_camel(hashtag[1:]).lower()  # Remove # and split
        text = text.replace(hashtag, normalized)

    return text

# Example
text = "Join the #ClimateStrike #SaveOurPlanet"
normalized = normalize_hashtags(text)  # "Join the climate strike save our planet"
```

#### Complete Text Pipeline

```python
def preprocess_text(text):
    # Translate emojis
    text = translate_emojis(text)

    # Normalize hashtags
    text = normalize_hashtags(text)

    # Lemmatize
    text = lemmatize_text(text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text
```

### Step 5: Multilingual Support

Use sentence transformers for multilingual text understanding:

#### Hugging Face Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

# Load multilingual model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Encode sentences
sentences = [
    "A peaceful protest in the city square",
    "Una protesta pacífica en la plaza de la ciudad",
    "Une manifestation pacifique sur la place de la ville"
]

embeddings = model.encode(sentences)

# Calculate similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(embeddings)
print(similarity_matrix)
```

#### LaBSE (Language-agnostic BERT Sentence Encoder)

```python
import tensorflow_hub as hub
import numpy as np

# Load LaBSE model
labse_model = hub.load('https://tfhub.dev/google/LaBSE/2')

# Encode multilingual sentences
sentences = [
    "People gathering for a cause",
    "Gente reunida por una causa"
]

embeddings = labse_model(sentences)

# Compute similarity
similarity = np.inner(embeddings[0], embeddings[1])
print(f"Similarity: {similarity:.4f}")
```

## Model Evaluation

### Classification Metrics

For label prediction tasks:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_classification(y_true, y_pred):
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'hamming_loss': hamming_loss(y_true, y_pred)
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Visualize
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return metrics

# Example usage
# y_true = true_labels
# y_pred = predicted_labels
# metrics = evaluate_classification(y_true, y_pred)
```

### Semantic Similarity Metrics

For caption generation tasks:

#### BERTScore

```python
from bert_score import score

def calculate_bertscore(candidates, references):
    P, R, F1 = score(candidates, references, lang='en', verbose=True)

    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }

# Example
candidates = ["A large crowd protesting in the streets"]
references = ["Many people demonstrating on the street"]
bert_metrics = calculate_bertscore(candidates, references)
```

#### Sentence Transformer Cosine Similarity

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_semantic_similarity(text1, text2):
    embeddings = model.encode([text1, text2])
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return similarity.item()

# Example
generated = "A peaceful protest in the city"
reference = "A calm demonstration in town"
similarity = calculate_semantic_similarity(generated, reference)
print(f"Similarity: {similarity:.4f}")
```

## Model Training and Fine-Tuning

### Data Preparation

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageCaptionDataset(Dataset):
    def __init__(self, image_paths, captions, transform=None):
        self.image_paths = image_paths
        self.captions = captions
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)

        caption = self.captions[idx]
        return image, caption

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create dataset and dataloader
dataset = ImageCaptionDataset(image_paths, captions, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Tracking Performance

```python
import wandb  # Weights & Biases for experiment tracking

# Initialize tracking
wandb.init(project="ml-pipeline-workshop")

# Log metrics during training
for epoch in range(num_epochs):
    train_loss = train_epoch(model, dataloader)
    val_metrics = evaluate(model, val_dataloader)

    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_accuracy': val_metrics['accuracy'],
        'val_f1': val_metrics['f1_score']
    })
```

## Required Resources

- **GPU Access**: Jetstream2 allocation through CyVerse
- **LLM Access**: Google Gemini API or VERDE platform
- **Hugging Face Account**: For model downloads
- **Storage**: Sufficient space for models and data

## Best Practices

### Model Selection
1. Start with pre-trained models
2. Fine-tune on your specific domain
3. Compare multiple models
4. Consider computational constraints

### Prompt Engineering
1. Clear and specific instructions
2. Include examples (few-shot prompting)
3. Iterate based on outputs
4. Document successful prompts

### Evaluation Strategy
1. Use multiple metrics
2. Include human evaluation
3. Test on diverse examples
4. Monitor for biases

## Outputs

By the end of this module, you should have:

- [ ] Generated labels for all images
- [ ] Generated captions for all images
- [ ] Preprocessed and normalized text data
- [ ] Evaluation metrics and visualizations
- [ ] Model performance documentation

## Next Steps

Once model development is complete:

1. Analyze model outputs for patterns
2. Proceed to [Model Deployment](model_deployment.md)
3. Build semantic networks from generated data

## Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [BERTScore Paper](https://arxiv.org/abs/1904.09675)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

## Troubleshooting

### Memory Issues
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training
- Clear GPU cache regularly

### Poor Model Performance
- Increase training data
- Adjust learning rate
- Try different model architectures
- Review data quality

### API Rate Limits
- Implement exponential backoff
- Cache responses
- Batch requests efficiently
- Monitor quota usage

---

Questions? Check the [CyVerse Learning Center](https://learning.cyverse.org) or ask during the workshop.
