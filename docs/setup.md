# Workshop Setup Guide

This guide will help you set up your environment for the AI/ML Pipeline Workshop on CyVerse Discovery Environment.

## Prerequisites

Before you begin, make sure you have:

- [x] Created a CyVerse account ([instructions here](cyverse_account.md))
- [x] Registered for the workshop at [https://user.cyverse.org/workshops/208](https://user.cyverse.org/workshops/208)
- [x] A Google Gemini API key (we'll cover how to get this)

## Step 1: Access CyVerse Discovery Environment

1. Navigate to [https://de.cyverse.org](https://de.cyverse.org)
2. Log in with your CyVerse credentials
3. You should see the Discovery Environment dashboard

## Step 2: Launch Jupyter Lab PyTorch GPU

This workshop requires GPU acceleration for model training and inference.

1. In the Discovery Environment, click **"Apps"** in the left sidebar
2. Look for the **"Instant Launches"** section
3. Click on **"Jupyter Lab PyTorch GPU"**
4. Wait for the environment to initialize (this may take 2-5 minutes)

The Jupyter Lab interface will open in a new browser tab once ready.

## Step 3: Clone the Workshop Repository

In Jupyter Lab:

### Option A: Using Terminal

1. Click the **Terminal** icon in the Launcher (or File > New > Terminal)
2. Navigate to your desired directory:
   ```bash
   cd /home/jovyan/data-store
   ```
3. Clone the repository:
   ```bash
   git clone https://github.com/lwdozal/AI-ML_PipelineWorkshop.git
   ```
4. Navigate into the repository:
   ```bash
   cd AI-ML_PipelineWorkshop
   ```

### Option B: Using Git Extension

1. Click the Git icon in the left sidebar
2. Click "Clone a Repository"
3. Enter the repository URL: `https://github.com/lwdozal/AI-ML_PipelineWorkshop.git`
4. Choose the destination folder: `/home/jovyan/data-store`
5. Click "Clone"

## Step 4: Create a Virtual Environment

Creating a virtual environment isolates your project dependencies:

```bash
# Navigate to the repository directory
cd /home/jovyan/data-store/AI-ML_PipelineWorkshop

# Create a virtual environment named 'workshop_env'
python -m venv workshop_env

# Activate the virtual environment
source workshop_env/bin/activate
```

You should see `(workshop_env)` in your terminal prompt, indicating the environment is active.

## Step 5: Install Required Packages

Install all necessary Python packages using pip:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Key Packages Installed

The `requirements.txt` file includes:

- **torch & torchvision**: PyTorch for deep learning
- **transformers**: Hugging Face transformers library
- **sentence-transformers**: For semantic similarity
- **google-generativeai**: Google Gemini API client
- **PIL (Pillow)**: Image processing
- **opencv-python**: Computer vision tasks
- **langchain**: LLM orchestration
- **pandas & numpy**: Data manipulation
- **matplotlib & seaborn**: Data visualization
- **networkx**: Network analysis
- **jupyter**: Notebook interface

## Step 6: Set Up Google Gemini API Key

You'll need a Google Gemini API key to generate synthetic images.

### Get Your API Key

1. Go to [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

### Configure Your API Key

Create a `.env` file in the repository root:

```bash
# In the terminal, navigate to repository root
cd /home/jovyan/data-store/AI-ML_PipelineWorkshop

# Create .env file
nano .env
```

Add your API key:

```
GOOGLE_API_KEY=your_api_key_here
```

Save and exit (Ctrl+X, then Y, then Enter).

**Security Note**: Never commit your `.env` file to Git. The repository includes a `.gitignore` file to prevent this.

### Verify Installation

Test your setup by running this Python code in a new notebook:

```python
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Test connection
model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("Hello, this is a test.")
print(response.text)
```

If this runs without errors, your setup is complete.

## Step 7: Verify GPU Access

Verify that PyTorch can access the GPU:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
```

Expected output should show `CUDA available: True` and list the GPU device.

## Directory Structure

After setup, your directory should look like this:

```
AI-ML_PipelineWorkshop/
├── DataCollection/          # Synthetic data generation notebooks
├── docs/                    # Documentation (this site)
├── workshop_env/            # Virtual environment (not tracked by git)
├── .env                     # API keys (not tracked by git)
├── .gitignore              # Git ignore rules
├── requirements.txt         # Python dependencies
├── README.md
├── AUTHORS.md
└── LICENSE
```

## Troubleshooting

### Virtual Environment Issues

If you can't activate the virtual environment:

```bash
# Make sure you're in the right directory
cd /home/jovyan/data-store/AI-ML_PipelineWorkshop

# Try creating it again
python3 -m venv workshop_env
source workshop_env/bin/activate
```

### Package Installation Errors

If pip install fails:

```bash
# Upgrade pip first
pip install --upgrade pip

# Try installing packages individually
pip install torch torchvision
pip install transformers sentence-transformers
pip install google-generativeai
```

### GPU Not Available

If CUDA is not available:

1. Verify you launched the **PyTorch GPU** version of Jupyter Lab
2. Restart the kernel: Kernel > Restart Kernel
3. Check CyVerse status page for GPU availability
4. Contact workshop support if issues persist

### API Key Issues

If Gemini API doesn't work:

1. Verify your API key is correct in the `.env` file
2. Check you've enabled the Gemini API in Google Cloud Console
3. Ensure you have API quota remaining
4. Test with a simple curl request:
   ```bash
   curl -H 'Content-Type: application/json' \
        -d '{"contents":[{"parts":[{"text":"Hello"}]}]}' \
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=YOUR_API_KEY"
   ```

## Additional Resources

- [CyVerse Learning Center](https://learning.cyverse.org)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [Google Gemini API Documentation](https://ai.google.dev/docs)

## Getting Help

- **During the workshop**: Ask instructors or teaching assistants
- **CyVerse support**: support@cyverse.org
- **Repository issues**: [GitHub Issues](https://github.com/lwdozal/AI-ML_PipelineWorkshop/issues)

---

Once your setup is complete, you're ready to start with [Data Collection](data_collection.md).
