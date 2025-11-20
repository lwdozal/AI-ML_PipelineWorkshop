
# Synthetic Data Generation
\textbf{Scenario Generation}: The data used for this workshop is generated with the free version of Google Gemini via a student account. The contextual foundation of the images and comments generated are based on three existing forms of data. 
1. [Atropia data](https://odin.tradoc.army.mil/DATE/Caucasus/Atropia): News reports and stories a fictional country named atropia. This data was created for training purposes by the united states military.
2. [World Bank Synthetic Data](https://microdata.worldbank.org/index.php/catalog/5906/study-description) for an Imaginary Country: Data variables include imaginary individual and household demographic data.
3. Online Social Movement Image Posts: Images taken from public facing social media platforms to act as visual reference for creating social movement data.

### Implementation

1. Download/clone the repository and save to your desired folder 
2. Create a new virtual environment
3. Identify your API Key from your Google account with access to Gemini
4. Install required packages via pip install -r requirements.txt
5. Follow Jupyter Notebooks in the DataCollection folder to generate images and comments

## Data Evaluation
1. Spot check samples of generated images and comments for quality assurance
2. Evaluate image labels and descriptions of generated images based on initial contextual information
3. Create downloadable dataset for model development and downstream tasks.

