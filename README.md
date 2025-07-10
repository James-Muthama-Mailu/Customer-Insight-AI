<<<<<<< HEAD
# Customer Insight AI
**Automated Understanding of Customer Feedback Using AI**

## Project Overview
CustomerInsightAI is a project aimed at automating the understanding of customer feedback by converting customer care conversations into text. The system leverages artificial intelligence (AI) to analyze and categorize customer interactions based on predetermined categories set for the AI model. This tool is designed to enhance customer service operations by providing deeper insights into customer sentiments and concerns, helping businesses improve their customer care strategies effectively.

## Project Components

1. **Speech-to-Text Conversion**: The project includes functionality to convert speech from customer care conversations into text using OpenAI's Whisper model for speech recognition.
2. **AI-based Categorization**: The converted text is then processed by an AI model to understand and categorize the content based on predefined categories.
3. **Flask Web Application**: A Flask web application handles user input and allows users to interact with the system, including uploading audio files for processing and viewing categorized results.
4. **Backend with MongoDB**: The backend, supported by MongoDB, stores the categorized data obtained from the AI model, allowing for efficient retrieval and analysis of categorized customer care conversations.

## File Structure
The project structure is organized as follows:
```plaintext
├── .venv # Virtual environment
├── audio_to_text # Directory for audio to text conversion scripts
├── Customer_Insight_AI_backend
│ ├── admin_backend # Admin backend scripts
│ ├── call_categorise_backend # Call categorization backend scripts
│ ├── call_file_categorise_backend # File categorization backend scripts
│ ├── company_backend # Company backend scripts
│ ├── .env # Environment variables
│ ├── init.py # Initialization script for the backend
│ ├── connection.py # Database connection script
├── models
│ ├── categorisation_model # Model for categorization
│ ├── dataset # Dataset related scripts and files
│ ├── pickle # Pickle files for model serialization
│ ├── word2vec_model # Word2Vec model files
│ ├── init.py # Initialization script for models
├── static
│ ├── css # CSS files for styling
│ ├── images # Image files
├── templates # HTML templates for the web application
├── app.py # Main application script
├── init.py # Initialization script
├── README.md # Project documentation
├── requirements.txt # Python dependencies
```

## Usage
The primary goal of CustomerInsightAI is to streamline customer call analysis, providing valuable insights into customer sentiments and concerns. This can be particularly useful in the data visualisation of customer care calls.

## Dataset Used
The project utilizes a dataset from Kaggle, found [here](https://www.kaggle.com/datasets/bitext/training-dataset-for-chatbotsvirtual-assistants).

## Requirements
Below are the required Python libraries as specified in the `requirements.txt` file:

```plaintext
absl-py==2.1.0
aiohttp==3.9.5
aiohttp-retry==2.8.3
aiosignal==1.3.1
astunparse==1.6.3
attrs==23.2.0
blinker==1.8.2
bokeh==3.4.2
certifi==2024.2.2
charset-normalizer==3.3.2
click==8.1.7
colorama==0.4.6
contourpy==1.2.1
dnspython==2.6.1
et-xmlfile==1.1.0
ffmpeg-python==0.2.0
filelock==3.15.4
Flask==3.0.3
flatbuffers==24.3.25
frozenlist==1.4.1
fsspec==2024.6.1
future==1.0.0
gast==0.5.5
google-pasta==0.2.0
grpcio==1.64.1
h5py==3.11.0
idna==3.7
inexactsearch==1.0.2
intel-openmp==2021.4.0
itsdangerous==2.2.0
Jinja2==3.1.4
joblib==1.4.2
keras==3.4.0
libclang==18.1.1
llvmlite==0.43.0
Markdown==3.6
markdown-it-py==3.0.0
MarkupSafe==2.1.5
mdurl==0.1.2
mkl==2021.4.0
ml-dtypes==0.4.0
more-itertools==10.3.0
mpmath==1.3.0
multidict==6.0.5
namex==0.0.8
networkx==3.3
nltk==3.8.1
numba==0.60.0
numpy==1.26.4
openai-whisper==20231117
openpyxl==3.1.4
opt-einsum==3.3.0
optree==0.11.0
packaging==24.1
pandas==2.2.2
passlib==1.7.4
pillow==10.3.0
plotly==5.22.0
protobuf==4.25.3
Pygments==2.18.0
PyJWT==2.8.0
pymongo==4.7.2
pyotp==2.9.0
pyspellchecker==0.8.1
python-dateutil==2.9.0.post0
python-dotenv==1.0.1
pytz==2024.1
PyYAML==6.0.1
regex==2024.5.15
requests==2.32.2
rich==13.7.1
scikit-learn==1.5.0
scipy==1.14.0
setuptools==70.1.1
silpa_common==0.3
six==1.16.0
soundex==1.1.3
spellchecker==0.4
sympy==1.12.1
tbb==2021.13.0
tenacity==8.4.2
tensorboard==2.16.2
tensorboard-data-server==0.7.2
tensorflow==2.16.1
tensorflow-intel==2.16.1
termcolor==2.4.0
tflearn==0.5.0
threadpoolctl==3.5.0
tiktoken==0.7.0
torchvision==0.18.1
tornado==6.4.1
tqdm==4.66.4
typing_extensions==4.12.2
tzdata==2024.1
urllib3==2.2.1
Werkzeug==3.0.3
wheel==0.43.0
whisper==1.1.10
wrapt==1.16.0
xyzservices==2024.6.0
yarl==1.9.4
```

## How to Install the Project Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/James-Muthama/CustomerInsightAI.git
   cd CustomerInsightAI
   ```

2. Create and activate a virtual environment:
   On Windows:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   On macOS and Linux::
    ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the project:
     Ensure you have set up any necessary environment variables and configured the AI models as required. You can use the .env file to manage environment variables.
     ```bash
   python app.py
   ```

## License
This project is licensed under the MIT License. See the <a href="MIT_license.txt">License</a> file for details.

This project is licensed under the Apache License. See the <a href="Apache_license.txt">License</a> file for details.

This project is licensed under the BSD-3-Clause License. See the <a href="BSD_3-Clause_license.txt">License</a> file for details.



For any issues or contributions, feel free to reach out via the GitHub repository (https://github.com/James-Muthama/CustomerInsightAI/) or email (jamesmuthaiks@gmail.com)
=======
# Customer-Insight-AI
>>>>>>> 
