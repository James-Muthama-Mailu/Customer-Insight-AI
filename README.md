# Customer Insight AI
**Automated Understanding of Customer Feedback Using AI**

## Project Overview
CustomerInsightAI is a project aimed at automating the understanding of customer feedback by converting customer care conversations into text. The system leverages artificial intelligence (AI) to analyze and categorize customer interactions based on predetermined categories set for the AI model. This tool is designed to enhance customer service operations by providing deeper insights into customer sentiments and concerns, helping businesses improve their customer care strategies effectively.

## Project Components

1. **Speech-to-Text Conversion**: The project includes functionality to convert speech from customer care conversations into text using OpenAI's Whisper model for speech recognition.
2. **AI-based Categorization**: The converted text is then processed by an AI model to understand and categorize the content based on predefined categories.

## Usage
The primary goal of CustomerInsightAI is to streamline customer call analysis, providing valuable insights into customer sentiments and concerns. This can be particularly useful in the data visualisation of customer care calls.

## Dataset Used
The project utilizes a dataset from Kaggle, found [here](https://www.kaggle.com/datasets/bitext/training-dataset-for-chatbotsvirtual-assistants).

## Requirements
Below are the required Python libraries as specified in the `requirements.txt` file:

```plaintext
aiohttp==3.9.5
aiohttp-retry==2.8.3
aiosignal==1.3.1
attrs==23.2.0
blinker==1.8.2
certifi==2024.2.2
charset-normalizer==3.3.2
click==8.1.7
colorama==0.4.6
dnspython==2.6.1
Flask==3.0.3
frozenlist==1.4.1
idna==3.7
itsdangerous==2.2.0
Jinja2==3.1.4
MarkupSafe==2.1.5
multidict==6.0.5
passlib==1.7.4
PyJWT==2.8.0
pymongo==4.7.2
pyotp==2.9.0
python-dotenv==1.0.1
requests==2.32.2
twilio==9.1.1
urllib3==2.2.1
Werkzeug==3.0.3
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
This project is licensed under the MIT License. See the <a href="license.txt">License</a> file for details.

For any issues or contributions, feel free to reach out via the GitHub repository (https://github.com/James-Muthama/CustomerInsightAI/) or email (jamesmuthaiks@gmail.com)
