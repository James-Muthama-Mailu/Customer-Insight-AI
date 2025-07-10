from huggingface_hub import snapshot_download
import os

# Specify the model name and local directory
model_name = "intfloat/multilingual-e5-large-instruct"
local_dir = r"C:\Users\james\PycharmProjects\CustomerInsightAI\Customer-Insight-AI\models\embeddings_model"

# Ensure the target directory exists
os.makedirs(local_dir, exist_ok=True)

# Download the model
snapshot_download(repo_id=model_name, local_dir=local_dir)

print(f"Model downloaded to {local_dir}")