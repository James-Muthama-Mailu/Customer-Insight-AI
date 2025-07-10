from huggingface_hub import snapshot_download

# Specify the model name and local directory
model_name = "intfloat/multilingual-e5-large-instruct"
local_dir = r"C:\Users\james\PycharmProjects\CustomerInsightAI\Customer-Insight-AI\models\word2vec_model"

# Download the model
snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)

print(f"Model downloaded to {local_dir}")