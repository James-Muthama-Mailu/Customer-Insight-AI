import sys
import os

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from emotion_categorise_validator import emotion_categorise_validator
from Customer_Insight_AI_backend.connection import client


# passing validator into clients
client.Customer_Insight_AI.command("collMod", "Emotion_Call_Categorises", validator=emotion_categorise_validator)
