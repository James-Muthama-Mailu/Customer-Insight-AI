import sys
import os

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)


from call_file_categorise_validator import call_file_categorise_validator
from Customer_Insight_AI_backend.connection import client


# passing validator into clients
client.Customer_Insight_AI.command("collMod", "Call_Files", validator=call_file_categorise_validator)
