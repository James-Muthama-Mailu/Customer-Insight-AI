import sys
import os

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)


from admin_validator import admin_validator
from Customer_Insight_AI_backend.connection import client


# passing validator into clients
client.Customer_Insight_AI.command("collMod", "Admin", validator=admin_validator)
