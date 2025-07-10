import sys
import os

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from Customer_Insight_AI_backend.connection import client
from company_validator import customer_validator

# Passing validator into clients
client.Customer_Insight_AI.command("collMod", "Customer", validator=customer_validator)