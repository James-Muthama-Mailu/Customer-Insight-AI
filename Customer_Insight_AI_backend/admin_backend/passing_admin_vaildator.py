from Customer_Insight_AI_backend.admin_backend.admin_validator import admin_validator
from Customer_Insight_AI_backend.connection import client


# passing validator into clients
client.Customer_Insight_AI.command("collMod", "Admin", validator=admin_validator)
