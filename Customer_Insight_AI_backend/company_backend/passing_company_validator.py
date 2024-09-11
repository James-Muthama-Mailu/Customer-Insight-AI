from Customer_Insight_AI_backend.company_backend.company_validator import customer_validator
from Customer_Insight_AI_backend.connection import client


# passing validator into clients
client.Customer_Insight_AI.command("collMod", "Customer", validator=customer_validator)
