from Customer_Insight_AI_backend.call_categorise_backend.call_categorise_validator import call_categorise_validator
from Customer_Insight_AI_backend.connection import client


# passing validator into clients
client.Customer_Insight_AI.command("collMod", "Call_Categorises", validator=call_categorise_validator)
