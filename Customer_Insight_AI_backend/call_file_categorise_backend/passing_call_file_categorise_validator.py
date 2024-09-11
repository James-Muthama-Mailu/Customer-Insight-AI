from Customer_Insight_AI_backend.call_file_categorise_backend.call_file_categorise_validator import call_file_categorise_validator
from Customer_Insight_AI_backend.connection import client


# passing validator into clients
client.Customer_Insight_AI.command("collMod", "Call_Files", validator=call_file_categorise_validator)
