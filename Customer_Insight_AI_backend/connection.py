from dotenv import load_dotenv, find_dotenv
import os
from pymongo import MongoClient
from urllib.parse import quote_plus

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Retrieve password from environment variables
password = os.environ.get("MONGO_PWD")

# URL encode the password
encoded_password = quote_plus(password)

# Form the connection string with SSL enabled
connection_string = f"mongodb+srv://jamesmuthaiks:{encoded_password}@customer-insight-ai.cge8ruu.mongodb.net/?retryWrites=true&w=majority&appName=Customer-Insight-AI"

# Create a MongoClient
client = MongoClient(connection_string)

# Attempt to list databases
try:
    dbs = client.list_database_names()
    print("Databases:", dbs)
except Exception as e:
    print("Error listing databases:", e)

# Attempt to list collections in SmartNanny database
try:
    collections = client.Customer_Insight_AI.list_collection_names()
    print("Collections:", collections)
except Exception as e:
    print("Error listing collections:", e)
