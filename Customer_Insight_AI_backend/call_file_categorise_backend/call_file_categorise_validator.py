call_file_categorise_validator = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["customer_id", "video_file_name", "category", "probability"],
        "properties": {
            "customer_id": {
                "bsonType": "objectId",
                "description": "Enter the customer's ID as an ObjectId"
            },
            "video_file_name": {
                "bsonType": "string",
                "description": "Enter a valid video file name"
            },
            "category": {
                "bsonType": "string",
                "description": "Enter a valid category"
            },
            "probability": {
                "bsonType": ["double", "int"],  # Accept both double and int
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Valid Probability between 0 and 1"
            }
        }
    }
}