emotion_categorise_validator = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["customer_id"],
        "properties": {
            "customer_id": {
                "bsonType": "objectId",
                "description": "Enter the customer's ID as an ObjectId"
            },
            "neutral": {
                "bsonType": ["object", "null"],
                "description": "Enter the track_refund value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for track_refund"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "happy": {
                "bsonType": ["object", "null"],
                "description": "Enter the review value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for review"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "angry": {
                "bsonType": ["object", "null"],
                "description": "Enter the cancel_order value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for cancel_order"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "sad": {
                "bsonType": ["object", "null"],
                "description": "Enter the switch_account value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for switch_account"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            }
        }
    }
}