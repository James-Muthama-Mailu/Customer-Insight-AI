intent_call_categorise_validator = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["customer_id"],
        "properties": {
            "customer_id": {
                "bsonType": "objectId",
                "description": "Enter the customer's ID as an ObjectId"
            },
            "track_refund": {
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
            "review": {
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
            "cancel_order": {
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
            "switch_account": {
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
            },
            "edit_account": {
                "bsonType": ["object", "null"],
                "description": "Enter the edit_account value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for edit_account"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "contact_customer_service": {
                "bsonType": ["object", "null"],
                "description": "Enter the contact_customer_service value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for contact_customer_service"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "place_order": {
                "bsonType": ["object", "null"],
                "description": "Enter the place_order value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for place_order"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "check_payment_methods": {
                "bsonType": ["object", "null"],
                "description": "Enter the check_payment_methods value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for check_payment_methods"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "payment_issue": {
                "bsonType": ["object", "null"],
                "description": "Enter the payment_issue value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for payment_issue"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "intent": {
                "bsonType": ["object", "null"],
                "description": "Enter the intent value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for intent"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "contact_human_agent": {
                "bsonType": ["object", "null"],
                "description": "Enter the contact_human_agent value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for contact_human_agent"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "complaint": {
                "bsonType": ["object", "null"],
                "description": "Enter the complaint value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for complaint"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "recover_password": {
                "bsonType": ["object", "null"],
                "description": "Enter the recover_password value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for recover_password"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "delivery_period": {
                "bsonType": ["object", "null"],
                "description": "Enter the delivery_period value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for delivery_period"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "check_invoices": {
                "bsonType": ["object", "null"],
                "description": "Enter the check_invoices value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for check_invoices"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "track_order": {
                "bsonType": ["object", "null"],
                "description": "Enter the track_order value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for track_order"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "delete_account": {
                "bsonType": ["object", "null"],
                "description": "Enter the delete_account value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for delete_account"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "get_invoice": {
                "bsonType": ["object", "null"],
                "description": "Enter the get_invoice value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for get_invoice"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "change_order": {
                "bsonType": ["object", "null"],
                "description": "Enter the change_order value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for change_order"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "delivery_options": {
                "bsonType": ["object", "null"],
                "description": "Enter the delivery_options value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for delivery_options"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "check_invoice": {
                "bsonType": ["object", "null"],
                "description": "Enter the delivery_options value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for delivery_options"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "create_account": {
                "bsonType": ["object", "null"],
                "description": "Enter the delivery_options value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for delivery_options"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "set_up_shipping_address": {
                "bsonType": ["object", "null"],
                "description": "Enter the delivery_options value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for delivery_options"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "check_refund_policy": {
                "bsonType": ["object", "null"],
                "description": "Enter the delivery_options value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for delivery_options"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "newsletter_subscription":{
                "bsonType": ["object", "null"],
                "description": "Enter the delivery_options value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for delivery_options"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "get_refund":{
                "bsonType": ["object", "null"],
                "description": "Enter the delivery_options value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for delivery_options"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "check_cancellation_fee": {
                "bsonType": ["object", "null"],
                "description": "Enter the delivery_options value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for delivery_options"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "registration_problems": {
                "bsonType": ["object", "null"],
                "description": "Enter the delivery_options value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for delivery_options"
                    },
                    "dates": {
                        "bsonType": "array",
                        "description": "Array of dates corresponding to each value"
                    }
                }
            },
            "change_shipping_address": {
                "bsonType": ["object", "null"],
                "description": "Enter the delivery_options value as an object or null",
                "properties": {
                    "values": {
                        "bsonType": "array",
                        "description": "Array of integer values for delivery_options"
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