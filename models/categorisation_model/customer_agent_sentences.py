import pickle

# Import functions from external files
from models.word2vec_model.pre_processing_dataset import clean_text

# Sample sentences from customer care agents
agent_sentences = [
    "Good morning! Thank you for reaching out to us today. How can I assist you?",
    "I see that you're experiencing an issue with your account. Let's get that sorted out for you.",
    "Could you please confirm your contact information so we can update our records?",
    "I apologize for the inconvenience caused. Let's work together to find a solution.",
    "I'll need to gather some details from you to proceed with resolving this matter.",
    "Thank you for your patience while I review your account information.",
    "I appreciate you bringing this matter to our attention. Let's see how we can help.",
    "We aim to provide excellent service. Please let me know how I can assist you further.",
    "I'll document all the details of our conversation to ensure we address your concerns.",
    "Rest assured, we're committed to resolving this issue promptly.",
    "Hello, thank you for contacting us today. How can I make your experience better?",
    "I understand this issue is frustrating. Let's work together to find a solution.",
    "Could you please verify your account details so I can access your information securely?",
    "Let me check the status of your recent order for you.",
    "I'll investigate this issue further and provide you with an update shortly.",
    "I'm here to assist you with any questions or concerns you may have.",
    "Thank you for your patience. I'll do everything I can to resolve this quickly.",
    "To better assist you, could you please describe the problem in detail?",
    "I'll escalate your case to our technical team for immediate attention.",
    "We value your feedback and will use it to improve our service.",
    "I'll ensure that your request is handled with the utmost priority.",
    "Please hold while I access the necessary information to assist you.",
    "I'll review your account history to better understand the situation.",
    "Let's go through the steps together to troubleshoot this issue.",
    "I'll make a note of your preferences to personalize your experience.",
    "Thank you for bringing this matter to our attention. We'll investigate and follow up.",
    "Could you provide your email address so we can send you a confirmation?",
    "I'll update your account details as per your request.",
    "I appreciate your patience while I locate the information you need.",
    "Let me confirm the details of your previous interaction with us.",
    "I'll ensure that you receive a replacement for the damaged item.",
    "Please confirm your shipping address so we can proceed with your order.",
    "I'll send you an email summarizing our conversation for your records.",
    "Thank you for waiting. I'm reviewing the notes from our previous communication.",
    "I'll connect you with a specialist who can provide further assistance.",
    "Could you please verify your identity for security purposes?",
    "I'll investigate this issue and provide you with a resolution plan.",
    "Let's review your options to find the best solution for your situation.",
    "I'll monitor this situation closely and keep you updated on progress.",
    "I'll provide you with detailed instructions to troubleshoot the issue.",
    "Thank you for your understanding as we work to resolve this.",
    "I'll ensure that your feedback is shared with our product development team.",
    "Let me know if there's anything else I can assist you with today.",
    "I'll schedule a callback at your convenience to discuss this further.",
    "I'll make sure that your concerns are addressed by our management team.",
    "Please let me know if there's anything specific you'd like us to focus on.",
    "I'll check our system to see if there are any ongoing issues in your area.",
    "Thank you for notifying us. We'll take immediate action to rectify the situation.",
    "I'll verify the terms of your warranty to ensure you receive the appropriate service.",
    "Let me know if there's anything more I can do to assist you today.",
    "Thank you for contacting us. How can I make your experience with us better?",
    "I'll personally handle your request to ensure it's resolved quickly and efficiently.",
    "Could you please provide more details so I can assist you accurately?",
    "I'll ensure that your feedback is relayed to our product development team.",
    "Let me double-check your account information to prevent any further issues.",
    "I'll update our records with the information you've provided.",
    "Please let me know if there's anything specific you'd like me to prioritize.",
    "I'll make a note of your preferences to tailor our service to your needs.",
    "Thank you for your loyalty. We appreciate your continued business.",
    "I'll follow up with you personally to ensure everything has been resolved to your satisfaction."
]

# Preprocess each customer agent sentence
customer_agent_sentences = []
for sentence in agent_sentences:
    processed_sentence = clean_text(sentence)
    customer_agent_sentences.append(processed_sentence)

with open('../pickle/customer_agent_sentences.pkl', 'wb') as f:
    pickle.dump(customer_agent_sentences, f)
