
import smtplib
import uuid
from collections import defaultdict
from datetime import timedelta
from email.mime.text import MIMEText
from functools import wraps
from io import BytesIO
import os
from pathlib import Path

from flask import flash, redirect, request, url_for, session, render_template
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId
import datetime
from nltk import sent_tokenize
import numpy as np
import plotly.graph_objs as go
import pyotp
import requests
from bson import Binary
from dotenv import load_dotenv, find_dotenv
from flask import Flask, send_file
from passlib.hash import pbkdf2_sha256

from Customer_Insight_AI_backend.connection import client
from audio_to_text.audio_to_text import audio_transcription
from audio_to_text.audio_to_wav import mp3_to_wav
from models.intent_categorisation_model.making_prediction import classify_sentences
from models.emotion_categorisation_model.emotion_prediction import get_emotion_prediction

# Define the allowed extensions for images
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

UPLOAD_FOLDER = 'C:/Users/james/PycharmProjects/CustomerInsightAI/Customer-Insight-AI'
ALLOWED_EXTENSIONS = {'mp3'}

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Declaration of collections
company_collection = client.Customer_Insight_AI.Customer
admin_collection = client.Customer_Insight_AI.Admin
intent_call_categorise = client.Customer_Insight_AI.Intent_Call_Categorises
call_files = client.Customer_Insight_AI.Call_Files
emotion_call_categories = client.Customer_Insight_AI.Emotion_Call_Categorises

# Email configuration
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")

# reCAPTCHA keys
RECAPTCHA_SITE_KEY = os.environ.get("RECAPTCHA_SITE_KEY")
RECAPTCHA_SECRET_KEY = os.environ.get("RECAPTCHA_SECRET_KEY")

# Define allowed IP address
allowed_ip = os.environ.get("allowed_ip")


def send_email(to_address, subject, body):
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to_address

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, to_address, msg.as_string())

        return True  # Email sent successfully
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False  # Failed to send email


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def get_summed_emotion_data(field_name):
    """
    Retrieves and aggregates data for a specific emotion field from a MongoDB collection,
    summing values by day and formatting dates as 'Month Day'.

    Args:
        field_name (str): The name of the emotion field to query (e.g., 'happy').

    Returns:
        tuple: (values, formatted_dates)
            - values: List of summed values for each day.
            - formatted_dates: List of dates formatted as 'Month Day' (e.g., 'June 25').
    """
    # Dictionary to accumulate sums for each day
    daily_sums = defaultdict(int)

    # Query the MongoDB collection
    cursor = emotion_call_categories.find({}, {field_name + '.values': 1, field_name + '.dates': 1, '_id': 0})

    for doc in cursor:
        if field_name in doc:
            field_data = doc[field_name]
            if field_data and 'values' in field_data and 'dates' in field_data:
                for value, date in zip(field_data['values'], field_data['dates']):
                    # Truncate to day (remove time component)
                    day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
                    daily_sums[day_start] += value

    if not daily_sums:
        return [], []

    # Sort by date and prepare output
    sorted_days = sorted(daily_sums.items())
    dates, values = zip(*sorted_days)

    # Format dates as 'Month Day'
    formatted_dates = [dt.strftime("%B %d") for dt in dates]

    return list(values), list(formatted_dates)

def get_summed_intent_data(field_name):
    """
    Retrieves and aggregates data for a specific emotion field from a MongoDB collection,
    summing values by day and formatting dates as 'Month Day'.

    Args:
        field_name (str): The name of the emotion field to query (e.g., 'happy').

    Returns:
        tuple: (values, formatted_dates)
            - values: List of summed values for each day.
            - formatted_dates: List of dates formatted as 'Month Day' (e.g., 'June 25').
    """
    # Dictionary to accumulate sums for each day
    daily_sums = defaultdict(int)

    # Query the MongoDB collection
    cursor = intent_call_categorise.find({}, {field_name + '.values': 1, field_name + '.dates': 1, '_id': 0})

    for doc in cursor:
        if field_name in doc:
            field_data = doc[field_name]
            if field_data and 'values' in field_data and 'dates' in field_data:
                for value, date in zip(field_data['values'], field_data['dates']):
                    # Truncate to day (remove time component)
                    day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
                    daily_sums[day_start] += value

    if not daily_sums:
        return [], []

    # Sort by date and prepare output
    sorted_days = sorted(daily_sums.items())
    dates, values = zip(*sorted_days)

    # Format dates as 'Month Day'
    formatted_dates = [dt.strftime("%B %d") for dt in dates]

    return list(values), list(formatted_dates)


def get_specific_emotion_field_data(field_name):
    """
    Retrieves and aggregates data for a specific field from a MongoDB collection,
    grouping values into 30-minute intervals.

    Args:
        field_name (str): The name of the field to query (e.g., 'check_cancellation_fee').

    Returns:
        tuple: Two NumPy arrays:
            - interval_values_array: Aggregated values for each 30-minute interval.
            - interval_dates_array: Corresponding start times of intervals, formatted as strings.
    """
    # Initialize empty lists to store values and dates (though not used directly in final output)
    values = []
    dates = []

    # Query the MongoDB collection 'call_categorise' to retrieve values and dates for the specified field
    # Excludes '_id' and projects only the field's 'values' and 'dates' subfields
    cursor = emotion_call_categories.find({}, {field_name + '.values': 1, field_name + '.dates': 1, '_id': 0})

    # Create a defaultdict to accumulate sums of values for each 30-minute interval
    interval_sums = defaultdict(int)
    # List to store interval start times (though not used directly in final output)
    interval_dates = []

    # Iterate through each document in the query result
    for doc in cursor:
        # Check if the specified field exists in the document
        if field_name in doc:
            # Extract the field data (contains 'values' and 'dates')
            field_data = doc[field_name]
            # Validate that field_data exists and contains both 'values' and 'dates'
            if field_data and 'values' in field_data and 'dates' in field_data:
                # Iterate through paired values and dates using zip
                for value, date in zip(field_data['values'], field_data['dates']):
                    # Calculate the start of the 30-minute interval for the given date
                    # Rounds down to the nearest 30-minute boundary by removing minutes, seconds, and microseconds
                    interval_start = date - timedelta(minutes=date.minute % 30, seconds=date.second,
                                                      microseconds=date.microsecond)
                    # Calculate the end of the 30-minute interval (not used but computed for clarity)
                    interval_end = interval_start + timedelta(minutes=30)
                    # Add the value to the sum for the corresponding interval start time
                    interval_sums[interval_start] += value

    # Sort the interval sums by date (keys) to ensure chronological order
    sorted_intervals = sorted(interval_sums.items())
    # Unzip the sorted intervals into separate lists of dates and values
    interval_dates, interval_values = zip(*sorted_intervals)

    # Format each datetime object as a string in 'YYYY-MM-DD HH:MM:SS' format
    interval_dates = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in interval_dates]

    # Convert the values and dates to NumPy arrays for efficient processing and compatibility
    interval_values_array = np.array(interval_values)
    interval_dates_array = np.array(interval_dates)

    # Return the arrays of aggregated values and formatted dates
    return interval_values_array, interval_dates_array

def get_specific_field_data(field_name):
    """
    Retrieves and aggregates data for a specific field from a MongoDB collection,
    grouping values into 30-minute intervals.

    Args:
        field_name (str): The name of the field to query (e.g., 'check_cancellation_fee').

    Returns:
        tuple: Two NumPy arrays:
            - interval_values_array: Aggregated values for each 30-minute interval.
            - interval_dates_array: Corresponding start times of intervals, formatted as strings.
    """
    # Initialize empty lists to store values and dates (though not used directly in final output)
    values = []
    dates = []

    # Query the MongoDB collection 'call_categorise' to retrieve values and dates for the specified field
    # Excludes '_id' and projects only the field's 'values' and 'dates' subfields
    cursor = intent_call_categorise.find({}, {field_name + '.values': 1, field_name + '.dates': 1, '_id': 0})

    # Create a defaultdict to accumulate sums of values for each 30-minute interval
    interval_sums = defaultdict(int)
    # List to store interval start times (though not used directly in final output)
    interval_dates = []

    # Iterate through each document in the query result
    for doc in cursor:
        # Check if the specified field exists in the document
        if field_name in doc:
            # Extract the field data (contains 'values' and 'dates')
            field_data = doc[field_name]
            # Validate that field_data exists and contains both 'values' and 'dates'
            if field_data and 'values' in field_data and 'dates' in field_data:
                # Iterate through paired values and dates using zip
                for value, date in zip(field_data['values'], field_data['dates']):
                    # Calculate the start of the 30-minute interval for the given date
                    # Rounds down to the nearest 30-minute boundary by removing minutes, seconds, and microseconds
                    interval_start = date - timedelta(minutes=date.minute % 30, seconds=date.second,
                                                      microseconds=date.microsecond)
                    # Calculate the end of the 30-minute interval (not used but computed for clarity)
                    interval_end = interval_start + timedelta(minutes=30)
                    # Add the value to the sum for the corresponding interval start time
                    interval_sums[interval_start] += value

    # Sort the interval sums by date (keys) to ensure chronological order
    sorted_intervals = sorted(interval_sums.items())
    # Unzip the sorted intervals into separate lists of dates and values
    interval_dates, interval_values = zip(*sorted_intervals)

    # Format each datetime object as a string in 'YYYY-MM-DD HH:MM:SS' format
    interval_dates = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in interval_dates]

    # Convert the values and dates to NumPy arrays for efficient processing and compatibility
    interval_values_array = np.array(interval_values)
    interval_dates_array = np.array(interval_dates)

    # Return the arrays of aggregated values and formatted dates
    return interval_values_array, interval_dates_array


# Create Flask app 'template_folder' specifies the folder where the HTML templates are stored. 'static_folder'
# specifies the folder where static files (CSS, JS, images) are stored.'static_url_path' sets the URL path that
# serves the static files.
app = Flask(__name__, template_folder="templates", static_folder='static', static_url_path='/')

#session secret key
app.secret_key = os.environ.get("app.secret_key")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


#Decorators for checking logged in to access homepage
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            return redirect('/')

    return wrap


# Route for the home page
@app.route('/')
def home():
    session.clear()
    return render_template('index.html')


# Route for the signup page
@app.route('/signup/')
def signup():
    return render_template('signup.html', RECAPTCHA_SITE_KEY=RECAPTCHA_SITE_KEY)


# Route for the homepage after successful signup or login
@app.route('/homepage/')
@login_required
def homepage():
    email = session['email']
    user = company_collection.find_one({"email": email})
    session['user'] = user['name']

    return render_template('homepage.html')


# Route for displaying login page
@app.route('/login')
def login():
    return render_template('login.html', RECAPTCHA_SITE_KEY=RECAPTCHA_SITE_KEY)


# Route for handling user login
@app.route('/user/login', methods=['POST'])
def user_login():
    if request.method == 'POST':
        # Verify reCAPTCHA
        recaptcha_response = request.form.get('g-recaptcha-response')
        data = {
            'secret': RECAPTCHA_SECRET_KEY,
            'response': recaptcha_response
        }
        response = requests.post('https://www.google.com/recaptcha/api/siteverify', data=data)
        result = response.json()

        if not result.get('success'):
            flash('Invalid reCAPTCHA. Please try again.', 'error')
            return redirect(url_for('login'))

        email = request.form.get('email')
        password = request.form.get('password')

        user = company_collection.find_one({"email": email})

        if user and pbkdf2_sha256.verify(password, user['password']):
            # Generate a valid base32 secret key for TOTP
            totp_secret = pyotp.random_base32()

            totp = pyotp.TOTP(totp_secret)
            otp = totp.now()

            session['totp_secret'] = totp_secret
            session['verify'] = True
            session['email'] = email

            send_email(email, 'Log In Verification Code', f'Your Verification Code is {otp}')

            return redirect(url_for('two_factor_authentication_login'))
        else:
            flash('Invalid email or password.', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')


# Route for handling the user signup form submission
@app.route('/user/signup/', methods=['GET', 'POST'])
def user_signup():
    if request.method == 'POST':
        # Verify reCAPTCHA
        recaptcha_response = request.form.get('g-recaptcha-response')
        data = {
            'secret': RECAPTCHA_SECRET_KEY,
            'response': recaptcha_response
        }
        response = requests.post('https://www.google.com/recaptcha/api/siteverify', data=data)
        result = response.json()

        if not result.get('success'):
            flash('Invalid reCAPTCHA. Please try again.', 'error')
            return redirect(url_for('user_signup'))

        try:
            user = {
                "_id": uuid.uuid4().hex[:24],
                "name": request.form.get('name'),
                "email": request.form.get('email'),
                "phone_no": request.form.get('phone_no'),
                "password": request.form.get('password'),
            }

            user['password'] = pbkdf2_sha256.hash(user['password'])

            if company_collection.find_one({"$or": [{"email": user['email']}, {"name": user['name']}]}):
                flash("Information filled is already in use.", "error")
                return redirect(url_for('user_signup'))
            else:
                company_collection.insert_one(user)
                flash("User signed up successfully. Please log in.", "success")
                return redirect(url_for('login'))
        except Exception as e:
            flash(f"Error occurred: {str(e)}", "error")
            return redirect(url_for('user_signup'))

    return render_template('signup.html', RECAPTCHA_SITE_KEY=RECAPTCHA_SITE_KEY)


@app.route('/forgot_password', methods=['GET', 'POST'])
# function that takes user emil and sends otp code to email
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = company_collection.find_one({"email": email})

        if user:
            # Generate a valid base32 secret key for TOTP
            totp_secret = pyotp.random_base32()

            totp = pyotp.TOTP(totp_secret)
            otp = totp.now()
            send_email(user['email'], 'Change Password Verification Code', f'Your Verification Code is {otp}')

            # Store the TOTP secret in session
            session['reset_email'] = email
            session['totp_secret'] = totp_secret

            return redirect(url_for('verify_otp'))
        else:
            flash('Email not found.', 'error')
            return render_template('forgot_pass.html')
    elif request.method == 'GET':
        return render_template('forgot_pass.html')

    return render_template('forgot_pass.html')


@app.route('/verify_otp', methods=['GET', 'POST'])
# function that takes in user OTP code and verifies the OTP code
def verify_otp():
    if 'reset_email' not in session:
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        otp = request.form.get('otp')
        email = session['reset_email']
        totp_secret = session['totp_secret']
        user = company_collection.find_one({"email": email})

        if user:
            totp = pyotp.TOTP(totp_secret)
            if totp.verify(otp, valid_window=1):
                session.pop('totp_secret', None)
                return redirect(url_for('change_password'))
            else:
                flash('Invalid OTP.', 'error')

    return render_template('verify_otp.html')


@app.route('/change_password', methods=['GET', 'POST'])
# function that runs once the OTP code is valid to allow user to change to
def change_password():
    if 'reset_email' not in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        email = session['reset_email']

        if new_password != confirm_password:
            flash('Passwords do not match.', 'error')
        else:
            hashed_password = pbkdf2_sha256.hash(new_password)
            company_collection.update_one({'email': email}, {'$set': {'password': hashed_password}})
            flash('Password reset successful. Please log in with your new password.', 'success')
            session.pop('reset_email', None)
            return redirect(url_for('login'))

    return render_template('change_password.html')


# Checking 2-factor authentication code
@app.route('/verify_2_fa_login', methods=['GET', 'POST'])
def two_factor_authentication_login():
    if 'verify' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        otp = request.form.get('otp')
        totp_secret = session.get('totp_secret')
        email = session.get('email')  # Change session['user'] to session['email']

        if not totp_secret or not email:
            flash('Session expired or invalid. Please log in again.', 'error')
            return redirect(url_for('login'))

        totp = pyotp.TOTP(totp_secret)
        if totp.verify(otp, valid_window=1):
            # Authentication successful, log the user in
            session['logged_in'] = True
            session['user'] = session['email']  # Update session['user'] to user['email']

            user = company_collection.find_one({"email": email})
            session['user_id'] = user['_id']

            # Generate current time as a datetime object
            current_time = datetime.datetime.now()
            customer_id = ObjectId(user['_id'])  # Convert to ObjectId

            # Insert into MongoDB
            customer_login = {
                "_id": uuid.uuid4().hex,
                "customer_id": customer_id,
                "login_date": current_time,
                "logout_date": None,
            }

            try:
                admin_collection.insert_one(customer_login)
                session.pop('verify', None)
                session.pop('totp_secret', None)
                return redirect(url_for('homepage'))
            except Exception as e:
                flash(f'Error inserting login record: {str(e)}', 'error')

        else:
            flash('Invalid OTP. Please try again.', 'error')

    return render_template('2_fa.html')


#admin route
@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'GET':
        if request.remote_addr != allowed_ip:
            return redirect(url_for('login'))
        else:
            # Generate a valid base32 secret key for TOTP
            totp_secret = pyotp.random_base32()

            totp = pyotp.TOTP(totp_secret)
            otp = totp.now()

            session['totp_secret'] = totp_secret
            session['admin_verify'] = True

            send_email('james.muthama@strathmore.edu', 'Log In Verification Code', f'Your Verification Code is {otp}')
            return redirect(url_for('verify_admin'))


@app.route('/verify_admin', methods=['GET', 'POST'])
# function that takes in user OTP code and verifies the OTP code
def verify_admin():
    if 'admin_verify' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        otp = request.form.get('otp')
        totp_secret = session['totp_secret']

        totp = pyotp.TOTP(totp_secret)
        if totp.verify(otp, valid_window=1):
            session.pop('admin_verify', None)
            session.pop('totp_secret', None)
            customers = list(company_collection.find({}))  # Fetch all documents in the collection

            admins = list(admin_collection.find({}))  # Fetch all documents in the collection

            return render_template('admin.html', admins=admins, customers=customers)
        else:
            flash('Invalid OTP.', 'error')

    return render_template('verify_admin.html')


#logging out the user
@app.route('/sign/out')
def sign_out():
    session.clear()
    return redirect('/')


#logging out the user
@app.route('/log/out')
def logout():
    if 'user_id' in session:
        email = session['email']  # Change session['user_id'] to session['email']

        # Update the logout_date for the admin document with customer_id equal to session['user_id']
        current_time = datetime.datetime.now()
        admin_collection.update_one(
            {'customer_id': ObjectId(session['user_id'])},
            # Change to session['user_id'] to ObjectId(session['user_id'])
            {'$set': {'logout_date': current_time}}
        )

        # Clear session
        session.clear()

    return redirect(url_for('home'))  # Redirect to your desired endpoint after logout


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        try:
            # Check if file is present in the request
            if 'file' not in request.files:
                flash('No file part', 'error')
                return redirect(request.url)

            uploaded_file = request.files['file']

            # Check if a file was actually selected
            if uploaded_file.filename == '':
                flash('No selected file', 'error')
                return redirect(request.url)

            customer_id = ObjectId(session['user_id'])

            # Validate file type and process if valid
            if uploaded_file and allowed_file(uploaded_file.filename):
                # Save the uploaded MP3 file securely
                filename = secure_filename(uploaded_file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                uploaded_file.save(file_path)



                # Print emotions found
                print("\n" + "=" * 50)
                print("Performing emotion classification...")
                print("=" * 50)

                # Convert forward slashes to proper path format
                normalized_file_path = str(Path(file_path).resolve())

                # Use the backward compatible function (minimal changes to your code)
                wav_path = mp3_to_wav(normalized_file_path)

                # Continue with emotion prediction
                emotion_results = get_emotion_prediction(wav_path)

                print("\n" + "=" * 50)
                print("MOST FREQUENT EMOTION")
                print("\n" + "=" * 50)
                print(f"Most frequent emotion: {emotion_results['most_frequent_emotion']}")
                print(f"Emotion counts: {dict(emotion_results['emotion_counts'])}")
                print(f"Emotion percentage: {dict(emotion_results['emotion_percentage'])}")

                # Get current timestamp for this prediction
                current_date = datetime.datetime.now()

                # Prepare MongoDB update query to add new data point
                # This pushes a new value (1) and corresponding date to the emotion's arrays
                emotion_category = emotion_results['most_frequent_emotion']
                update_query = {
                    "$push": {
                        f"{emotion_category}.values": 1,  # Increment count for this emotion
                        f"{emotion_category}.dates": current_date  # Record when this prediction was made
                    }
                }

                # Update the emotion_call_categories collection
                # Uses upsert=True to create document if it doesn't exist
                emotion_call_categories.update_one(
                    {"customer_id": customer_id},
                    update_query,
                    upsert=True
                )

                # Also store individual call record in call_files collection
                call_files.insert_one({
                    "customer_id": customer_id,
                    "video_file_name": filename,
                    "category": emotion_category,  # Emotions category
                    "probability": float(emotion_results['emotion_percentage'][emotion_category]) / 100.0  # Convert percentage to probability (0-1),  # Emotion probability
                })

                # Print emotions found
                print("\n" + "=" * 50)
                print("Performing intent classification...")
                print("=" * 50)

                print("Starting transcription...")

                # Perform audio transcription
                customer_care_call = audio_transcription(file_path)

                print("\nTranscription complete!")
                print("-" * 50)
                print(customer_care_call)
                print("-" * 50)

                # Split transcription into sentences for classification
                customer_care_call_sentences = sent_tokenize(customer_care_call)

                # Perform text classification on each sentence
                categories, descriptions, class_probabilities = classify_sentences(customer_care_call_sentences)

                # FIXED: Handle different probability data types safely
                top_probabilities = []
                for i, probs in enumerate(class_probabilities):
                    try:
                        # Handle NumPy arrays (most common source of the error)
                        if hasattr(probs, 'max') and hasattr(probs, 'shape'):
                            # This is likely a NumPy array
                            if probs.shape == ():
                                # Scalar NumPy array (0-dimensional)
                                top_probabilities.append(float(probs.item()))
                            else:
                                # Multi-dimensional NumPy array
                                top_probabilities.append(float(probs.max()))
                        # Handle regular Python lists or tuples
                        elif isinstance(probs, (list, tuple)):
                            if len(probs) > 0:
                                top_probabilities.append(max(probs))
                            else:
                                top_probabilities.append(0.0)
                        # Handle single scalar values (int, float)
                        elif isinstance(probs, (int, float)):
                            top_probabilities.append(float(probs))
                        # Handle other iterable types
                        elif hasattr(probs, '__iter__'):
                            prob_list = list(probs)
                            if len(prob_list) > 0:
                                top_probabilities.append(max(prob_list))
                            else:
                                top_probabilities.append(0.0)
                        else:
                            # Fallback: try to convert to float
                            top_probabilities.append(float(probs))
                    except Exception as prob_error:
                        # If all else fails, use 0.0 as default probability
                        print(f"Warning: Could not process probability at index {i}: {prob_error}")
                        top_probabilities.append(0.0)

                # Combine categories with their corresponding probabilities
                results = list(zip(categories, top_probabilities))

                # Sort results by probability in descending order (highest confidence first)
                sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

                # Extract the top three predictions
                top_three = sorted_results[:3]

                # Display the top 3 predictions with their confidence scores
                print("\n" + "=" * 50)
                print("\nTop 3 Predicted Categories and Probabilities (Highest to Lowest):")
                print("\n" + "=" * 50)

                for category, probability in top_three:
                    print(f"Category: {category}, Probability: {probability:.4f}")

                # Store results in MongoDB if we have valid predictions
                if categories and top_probabilities:

                    # Process each category-probability pair
                    for category, probability in zip(categories, top_probabilities):
                        # Get current timestamp for this prediction
                        current_date = datetime.datetime.now()

                        # Prepare MongoDB update query to add new data point
                        # This pushes a new value (1) and corresponding date to the category's arrays
                        update_query = {
                            "$push": {
                                f"{category}.values": 1,  # Increment count for this category
                                f"{category}.dates": current_date  # Record when this prediction was made
                            }
                        }

                        # Update the call_categorise collection
                        # Uses upsert=True to create document if it doesn't exist
                        intent_call_categorise.update_one(
                            {"customer_id": customer_id},
                            update_query,
                            upsert=True
                        )

                        # Also store individual call record in call_files collection
                        call_files.insert_one({
                            "customer_id": customer_id,
                            "video_file_name": filename,
                            "category": category,
                            "probability": float(probability),  # Ensure it's stored as Python float
                        })

                    flash('File successfully uploaded and categorized', 'success')
                else:
                    flash('No Category Found', 'error')

            else:
                flash('Allowed file types are mp3 only', 'error')
                return redirect(request.url)

        except Exception as e:
            # Enhanced error handling with debug information
            error_message = f"Error occurred: {str(e)}"
            flash(error_message, 'error')

            # Print detailed error information for debugging
            print(f"Upload Error: {error_message}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")

            return redirect(request.url)

        # Redirect back to upload page after successful processing
        return redirect(url_for('upload_file'))

    elif request.method == 'GET':
        # Handle GET request to display upload page with previous uploads
        customer_id = ObjectId(session['user_id'])

        # Fetch all previous call records for this customer
        customer_data = call_files.find({"customer_id": customer_id})

        # Prepare data for template display
        customers = []
        for data in customer_data:
            customers.append({
                'video_file_name': data.get('video_file_name', 'N/A'),
                'category': data.get('category', 'No category found'),
                'probability': data.get('probability', 'N/A')
            })

        # Render the upload template with previous upload history
        return render_template('upload.html', customers=customers)


@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')


@app.route('/profile_image')
@login_required
def profile_image():
    email = session['email']
    profile_data = company_collection.find_one({'email': email})

    if profile_data and 'profile_image' in profile_data:
        image_data = profile_data['profile_image']
        return send_file(BytesIO(image_data), mimetype='image/jpeg')
    else:
        # Return a default image or a placeholder
        return redirect(url_for('static', filename='images/default_profile_photo.jpg'))


@app.route('/send_otp', methods=['GET', 'POST'])
# function that takes in user OTP code and verifies the OTP code
@login_required
def send_email_verification():
    email = session['email']

    # Generate a valid base32 secret key for TOTP
    totp_secret = pyotp.random_base32()

    totp = pyotp.TOTP(totp_secret)
    otp = totp.now()
    print(otp)

    session['totp_secret'] = totp_secret

    send_email(email, 'Email Confirmation Verification Code', f'Your Verification Code is {otp}')
    return redirect(url_for('verify_user'))


@app.route('/verify_user', methods=['GET', 'POST'])
# function that takes in user OTP code and verifies the OTP code
@login_required
def verify_user():
    if request.method == 'POST':
        otp = request.form.get('otp')
        email = session['email']
        totp_secret = session['totp_secret']
        user = company_collection.find_one({"email": email})

        if user:
            totp = pyotp.TOTP(totp_secret)
            if totp.verify(otp, valid_window=1):
                session.pop('totp_secret', None)
                session['reset_email'] = email
                return redirect(url_for('change_password'))
            else:
                flash('Invalid OTP.', 'error')

    else:
        flash('Check Email for OTP Code', 'error')
        return render_template('verify_user.html')


# Route for uploading an image
@app.route('/upload_image/', methods=['GET', 'POST'])
@login_required
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        image_file = request.files['image']

        if image_file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if image_file and allowed_image_file(image_file.filename):
            # Secure the filename
            filename = secure_filename(image_file.filename)

            # Read the file content
            image_content = image_file.read()

            # Get user ID from session
            email = session['email']

            # Store the image in the database as binary data
            company_collection.update_one({'email': email}, {'$set': {'profile_image': Binary(image_content)}})

            flash('Image uploaded successfully', 'success')

            return redirect(url_for('profile'))

        else:
            flash('Allowed file types are png, jpg, jpeg, gif', 'error')
            return redirect(request.url)

    # Handle GET request to show the upload form
    return render_template('upload_image.html')


# Define the Flask route for data visualization, handling both GET and POST requests
# The @login_required decorator ensures only authenticated users can access this route
@app.route('/data_visualisation/', methods=['GET', 'POST'])
@login_required
def data_visualisation():
    # Dictionary mapping database field names (keys) to user-friendly category names (values)
    # Used to display readable category names in the UI and map user selections back to database fields
    intent_names = {
        "cancel_order": "Cancel Order",
        "change_order": "Change Order",
        "change_shipping_address": "Change Shipping Address",
        "check_cancellation_fee": "Check Cancellation Fee",
        "check_invoices": "Check Invoices",
        "check_payment_methods": "Check Payment Methods",
        "check_refund_policy": "Check Refund Policy",
        "complaint": "Complaint",
        "contact_customer_service": "Contact Customer Service",
        "contact_human_agent": "Contact Human Agent",
        "create_account": "Create Account",
        "delete_account": "Delete Account",
        "delivery_options": "Delivery Options",
        "delivery_period": "Delivery Period",
        "edit_account": "Edit Account",
        "get_invoice": "Get Invoice",
        "get_refund": "Get Refund",
        "newsletter_subscription": "Newsletter Subscription",
        "payment_issue": "Payment Issue",
        "place_order": "Place Order",
        "recover_password": "Recover Password",
        "registration_problems": "Registration Problems",
        "review": "Review",
        "set_up_shipping_address": "Set Up Shipping Address",
        "switch_account": "Switch Account",
        "track_order": "Track Order",
        "track_refund": "Track Refund"
    }

    # Wrap the main logic in a try-except block to handle potential errors gracefully
    try:
        if request.method == "POST":
            category = request.form.get('category')
            if not category:
                flash("Please select a category.", "error")
                return render_template('data_visualisation.html', plot_div="", field_names=intent_names)

            key = next((k for k, v in intent_names.items() if v == category), None)
            if not key:
                flash(f"Invalid category: {category}", "error")
                return render_template('data_visualisation.html', plot_div="", field_names=intent_names)

            # Use modified function to get daily summed data
            values, dates = get_summed_intent_data(key)
            if not values or not dates:
                flash(f"No data available for {category}", "warning")
                return render_template('data_visualisation.html', plot_div="", field_names=intent_names)

            # Create bar chart with daily values
            fig = go.Figure()
            fig.add_trace(go.Bar(x=dates, y=values, name=category))
            fig.update_layout(
                title=f'Daily {category} Calls Trends',
                xaxis_title='Date',
                yaxis_title='Total Calls',
                xaxis=dict(
                    tickformat='%B %d',  # Format as "Month Day" (e.g., June 25)
                    tickangle=45
                )
            )
            plot_div = fig.to_html(full_html=False, include_plotlyjs=True)

        # Handle GET requests (initial page load)
        else:
            # GET request - Create pie chart showing top 5 intent distribution
            intent_totals = {}

            # Get data for all intents
            for field_name, display_name in intent_names.items():
                values, dates = get_specific_field_data(field_name)
                if values is not None and len(values) > 0:
                    # Sum all values for this intent across all intervals
                    intent_totals[display_name] = int(np.sum(values))
                else:
                    intent_totals[display_name] = 0

            # Check if we have any data
            total_sum = sum(intent_totals.values())

            if total_sum == 0:
                flash("No intent data available", "warning")
                return render_template('data_visualisation.html', plot_div="", field_names=intent_names)

            # Debug prints
            print("Intent totals:", intent_totals)
            print("Total sum:", total_sum)

            # Get top 5 intents
            top_intents = sorted(intent_totals.items(), key=lambda x: x[1], reverse=True)[:5]
            labels, values = zip(*top_intents) if top_intents else ([], [])

            # Define colors for the top 5 intents
            colors = ['#4682B4', '#32CD32', '#DC143C', '#B22222',
                      '#FFD700']  # Steel Blue, Lime Green, Crimson, Firebrick, Gold

            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0,  # Set to 0 for full pie chart, increase (e.g., 0.3) for donut chart
                textinfo='label+percent',
                textposition='auto',
                marker=dict(colors=colors[:len(labels)]),  # Use only as many colors as needed
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )])

            fig.update_layout(
                title={
                    'text': 'Top 5 Intent Distribution',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.01
                ),
                margin=dict(t=60, b=40, l=40, r=100),  # Adjust margins for legend
                height=450,  # Set a fixed height for consistency
                width=550  # Set a fixed width for consistency
            )

            # Generate HTML for the pie chart
            plot_div = fig.to_html(full_html=False, include_plotlyjs=True)

        return render_template('data_visualisation.html', plot_div=plot_div, field_names=intent_names)

    except Exception as e:
        flash(f"Error generating visualization: {str(e)}", "error")
        return render_template('data_visualisation.html', plot_div="", field_names=intent_names)


@app.route('/emotion_visualisation/', methods=['GET', 'POST'])
@login_required
def emotion_visualisation():
    emotion_names = {
        "neutral": "Neutral",
        "happy": "Happy",
        "angry": "Angry",
        "sad": "Sad"
    }

    try:
        if request.method == "POST":
            category = request.form.get('category')
            if not category:
                flash("Please select a category.", "error")
                return render_template('emotion_visualization.html', plot_div="", field_names=emotion_names)

            key = next((k for k, v in emotion_names.items() if v == category), None)
            if not key:
                flash(f"Invalid category: {category}", "error")
                return render_template('emotion_visualization.html', plot_div="", field_names=emotion_names)

            # Use modified function to get daily summed data
            values, dates = get_summed_emotion_data(key)
            if not values or not dates:
                flash(f"No data available for {category}", "warning")
                return render_template('emotion_visualization.html', plot_div="", field_names=emotion_names)

            # Create bar chart with daily values
            fig = go.Figure()
            fig.add_trace(go.Bar(x=dates, y=values, name=category))
            fig.update_layout(
                title=f'Daily {category} Calls Trends',
                xaxis_title='Date',
                yaxis_title='Total Calls',
                xaxis=dict(
                    tickformat='%B %d',  # Format as "Month Day" (e.g., June 25)
                    tickangle=45
                )
            )
            plot_div = fig.to_html(full_html=False, include_plotlyjs=True)

        else:
            # GET request - Create pie chart showing emotion distribution
            emotion_totals = {}

            # Get data for all emotions
            for field_name, display_name in emotion_names.items():
                values, dates = get_specific_emotion_field_data(field_name)
                if values is not None and len(values) > 0:
                    # Sum all values for this emotion across all intervals
                    emotion_totals[display_name] = int(np.sum(values))
                else:
                    emotion_totals[display_name] = 0

            # Check if we have any data
            total_sum = sum(emotion_totals.values())

            if total_sum == 0:
                flash("No emotion data available", "warning")
                return render_template('emotion_visualization.html', plot_div="", field_names=emotion_names)

            # Debug prints
            print("Emotion totals:", emotion_totals)
            print("Total sum:", total_sum)

            # Define colors to match different emotions
            colors = ['#4682B4', '#32CD32', '#DC143C',
                      '#B22222']  # Steel Blue (Neutral), Lime Green (Happy), Crimson (Angry), Firebrick (Sad)

            # Create pie chart
            labels = list(emotion_totals.keys())  # Fixed: Changed emotion_labels to emotion_totals
            values = list(emotion_totals.values())

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0,  # Set to 0 for full pie chart, increase (e.g., 0.3) for donut chart
                textinfo='label+percent',
                textposition='auto',
                marker=dict(colors=colors),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )])

            fig.update_layout(
                title={
                    'text': 'Emotion Distribution',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.01
                ),
                margin=dict(t=60, b=40, l=40, r=100),  # Adjust margins for legend
                height=450,  # Set a fixed height for consistency
                width=550  # Set a fixed width for consistency
            )

            # Generate HTML for the pie chart
            plot_div = fig.to_html(full_html=False, include_plotlyjs=True)

        return render_template('emotion_visualization.html', plot_div=plot_div, field_names=emotion_names)

    except Exception as e:
        flash(f"Error generating visualization: {str(e)}", "error")
        return render_template('emotion_visualization.html', plot_div="", field_names=emotion_names)


if __name__ == "__main__":
    app.run(debug=True)
