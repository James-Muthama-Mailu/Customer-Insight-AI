import datetime
import smtplib
import uuid
from email.mime.text import MIMEText
from functools import wraps
from werkzeug.utils import secure_filename
from bson import ObjectId
import pyotp
import requests
from passlib.hash import pbkdf2_sha256
from Customer_Insight_AI_backend.connection import client
from dotenv import load_dotenv, find_dotenv
import os
from flask import Flask, request, redirect, url_for, flash, session, render_template, send_file, jsonify
from audio_to_text.audio_to_text import audio_transcription
from models.categorisation_model.making_prediction import classify_text_results
from bson import Binary
from io import BytesIO
import plotly.graph_objs as go
from datetime import timedelta
from collections import defaultdict
import numpy as np

# Define the allowed extensions for images
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

UPLOAD_FOLDER = 'C:/Users/James Muthama/ICS Project 1/ICS Project 1'
ALLOWED_EXTENSIONS = {'mp3'}

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Declaration of collections
company_collection = client.Customer_Insight_AI.Customer
admin_collection = client.Customer_Insight_AI.Admin
call_categorise = client.Customer_Insight_AI.Call_Categorises
call_files = client.Customer_Insight_AI.Call_Files

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


def get_specific_field_data(field_name):
    values = []
    dates = []

    cursor = call_categorise.find({}, {field_name + '.values': 1, field_name + '.dates': 1, '_id': 0})

    interval_sums = defaultdict(int)
    interval_dates = []

    for doc in cursor:
        if field_name in doc:
            field_data = doc[field_name]
            if field_data and 'values' in field_data and 'dates' in field_data:
                for value, date in zip(field_data['values'], field_data['dates']):
                    interval_start = date - timedelta(minutes=date.minute % 30, seconds=date.second,
                                                      microseconds=date.microsecond)
                    interval_end = interval_start + timedelta(minutes=30)
                    interval_sums[interval_start] += value

    sorted_intervals = sorted(interval_sums.items())
    interval_dates, interval_values = zip(*sorted_intervals)

    # Format datetime values to include year, month, day, hour, and minute
    interval_dates = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in interval_dates]

    # Convert to NumPy arrays
    interval_values_array = np.array(interval_values)
    interval_dates_array = np.array(interval_dates)

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
    return render_template('signup.html')


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
    return render_template('login.html')


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

    return render_template('signup.html')


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
            if 'file' not in request.files:
                flash('No file part', 'error')
                return redirect(request.url)

            uploaded_file = request.files['file']

            if uploaded_file.filename == '':
                flash('No selected file', 'error')
                return redirect(request.url)

            if uploaded_file and allowed_file(uploaded_file.filename):
                # Save the uploaded MP3 file to a temporary file
                filename = secure_filename(uploaded_file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                uploaded_file.save(file_path)

                # Perform audio transcription using the file path
                customer_care_call = audio_transcription(file_path)

                # Perform text classification
                categories, descriptions = classify_text_results(customer_care_call)

                print(categories)
                print(descriptions)

                if categories and descriptions:
                    customer_id = ObjectId(session['user_id'])  # Convert to ObjectId

                    for category, description in zip(categories, descriptions):
                        # Prepare values and dates
                        current_date = datetime.datetime.now()
                        update_query = {
                            "$push": {
                                f"{category}.values": 1,  # Example value, adjust as needed
                                f"{category}.dates": current_date
                            }
                        }

                        # Update MongoDB collection
                        call_categorise.update_one(
                            {"customer_id": customer_id},
                            update_query,
                            upsert=True  # Create a new document if it doesn't exist
                        )

                        # Insert into call_files collection
                        call_files.insert_one({
                            "customer_id": customer_id,
                            "video_file_name": filename,
                            "category": category,
                            "probability": description,
                        })

                    flash('File successfully uploaded and categorized', 'success')
                else:
                    flash('No Category Found', 'error')

            else:
                flash('Allowed file types are mp3 only', 'error')
                return redirect(request.url)

        except Exception as e:
            flash(f"Error occurred: {str(e)}", 'error')
            return redirect(request.url)

        return redirect(url_for('upload_file'))

    elif request.method == 'GET':
        customer_id = ObjectId(session['user_id'])  # Convert to ObjectId

        # Fetch data from call_files collection
        customer_data = call_files.find({"customer_id": customer_id})

        customers = []
        for data in customer_data:
            customers.append({
                'video_file_name': data.get('video_file_name', 'N/A'),
                'category': data.get('category', 'No category found'),
                'probability': data.get('probability', 'N/A')
            })

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


@app.route('/data_visualisation/', methods=['GET', 'POST'])
@login_required
def data_visualisation():
    if request.method == "POST":
        field_names = {
            "track_refund": "Track Refund",
            "review": "Review",
            "cancel_order": "Cancel Order",
            "switch_account": "Switch Account",
            "edit_account": "Edit Account",
            "contact_customer_service": "Contact Customer Service",
            "place_order": "Place Order",
            "check_payment_methods": "Check Payment Methods",
            "payment_issue": "Payment Issue",
            "contact_human_agent": "Contact Human Agent",
            "complaint": "Complaint",
            "recover_password": "Recover Password",
            "delivery_period": "Delivery Period",
            "check_invoices": "Check Invoices",
            "track_order": "Track Order",
            "delete_account": "Delete Account",
            "get_invoice": "Get Invoice",
            "change_order": "Change Order",
            "delivery_options": "Delivery Options",
            "check_invoice": "Check Invoice",
            "create_account": "Create Account",
            "set_up_shipping_address": "Set Up Shipping Address",
            "check_refund_policy": "Check Refund Policy",
            "newsletter_subscription": "Newsletter Subscription",
            "get_refund": "Get Refund",
            "check_cancellation_fee": "Check Cancellation Fee",
            "registration_problems": "Registration Problems",
            "change_shipping_address": "Change Shipping Address"
        }

        category = request.form.get('category')

        # Find the key for the value category
        value_to_find = category
        key = None

        # Reverse the dictionary and look up the key
        for k, v in field_names.items():
            if v == value_to_find:
                key = k
                break

        values, dates = get_specific_field_data(key)

        # Create the Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=values, mode='lines', name='Line'))
        fig.update_layout(title='Line Chart for Catgeory: ' + category, xaxis_title='Dates', yaxis_title='Values')

        # Update date format to display hours as the smallest measure
        fig.update_xaxes(type='date', tickformat='%Y-%m-%d %H:%M:%S')

        # Convert the plot to HTML directly embedding the div
        plot_div = fig.to_html(full_html=False)

        return render_template('data_visualisation.html', plot_div=plot_div, field_names=field_names)
    else:
        field_names = {
            "track_refund": "Track Refund",
            "review": "Review",
            "cancel_order": "Cancel Order",
            "switch_account": "Switch Account",
            "edit_account": "Edit Account",
            "contact_customer_service": "Contact Customer Service",
            "place_order": "Place Order",
            "check_payment_methods": "Check Payment Methods",
            "payment_issue": "Payment Issue",
            "contact_human_agent": "Contact Human Agent",
            "complaint": "Complaint",
            "recover_password": "Recover Password",
            "delivery_period": "Delivery Period",
            "check_invoices": "Check Invoices",
            "track_order": "Track Order",
            "delete_account": "Delete Account",
            "get_invoice": "Get Invoice",
            "change_order": "Change Order",
            "delivery_options": "Delivery Options",
            "check_invoice": "Check Invoice",
            "create_account": "Create Account",
            "set_up_shipping_address": "Set Up Shipping Address",
            "check_refund_policy": "Check Refund Policy",
            "newsletter_subscription": "Newsletter Subscription",
            "get_refund": "Get Refund",
            "check_cancellation_fee": "Check Cancellation Fee",
            "registration_problems": "Registration Problems",
            "change_shipping_address": "Change Shipping Address"
        }

        field_name = "check_cancellation_fee"

        category = field_names[field_name]

        values, dates = get_specific_field_data(field_name)

        # Create the Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=values, mode='lines', name='Line'))
        fig.update_layout(title='Line Chart for Catgeory: ' + category, xaxis_title='Dates', yaxis_title='Values')

        # Update date format to display hours as the smallest measure
        fig.update_xaxes(type='date', tickformat='%Y-%m-%d %H:%M:%S')

        # Convert the plot to HTML directly embedding the div
        plot_div = fig.to_html(full_html=False)

        # Render the HTML template with the plot div
        return render_template('data_visualisation.html', plot_div=plot_div, field_names=field_names)


if __name__ == "__main__":
    app.run(debug=True)
