import os
import pickle

import mysql.connector
import numpy as np
import pandas as pd
from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   session, url_for)
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_login import (LoginManager, UserMixin, current_user, login_required,
                         login_user, logout_user)

app = Flask(__name__)
app.secret_key = '1234'
CORS(app)


# MySQL connection
def get_db_conn():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="appointment",
        auth_plugin='mysql_native_password'
    )


# --- Existing Prediction Model & User Auth Logic ---

# Load model and datasets
svc = pickle.load(open('models/svc.pkl', 'rb'))
sys_des = pd.read_csv('datasets/symtoms_df.csv')
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medication = pd.read_csv("datasets/medications.csv")
diets = pd.read_csv("datasets/diets.csv")

# Flask Login and Bcrypt
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin):
    def __init__(self, user_dict):
        if user_dict:
            self.id = str(user_dict["id"])
            self.username = user_dict["username"]
        else:
            self.id = None
            self.username = None


@login_manager.user_loader
def load_user(user_id):
    conn = get_db_conn()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user_data = cursor.fetchone()
    cursor.close()
    conn.close()
    return User(user_data) if user_data else None


# (Symptoms and Disease Prediction logic remains the same)
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4,
                 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9,
                 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13,
                 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18,
                 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22,
                 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27,
                 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
                 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37,
                 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42,
                 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46,
                 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50,
                 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54,
                 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
                 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
                 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66,
                 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70,
                 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74,
                 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78,
                 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
                 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86,
                 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89,
                 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92,
                 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96,
                 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100,
                 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103,
                 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
                 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110,
                 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113,
                 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
                 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120,
                 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124,
                 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127,
                 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
                 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma',
                 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
                 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A',
                 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis',
                 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)',
                 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism',
                 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis',
                 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection',
                 35: 'Psoriasis', 27: 'Impetigo'}


def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for sym in patient_symptoms:
        if sym in symptoms_dict:
            input_vector[symptoms_dict[sym]] = 1
    return diseases_list[svc.predict([input_vector])[0]]


def helper(disease):
    desc = " ".join(
        description[description['Disease'] == disease]['Description'])
    pre = precautions[precautions['Disease']
                      == disease].iloc[:, 1:].values.tolist()
    pre = pre[0] if pre else []
    meds = medication[medication['Disease'] == disease]['Medication'].tolist()
    diet = diets[diets['Disease'] == disease]['Diet'].tolist()
    wrk = workout[workout['disease'] == disease]['workout'].tolist()
    return desc, pre, meds, diet, wrk


# --- Existing Routes ---

@app.route('/')
@login_required
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    symptoms = request.form.get('symptoms')
    cleanSymptoms = ""
    for char in symptoms:
        if char == ' ':
            cleanSymptoms += "_"
        else:
            cleanSymptoms += char
    if not cleanSymptoms or cleanSymptoms.strip().lower() == "symptoms":
        return render_template('index.html', message="Please write correct symptoms")
    user_symptoms = [s.strip("[]' ") for s in cleanSymptoms.split(',')]
    predicted_disease = get_predicted_value(user_symptoms)
    dis_des, pre, meds, diet, wrk = helper(predicted_disease)
    return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des, my_precautions=pre,
                           my_medications=meds, my_diet=diet, workout=wrk)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            flash('User already exists', 'danger')
        else:
            hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, hashed_pw))
            conn.commit()
            flash('Registered successfully! Please login.', 'success')
            return redirect(url_for('login'))
        cursor.close()
        conn.close()
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user_data = cursor.fetchone()
        cursor.close()
        conn.close()
        if user_data and bcrypt.check_password_hash(user_data['password_hash'], password):
            user_obj = User(user_data)
            login_user(user_obj)
            return redirect(url_for('index'))
        flash('Invalid credentials', 'danger')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for('login'))


@app.route('/session')
def check_session():
    is_auth = current_user.is_authenticated
    username = current_user.username if is_auth and hasattr(
        current_user, 'username') else None
    return jsonify({'authenticated': is_auth, 'username': username})


@app.route("/api/doctors", methods=["GET"])
@login_required
def get_doctors():
    conn = get_db_conn()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM doctors")
    doctors = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(doctors)


# ... (other existing doctor and appointment APIs) ...
@app.route("/api/appointments", methods=["GET"])
def get_appointments():
    conn = get_db_conn()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM doctor_appointment")
    appointments = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(appointments)


@app.route("/api/appointments", methods=["POST"])
def create_appointment():
    data = request.json
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO doctor_appointment (name, doctor, date, time, type) VALUES (%s, %s, %s, %s, %s)",
                   (data["name"], data["doctor"], data["date"], data["time"], data["type"]))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "Appointment created"}), 201


# --- NEW: Medicine Delivery APIs ---

@app.route('/api/medicines', methods=['GET'])
def get_medicines():
    """Fetches all available medicines from the database."""
    conn = get_db_conn()
    cursor = conn.cursor(dictionary=True)
    query = "SELECT id, name, description, category, price, discount_price, requires_prescription, image_url FROM medicines WHERE stock_quantity > 0"
    cursor.execute(query)
    medicines = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(medicines)


@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Fetches distinct categories to avoid duplicates shown in your db output."""
    conn = get_db_conn()
    cursor = conn.cursor(dictionary=True)
    # Using DISTINCT to handle the duplicate data in your table
    cursor.execute(
        "SELECT DISTINCT name, description, icon_class FROM categories ORDER BY name")
    categories = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(categories)


def get_or_create_cart_id(user_id):
    """Gets a user's cart ID, or creates a new cart if one doesn't exist."""
    conn = get_db_conn()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id FROM carts WHERE user_id = %s", (user_id,))
    cart = cursor.fetchone()
    if cart:
        cart_id = cart['id']
    else:
        cursor.execute("INSERT INTO carts (user_id) VALUES (%s)", (user_id,))
        conn.commit()
        cart_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return cart_id


@app.route('/api/cart', methods=['GET'])
@login_required
def get_cart():
    """Fetches the current user's cart items and total price."""
    user_id = current_user.id
    cart_id = get_or_create_cart_id(user_id)

    conn = get_db_conn()
    cursor = conn.cursor(dictionary=True)
    # Join with medicines to get full details
    cursor.execute("""
        SELECT ci.medicine_id, m.name, m.image_url, ci.quantity, 
               COALESCE(m.discount_price, m.price) as price_each
        FROM cart_items ci
        JOIN medicines m ON ci.medicine_id = m.id
        WHERE ci.cart_id = %s
    """, (cart_id,))
    cart_items = cursor.fetchall()
    cursor.close()
    conn.close()

    # Calculate total price on the server for accuracy
    total_price = sum(float(item['price_each']) *
                      item['quantity'] for item in cart_items)
    return jsonify({'items': cart_items, 'total_price': total_price})


@app.route('/api/cart/add', methods=['POST'])
@login_required
def add_to_cart():
    """Adds a medicine to the cart or updates its quantity."""
    data = request.json
    medicine_id = data.get('medicine_id')
    quantity_to_add = int(data.get('quantity', 1))

    if not medicine_id or quantity_to_add <= 0:
        return jsonify({'error': 'Invalid medicine ID or quantity'}), 400

    user_id = current_user.id
    cart_id = get_or_create_cart_id(user_id)
    conn = get_db_conn()
    cursor = conn.cursor(dictionary=True)

    # Check if item already in cart
    cursor.execute(
        "SELECT quantity FROM cart_items WHERE cart_id = %s AND medicine_id = %s", (cart_id, medicine_id))
    existing_item = cursor.fetchone()

    # Check for available stock
    cursor.execute(
        "SELECT stock_quantity, name FROM medicines WHERE id = %s", (medicine_id,))
    medicine = cursor.fetchone()
    if not medicine:
        cursor.close()
        conn.close()
        return jsonify({'error': 'Medicine not found'}), 404

    new_quantity = (existing_item['quantity']
                    if existing_item else 0) + quantity_to_add

    if medicine['stock_quantity'] < new_quantity:
        cursor.close()
        conn.close()
        return jsonify(
            {'error': f"Not enough stock for {medicine['name']}. Available: {medicine['stock_quantity']}"}), 400

    if existing_item:
        cursor.execute("UPDATE cart_items SET quantity = %s WHERE cart_id = %s AND medicine_id = %s",
                       (new_quantity, cart_id, medicine_id))
    else:
        cursor.execute("INSERT INTO cart_items (cart_id, medicine_id, quantity) VALUES (%s, %s, %s)",
                       (cart_id, medicine_id, quantity_to_add))

    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({'message': 'Item added to cart successfully'}), 200


@app.route('/api/cart/update/<int:medicine_id>', methods=['PUT'])
@login_required
def update_cart_item(medicine_id):
    """Updates the quantity of a specific item in the cart."""
    data = request.json
    new_quantity = int(data.get('quantity'))

    if new_quantity <= 0:  # Let frontend call remove if quantity is 0
        return remove_from_cart(medicine_id)

    user_id = current_user.id
    cart_id = get_or_create_cart_id(user_id)
    conn = get_db_conn()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        "SELECT stock_quantity, name FROM medicines WHERE id = %s", (medicine_id,))
    medicine = cursor.fetchone()
    if not medicine:
        cursor.close()
        conn.close()
        return jsonify({'error': 'Medicine not found'}), 404

    if medicine['stock_quantity'] < new_quantity:
        cursor.close()
        conn.close()
        return jsonify(
            {'error': f"Not enough stock for {medicine['name']}. Available: {medicine['stock_quantity']}"}), 400

    cursor.execute("UPDATE cart_items SET quantity = %s WHERE cart_id = %s AND medicine_id = %s",
                   (new_quantity, cart_id, medicine_id))
    conn.commit()

    cursor.close()
    conn.close()
    return jsonify({'message': 'Cart item updated'}), 200


@app.route('/api/cart/remove/<int:medicine_id>', methods=['DELETE'])
@login_required
def remove_from_cart(medicine_id):
    """Removes an item completely from the cart."""
    user_id = current_user.id
    cart_id = get_or_create_cart_id(user_id)

    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM cart_items WHERE cart_id = %s AND medicine_id = %s", (cart_id, medicine_id))
    conn.commit()

    affected_rows = cursor.rowcount
    cursor.close()
    conn.close()

    if affected_rows == 0:
        return jsonify({'error': 'Item not found in cart'}), 404
    return jsonify({'message': 'Item removed from cart'}), 200


@app.route('/api/orders/create', methods=['POST'])
@login_required
def create_order():
    """Creates an order from the user's cart. This is a transactional operation."""
    data = request.json
    delivery_address = data.get('delivery_address')

    if not delivery_address or not delivery_address.strip():
        return jsonify({'error': 'Delivery address is required'}), 400

    user_id = current_user.id
    cart_id = get_or_create_cart_id(user_id)
    conn = get_db_conn()
    cursor = conn.cursor(dictionary=True)
    order_id = None

    try:
        conn.start_transaction()
        cursor.execute("""
            SELECT ci.medicine_id, m.name, ci.quantity, 
                   COALESCE(m.discount_price, m.price) as price_each,
                   m.stock_quantity
            FROM cart_items ci JOIN medicines m ON ci.medicine_id = m.id
            WHERE ci.cart_id = %s
        """, (cart_id,))
        cart_items = cursor.fetchall()

        if not cart_items:
            raise ValueError("Cart is empty")

        total_amount = 0
        for item in cart_items:
            if item['stock_quantity'] < item['quantity']:
                raise ValueError(f"Not enough stock for {item['name']}")
            total_amount += float(item['price_each']) * item['quantity']

        cursor.execute(
            "INSERT INTO orders (user_id, total_amount, delivery_address, payment_method) VALUES (%s, %s, %s, %s)",
            (user_id, total_amount, delivery_address, 'Cash on Delivery'))
        order_id = cursor.lastrowid

        for item in cart_items:
            cursor.execute("INSERT INTO order_items (order_id, medicine_id, quantity, price) VALUES (%s, %s, %s, %s)",
                           (order_id, item['medicine_id'], item['quantity'], item['price_each']))
            cursor.execute("UPDATE medicines SET stock_quantity = stock_quantity - %s WHERE id = %s",
                           (item['quantity'], item['medicine_id']))

        cursor.execute("DELETE FROM cart_items WHERE cart_id = %s", (cart_id,))
        conn.commit()

        return jsonify({'message': 'Order created successfully!', 'order_id': order_id}), 201

    except (mysql.connector.Error, ValueError) as err:
        conn.rollback()
        return jsonify({'error': str(err)}), 400
    finally:
        cursor.close()
        conn.close()


# --- Static Pages Routes ---

@app.route('/about')
def about(): return render_template('about.html')


@app.route('/contact')
def contact(): return render_template('contact.html')


@app.route('/developer')
def developer(): return render_template('developer.html')


@app.route('/blog')
def blog(): return render_template('blog.html')


@app.route('/medicine-delivery')
@login_required  # Require login to see the delivery page
def medicine_delivery():
    return render_template('delivery.html')


@app.route('/doctor-appointment')
@login_required  # Require login to see the appointment page
def doctor_appointment():
    return render_template('doctor_appointment.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9000, debug=True)
