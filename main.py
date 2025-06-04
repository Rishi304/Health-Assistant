import pickle

import mysql.connector
import numpy as np
import pandas as pd
from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   url_for)
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
        database="appointment"
    )


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
        self.id = str(user_dict["id"])
        self.username = user_dict["username"]


@login_manager.user_loader
def load_user(user_id):
    conn = get_db_conn()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return User(user) if user else None


symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4,
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
    'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}

# Diseases list
diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
    33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma',
    23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
    28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A',
    19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis',
    36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)',
    18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism',
    25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis',
    0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection',
    35: 'Psoriasis', 27: 'Impetigo'
}


def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for sym in patient_symptoms:
        if sym in symptoms_dict:
            input_vector[symptoms_dict[sym]] = 1
    prediction = svc.predict([input_vector])[0]
    return diseases_list[prediction]


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

# Routes


@app.route('/')
@login_required
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    symptoms = request.form.get('symptoms')
    if not symptoms or symptoms.strip().lower() == "symptoms":
        return render_template('index.html', message="Please write correct symptoms")
    user_symptoms = [s.strip("[]' ") for s in symptoms.split(',')]
    predicted_disease = get_predicted_value(user_symptoms)
    dis_des, pre, meds, diet, wrk = helper(predicted_disease)
    return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                           my_precautions=pre, my_medications=meds, my_diet=diet, workout=wrk)


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
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        if user and bcrypt.check_password_hash(user['password_hash'], password):
            login_user(User(user))
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
    return jsonify({'authenticated': current_user.is_authenticated, 'username': current_user.username if current_user.is_authenticated else None})

# Doctor APIs


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


@app.route("/api/doctors", methods=["POST"])
@login_required
def add_doctor():
    data = request.json
    if not all(k in data for k in ("name", "specialty", "email")):
        return jsonify({"error": "Missing fields"}), 400
    conn = get_db_conn()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM doctors WHERE email = %s", (data["email"],))
    if cursor.fetchone():
        return jsonify({"error": "Doctor already exists"}), 400
    cursor.execute("INSERT INTO doctors (name, specialty, email) VALUES (%s, %s, %s)",
                   (data["name"], data["specialty"], data["email"]))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify(data), 201


@app.route("/api/doctors/<int:id>", methods=["PUT"])
@login_required
def update_doctor(id):
    data = request.json
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("UPDATE doctors SET name=%s, specialty=%s, email=%s WHERE id=%s",
                   (data["name"], data["specialty"], data["email"], id))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "Doctor updated"})


@app.route("/api/doctors/<int:id>", methods=["DELETE"])
@login_required
def delete_doctor(id):
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM doctors WHERE id = %s", (id,))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "Doctor deleted"})

# Appointment APIs


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


@app.route("/api/appointments/<int:id>", methods=["PUT"])
def update_appointment(id):
    data = request.json
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("UPDATE doctor_appointment SET name=%s, doctor=%s, date=%s, time=%s, type=%s WHERE id=%s",
                   (data["name"], data["doctor"], data["date"], data["time"], data["type"], id))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "Appointment updated"})


@app.route("/api/appointments/<int:id>", methods=["DELETE"])
def delete_appointment(id):
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM doctor_appointment WHERE id = %s", (id,))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "Appointment deleted"})

# Static Pages


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/developer')
def developer():
    return render_template('developer.html')


@app.route('/blog')
def blog():
    return render_template('blog.html')


@app.route('/medicine-delivery')
def medicine_delivery():
    return render_template('delivery.html')


@app.route('/doctor-appointment')
def doctor_appointment():
    return render_template('doctor_appointment.html')


@app.route('/manage-doctors')
@login_required
def manage_doctors():
    return render_template('manage_doctors.html')


if __name__ == "__main__":
    app.run(port=9000, debug=True)
