from urllib.parse import quote_plus
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import numpy as np
import pandas as pd
import pickle
from flask_pymongo import PyMongo
from flask_cors import CORS
from bson.objectid import ObjectId

from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from flask_bcrypt import Bcrypt

app = Flask(__name__)
CORS(app)

username = "rishabkumar"
password = "pass@1234"

escaped_username = quote_plus(username)
escaped_password = quote_plus(password)

app.config["MONGO_URI"] = (
    f"mongodb+srv://{escaped_username}:{escaped_password}@appointments.llruxte.mongodb.net/appointment"
    "?retryWrites=true&w=majority&appName=Appointments"
)
app.secret_key = '1234'

# MongoDB instance is created via PyMongo
mongo = PyMongo(app)
appointments = mongo.db["doctor appointment"]
doctors = mongo.db["doctors"]
users = mongo.db["users"]

# Load datasets
sys_des = pd.read_csv('datasets/symtoms_df.csv')
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medication = pd.read_csv("datasets/medications.csv")
diets = pd.read_csv("datasets/diets.csv")

# Load model
svc = pickle.load(open('models/svc.pkl', 'rb'))

# Bcrypt and LoginManager
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# --- Flask-Login User Loader ---
class User(UserMixin):
    def __init__(self, user_dict):
        self.id = str(user_dict["_id"])
        self.username = user_dict["username"]
        self.password_hash = user_dict.get("password_hash", "")


@login_manager.user_loader
def load_user(user_id):
    user = users.find_one({"_id": ObjectId(user_id)})
    if user:
        return User(user)
    return None


# Helper function
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values[0]]  # Convert to list of strings

    med = medication[medication['Disease'] == dis]['Medication']
    med = med.tolist()  # Convert to list of strings

    die = diets[diets['Disease'] == dis]['Diet']
    die = die.tolist()  # Convert to list of strings

    wrkout = workout[workout['disease'] == dis]['workout']
    wrkout = wrkout.tolist()  # Convert to list of strings

    return desc, pre, med, die, wrkout


# Symptoms dictionary and diseases list (unchanged from your original code)
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


# Model prediction function (unchanged)
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]


# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user_data = users.find_one({'username': username})

        if user_data:
            try:
                # Check if password_hash exists and is valid
                if 'password_hash' not in user_data or not user_data['password_hash']:
                    flash('Invalid user configuration', 'danger')
                elif bcrypt.check_password_hash(user_data['password_hash'], password):
                    user_obj = User(user_data)
                    login_user(user_obj)
                    flash('Logged in successfully!', 'success')
                    return redirect(url_for('index'))
                else:
                    flash('Invalid username or password', 'danger')
            except ValueError as e:
                flash('Invalid user configuration. Please contact support.', 'danger')
                app.logger.error(f"Invalid hash for user {username}: {str(e)}")
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if users.find_one({'username': username}):
            flash('Username already exists.', 'danger')
        else:
            # Ensure password is hashed properly
            password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
            users.insert_one({
                'username': username,
                'password_hash': password_hash
            })
            flash('Registration successful. Please login.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

# --- Session Check (for auto-login) ---
@app.route('/session')
def check_session():
    if current_user.is_authenticated:
        return jsonify({'authenticated': True, 'username': current_user.username})
    else:
        return jsonify({'authenticated': False})

# --- Example: Protect Doctors Management ---

# --- MongoDB doctor CRUD ---
@app.route("/api/doctors", methods=["GET"])
@login_required
def get_doctors():
    try:
        doctor_list = []
        for doc in doctors.find():
            doc["_id"] = str(doc["_id"])
            doctor_list.append(doc)
        return jsonify(doctor_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/doctors", methods=["POST"])
@login_required
def add_doctor():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        required_fields = ["name", "specialty", "email"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400
        if doctors.find_one({"email": data["email"]}):
            return jsonify({"error": "Doctor with this email already exists"}), 400
        result = doctors.insert_one({
            "name": data["name"],
            "specialty": data["specialty"],
            "email": data["email"]
        })
        return jsonify({
            "_id": str(result.inserted_id),
            "name": data["name"],
            "specialty": data["specialty"],
            "email": data["email"]
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/doctors/<id>", methods=["PUT"])
@login_required
def update_doctor(id):
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        update_data = {}
        if "name" in data:
            update_data["name"] = data["name"]
        if "specialty" in data:
            update_data["specialty"] = data["specialty"]
        if "email" in data:
            update_data["email"] = data["email"]
        if not update_data:
            return jsonify({"error": "No valid fields to update"}), 400
        result = doctors.update_one(
            {"_id": ObjectId(id)},
            {"$set": update_data}
        )
        if result.matched_count == 0:
            return jsonify({"error": "Doctor not found"}), 404
        return jsonify({"message": "Doctor updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/doctors/<id>", methods=["DELETE"])
@login_required
def delete_doctor(id):
    try:
        result = doctors.delete_one({"_id": ObjectId(id)})
        if result.deleted_count == 0:
            return jsonify({"error": "Doctor not found"}), 404
        return jsonify({"message": "Doctor deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/appointments", methods=["POST"])
def create_appointment():
    data = request.json
    result = appointments.insert_one({
        "name": data.get("name"),
        "doctor": data.get("doctor"),
        "date": data.get("date"),
        "time": data.get("time"),
        "type": data.get("type")
    })
    return jsonify({"_id": str(result.inserted_id)}), 201

# READ (all)
@app.route("/api/appointments", methods=["GET"])
def get_appointments():
    appt_list = []
    for appt in appointments.find():
        appt["_id"] = str(appt["_id"])
        appt_list.append(appt)
    return jsonify(appt_list)

# UPDATE
@app.route("/api/appointments/<id>", methods=["PUT"])
def update_appointment(id):
    data = request.json
    appointments.update_one(
        {"_id": ObjectId(id)},
        {"$set": {
            "name": data.get("name"),
            "doctor": data.get("doctor"),
            "date": data.get("date"),
            "time": data.get("time"),
            "type": data.get("type")
        }}
    )
    return jsonify({"msg": "updated"})

# DELETE
@app.route("/api/appointments/<id>", methods=["DELETE"])
def delete_appointment(id):
    appointments.delete_one({"_id": ObjectId(id)})
    return jsonify({"msg": "deleted"})
@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if symptoms == "Symptoms":
            message = "Please write correct symptoms"
            return render_template('index.html', message=message)
        else:
            # Split the user's input into a list of symptoms (assuming they are comma-separated)
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            # Remove any extra characters, if any
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

            # Pass the lists directly to the template
            return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=precautions, my_medications=medications, my_diet=rec_diet,
                                   workout=workout)

    return render_template('index.html')

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

# @app.route('/login')
# def login():
#     return render_template('login.html')

@app.route('/doctor-appointment')
def doctor_appointment():
    return render_template('doctor_appointment.html')

@app.route('/manage-doctors')
@login_required
def manage_doctors():
    return render_template('manage_doctors.html')

# Python main
if __name__ == "__main__":
    app.run(debug=True)