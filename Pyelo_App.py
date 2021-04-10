#Core Pkgs
import streamlit as st


#EDA Pkgs
import pandas as pd
import numpy as np

#Utils
import os
import joblib
import hashlib
#passlib,bcrypt

#Data Viz Pckgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# DB
from managed_db import *

#Password
def generate_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def verify_hashes(password, hashed_text):
    if generate_hashes(password) == hashed_text:
        return hashed_text
    return False

feature_names_best = ['OR_Time', 'Age_sx_Mos', 'Preop_US', 'ABdo_Xray', 'RPG', 'Preop_nuc_scan', 'Side', 'Intraop US', 'Intraop_finding', 'Approach', 'Sex', 'VCUG', 'Nephrostogram', 'Salle_Stent', 'JJ_Stent']

gender_dict = {"Male":1, "Female":2}
side_dict = {"Left":1, "Right":2}
feature_dict = {"Yes":1,"No":0}
approach_dict = {"Laparoscopic":1,"Open":2, "Robotic":3, "Laparoscopic-assisted open":4}
intraop_dict = {"Crossing vessel":1, "Intrinsic narrowing":2}

def get_value(val,my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

def get_key(val):
    for key, value in my_dict.items():
        if val == key:
            return value
        
def get_feature_value(val):
    feature_dict = {"Yes":1, "No":2}
    for key, value in feature_dict.items():
        if val == key:
            return value

# Load ML models        
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

def main():
    """Pyeloplasty Failure Prediction App"""
    st.title("Pyeloplasty Failure Prediction App")
    
    menu = ["Home", "Login", "SignUp"]
    submenu = ["Prediction", "Metrics"]
    
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Home")
        st.text ("What is pyeloplasty?")
        
    
                
        activity = st.selectbox("Activity", submenu)
        if activity == "Prediction":
                st.header("Predictive Analytics")
                    
                st.subheader ("Patient characteristics")
                Age_sx_Mos = st.number_input("Age (Months)",0,216)
                Sex = st.radio("Sex",tuple(gender_dict.keys()))
                Side = st.radio("Side",tuple(side_dict.keys()))
                    
                st.subheader ("Preoperative investigations")
                Preop_US = st.radio("Ultrasound", tuple(feature_dict.keys()))
                ABdo_Xray = st.radio("Abdominal X-ray", tuple(feature_dict.keys()))
                RPG = st.radio("Retrograde pyelogram", tuple(feature_dict.keys()))
                Preop_nuc_scan = st.radio("Nuclear scan", tuple(feature_dict.keys()))
                VCUG = st.radio("Voiding cystourethrogram", tuple(feature_dict.keys()))
                Nephrostogram = st.radio("Nephrostogram", tuple(feature_dict.keys()))
                    
                st.subheader ("Intraoperative factors")
                Intraop_US = st.radio("Intraoperative Ultrasound", tuple(feature_dict.keys()))
                Intraop_finding = st.radio("Intraoperative Findings", tuple(intraop_dict.keys()))
                Approach = st.radio("Operative Approach", tuple(approach_dict.keys()))
                OR_Time = st.number_input("Operative Time (Minutes)", 0,800)
                JJ_Stent = st.radio("JJ stent", tuple(feature_dict.keys()))
                Salle_Stent = st.radio("Salle stent", tuple(feature_dict.keys()))
                    
                feature_list = [Age_sx_Mos, get_value(Sex,gender_dict), get_value(Side,side_dict), get_feature_value(Preop_US), get_feature_value(ABdo_Xray), get_feature_value(RPG), get_feature_value(Preop_nuc_scan), get_feature_value(VCUG), get_feature_value(Nephrostogram), get_feature_value(Intraop_US), get_value(Intraop_finding,intraop_dict), get_value(Approach, approach_dict), OR_Time, get_feature_value(JJ_Stent), get_feature_value(Salle_Stent)]
                   
                st.write(feature_list)
                pretty_result = {"Age":Age_sx_Mos, "Sex":Sex, "Side":Side, "Ultrasound":Preop_US, "Retrograde pyelogram":RPG, "Nuclear scan":Preop_nuc_scan, "Voiding cystourethrogram":VCUG, "Nephrostogram":Nephrostogram, "Intraoperative Ultrasound":Intraop_US, "Intraoperative Findings":Intraop_finding, "Operative Approach":Approach, "Operative Time":OR_Time, "JJ Stent": JJ_Stent, "Salle Stent": Salle_Stent}
                st.json(pretty_result)
                single_sample = np.array(feature_list).reshape(1,-1)
                 
                #ML
                model_choice = st.selectbox("Select Model", ["SVM"])
                if st.button("Predict"):
                    if model_choice == "SVM":
                        loaded_model = load_model("Pickle_SVM_Pyeloplasty2.pkl")
                        prediction = loaded_model.predict(single_sample)
                        pred_prob = loaded_model.predict_proba(single_sample)
                            
                    st.write(prediction)
                    if prediction == 1:
                        st.warning("Patient will likely need reoperation")
                        pred_probability_score = {"Failure":pred_prob[0][1]*100, "Success":pred_prob[0][0]*100}
                        st.subheader("Prediction Probability Score using {}".format(model_choice))
                        st.json(pred_probability_score)
                    elif prediction == 0:
                        st.success("Patient will likely have a successful pyeloplasty")
                        pred_probability_score = {"Failure":pred_prob[0][1]*100, "Success":pred_prob[0][0]*100}
                        st.subheader("Prediction Probability Score using {}".format(model_choice))
                        st.json(pred_probability_score)
                            
    


if __name__ == '__main__':
    main()
