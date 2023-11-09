import pandas as pd
import streamlit as st
import joblib
from dummies_dict import *


st.title('''Inpatient Analysis & Predicting Length of Stay Model\n Data Science Project\n 
         Project prepared by: Diaa Aldein Alsayed Ibrahim Osman\n 
         Prepared for: Epsilon AI Institute''')

st.info('''
        The aim of this data science project is to perform a comprehensive analysis of inpatient data for the state of New York in 2015, and develop 
        a predictive model for the length of stay. The project will leverage advanced data analytics and machine learning techniques to provide valuable insights for healthcare providers and policymakers.
        

        *** IMPORTANT NOTICE: This model shows high performance during small-sample testing  but is not yet trusted to be generalized. More time and more comprehensive study are needed for generalization. *** 
        ''')

st.header('User Input Features For the Model: ')

age_group_sel = st.selectbox('Select Patient Age Group', ['18 to 29','30 to 49','50 to 69','70 or Older'])
age_group = dict_age_group[age_group_sel]


gender_sel = st.selectbox('Select Patient Gender', ['F','M'])
gender_M = dict_gender[gender_sel]


type_of_admission_sel = st.selectbox('Select Type of Admission', ['Emergency', 'Urgent', 'Elective', 'Not Available', 'Trauma', 'Newborn'])
type_of_admission = dict_type_of_admission[type_of_admission_sel]

diagnosis_description_list = []
for key in dict_ccs_diagnosis_description.keys():
    diagnosis_description_list.append(key)

ccs_diagnosis_description_sel = st.selectbox('Select CCS Diagnosis Description', diagnosis_description_list)
ccs_diagnosis_description = dict_ccs_diagnosis_description[ccs_diagnosis_description_sel]

ccs_procedure_description_list = []
for key in dict_ccs_procedure_description.keys():
    ccs_procedure_description_list.append(key)

ccs_procedure_description_sel = st.selectbox('Select CCS Procedure Description', ccs_procedure_description_list)
ccs_procedure_description = dict_ccs_procedure_description[ccs_procedure_description_sel]

apr_drg_description_lilst = []
for key in dict_apr_drg_description.keys():
    apr_drg_description_lilst.append(key)

apr_drg_description_sel = st.selectbox('Select APR DRG Description', apr_drg_description_lilst)
apr_drg_description = dict_apr_drg_description[apr_drg_description_sel]


apr_mdc_description_lilst = []
for key in dict_apr_mdc_description.keys():
    apr_mdc_description_lilst.append(key)

apr_mdc_description_sel = st.selectbox('Select APR MDC Description', apr_mdc_description_lilst)
apr_mdc_description = dict_apr_mdc_description[apr_mdc_description_sel]

apr_severity_of_illness_description_sel = st.selectbox('Select APR Severity of Illness Description', ['Minor', 'Moderate', 'Major', 'Extreme'])
apr_severity_of_illness_description = dict_apr_severity_of_illness_description[apr_severity_of_illness_description_sel]

apr_risk_of_mortality_sel = st.selectbox('Select APR Risk of Mortality', ['Minor', 'Moderate', 'Major', 'Extreme'])
apr_risk_of_mortality = dict_apr_risk_of_mortality[apr_risk_of_mortality_sel]

apr_medical_surgical_description_sel = st.selectbox('Select APR Medical Surgical Description', ['Medical', 'Surgical'])
apr_medical_surgical_description_Surgical = dict_apr_medical_surgical_description_list[apr_medical_surgical_description_sel]


emergency_department_indicator_sel = st.selectbox('Select Emergency Department Indicator', ['Y', 'N'])
emergency_department_indicator_Y = dict_emergency_department_indicator_list[emergency_department_indicator_sel]

data = [age_group,type_of_admission,ccs_diagnosis_description,ccs_procedure_description,apr_drg_description,apr_mdc_description,apr_severity_of_illness_description,apr_risk_of_mortality,gender_M,apr_medical_surgical_description_Surgical,emergency_department_indicator_Y]

scaler = joblib.load('scaler.h5') 
model = joblib.load('knnr.h5')

data_scaled = scaler.transform([data])

result = model.predict(data_scaled)

def show(result):
       if result == 120:
          return 'Patient expected to stay for 120 days or more'
       else:
          return f'Patient expected to stay for around {int(result)} day'

st.subheader('''Base on the above value entered The Model Prediction is:''')

if st.button('Predict'):
    st.write(show(result))
