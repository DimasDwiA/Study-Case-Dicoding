import streamlit as st
import joblib
import numpy as np

# Load saved model
model = joblib.load('random_forest_model.pkl')

# Function to get user input
def user_input_features():
    Application_mode = st.number_input('Application_mode')
    st.write("Description : 1–yes, 0–no")
    Debtor = st.selectbox('Debtor', (0, 1))
    st.write("Description : 1–yes, 0–no")
    Tuition_fees_up_to_date = st.selectbox('Tuition_fees_up_to_date', (0, 1))
    st.write("D escription: 1–Male, 0–Female")
    Gender = st.selectbox('Gender', (0, 1))
    st.write("Description : 1–yes, 0–no")
    Scholarship_holder = st.selectbox('Scholarship_holder', (0, 1))
    Age_at_enrollment = st.number_input('Age_at_enrollment')
    Previous_qualification_grade = st.number_input('Previous_qualification_grade')
    Admission_grade = st.number_input('Admission_grade')
    st.write("Description : 1–yes, 0–no")
    Displaced = st.selectbox('Displaced', (0, 1))
    Curricular_units_1st_sem_enrolled = st.number_input('Curricular_units_1st_sem_enrolled')
    Curricular_units_1st_sem_approved = st.number_input('Curricular_units_1st_sem_approved')
    Curricular_units_1st_sem_grade = st.number_input('Curricular_units_1st_sem_grade')
    Curricular_units_2nd_sem_enrolled = st.number_input('Curricular_units_2nd_sem_enrolled')
    Curricular_units_2nd_sem_approved = st.number_input('Curricular_units_2nd_sem_approved')
    Curricular_units_2nd_sem_grade = st.number_input('Curricular_units_2nd_sem_grade')
    
    # Buat array fitur dari input pengguna
    features = np.array([[Curricular_units_2nd_sem_enrolled, Curricular_units_2nd_sem_approved, Curricular_units_2nd_sem_grade,
                          Curricular_units_1st_sem_enrolled, Curricular_units_1st_sem_approved, Curricular_units_1st_sem_grade,
                          Admission_grade, Previous_qualification_grade, Age_at_enrollment,
                          Tuition_fees_up_to_date, Scholarship_holder, Gender, Debtor,
                          Application_mode, Displaced]])
    return features

def data_raw(Application_mode, Debtor, Tuition_fees_up_to_date, Gender, 
            Scholarship_holder, Age_at_enrollment, Previous_qualification_grade, Admission_grade, Displaced, 
            Curricular_units_1st_sem_enrolled, Curricular_units_1st_sem_approved, Curricular_units_1st_sem_grade, Curricular_units_2nd_sem_enrolled,
            Curricular_units_2nd_sem_approved, Curricular_units_2nd_sem_grade):
    data = {
        'Application_mode': Application_mode,
        'Debtor': Debtor,
        'Tuition_fees_up_to_date': Tuition_fees_up_to_date,
        'Gender': Gender,
        'Scholarship_holder': Scholarship_holder,
        'Age_at_enrollment': Age_at_enrollment,
        'Previous_qualification_grade': Previous_qualification_grade,
        'Admission_grade': Admission_grade,
        'Displaced': Displaced,
        'Curricular_units_1st_sem_enrolled': Curricular_units_1st_sem_enrolled,
        'Curricular_units_1st_sem_approved': Curricular_units_1st_sem_approved,
        'Curricular_units_1st_sem_grade': Curricular_units_1st_sem_grade,
        'Curricular_units_2nd_sem_enrolled': Curricular_units_2nd_sem_enrolled,
        'Curricular_units_2nd_sem_approved': Curricular_units_2nd_sem_approved,
        'Curricular_units_2nd_sem_grade': Curricular_units_2nd_sem_grade,
    }

    return data

# Streamlit UI
st.title("Students Dropout and Academic Sucess Prediction App")
st.write("""
Aplikasi ini memprediksi kemungkinan seorang mahasiswa akan Dropout, Graduated, atau Enrolled.
""")

# Ambil input dari pengguna
input_data = user_input_features()

# Ambil data input mentah untuk ditampilkan
raw_data = data_raw(*input_data[0])

# Menampilkan input data dengan expander
with st.expander("View the Raw Data"):
    st.dataframe(data=raw_data, width=800, height=300)

# Button untuk memprediksi
if st.button('Predict'):
    # Predict result
    prediction = model.predict(input_data)

    # Mapping prediksi ke label
    status_map = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduated'}
    prediction_label = status_map[prediction[0]]

    # Tampilkan hasil prediksi
    st.subheader('Prediction Result')
    st.write(f"Status: {prediction_label}")

