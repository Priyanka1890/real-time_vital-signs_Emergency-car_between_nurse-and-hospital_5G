import streamlit as st
import json
import pandas as pd
from datetime import datetime

# Function to load data from JSON file
def load_json_data(filepath):
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        st.warning(f"No data found in {filepath}.")
        return None

# Function to convert data to a DataFrame for display
def convert_to_dataframe(data, data_type):
    if not data:
        return None

    if data_type == "eye_tracking":
        # Convert Eye Tracking data to DataFrame
        df = pd.DataFrame(data["results"])
        df['Final Score'] = data["final_score"]
        return df

    elif data_type == "voice_recognition":
        # Convert Voice Recognition data to DataFrame
        df = pd.DataFrame(data["results"])
        df['Final Score'] = data["final_score"]
        return df

    elif data_type == "nurse_questionnaire":
        # Convert Nurse Questionnaire data to DataFrame
        qa_data = data["questions_and_answers"]
        df = pd.DataFrame(qa_data)
        df['Final Score'] = data["final_score"]
        return df

    return None

# Function to log department decision to JSON
def log_department_decision(department, filepath="department_decision.json"):
    data = {
        "department": department,
        "decision_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)

# Main function for HospitalApp
def main():
    st.title("Hospital Dashboard for Doctor")
    st.write("Here the doctor can see the DiagnosisApp results and decide the patient's department.")

    # Tabs for each result type
    tab1, tab2, tab3 = st.tabs(["Eye Tracking Results", "Voice Recognition Results", "Nurse Questionnaire Results"])

    # Load the data from JSON files
    eye_tracking_data = load_json_data("eye_tracking_results.json")
    voice_recognition_data = load_json_data("voice_recognition_results.json")
    nurse_questionnaire_data = load_json_data("nurse_questionnaire_results.json")

    # Display Eye Tracking Results in Tab 1
    with tab1:
        st.header("Eye Tracking Results")
        if eye_tracking_data:
            st.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
            st.write(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            df_eye_tracking = convert_to_dataframe(eye_tracking_data, "eye_tracking")
            if df_eye_tracking is not None:
                st.table(df_eye_tracking)
        else:
            st.write("No eye tracking data available.")

    # Display Voice Recognition Results in Tab 2
    with tab2:
        st.header("Voice Recognition Results")
        if voice_recognition_data:
            st.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
            st.write(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            df_voice_recognition = convert_to_dataframe(voice_recognition_data, "voice_recognition")
            if df_voice_recognition is not None:
                st.table(df_voice_recognition)
        else:
            st.write("No voice recognition data available.")

    # Display Nurse Questionnaire Results in Tab 3
    with tab3:
        st.header("Nurse Questionnaire Results")
        if nurse_questionnaire_data:
            st.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
            st.write(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            df_nurse_questionnaire = convert_to_dataframe(nurse_questionnaire_data, "nurse_questionnaire")
            if df_nurse_questionnaire is not None:
                st.table(df_nurse_questionnaire)
        else:
            st.write("No nurse questionnaire data available.")

    # Doctor's Decision Section
    st.header("Doctor's Department Decision")

    st.write("Please select the department where the patient should be directed:")

    # Department selection buttons
    if st.button("Emergency"):
        log_department_decision("Emergency")
        st.success("Patient directed to Emergency Department.")

    if st.button("Cardiology"):
        log_department_decision("Cardiology")
        st.success("Patient directed to Cardiology Department.")

    if st.button("Neurology"):
        log_department_decision("Neurology")
        st.success("Patient directed to Neurology Department.")

    if st.button("Orthopedics"):
        log_department_decision("Orthopedics")
        st.success("Patient directed to Orthopedics Department.")

    if st.button("General Medicine"):
        log_department_decision("General Medicine")
        st.success("Patient directed to General Medicine Department.")

    # Display the last department decision
    st.header("Latest Department Decision")
    department_decision = load_json_data("department_decision.json")
    if department_decision:
        st.write(f"Department: {department_decision['department']}")
        st.write(f"Decision Time: {department_decision['decision_time']}")
    else:
        st.write("No department decision made yet.")

if __name__ == "__main__":
    main()
