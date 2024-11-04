import cv2
import os
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import streamlit as st
from PIL import Image
import time
import speech_recognition as sr
import pandas as pd
import json


# Eye Aspect Ratio calculation function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Set up MediaPipe Face Mesh for facial landmarks detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5)

# Indices for the left and right eyes in the facial landmarks
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]


# Function to log eye tracking results to a JSON file
def log_eye_tracking_to_json(results, final_score, filepath="eye_tracking_results.json"):
    data = {
        "results": results,
        "final_score": final_score
    }
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)


# Function to handle eye tracking, Eye Tracking score calculation, and display updates (with table output)
def run_eye_tracking(cap, video_placeholder, status_placeholder, score_placeholder, tracking_duration):
    eye_results = []  # List to store results for table
    eye_open_count = 0
    eye_semi_closed_count = 0
    eye_closed_count = 0
    total_count = 0

    while st.session_state.get('tracking', False):
        current_time = time.time()
        elapsed_time = current_time - st.session_state['start_time']

        # Stop tracking after 3 minutes
        if elapsed_time > tracking_duration:
            st.session_state['tracking'] = False
            break

        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array(
                    [(int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])) for pt in face_landmarks.landmark])

                left_eye = landmarks[left_eye_indices]
                right_eye = landmarks[right_eye_indices]

                leftEAR = eye_aspect_ratio(left_eye)
                rightEAR = eye_aspect_ratio(right_eye)

                ear = (leftEAR + rightEAR) / 2.0

                # Track eye state and update counts
                if ear < 0.2:
                    status = "Eyes Closed"
                    eye_closed_count += 1
                elif ear < 0.3:
                    status = "Eyes Semi-Closed"
                    eye_semi_closed_count += 1
                else:
                    status = "Eyes Open"
                    eye_open_count += 1

                # Add result to the list for table display
                eye_results.append({
                    'Time': time.strftime('%H:%M:%S', time.localtime(current_time)),
                    'Eye Status': status
                })

                # Update the status display
                status_placeholder.text(f"Status: {status}")

                # Draw circles around eyes
                for (x, y) in np.concatenate((left_eye, right_eye), axis=0):
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        video_placeholder.image(img)

        total_count += 1

    # Calculate the final score after the tracking duration (3 minutes)
    final_score = (
        (3 * eye_open_count) +
        (2 * eye_semi_closed_count) +
        (1 * eye_closed_count)
    ) / total_count if total_count > 0 else 0

    # Store the final score in session state
    st.session_state['eye_tracking_score'] = round(final_score, 2)

    # Display the final Eye Tracking Score
    score_placeholder.text(f"Final Eye Tracking Score: {st.session_state['eye_tracking_score']}")

    cap.release()

    # Save results to JSON
    log_eye_tracking_to_json(eye_results, st.session_state['eye_tracking_score'])

    # Convert the results to a DataFrame and display as a table
    if eye_results:
        eye_df = pd.DataFrame(eye_results)
        st.table(eye_df)


# Function to classify voice urgency based on the detected words
def classify_voice_urgency(text):
    # Positive words and phrases -> Score 3
    positive_words = ["well", "good", "fine", "better", "okay"]

    # Neutral or medium urgency words -> Score 2
    neutral_words = ["pain", "little", "tired", "unwell", "dizzy"]

    # Negative or high urgency words -> Score 1
    negative_words = ["help", "bad", "emergency", "urgent", "hurt"]

    text_lower = text.lower()

    if any(word in text_lower for word in positive_words):
        return 3
    elif any(word in text_lower for word in neutral_words):
        return 2
    elif any(word in text_lower for word in negative_words):
        return 1
    return 2  # Default to medium urgency


# Voice recognition function that logs results to a JSON file
def log_voice_results_to_json(results, final_score, filepath="voice_recognition_results.json"):
    data = {
        "results": results,
        "final_score": final_score
    }
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)


# Voice recognition function that listens for 30 seconds and classifies the urgency (with table output)
# Voice recognition function that listens for 30 seconds and classifies the urgency (with table output)
def listen_and_recognize_30_seconds(recognizer, microphone):
    st.text("Listening for 30 seconds...")
    st.session_state['voice_results'] = []
    voice_results = []  # List to store results for table
    start_time = time.time()

    while time.time() - start_time < 30:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                st.text("Processing...")
                text = recognizer.recognize_google(audio)
                score = classify_voice_urgency(text)
                st.session_state['voice_results'].append((text, score))

                # Add result to the list for table display
                voice_results.append({
                    'Detected Speech': text,
                    'Urgency Score': score
                })

            except sr.UnknownValueError:
                st.session_state['voice_results'].append(("Could not understand audio", 2))
                voice_results.append({
                    'Detected Speech': "Could not understand audio",
                    'Urgency Score': 2  # Set default score to 2 for unknown value
                })
            except sr.RequestError:
                st.session_state['voice_results'].append(("API unavailable", 2))
                voice_results.append({
                    'Detected Speech': "API unavailable",
                    'Urgency Score': 2  # Set default score to 2 for API errors
                })
            except sr.WaitTimeoutError:
                st.session_state['voice_results'].append(("Listening timed out", 2))
                voice_results.append({
                    'Detected Speech': "Listening timed out",
                    'Urgency Score': 2  # Set default score to 2 for timeout errors
                })

    # Ensure all scores are integers and calculate the average score
    valid_scores = [score for _, score in voice_results if isinstance(score, int)]
    if len(valid_scores) > 0:
        avg_score = sum(valid_scores) / len(valid_scores)
    else:
        avg_score = 0  # In case no valid scores were collected

    # Save results to JSON
    log_voice_results_to_json(voice_results, avg_score)

    # Convert the results to a DataFrame and display as a table
    if voice_results:
        voice_df = pd.DataFrame(voice_results)
        st.table(voice_df)

    return avg_score



# Nurse questionnaire function that logs results to a JSON file
def log_nurse_results_to_json(questions, answers, scores, final_score, filepath="nurse_questionnaire_results.json"):
    data = {
        "questions_and_answers": [{"Question": q, "Answer": a, "Score": s} for q, a, s in zip(questions, answers, scores)],
        "final_score": final_score
    }
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)


# Define the questions and answers in English and German
questions_answers = {
    "en": {
        "questions": [
            "How are you feeling overall?",
            "Do you feel any pain?",
            "Where do you feel pain?",
            "How would you rate your energy level?",
            "Are you experiencing any dizziness or nausea?"
        ],
        "answers": {
            "Feeling good": 3,
            "Feeling okay": 2,
            "Feeling unwell": 1,
            "No pain": 3,
            "Mild pain": 2,
            "Severe pain": 1,
            "Energetic": 3,
            "A bit tired": 2,
            "Very tired": 1,
            "No dizziness or nausea": 3,
            "Mild dizziness/nausea": 2,
            "Severe dizziness/nausea": 1,
            "No specific pain": 3,
            "Mild pain in head/neck": 2,
            "Severe pain in body": 1
        }
    },
    "de": {
        "questions": [
            "Wie fühlen Sie sich insgesamt?",
            "Haben Sie Schmerzen?",
            "Wo haben Sie Schmerzen?",
            "Wie würden Sie Ihr Energieniveau bewerten?",
            "Erleben Sie Schwindel oder Übelkeit?"
        ],
        "answers": {
            "Fühle mich gut": 3,
            "Fühle mich okay": 2,
            "Fühle mich unwohl": 1,
            "Keine Schmerzen": 3,
            "Leichte Schmerzen": 2,
            "Starke Schmerzen": 1,
            "Energisch": 3,
            "Etwas müde": 2,
            "Sehr müde": 1,
            "Kein Schwindel oder Übelkeit": 3,
            "Leichter Schwindel/Übelkeit": 2,
            "Starker Schwindel/Übelkeit": 1,
            "Keine spezifischen Schmerzen": 3,
            "Leichte Schmerzen in Kopf/Nacken": 2,
            "Starke Schmerzen im Körper": 1
        }
    }
}

# Function to handle the multilingual nurse questionnaire and save results to a JSON file
def nurse_questionnaire_multilingual():
    st.title("Nurse Questionnaire")

    # Language selection
    languages = {"English": "en", "German": "de"}
    selected_language = st.selectbox("Choose Language", list(languages.keys()))
    lang_code = languages[selected_language]

    # Fetch the questions and answers based on selected language
    questions = questions_answers[lang_code]["questions"]
    answers = questions_answers[lang_code]["answers"]

    # Collect the user responses
    user_answers = []
    scores = []

    # Time limit of 1 minute for answering
    start_time = time.time()
    time_limit = 60  # 1 minute

    # Ask each question, collect the answer and score
    for question in questions:
        if time.time() - start_time > time_limit:
            st.warning("Time is up!")
            break

        answer = st.radio(question, list(answers.keys()), key=question)
        user_answers.append(answer)
        scores.append(answers[answer])

    # Create a table of the questions, answers, and scores
    df = pd.DataFrame({
        "Question": questions[:len(user_answers)],
        "Answer": user_answers,
        "Score": scores
    })

    # Display the table
    st.table(df)

    # Calculate and display the final average score
    if scores:
        final_score = sum(scores) / len(scores)
        st.write(f"Final Average Score: {final_score:.2f}/3")

# Nurse questionnaire function that logs results to a JSON file
def log_nurse_results_to_json(questions, answers, scores, final_score, filepath="nurse_questionnaire_results.json"):
    data = {
        "questions_and_answers": [{"Question": q, "Answer": a, "Score": s} for q, a, s in zip(questions, answers, scores)],
        "final_score": final_score
    }
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)

def connect_usg_device():
    st.title("USG Device Connection")

    # Simulate device connection
    if st.button("Connect USG Device"):
        # Here, you can add the actual logic to connect to a real USG device
        st.success("USG device connected successfully!")


# Function to load the latest department decision from JSON file
def load_department_decision(filepath="department_decision.json"):
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        return None


# Function to display the latest department decision to the nurse
def display_department_decision():
    st.header("Doctor's Department Decision")
    st.write("The nurse will see the doctor's department decision here and inform the ambulance.")

    # Continuously check for updates to the department decision
    department_decision = load_department_decision()

    # Display the latest decision if it exists
    if department_decision:
        st.success(f"Department: {department_decision['department']}")
        st.info(f"Decision Time: {department_decision['decision_time']}")
        st.write("Please inform the ambulance to go to the specified department.")
    else:
        st.warning("No department decision has been made yet.")


# Main Streamlit App with Tabs
def main():
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Eye Tracking", "Voice Recognition", "Nurse Questionnaire", "Connect Usg Device", "Video Consultation", "Department Decision"])

    # Tab 1: AI Eye Tracking with Table Output
    with tab1:
        st.title("AI-powered Eye Tracking System")
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        score_placeholder = st.empty()

        tracking_duration = 0.5 * 60  # Set to 3o secs

        tracking_state = st.session_state.get('tracking', False)
        st.session_state.setdefault('eye_tracking_score', 3)
        st.session_state.setdefault('start_time', None)

        start_button = st.button("Start Eye Tracking")
        stop_button = st.button("Stop Eye Tracking")

        if start_button and not tracking_state:
            st.session_state['tracking'] = True
            st.session_state['start_time'] = time.time()

        if stop_button and tracking_state:
            st.session_state['tracking'] = False

        if tracking_state:
            cap = cv2.VideoCapture(0)
            run_eye_tracking(cap, video_placeholder, status_placeholder, score_placeholder, tracking_duration)

    # Tab 2: AI Voice Recognition with Table Output
    with tab2:
        st.title("Voice Recognition")
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()

        if st.button("Start Voice Recognition"):
            results = listen_and_recognize_30_seconds(recognizer, microphone)
            st.write("Voice Tracking Results (over 30 seconds):")
            # Results will be displayed as a table inside the function

    # Tab 3: Nurse Questionnaire with Multilingual Support
    with tab3:
        nurse_questionnaire_multilingual()

    # Tab 4: USG Device Connection
    with tab4:
        connect_usg_device()

    # Tab 5: Video Consultation (Placeholder)
    with tab5:
        st.title("Video Consultation")
        st.markdown("Join Video Conference [Telko Live](https://telko.live/5bc36d47b87dad6c)")

    # Tab 6: Doctor's Department Decision Display
    with tab6:
        display_department_decision()


if __name__ == "__main__":
    main()