import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import random
from datetime import datetime

from src.preprocess import load_and_preprocess_data
from src.predict import predict_cgpa

# Custom CSS for better UI
st.set_page_config(
    page_title="Student Mental Health Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    .chat-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .resource-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        color: #2c3e50;
    }
    .emergency-card {
        background: #fff5f5;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #e74c3c;
        color: #2c3e50;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-top: 3px solid #667eea;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_dataset(path: str):
    return pd.read_excel(path)

@st.cache_resource
def load_artifacts():
    model = joblib.load("model/random_forest_regressor.joblib") if os.path.exists("model/random_forest_regressor.joblib") else None
    scaler = joblib.load("model/scaler.joblib") if os.path.exists("model/scaler.joblib") else None
    return model, scaler

@st.cache_data
def get_training_columns():
    try:
        # Use preprocessing to obtain training feature columns from the dataset
        *_rest, feature_names = load_and_preprocess_data("data/Education_Dataset.xlsx")
        return list(feature_names)
    except Exception:
        return None

def get_mental_health_response(user_input):
    """Generate dynamic responses for mental health chatbot"""
    user_input_lower = user_input.lower()
    
    # Track conversation context
    if "conversation_count" not in st.session_state:
        st.session_state.conversation_count = 0
    st.session_state.conversation_count += 1
    
    # Check for follow-up questions
    if st.session_state.conversation_count > 1:
        if any(word in user_input_lower for word in ["thanks", "thank you", "helped", "good advice"]):
            return random.choice([
                "You're welcome! I'm glad I could help. Remember to take care of yourself. 💙",
                "Anytime! Is there anything else you'd like to talk about?",
                "I'm here for you. Don't hesitate to reach out again if you need support."
            ])
    
    # POSITIVE EMOTIONS - Handle first
    if any(word in user_input_lower for word in ["happy", "good", "great", "excited", "wonderful", "amazing", "fantastic", "joy", "positive"]):
        responses = [
            "That's wonderful to hear! It's great that you're feeling positive. What's been going well for you?",
            "I'm so glad you're feeling happy! Positive moments are worth celebrating. What's making you feel good today?",
            "That's fantastic! It's important to acknowledge when we're feeling good. What's contributing to your positive mood?",
            "Wonderful! Happiness is something to cherish. What's been the highlight of your day?"
        ]
        return random.choice(responses)
    
    # Enhanced depression and sadness responses - MOVED UP
    elif any(word in user_input_lower for word in ["sad", "depressed", "unhappy", "down", "empty", "numb", "low", "terrible", "awful", "miserable"]):
        intensity = "high" if any(word in user_input_lower for word in ["empty", "numb", "can't", "nothing", "terrible", "awful", "miserable"]) else "moderate"
        
        if intensity == "high":
            responses = [
                "I'm really concerned about what you're sharing. Feeling empty or numb can be frightening. You don't have to carry this alone. Please consider reaching out to a counselor or trusted person.",
                "It sounds like you're going through something really difficult. These feelings are heavy, but they don't define you. Your life has value, even when it doesn't feel like it.",
                "I hear your pain, and I want you to know that help is available. What you're experiencing is treatable, and many people have found relief through support."
            ]
        else:
            responses = [
                "I'm sorry you're feeling this way. Emotions come and go like waves - this feeling won't last forever, even if it feels that way now.",
                "When you're feeling down, even basic tasks can feel overwhelming. That's okay. Try setting a tiny goal - maybe just getting up to stretch.",
                "It's brave to acknowledge these feelings. Consider writing down 3 things, no matter how small, that you managed to do today."
            ]
        return random.choice(responses)
    
    # Enhanced study and academic pressure responses
    elif any(word in user_input_lower for word in ["study", "exam", "grades", "academic", "pressure", "failing", "fail", "failed", "test", "class", "professor"]):
        if any(word in user_input_lower for word in ["failing", "fail", "failed", "terrible", "awful"]):
            responses = [
                "I hear that you're really worried about your academic performance. One grade or exam doesn't define your intelligence or worth. Let's focus on what you can learn from this.",
                "Academic setbacks feel devastating, but they're not permanent. Many successful people have failed exams. What specific subject is giving you trouble?",
                "It sounds like you're carrying a heavy academic burden. Your worth extends far beyond your grades. Have you considered talking to your professors?"
            ]
        else:
            responses = [
                f"Academic pressure can be intense. Let me ask - how many hours are you currently studying per day? Many students find that {random.choice(['6-8 hours with breaks', 'focused 25-minute sessions'])} works better.",
                "Try the Pomodoro technique: Study for 25 minutes, then take a 5-minute break. This maintains focus while preventing burnout. What subject are you working on?",
                "Balance is crucial for sustainable academic success. Are you getting enough sleep, nutrition, and social time?"
            ]
        return random.choice(responses)
    
    # Enhanced stress and anxiety responses
    elif any(word in user_input_lower for word in ["stressed", "anxious", "worry", "overwhelmed", "panic", "nervous", "scared"]):
        stress_level = "high" if any(word in user_input_lower for word in ["overwhelmed", "panic", "can't", "scared"]) else "moderate"
        
        if stress_level == "high":
            responses = [
                "I hear that you're feeling really overwhelmed right now. Let's try something immediate: Take 5 deep breaths - inhale for 4 counts, hold for 7, exhale for 8.",
                "It sounds like you're experiencing intense stress. Try the 5-4-3-2-1 grounding technique: Name 5 things you can see, 4 you can touch, 3 you can hear.",
                "When stress feels overwhelming, your nervous system is in overdrive. Let's reset together with some breathing exercises."
            ]
        else:
            responses = [
                "I understand you're feeling stressed. This is your body's response to pressure. Try breaking your current situation into the smallest possible step.",
                "Stress is a normal part of student life. Consider what's within your control vs. what isn't. Focus on what you can influence.",
                "Your feelings are valid. Many students experience this. Let's think about what's helped you cope with stress in the past."
            ]
        return random.choice(responses)
    
    # Enhanced sleep issues
    elif any(word in user_input_lower for word in ["sleep", "insomnia", "tired", "fatigue", "can't sleep", "awake", "exhausted"]):
        if any(word in user_input_lower for word in ["can't sleep", "awake", "all night"]):
            responses = [
                "Lying awake at night can be so frustrating. Instead of trying to force sleep, try getting up for 15 minutes and doing something calming.",
                "When your mind won't shut off at night, try writing down all your worries on paper, then physically putting the paper aside.",
                "If you've been awake for more than 20 minutes, don't stay in bed tossing and turning. Try a calming activity until you feel sleepy."
            ]
        else:
            responses = [
                "Sleep issues are so common among students. Consider your 'sleep hygiene' - avoid screens 1 hour before bed, keep your room cool.",
                "Progressive muscle relaxation can help: Tense your toes for 5 seconds, then release. Move up through your body.",
                "What time are you usually going to bed? Consistency is key for your internal body clock."
            ]
        return random.choice(responses)
    
    # Enhanced social issues
    elif any(word in user_input_lower for word in ["lonely", "friends", "social", "alone", "isolated", "no friends", "connection"]):
        if any(word in user_input_lower for word in ["no friends", "completely alone", "nobody"]):
            responses = [
                "I hear how painful loneliness feels. You're not alone in feeling this way. What's one small social interaction you could try this week?",
                "Feeling completely isolated is really hard. Consider that building connections takes time and often starts small.",
                "Your feelings of loneliness are valid and real. Many people put on a brave face while struggling inside."
            ]
        else:
            responses = [
                "Building social connections as a student can be challenging. Consider joining clubs related to your interests.",
                "Social connections often start with small steps. Try complimenting a classmate or asking someone about their weekend.",
                "Many students feel lonely even when surrounded by people. It's about meaningful connection, not just being around others."
            ]
        return random.choice(responses)
    
    # Enhanced general wellness
    elif any(word in user_input_lower for word in ["how are you", "help", "advice", "tips", "struggling"]):
        responses = [
            f"I'm here to support you! It sounds like you're looking for guidance. Since this is our {st.session_state.conversation_count}{'st' if st.session_state.conversation_count == 1 else 'nd' if st.session_state.conversation_count == 2 else 'rd' if st.session_state.conversation_count == 3 else 'th'} conversation, I want you to know that reaching out shows strength.",
            "Thank you for trusting me with your concerns. Student life brings unique challenges, and you're not alone in facing them.",
            "I appreciate you opening up. Mental health is just as important as physical health. What feels most pressing for you today?"
        ]
        return random.choice(responses)
    
    # Dynamic responses for other inputs
    else:
        # Check if it's a question
        if "?" in user_input or any(word in user_input_lower for word in ["what", "why", "how", "when", "where"]):
            responses = [
                "That's a thoughtful question. While I'm not a mental health professional, I can share some general insights and resources.",
                "I appreciate you asking that. Mental health and academic success are interconnected. What specific challenge are you facing?",
                "That's an important question. Many students wonder about similar things. What's your specific concern?"
            ]
        else:
            responses = [
                f"Thank you for sharing that with me. It takes courage to express yourself. Based on what you've mentioned, I'm here to listen. What would be most helpful to discuss?",
                "I hear you. Student life comes with many ups and downs, and your feelings are completely valid. I'm here to support you.",
                f"This is our {st.session_state.conversation_count}{'st' if st.session_state.conversation_count == 1 else 'nd' if st.session_state.conversation_count == 2 else 'rd' if st.session_state.conversation_count == 3 else 'th'} conversation, and I want you to know I'm here to support you."
            ]
        return random.choice(responses)

st.markdown('<h1 class="main-header">🎓 Student Wellness Hub</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #7f8c8d; font-size: 1.1rem; margin-bottom: 2rem;">Your comprehensive platform for academic success and mental well-being</p>', unsafe_allow_html=True)

# Enhanced sidebar navigation
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Choose a feature:",
        ("📊 Predict CGPA", "🧠 Mental Health Chatbot"),
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    
    # Load data for stats
    try:
        df = load_dataset("data/Education_Dataset.xlsx")
        if 'CGPA' in df.columns:
            avg_cgpa = df['CGPA'].mean()
            st.metric("Average CGPA", f"{avg_cgpa:.2f}")
        
        total_students = len(df)
        st.metric("Total Students", total_students)
        
        if 'Gender' in df.columns:
            gender_dist = df['Gender'].value_counts()
            st.metric("Gender Distribution", f"{len(gender_dist)} categories")
    except:
        st.warning("Data not available for stats")
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("💤 **Sleep Tip**: Aim for 7-9 hours of sleep per night for optimal academic performance!")
    
    st.markdown("---")
    st.markdown("### 📞 Need Help?")
    st.warning("🚨 **Crisis Hotline**: 988 (US)")

# Load data and artifacts
DATA_PATH = "data/Education_Dataset.xlsx"
df = load_dataset(DATA_PATH)
model, scaler = load_artifacts()
training_cols = get_training_columns()

if page == "📊 Predict CGPA":
    st.markdown('<h2 class="section-header">📊 Academic Performance Predictor</h2>', unsafe_allow_html=True)
    
    if model is None or scaler is None:
        st.markdown('<div class="resource-card">⚠️ **Model Not Available**: Please train the model first using main.py (option 1) or the notebook.</div>', unsafe_allow_html=True)
    else:
        # Add some context about the prediction
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<div class="resource-card">📚 **About This Tool**: This predictor uses machine learning to estimate your CGPA based on various factors including study habits, lifestyle, and demographic information.</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">🎯 **Accuracy**: ~85-90%</div>', unsafe_allow_html=True)
        
        st.markdown("### 📝 Enter Your Information")
        
        # Build input form dynamically from dataset columns
        work = df.copy()
        if "Student_ID" in work.columns:
            work = work.drop("Student_ID", axis=1)

        # Identify categorical columns we expect
        cat_cols = [
            "Gender", "Course", "Living_Type", "Club_Participation", "Counseling_Access"
        ]
        present_cat = [c for c in cat_cols if c in work.columns]

        # Split numeric/categorical for UI
        numeric_cols = [c for c in work.select_dtypes(include=[np.number]).columns if c != "CGPA"]
        cat_ui = {}
        num_ui = {}

        with st.form("input_form"):
            st.markdown("#### 🎯 Personal Information")
            cols = st.columns(2)
            
            # Categorical inputs first
            for i, col in enumerate(present_cat):
                with cols[i % 2]:
                    options = sorted([x for x in work[col].dropna().unique()])
                    default_opt = options[0] if options else "Unknown"
                    cat_ui[col] = st.selectbox(f"👤 {col}", options=options if options else [default_opt])
            
            st.markdown("#### 📊 Academic & Lifestyle Factors")
            # Numeric inputs
            for i, col in enumerate(numeric_cols):
                with cols[i % 2]:
                    default_val = float(work[col].dropna().median()) if work[col].notna().any() else 0.0
                    num_ui[col] = st.number_input(f"🔢 {col}", value=default_val, format="%.2f")

            submitted = st.form_submit_button("🚀 Predict My CGPA", use_container_width=True)

        if submitted:
            # Build single-row DataFrame
            row = {**num_ui, **cat_ui}
            new_data = pd.DataFrame([row])

            # Impute missing and encode
            for col in new_data.columns:
                if new_data[col].dtype == "object":
                    if new_data[col].isna().any():
                        new_data[col] = new_data[col].fillna(new_data[col].mode().iloc[0])
                else:
                    if new_data[col].isna().any():
                        new_data[col] = new_data[col].fillna(new_data[col].mean())

            new_data = pd.get_dummies(new_data, columns=present_cat, drop_first=True)

            # Align to training columns
            if training_cols is None:
                st.error("Unable to resolve training feature columns. Please retrain the model.")
            else:
                for col in training_cols:
                    if col not in new_data.columns:
                        new_data[col] = 0
                new_data = new_data[training_cols]

                # Scale and predict
                X_scaled = scaler.transform(new_data)
                pred = model.predict(X_scaled)[0]
                
                # Display result with enhanced styling
                st.markdown(f'<div class="prediction-result">🎓 Predicted CGPA: {pred:.2f}</div>', unsafe_allow_html=True)
                
                # Add recommendations based on CGPA
                if pred >= 3.5:
                    st.markdown('<div class="resource-card">🌟 **Excellent!** Keep up the great work! Consider mentoring other students.</div>', unsafe_allow_html=True)
                elif pred >= 3.0:
                    st.markdown('<div class="resource-card">👍 **Good Job!** You\'re doing well. Focus on consistency and continue your current study habits.</div>', unsafe_allow_html=True)
                elif pred >= 2.5:
                    st.markdown('<div class="resource-card">📈 **Room for Improvement**: Consider forming study groups and meeting with professors during office hours.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="resource-card">💪 **Support Available**: Consider academic counseling, tutoring services, and time management workshops.</div>', unsafe_allow_html=True)

elif page == "🧠 Mental Health Chatbot":
    st.markdown('<h2 class="section-header">🧠 Mental Health Support Chatbot</h2>', unsafe_allow_html=True)
    
    # Welcome message
    st.markdown('<div class="resource-card">💬 **Welcome!** I\'m here to provide support and resources for your mental well-being. Remember, taking care of your mental health is just as important as your academic success.</div>', unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your mental health support assistant. How are you feeling today? Remember, I'm here to listen and provide support. 💙"}
        ]
    
    # Display chat messages
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧠" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("How are you feeling today? 💭"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Generate bot response
        with st.chat_message("assistant", avatar="🧠"):
            response = get_mental_health_response(prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Enhanced resources section
    st.markdown("### 📚 Mental Health Resources")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="resource-card">🧘‍♀️ **Quick Tips**<br><br>• Take regular breaks<br>• Practice deep breathing<br>• Maintain sleep schedule<br>• Stay hydrated<br>• Connect with others</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="resource-card">⚠️ **When to Seek Help**<br><br>• Overwhelmed >2 weeks<br>• Sleep/appetite changes<br>• Self-harm thoughts<br>• Can\'t focus daily<br>• Anxious/depressed often</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="resource-card">📱 **Apps & Tools**<br><br>• Headspace (meditation)<br>• Calm (sleep/stress)<br>• Moodpath (tracking)<br>• Talkspace (therapy)<br>• 7 Cups (listening)</div>', unsafe_allow_html=True)
    
    # Emergency contacts section
    st.markdown("### 🚨 Crisis Support")
    st.markdown('<div class="emergency-card">⚡ **If you\'re in crisis, please contact immediately:**<br><br>• **Emergency Services**: 📞 911<br>• **Crisis Hotline**: 📞 988 (US)<br>• **School Counseling**: 🏫 Check your school website<br>• **Trusted Contact**: 👥 Reach out to family/friends<br><br>🌟 You are not alone. Help is available 24/7.</div>', unsafe_allow_html=True)
    
    # Mood tracker
    st.markdown("### 📊 Daily Mood Tracker")
    mood_col1, mood_col2, mood_col3 = st.columns(3)
    
    with mood_col1:
        if st.button("😊 Good", use_container_width=True):
            st.session_state.mood = "Good"
            st.success("Great to hear you're feeling good! Keep it up! 🌟")
    
    with mood_col2:
        if st.button("😐 Okay", use_container_width=True):
            st.session_state.mood = "Okay"
            st.info("It's okay to have okay days. Be gentle with yourself. 💙")
    
    with mood_col3:
        if st.button("😔 Struggling", use_container_width=True):
            st.session_state.mood = "Struggling"
            st.warning("I'm sorry you're struggling. Consider reaching out to someone you trust. You're not alone. 🤗")
    
    # Show current mood if set
    if "mood" in st.session_state:
        st.markdown(f'<div class="metric-card">Today\'s Mood: {st.session_state.mood}</div>', unsafe_allow_html=True)
