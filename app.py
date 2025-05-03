import streamlit as st
import torch
import joblib
import pandas as pd
import numpy as np
import re
import os

# Check model files
model_path = "chatbot_model"
required_files = ["config.json", "vocab.txt"]
weight_files = ["pytorch_model.bin", "model.safetensors"]
if not all(os.path.exists(os.path.join(model_path, f)) for f in required_files) or not any(os.path.exists(os.path.join(model_path, f)) for f in weight_files):
    st.error("Chatbot model files missing in 'chatbot_model/' (requires config.json, vocab.txt, and either pytorch_model.bin or model.safetensors). Please run 'train_chatbot.py'.")
    st.stop()

# Load models and category map
@st.cache_resource
def load_chatbot_model():
    try:
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        import torch
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load chatbot model: {e}")
        st.stop()

@st.cache_resource
def load_pricing_model():
    try:
        pricing_model = joblib.load("pricing_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return pricing_model, scaler
    except Exception as e:
        st.error(f"Failed to load pricing model: {e}")
        st.stop()

@st.cache_data
def load_category_map():
    try:
        return pd.read_csv("category_map.csv", index_col="code")["category"].to_dict()
    except Exception as e:
        st.error(f"Failed to load category map: {e}")
        st.stop()

tokenizer, model = load_chatbot_model()
pricing_model, scaler = load_pricing_model()
category_map = load_category_map()

# Initialize session state for conversation
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "context" not in st.session_state:
    st.session_state.context = {
        "category": None,
        "destination": None,
        "date": None,
        "step": "initial"
    }
if "query_key" not in st.session_state:
    st.session_state.query_key = 0

# Function to extract metadata from query
def extract_metadata(query):
    destinations = ["Lagos", "Abuja", "Accra", "London", "New York", "Paris"]
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    destination = next((d for d in destinations if d.lower() in query.lower()), None)
    date = re.search(date_pattern, query)
    date = date.group(0) if date else None
    return destination, date

# Function to generate response based on category, context, and metadata
def generate_response(category, query, context):
    destination, date = extract_metadata(query)
    if destination:
        context["destination"] = destination
    if date:
        context["date"] = date

    if context["step"] == "initial":
        context["category"] = category
        if category == "price_inquiry":
            if not context["destination"]:
                context["step"] = "ask_destination"
                return "Please specify your destination city (e.g., Lagos, London)."
            if not context["date"]:
                context["step"] = "ask_date"
                return f"When are you traveling to {context['destination']}? (e.g., 2025-06-01)"
            return f"Flights to {context['destination']} on {context['date']} start at ~$200. Use the price predictor below for details!"
        elif category == "cancellation":
            if not context["destination"]:
                context["step"] = "ask_destination"
                return "Which flight would you like to cancel? Please specify the destination."
            return f"To cancel your flight to {context['destination']}, please provide your booking ID. Continue?"
        elif category == "flight_status":
            if not context["destination"]:
                context["step"] = "ask_destination"
                return "Which flight’s status would you like to check? Please specify the destination."
            return f"Checking status for flights to {context['destination']}. Please provide the flight number."
        elif category == "baggage":
            return "Baggage allowance is 23kg for economy, 32kg for business. Need details for a specific flight?"
        elif category == "loyalty":
            return "You earn 1 mile per km flown. Want to check miles for a specific destination?"
        elif category == "refunds":
            if not context["destination"]:
                context["step"] = "ask_destination"
                return "Which flight needs a refund? Please specify the destination."
            return f"Refunds for {context['destination']} flights depend on ticket type. Provide your booking ID to proceed."
        elif category == "check_in":
            if not context["destination"]:
                context["step"] = "ask_destination"
                return "Which flight are you checking in for? Please specify the destination."
            return f"Online check-in for {context['destination']} opens 24 hours before departure. Start now?"
        elif category == "special_requests":
            return "Special requests (e.g., meals, seating) can be added during booking. What do you need?"
        elif category == "booking_change":
            if not context["destination"]:
                context["step"] = "ask_destination"
                return "Which flight would you like to change? Please specify the destination."
            return f"To change your flight to {context['destination']}, please provide new travel dates."
        else:  # other
            return "I’m not sure how to help with that. Try asking about flights, bookings, or baggage, or contact support."

    elif context["step"] == "ask_destination":
        if destination:
            context["destination"] = destination
            context["step"] = "ask_date" if context["category"] in ["price_inquiry", "booking_change"] else "initial"
            if context["category"] == "price_inquiry":
                return f"When are you traveling to {destination}? (e.g., 2025-06-01)"
            elif context["category"] == "cancellation":
                return f"To cancel your flight to {destination}, please provide your booking ID. Continue?"
            elif context["category"] == "flight_status":
                return f"Checking status for flights to {destination}. Please provide the flight number."
            elif context["category"] == "refunds":
                return f"Refunds for {destination} flights depend on ticket type. Provide your booking ID."
            elif context["category"] == "check_in":
                return f"Online check-in for {destination} opens 24 hours before departure. Start now?"
            elif context["category"] == "booking_change":
                return f"To change your flight to {destination}, please provide new travel dates."
        return "Please specify a valid destination city (e.g., Lagos, London)."

    elif context["step"] == "ask_date":
        if date:
            context["date"] = date
            context["step"] = "initial"
            if context["category"] == "price_inquiry":
                return f"Flights to {context['destination']} on {date} start at ~$200. Use the price predictor below!"
            elif context["category"] == "booking_change":
                return f"Changed your flight to {context['destination']} on {date}. Confirm with booking ID."
        return "Please provide a travel date (e.g., 2025-06-01)."

# Streamlit app layout
st.title("Airline Ticket AI Prototype")
st.markdown("This prototype handles customer queries conversationally and predicts ticket prices using AI.")

# Chatbot section
st.header("Conversational Chatbot")
st.write("Ask about flights, bookings, baggage, or more. The chatbot will follow up as needed!")
chat_container = st.container()
with chat_container:
    for message in st.session_state.conversation:
        if message["role"] == "user":
            st.write(f"**You**: {message['text']}")
        else:
            st.write(f"**Bot**: {message['text']}")

# Query input form
with st.form(key="query_form"):
    query = st.text_input("Enter your query:", placeholder="e.g., What’s the cheapest flight to Lagos?", key=f"query_{st.session_state.query_key}")
    submit_button = st.form_submit_button("Send")
    if submit_button and query:
        # Add user query to conversation
        st.session_state.conversation.append({"role": "user", "text": query})
        
        # Classify query
        inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        label = torch.argmax(outputs.logits, dim=1).item()
        category = category_map[label]
        
        # Generate response with context
        response = generate_response(category, query, st.session_state.context)
        st.session_state.conversation.append({"role": "bot", "text": response})
        
        # Increment query key to clear input
        st.session_state.query_key += 1

# Reset conversation button
if st.button("Reset Conversation"):
    st.session_state.conversation = []
    st.session_state.context = {"category": None, "destination": None, "date": None, "step": "initial"}
    st.session_state.query_key = 0
    st.rerun()

# Pricing section
st.header("Ticket Price Prediction")
st.write("Enter details to predict a ticket price.")
day = st.selectbox("Day of Week", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
demand = st.selectbox("Demand Level", options=["Low", "High"])
distance = st.number_input("Distance (km)", min_value=100, max_value=5000, value=500, step=100)
if st.button("Predict Price"):
    day_idx = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day)
    demand_encoded = 1 if demand == "High" else 0
    features = np.array([[day_idx, demand_encoded, distance]])
    scaled_features = scaler.transform(features)
    price = pricing_model.predict(scaled_features)[0]
    st.write(f"**Predicted Price**: ${price:.2f}")