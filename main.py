# main.py

import folium
import streamlit as st
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium

from agents import (
    OPENWEATHER_API_KEY,
    describe_image_with_gemini,
    get_weather_by_coords,
    route_query,
    text_to_speech,
    TourPlannerAgent,
)

load_dotenv()

st.set_page_config(page_title="VirtuTrek: AI-Powered Virtual Tour Assistant", layout="wide")

st.markdown("""
    <style>
    .stSidebar { background-color: #2c3e50; color: white; }
    .stButton { color: white; border-radius: 5px; font-weight: bold; }
    .stButton:hover { transition: background-color 0.3s ease; }
    .stButton[data-baseweb="button"] {
        background-color: #e74c3c; color: white; font-size: 16px;
        padding: 12px 30px; border-radius: 8px;
    }
    h1 { color: #e74c3c; font-size: 36px; text-align: center; }
    .stSidebar h2 { color: #ecf0f1; }
    .stMarkdown { font-size: 18px; color: #ecf0f1; line-height: 1.6; }
    .stTextInput, .stTextArea, .stMultiSelect {
        background-color: #34495e; border-radius: 5px; padding: 10px; color: white;
    }
    .stTextInput::placeholder, .stTextArea::placeholder { color: #bdc3c7; }
    .stSidebar .stMarkdown { color: #ecf0f1; }
    .stSidebar h1 { color: #ecf0f1; }
    .stButton p { color: white; }
    a { color: #3498db; }
    a:hover { color: #2980b9; }
    .stTextInput label, .stTextArea label, .stMultiSelect label { color: white; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("VirtuTrek")
st.sidebar.markdown("""
Your AI-powered virtual tour assistant. 🌍

- 💬 **AI Tour Guide** — ask about history, architecture, travel, food, or weather.
- 🖼 **Image Analysis** — upload a landmark photo for instant insights.
- 🗺 **Tour Planner** — a personalized itinerary built from your mood & interests.
""")

st.title("🌍 VirtuTrek: AI-Powered Virtual Tour Assistant")

chat_tab, image_tab, planner_tab = st.tabs(["💬 AI Tour Guide", "🖼 Image Analysis", "🗺 Tour Planner"])


# ---------------------------------------------------------------------------
# Tab 1: AI Tour Guide chatbot
# ---------------------------------------------------------------------------
with chat_tab:
    example_queries = [
        "Tell me about the history of the Taj Mahal.",
        "What architectural style is the Angkor Wat built in?",
        "How do I get to the Louvre and what are the entry fees?",
        "What's the weather like in Cairo right now?",
        "Where should I stay and eat near the Acropolis?",
        "Haha this app is pretty cool!",
    ]

    selected = st.selectbox("Try an example query", ["select a query"] + example_queries)
    default_text = selected if selected != "select a query" else ""
    topic = st.text_input("Ask something about a heritage site:", value=default_text, key="chat_topic")

    if st.button("Ask our AI Tour Guide"):
        if not topic.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Thinking..."):
                result = route_query(topic)
            st.caption(f"Category: {result['category']}")
            st.markdown(result["answer"])


# ---------------------------------------------------------------------------
# Tab 2: Image-to-Insight
# ---------------------------------------------------------------------------
with image_tab:
    st.markdown("Upload a photo of a landmark and let Gemini Vision identify it.")

    uploaded_file = st.file_uploader("Landmark photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
    user_location = st.text_input(
        "Your location (city, country) — optional, shows a route on the map",
        key="user_location",
    )

    if st.button("Analyze Image"):
        if not uploaded_file:
            st.warning("Please upload an image first.")
        else:
            bytes_data = uploaded_file.read()
            with st.spinner("Analyzing your image..."):
                analysis = describe_image_with_gemini(bytes_data)

            col1, col2 = st.columns([1, 1.4])
            with col1:
                st.image(bytes_data, use_container_width=True)
            with col2:
                st.subheader(f"🏛️ {analysis['landmark']}, {analysis['city']}, {analysis['country']}")
                st.markdown(analysis["description"])

                with st.spinner("🔊 Generating audio narration..."):
                    try:
                        audio_fp = text_to_speech(analysis["description"])
                        st.audio(audio_fp.read(), format="audio/mp3")
                    except Exception as e:
                        st.info(f"Audio narration unavailable: {e}")

            coords = analysis["coordinates"]
            if coords:
                st.subheader("🗺️ Map")
                user_coords = None
                if user_location.strip():
                    geolocator = Nominatim(user_agent="virtutrek_tour_agent")
                    location = geolocator.geocode(user_location)
                    if location:
                        user_coords = [location.latitude, location.longitude]
                    else:
                        st.info("Could not find that location — showing the landmark only.")

                map_obj = folium.Map(location=coords, zoom_start=12)
                folium.Marker(location=coords, tooltip=analysis["landmark"]).add_to(map_obj)
                if user_coords:
                    folium.Marker(
                        location=user_coords, tooltip="Your Location",
                        icon=folium.Icon(color="blue"),
                    ).add_to(map_obj)
                    folium.PolyLine([user_coords, coords], color="green", weight=2.5).add_to(map_obj)
                    map_obj.fit_bounds([user_coords, coords])
                st_folium(map_obj, width=None, height=420)

                st.subheader("🌦️ Weather at the Landmark")
                weather = get_weather_by_coords(*coords)
                st.write(weather["summary"])
                if weather["details"]:
                    st.json(weather["details"])
            else:
                st.info("No coordinates were returned for this landmark, so the map and weather are unavailable.")


# ---------------------------------------------------------------------------
# Tab 3: Personalized Tour Planner (RAG)
# ---------------------------------------------------------------------------
with planner_tab:
    st.markdown("Tell us where you're headed and how you're feeling — we'll build a tour around it.")

    city = st.text_input("Which city are you exploring?", key="planner_city")
    preferences = st.text_input(
        "Your interests (e.g., museums, food, history, nature)", key="planner_preferences"
    )
    mood = st.selectbox(
        "How are you feeling today?",
        ["adventurous", "relaxed", "curious", "romantic", "energetic"],
        key="planner_mood",
    )

    if st.button("Plan My Tour"):
        if not city.strip() or not preferences.strip():
            st.warning("Please fill in both the city and your interests.")
        else:
            with st.spinner("Researching your destination and building a personalized plan..."):
                plan = TourPlannerAgent().plan(city, preferences, mood)

            st.subheader(f"Your {mood} tour of {plan['city']}")
            st.markdown(plan["plan"])

            with st.expander("🌦️ Weather details"):
                st.json(plan["weather"])
            with st.expander("🍽️ Food recommendations"):
                st.markdown(plan["food"])
