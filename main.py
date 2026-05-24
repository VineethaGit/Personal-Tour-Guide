# main.py

import streamlit as st
from agents import (
    CategorizerAgent, GeneralAgent, LocationAgent,
    TimeAgent, TicketAgent, CultureInsightsAgent,
    TipsAgent, FacilitiesAgent, ExperienceAgent,
    RecommendationAgent, LanguageAgent, WriterAgent
)
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Heritage Site Explorer", layout="wide")



st.markdown("""
    <style>
    /* Sidebar Styling */
    .stSidebar {
        background-color: #2c3e50; /* Dark blue-gray for a modern look */
        color: white; /* White text for better readability */
    }

    /* Button Styling */
    .stButton {
        
        color: white; /* White text for visibility */
        border-radius: 5px; /* Slight rounded corners */
        font-weight: bold; /* Emphasize the button text */
    }

    /* Button Hover Effect */
    .stButton:hover {
        
        transition: background-color 0.3s ease; /* Smooth transition */
    }
    
    /* Submit Button Specific Styling */
    .stButton[data-baseweb="button"] {
        background-color: #e74c3c; /* Red background for the submit button */
        color: white; /* White text for visibility */
        font-size: 16px; /* Slightly larger font for emphasis */
        padding: 12px 30px; /* Adjust padding for the submit button */
        border-radius: 8px; /* More rounded corners */
    }

    /* Main Title Styling */
    h1 {
        color: #e74c3c; /* Bold red for the title */
        font-size: 36px; /* Larger font size */
        text-align: center; /* Center the title */
    }

    /* Sidebar Header Styling */
    .stSidebar h2 {
        color: #ecf0f1; /* Light gray for sidebar headers */
    }

    /* General Text Styling */
    .stMarkdown {
        font-size: 18px; /* Slightly larger text for readability */
        color: #ecf0f1; /* Light gray text for readability */
        line-height: 1.6; /* Better spacing for easier reading */
    }

    /* Input fields Styling */
    .stTextInput, .stTextArea, .stMultiSelect {
        background-color: #34495e; /* Darker background for input fields */
        border-radius: 5px; /* Slight rounded corners */
        padding: 10px; /* Add padding for better input field appearance */
        color: white; /* White text for better visibility */
    }

    /* Placeholder Text */
    .stTextInput::placeholder, .stTextArea::placeholder {
        color: #bdc3c7; /* Lighter color for placeholder text */
    }

    /* Select and Multi-Select Dropdown Styling */
    .stMultiSelect {
        background-color: #34495e; /* Darker background for multi-select */
        border-radius: 5px;
        padding: 10px;
        color: white; /* White text */
    }

    /* Sidebar Text Styling */
    .stSidebar .stMarkdown {
        color: #ecf0f1; /* Ensure visibility of text in sidebar */
    }

    /* Title in the Sidebar */
    .stSidebar h1 {
        color: #ecf0f1; /* Light gray for sidebar titles */
    }

    /* Text in the Buttons */
    .stButton p {
        color: white; /* White text inside buttons */
    }
    
    /* Make links more visible */
    a {
        color: #3498db; /* Blue for links */
    }
    a:hover {
        color: #2980b9; /* Darker blue for hover effect */
    }

    /* Ensure visibility of labels and inputs */
    .stTextInput label, .stTextArea label, .stMultiSelect label {
        color: white; /* White text for labels */
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Title and Description
st.sidebar.title("Heritage Info Assistant")
st.sidebar.markdown("""
Welcome to the Heritage Info Assistant! 🌍

This app helps you explore famous heritage sites by answering questions about:

- 📍 Location & accessibility  
- ⏰ Visiting hours  
- 🎟️ Tickets & pricing  
- 🏛️ History & cultural insights  
- 🧳 Travel tips & rules  
- 🗺️ Nearby attractions  
- 💬 Language & guides

Just type a question or click one of the examples to get started!
""")


st.title("🌍 Heritage Site Explorer")

# Title or header


# Example queries (one per category)
example_queries = [
    "Tell me about the Taj Mahal.",                             # General Information
    "Where is Angkor Wat located?",                              # Location & Accessibility
    "What are the opening hours of the Louvre?",                 # Visiting Hours & Timing
    "How much is the entry fee for the Acropolis?",              # Tickets & Pricing
    "Who built the Pyramids of Giza and why?",                   # Historical & Cultural Insights
    "What should I wear when visiting the Golden Temple?",       # Visitor Tips & Rules
    "What can I see near the Eiffel Tower?",                     # Facilities & Nearby Attractions
    "Can I get a private tour of the Red Fort?",                 # Custom Experience
    "Which is better to visit—Hampi or Badami?",                 # Comparison & Recommendations
    "What language is spoken at Hampi?"                          # Language & Culture
]


selected_query = None
query = "select a query"


selected =  st.sidebar.selectbox(query,example_queries)
if selected:
    selected_query = selected

# Input box – prefilled if user clicked a button
default_text = selected_query if selected_query else ""
topic = st.text_input("Ask something about a heritage site:", value=default_text)

# topic = st.text_input("Enter a heritage site name or query:")

if st.button("Ask our AI Tour Guide"):
    agent = CategorizerAgent()
    response = agent.categorize_topic(topic)
    category = response["category"]
    # st.json(category)
    
    if category == "General Information":
        generalizer = GeneralAgent()
        data = generalizer.general_topic(topic)
        
    elif category == "Location & Accessibility":
        generalizer = LocationAgent()
        data = generalizer.locate(topic)
        
    elif category == "Location & Accessibility":
        generalizer = LocationAgent()
        data = generalizer.locate(topic)
    
    elif category == "Visiting Hours & Timing":
        generalizer = TimeAgent()
        data = generalizer.time(topic)
        
    elif category == "Tickets & Pricing":
        generalizer = TicketAgent()
        data = generalizer.ticket(topic)
        
    elif category == "Historical & Cultural Insights":
        generalizer = CultureInsightsAgent()
        data = generalizer.culture(topic)
    
    elif category == "Visitor Tips & Rules":
        generalizer = TipsAgent()
        data = generalizer.tips(topic)
    
    elif category == "Facilities & Nearby Attractions":
        generalizer = FacilitiesAgent() 
        data = generalizer.facility(topic)
        
    elif category == "Custom Experience":
        generalizer = ExperienceAgent()
        data = generalizer.experience(topic)
        
    elif category == "Comparison & Recommendations":
        generalizer = RecommendationAgent()
        data = generalizer.recommend(topic)
        
    elif category == "Language & Culture":
        generalizer = LanguageAgent()
        data = generalizer.language(topic)

    writer = WriterAgent()
    article = writer.write_article(data)
    
    st.markdown(article)
