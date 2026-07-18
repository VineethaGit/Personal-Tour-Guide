import io
import json
import os
import re
import time
from io import BytesIO

import google.generativeai as genai
import nltk
import numpy as np
import requests
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
from gtts import gTTS
from langchain.agents import AgentType, initialize_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from PIL import Image

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

MODEL_NAME = "gemini-flash-latest"


# ---------------------------------------------------------------------------
# Shared tools (used by the chatbot's ReAct agents)
# ---------------------------------------------------------------------------

def search_google(query: str) -> str:
    """Search Google using SerpAPI."""
    serp = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
    return serp.run(query)


def _fetch_openweather(params: dict) -> dict | None:
    """Call OpenWeatherMap's current-weather endpoint with either a `q` (city) or `lat`/`lon` param."""
    if not OPENWEATHER_API_KEY:
        return None
    params = {**params, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    response = requests.get("https://api.openweathermap.org/data/2.5/weather", params=params, timeout=15).json()
    return response if response.get("main") else None


def get_weather_by_city(city: str) -> str:
    """Get current weather for a city as a human-readable summary."""
    if not OPENWEATHER_API_KEY:
        return "Weather lookup is unavailable: OPENWEATHER_API_KEY is not configured."
    try:
        response = _fetch_openweather({"q": city})
        if response:
            main, weather, wind = response["main"], response["weather"][0], response["wind"]
            return (
                f"Weather in {city}:\n"
                f"- Condition: {weather['description'].capitalize()}\n"
                f"- Temperature: {main['temp']}°C\n"
                f"- Feels Like: {main['feels_like']}°C\n"
                f"- Humidity: {main['humidity']}%\n"
                f"- Wind Speed: {wind.get('speed', 'N/A')} m/s"
            )
        return "Weather data not available."
    except Exception as e:
        return f"Weather data not available: {e}"


def get_weather_summary_dict(city: str) -> dict:
    """Get current weather for a city as a structured dict (used by the tour planner)."""
    if not OPENWEATHER_API_KEY:
        return {"description": "Weather lookup unavailable (OPENWEATHER_API_KEY not configured)."}
    try:
        response = _fetch_openweather({"q": city})
        if response:
            main, weather, wind = response["main"], response["weather"][0], response["wind"]
            return {
                "description": weather["description"],
                "temperature": main["temp"],
                "humidity": main["humidity"],
                "wind": wind.get("speed", "N/A"),
            }
        return {"description": "Weather data not available."}
    except Exception as e:
        return {"description": f"Weather data not available: {e}"}


def get_weather_by_coords(lat: float, lon: float) -> dict:
    """Get current weather for coordinates using OpenWeatherMap (used by the image module)."""
    if not OPENWEATHER_API_KEY:
        return {"summary": "Weather lookup is unavailable: OPENWEATHER_API_KEY is not configured.", "details": None}
    try:
        response = _fetch_openweather({"lat": lat, "lon": lon})
        if response:
            temp = response["main"]["temp"]
            desc = response["weather"][0]["description"]
            humidity = response["main"]["humidity"]
            wind_speed = response["wind"]["speed"]
            return {
                "summary": f"{desc.capitalize()} with temperature of {temp}°C",
                "details": {
                    "temperature": temp,
                    "description": desc,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                },
            }
        return {"summary": "Weather data not available.", "details": None}
    except Exception as e:
        return {"summary": f"Weather data not available: {e}", "details": None}


web_search = Tool(
    name="Web Search",
    func=search_google,
    description="A tool to search the web using Google. Input should be a string like 'What is the capital of France?'.",
)

weather_search = Tool(
    name="Weather Search",
    func=get_weather_by_city,
    description="A tool to get the weather of a heritage site or any city. Input should be a string like 'how is the weather in Paris?'.",
)

tools = [web_search, weather_search]


def _clean_json_like(text: str) -> str:
    return re.sub(r"^```json|```$", "", text).strip()


def _invoke_with_retry(fn, *args, max_retries: int = 3, base_delay: float = 20, **kwargs):
    """Call `fn` (an LLM/agent invoke) and retry on free-tier 429s with backoff."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except ResourceExhausted:
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (attempt + 1))


def _new_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GOOGLE_API_KEY)


def _new_react_agent(verbose: bool = False):
    llm = _new_llm()
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=verbose,
        handle_parsing_errors=True,
    )
    return llm, agent


# ---------------------------------------------------------------------------
# Module 1: Chatbot — categorizer + specialist subagents
# ---------------------------------------------------------------------------

CHAT_CATEGORIES = [
    "Historical Information",
    "Architectural Details",
    "Travel or Logistics",
    "Accommodation and Dining",
    "General Conversation",
    "Weather Information",
    "Unrecognized",
]


class CategorizerAgent:
    def __init__(self):
        self.llm, self.agent = _new_react_agent(verbose=False)

    def categorize_topic(self, text: str) -> str:
        prompt = f"""
        You are the Categorizer Agent, a specialized AI component responsible for analyzing the user's input and classifying it into one of the predefined categories based strictly on the content and intent of the message.

        Your job is to accurately detect the user's intent and return only the matching category label from the list below. You should not perform any processing or delegation yourself — your role is limited to categorization only.

        Available Categories (choose exactly one):
        - "Historical Information" - if the user is asking about the history of a place, monument, or heritage site.
        - "Architectural Details" - if the user is interested in the design, structure, style, or architecture of a place.
        - "Travel or Logistics" - if the user is asking how to reach a place, travel duration, entry fee, timings, routes, etc.
        - "Accommodation and Dining" - if the user is seeking places to stay, eat, nearby attractions, or leisure activities.
        - "General Conversation" - if the user is making casual remarks, greetings, jokes, or off-topic chit-chat.
        - "Weather Information" - if the user is asking about the weather of a place.
        - "Unrecognized" - if the input does not fit into any of the above categories.

        Output format: return only a single line of text with just the category label, nothing else. Do not explain or comment on your decision. Do not repeat the input.

        User input: {text}
        """
        response = _invoke_with_retry(self.llm.invoke, prompt).content.strip()
        return _clean_json_like(response)


class HistoryExpertAgent:
    def __init__(self):
        self.llm, self.agent = _new_react_agent(verbose=False)

    def history_info(self, text: str) -> str:
        prompt = f"""
        You are the History Expert Subagent, responsible for providing detailed, accurate, and insightful historical context for any heritage site, landmark, or culturally significant location the user inquires about.

        Your responsibilities:
        - Provide verified historical facts (e.g., date of construction, founder, purpose, major events).
        - Highlight the cultural and political significance of the location.
        - Mention any legends, folklore, or myths associated with the place, if applicable.
        - Be clear, concise, and structured — no fluff or filler.

        User input: {text}

        Return your output in this format:
        Name of Site:
        Historical Overview:
        Timeline of Key Events:
        Interesting Facts or Stories:
        Source Reliability: High / Medium / Low
        """
        response = _invoke_with_retry(self.llm.invoke, prompt).content.strip()
        return _clean_json_like(response)


class ArchitecturalExpertAgent:
    def __init__(self):
        self.llm, self.agent = _new_react_agent(verbose=False)

    def architecture_info(self, text: str) -> str:
        prompt = f"""
        You are the Architectural Expert Subagent, responsible for analyzing and explaining the structural, stylistic, and artistic aspects of a monument, temple, fort, or other built heritage.

        Your responsibilities:
        - Identify the architectural style (e.g., Mughal, Dravidian, Gothic, Colonial).
        - Mention the materials, structural techniques, and artistic features used.
        - Point out any symbolic design elements or layout significance.
        - Provide comparisons with similar structures if relevant.

        User input: {text}

        Return your output in this format:
        Name of Site:
        Architectural Style:
        Materials Used:
        Structural Highlights:
        Artistic Elements (e.g., carvings, frescoes, motifs):
        Special Features / Innovations:
        """
        response = _invoke_with_retry(self.llm.invoke, prompt).content.strip()
        return _clean_json_like(response)


class WeatherForecasterAgent:
    def __init__(self):
        self.llm, self.agent = _new_react_agent(verbose=False)

    def weather_info(self, text: str) -> str:
        location_prompt = f"""
        You are a Weather Location Extractor Subagent. Your job is to extract the name of the city or place the user is asking about in their query.

        Only return the name of the city or place - nothing else. The place name should be suitable to pass into a weather API.

        User input: {text}

        Extracted location:
        """
        location = _invoke_with_retry(self.llm.invoke, location_prompt).content.strip()
        location = _clean_json_like(location)

        raw_weather = get_weather_by_city(location)
        summary_prompt = f"""
        You are a weather forecaster. Present the following weather details in a clear, friendly, easy-to-read way for a traveler:
        {raw_weather}
        """
        return _invoke_with_retry(self.llm.invoke, summary_prompt).content.strip()


class TravelLogisticsAgent:
    def __init__(self):
        self.llm, self.agent = _new_react_agent(verbose=False)

    def travel_info(self, text: str) -> str:
        prompt = f"""
        You are the Travel Logistics Subagent, tasked with providing up-to-date and practical information on how to reach a specific tourist location.

        Your responsibilities:
        - Provide recommended modes of transport (train, bus, flight, taxi).
        - Mention nearest transport hubs (airport, railway station).
        - Approximate travel time and cost from common points (e.g., major cities).
        - Include entry fees, opening/closing times, and best visiting seasons.
        - Mention accessibility (elderly-friendly, wheelchair access, etc.) if applicable.

        User input: {text}

        Return your output in this format:
        Location:
        Nearest Airport/Station:
        Travel Options:
        Travel Duration & Cost Estimates:
        Entry Fee & Timings:
        Best Time to Visit:
        Accessibility Notes:
        """
        response = _invoke_with_retry(self.llm.invoke, prompt).content.strip()
        return _clean_json_like(response)


class AccommodationDiningAgent:
    def __init__(self):
        self.llm, self.agent = _new_react_agent(verbose=False)

    def accommodation_info(self, text: str) -> str:
        prompt = f"""
        You are the Accommodation and Dining Expert Subagent, responsible for suggesting where to stay, what to eat, and what else to explore nearby.

        Your responsibilities:
        - Recommend accommodations across budget ranges (luxury, mid-range, budget).
        - Suggest authentic or popular local dining spots.
        - Recommend nearby attractions and activities for tourists.
        - Mention safety tips and local etiquette if relevant.

        User input: {text}

        Return your output in this format:
        Location:
        Top Accommodation Picks:
        - Luxury:
        - Mid-range:
        - Budget:
        Recommended Eateries:
        - Local Cuisine:
        - Vegetarian/Vegan Options:
        Nearby Attractions/Activities:
        Safety Tips & Local Etiquette:
        """
        response = _invoke_with_retry(self.llm.invoke, prompt).content.strip()
        return _clean_json_like(response)


class ConversationalGuideAgent:
    def __init__(self):
        self.llm, self.agent = _new_react_agent(verbose=False)

    def convo_info(self, text: str) -> str:
        prompt = f"""
        You are a Conversational Tour Guide Subagent. Your role is to engage in friendly, casual conversation with users as if you're accompanying them on a relaxed tour.

        Your tone is warm, conversational, and attentive. You don't wait for commands or tasks; instead, you respond naturally to whatever the user says - like a human guide would during small talk.

        If the user shares a feeling, observation, or random thought, respond with something related, thoughtful, or playful. You are not here to perform tasks, give definitions, or answer deep factual queries - just keep the energy light, interesting, and human.

        Keep responses short and context-aware. Ask casual follow-up questions if it feels right. Never prompt the user to "ask a question." Just go with the flow.

        User input: {text}
        """
        response = _invoke_with_retry(self.llm.invoke, prompt).content.strip()
        return _clean_json_like(response)


def route_query(topic: str) -> dict:
    """Categorize a query and dispatch it to the matching specialist subagent."""
    category = CategorizerAgent().categorize_topic(topic)

    if category == "Historical Information":
        answer = HistoryExpertAgent().history_info(topic)
    elif category == "Architectural Details":
        answer = ArchitecturalExpertAgent().architecture_info(topic)
    elif category == "Weather Information":
        answer = WeatherForecasterAgent().weather_info(topic)
    elif category == "Travel or Logistics":
        answer = TravelLogisticsAgent().travel_info(topic)
    elif category == "Accommodation and Dining":
        answer = AccommodationDiningAgent().accommodation_info(topic)
    else:
        category = category if category in CHAT_CATEGORIES else "Unrecognized"
        answer = ConversationalGuideAgent().convo_info(topic)

    return {"category": category, "answer": answer}


# ---------------------------------------------------------------------------
# Module 2: Image-to-Insight — Gemini Vision landmark analysis
# ---------------------------------------------------------------------------

def describe_image_with_gemini(image_bytes: bytes) -> dict:
    """Identify a landmark from a photo and return structured details."""
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    image = Image.open(io.BytesIO(image_bytes))

    prompt = """
    You are a landmark detection assistant.
    Given an image, identify the landmark, its full name, city, and country.
    Return your answer in the following JSON format:

    {
        "landmark": "Eiffel Tower",
        "city": "Paris",
        "country": "France",
        "description": "A wrought iron lattice tower built on the Champ de Mars in Paris.",
        "coordinates": [48.8584, 2.2945]
    }

    For the "description" field, give a comprehensive overview covering its historical background,
    cultural importance, geographical features, famous attractions, and why it's worth visiting,
    in max 3 paragraphs. Provide the description in markdown.
    If you're unsure, write "Unknown" for the relevant fields.
    """

    try:
        response = model.generate_content([prompt, image])
        match = re.search(r"\{.*?\}", response.text, re.DOTALL)
        if not match:
            raise ValueError("Gemini response was not valid JSON.")
        output = json.loads(match.group())

        coords = output.get("coordinates")
        if coords and isinstance(coords, list) and len(coords) == 2:
            coords = tuple(coords)
        else:
            coords = None

        return {
            "landmark": output.get("landmark", "Unknown"),
            "city": output.get("city", "Unknown"),
            "country": output.get("country", "Unknown"),
            "description": output.get("description", "No description provided."),
            "coordinates": coords,
        }
    except Exception as e:
        return {
            "landmark": "Unknown",
            "city": "Unknown",
            "country": "Unknown",
            "description": f"Analysis failed: {e}",
            "coordinates": None,
        }


def text_to_speech(text: str) -> BytesIO:
    tts = gTTS(text)
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp


# ---------------------------------------------------------------------------
# Module 3: Personalized Tour Planner — Wikipedia + FAISS RAG
# ---------------------------------------------------------------------------

def detect_and_translate(city: str, preferences: str, llm: ChatGoogleGenerativeAI) -> tuple[str, str, str]:
    """Detect the input language once and translate both city and preferences to English in a single call."""
    prompt = f"""Detect the language the following two user inputs are written in (assume both are in the
    same language), and translate each into English.

    Respond with strict JSON only, no markdown, in exactly this shape:
    {{"language": "<detected language name>", "city": "<city translated to English>", "preferences": "<preferences translated to English>"}}

    City: {city}
    Preferences: {preferences}
    """
    response = _invoke_with_retry(llm.invoke, prompt).content.strip()
    response = _clean_json_like(response)
    try:
        data = json.loads(response)
        return (
            data.get("language", "English") or "English",
            data.get("city", city) or city,
            data.get("preferences", preferences) or preferences,
        )
    except json.JSONDecodeError:
        return "English", city, preferences


def translate_to_language(lang: str, text: str, llm: ChatGoogleGenerativeAI) -> str:
    """Translate `text` into `lang` if it isn't already in that language."""
    prompt = f"""If the given text is not in {lang}, translate it completely into {lang} and print only the
    translated text. Otherwise, print the text as it is, with nothing else added.
    Text: {text}
    """
    return _invoke_with_retry(llm.invoke, prompt).content.strip()


def fetch_wikipedia_summary(place: str) -> str:
    title = place.replace(" ", "_")
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "titles": title,
        "explaintext": True,
        "exlimit": 1,
    }
    # Wikipedia's API rejects/throttles requests without a descriptive User-Agent.
    headers = {"User-Agent": "VirtuTrek-TourPlanner/1.0 (https://github.com/; contact: streamlit-demo)"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        pages = response.json()["query"]["pages"]
        page = next(iter(pages.values()))
        return page.get("extract", "")
    except Exception:
        return ""


def get_food_recommendations(city: str) -> str:
    """One web search + one formatting call (a ReAct agent loop here would burn several LLM calls
    on a task that only ever needs a single search)."""
    try:
        search_results = search_google(f"Top vegetarian and non-vegetarian food in {city}")
    except Exception as e:
        search_results = f"(web search unavailable: {e})"

    llm = _new_llm()
    prompt = f"""Using the following web search results about food in {city}, write a short, friendly
    summary (3-5 bullet points) of top vegetarian and non-vegetarian food recommendations.

    Search results:
    {search_results}
    """
    return _invoke_with_retry(llm.invoke, prompt).content.strip()


def _ensure_nltk_punkt() -> None:
    for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
        try:
            nltk.data.find(resource)
            return
        except LookupError:
            continue
    nltk.download("punkt_tab", quiet=True)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preferring NLTK's tokenizer but falling back to a
    regex splitter if the punkt data can't be loaded/downloaded (e.g. no network, SSL issues)."""
    try:
        _ensure_nltk_punkt()
        from nltk.tokenize import sent_tokenize

        return sent_tokenize(text)
    except Exception:
        return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def chunk_text(text: str, chunk_size: int = 5) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return [text] if text else []
    return [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]


def build_tour_index(chunks: list[str]):
    import faiss

    embedder = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    embeddings = embedder.embed_documents(chunks)
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index, embedder


def retrieve_tour_snippets(query: str, chunks: list[str], index, embedder, k: int = 5) -> str:
    k = min(k, len(chunks))
    query_vec = np.array([embedder.embed_query(query)]).astype("float32")
    _, matched_indices = index.search(query_vec, k)
    return "\n\n".join(chunks[i] for i in matched_indices[0] if 0 <= i < len(chunks))


TOUR_PLAN_PROMPT = PromptTemplate(
    input_variables=["city", "mood", "preferences", "weather", "tour", "food"],
    template="""
    You are a friendly and knowledgeable AI travel guide helping users plan a personalized tour.

    Using the following details:
    - City: {city}
    - Mood: {mood}
    - Preferences: {preferences}
    - Weather: {weather}
    - Tour Info: {tour}
    - Food: {food}

    Generate a detailed and engaging travel plan.

    Make sure to:
    1. Suggest activities that match the mood and preferences.
    2. Adapt the plan based on the weather (e.g., indoor if rainy).
    3. Include must-see sights or hidden gems from the tour info.
    4. Recommend local dishes or food experiences tailored to the user.
    5. Use a friendly and conversational tone throughout.
    6. End with an inviting summary of the experience.

    Keep the tone warm, positive, and helpful - like a local friend planning a fun day!
    """,
)


class TourPlannerAgent:
    """Orchestrates the Wikipedia + weather + food + FAISS RAG tour-planning pipeline."""

    def __init__(self):
        self.llm = _new_llm()

    def plan(self, city: str, preferences: str, mood: str) -> dict:
        # `mood` always comes from a fixed English selectbox, so it needs no translation.
        lang, translated_city, translated_preferences = detect_and_translate(city, preferences, self.llm)
        translated_mood = mood

        wiki_text = fetch_wikipedia_summary(translated_city)
        if not wiki_text:
            wiki_text = f"{translated_city} is a notable travel destination with a variety of things to see and do."

        weather_info = get_weather_summary_dict(translated_city)
        food_recs = get_food_recommendations(translated_city)

        chunks = chunk_text(wiki_text)
        index, embedder = build_tour_index(chunks)
        tour_query = f"Suggest a {translated_mood} tour in {translated_city} that includes {translated_preferences}"
        tour_snippets = retrieve_tour_snippets(tour_query, chunks, index, embedder)

        chain = TOUR_PLAN_PROMPT | self.llm
        raw_plan = _invoke_with_retry(chain.invoke, {
            "city": translated_city,
            "mood": translated_mood,
            "preferences": translated_preferences,
            "weather": str(weather_info),
            "tour": tour_snippets,
            "food": str(food_recs),
        }).content

        # Only spend a call translating back if the user didn't write in English to begin with.
        final_plan = raw_plan if lang.strip().lower() in ("english", "en") else translate_to_language(lang, raw_plan, self.llm)

        return {
            "language": lang,
            "city": translated_city,
            "weather": weather_info,
            "food": food_recs,
            "plan": final_plan,
        }
