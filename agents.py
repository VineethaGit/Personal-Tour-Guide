from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Agent
from langchain.tools import Tool
from langchain.agents import AgentExecutor,ZeroShotAgent
from langchain_community.utilities import SerpAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
import requests
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
serp_api_key = os.getenv("SERPAPI_API_KEY")


def simple_calculator(x: str) -> str:
    """A simple calculator that can do basic math operations."""
    try:
        result = eval(x)
        return str(result)
    except Exception as e:
        return str(e)
    
def search_google(query: str) -> str:
    """Search Google using SerpAPI."""
    serp = SerpAPIWrapper()
    results = serp.run(query)
    return results


calculator = Tool(
    name = "Calculator",
    func=simple_calculator,
    description="A simple calculator that can do basic math operations. Input should be a string like '2 + 2'.",
)

web_search = Tool(
    name = "Web Search",
    func=search_google,
    description="A tool to search the web using Google. Input should be a string like 'What is the capital of France?'.",
)

tools = [calculator, web_search]


class CategorizerAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.tools = tools
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def categorize_topic(self, topic):
        prompt = f"""
        You are a Categorizer AI Agent that receives natural language queries from users about heritage or historical sites.

        Your task is to extract structured metadata from the query and return a **strictly valid JSON object** in the following format:

        {{
        "category": "<One of: General Information, Location & Accessibility, Visiting Hours & Timing, Tickets & Pricing, Historical & Cultural Insights, Visitor Tips & Rules, Facilities & Nearby Attractions, Custom Experience, Comparison & Recommendations, Language & Culture>",
        "site": "<The name of the heritage site mentioned, if any. If none specified, write 'Unknown'>",
        "intent": "<A short natural language phrase explaining what the user wants to know or achieve>",
        "question_type": "<One of: fact, opinion, recommendation, instruction, comparison, clarification>"
        }}

        Instructions:
        - Use only one category per query.
        - Focus only on heritage/tourist/historical-related topics.
        - Keep your output strictly in raw JSON (no markdown, no code block).
        - Do not explain or narrate anything outside the JSON object.
        - If the site is not mentioned, set "site" as "Unknown".

        Now, categorize the following user query:
        "{topic}"
        """
        response = self.llm.invoke(prompt).content.strip()
        
        # Clean markdown-style formatting (if any)
        response = re.sub(r"^```json|```$", "", response).strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format returned", "raw_response": response}
        
class GeneralAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.tools = tools
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True
        )

    def general_topic(self, topic):
        prompt = f"""You are an expert Research Agent specialized in gathering GENERAL INFORMATION about heritage sites across the world.

                      Your task is to search the web and extract clear, concise, and accurate information about a given heritage site and return only FACTUAL details in a structured JSON format. You DO NOT narrate, assume, or summarize creatively. You also DO NOT include opinion, user reviews, or travel blog content.

                      # INPUT:
                      {topic}

                      # TASKS:
                      - Retrieve factual information from credible sources (e.g., UNESCO, official tourism boards, government sites, academic resources).
                      - Extract key general facts including:
                        - Full Name of Site
                        - Country and City/Location
                        - Year of Establishment or Recognition
                        - Who built it or founded it (if applicable)
                        - Historical Significance
                        - Cultural Importance
                        - UNESCO World Heritage status (yes/no and year)
                        - Official Website (if available)

                      # OUTPUT FORMAT (Strict JSON):
                      {{
                        "site": "site",
                        "location": {{
                          "country": "...",
                          "city_or_region": "..."
                        }},
                        "established_year": "...",
                        "founded_by": "...",
                        "historical_significance": "...",
                        "cultural_importance": "...",
                        "unesco_status": {{
                          "is_unesco_site": true,
                          "designation_year": "..."
                        }},
                        "official_website": "..."
                      }}

                      # RULES:
                      - DO NOT include unrelated content, tips, travel advice, or opinions.
                      - NEVER make up facts. If data is missing, write `"unknown"` or `null`.
                      - Avoid promotional or subjective content.
                      - Only output the final structured JSON, no extra text.

                      Begin researching and return the structured general information for the site: **site**
                      """
        response = self.agent.run(prompt)
        return response
    
class LocationAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.tools = tools
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True
        )

    def locate(self, topic):
        prompt = f"""You are a Research Agent specialized in retrieving precise and factual LOCATION & ACCESSIBILITY information about global heritage sites.

                      Your job is to query the web and extract details that help a visitor understand where the heritage site is located and how to reach it. You will output the data in a strictly structured JSON format.

                      # INPUT:
                      {topic}

                      # TASKS:
                      Search and extract the following:
                      - Country and State/Region where the site is located
                      - Nearest Major City or Airport
                      - Popular modes of transportation to the site (road, train, air, etc.)
                      - Accessibility status (wheelchair accessible, senior-friendly, etc.)
                      - Distance from the nearest major city (if available)
                      - Common travel routes or transit hubs (e.g., rail station, bus terminals)
                      - Geo-coordinates (latitude and longitude) of the site

                      # OUTPUT FORMAT (Strict JSON):
                      {{
                        "site": "site",
                        "location": {{
                          "country": "...",
                          "state_or_region": "...",
                          "nearest_major_city": "...",
                          "distance_from_city_km": "...",
                          "geo_coordinates": {{
                            "latitude": "...",
                            "longitude": "..."
                          }}
                        }},
                        "transportation": {{
                          "available_modes": ["road", "rail", "air"],
                          "nearest_airport": "...",
                          "nearest_rail_station": "...",
                          "common_routes": "..."
                        }},
                        "accessibility": {{
                          "wheelchair_accessible": true,
                          "senior_friendly": true,
                          "note": "..."
                        }}
                      }}

                      # RULES:
                      - Use ONLY factual info from reliable sources (official tourism boards, Google Maps, transportation sites).
                      - DO NOT add opinions, travel tips, or promotional content.
                      - DO NOT speculate—if a data point is unavailable, use `"unknown"` or `null`.
                      - Only output the final structured JSON, nothing else.

                      Begin researching and return structured location & accessibility data for: **site**

                      """
        response = self.agent.run(prompt)
        return response
    

class TimeAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.tools = tools
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True
        )

    def time(self, topic):
        prompt = f"""You are a Research Agent specialized in retrieving precise and factual VISITING HOURS & TIMING information about global heritage sites.

                      Your job is to query the web and extract structured information to help travelers know when they can visit the site. You will output the data in a strictly structured JSON format.

                      # INPUT:
                      {topic}


                      # TASKS:
                      Search and extract the following:
                      - Opening days (e.g., Monday to Sunday, weekdays only, etc.)
                      - Opening and closing times for each day (include seasonal variations if any)
                      - Last entry time (if applicable)
                      - Holidays or closed days (e.g., public holidays, maintenance days)
                      - Time zone of the site
                      - Duration of an average visit (if available)
                      - Special night entry or evening programs (if applicable)

                      # OUTPUT FORMAT (Strict JSON):
                      {{
                        "site": "site",
                        "timing": {{
                          "time_zone": "...",
                          "weekly_schedule": {{
                            "monday": {{ "open": "...", "close": "..." }},
                            "tuesday": {{ "open": "...", "close": "..." }},
                            "wednesday": {{ "open": "...", "close": "..." }},
                            "thursday": {{ "open": "...", "close": "..." }},
                            "friday": {{ "open": "...", "close": "..." }},
                            "saturday": {{ "open": "...", "close": "..." }},
                            "sunday": {{ "open": "...", "close": "..." }}
                          }},
                          "last_entry_time": "...",
                          "closed_on": ["..."],
                          "special_events": {{
                            "night_entry_available": "true",
                            "description": "..."
                          }},
                          "average_visit_duration": "..."
                        }}
                      }}

                      # RULES:
                      - Use ONLY factual info from official tourism websites or the official site page.
                      - DO NOT include tips, travel suggestions, or opinions.
                      - DO NOT speculate—if data is missing, use `"unknown"` or `null`.
                      - Output ONLY the structured JSON response, nothing else.

                      Begin researching and return structured visiting hours & timing data for: **site**

                    """
        response = self.agent.run(prompt)
        return response
    
class TicketAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.tools = tools
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True
        )

    def ticket(self, topic):
        prompt = f"""You are a Research Agent specialized in retrieving precise and factual TICKETS & PRICING information about global heritage sites.

                      Your task is to query the web and collect detailed information about entry costs, booking methods, and ticketing rules for the site. Output everything in a strictly structured JSON format.

                      # INPUT:
                      {topic}
                     

                      # TASKS:
                      Search and extract the following:
                      - General entry ticket prices for adults, children, and seniors (local and foreign)
                      - Any ticket categories (e.g., guided tour, group ticket, fast track)
                      - Discounts or free entry policies (e.g., students, disabled, residents)
                      - Online booking options (website or platform)
                      - On-site purchase availability
                      - Currency used
                      - Extra charges (e.g., camera fees, parking, special exhibitions)
                      - Validity duration of the ticket (e.g., same day, multi-day pass)

                      # OUTPUT FORMAT (Strict JSON):
                      {{
                        "site": "site",
                        "ticketing": {{
                          "currency": "...",
                          "pricing": {{
                            "local_adult": "...",
                            "local_child": "...",
                            "local_senior": "...",
                            "foreign_adult": "...",
                            "foreign_child": "...",
                            "foreign_senior": "..."
                          }},
                          "ticket_types": [
                            {{
                              "type": "General Admission",
                              "price": "...",
                              "includes": "..."
                            }},
                            {{
                              "type": "Guided Tour",
                              "price": "...",
                              "includes": "..."
                            }}
                          ],
                          "discounts": {{
                            "available_for": ["students", "disabled", "residents"],
                            "details": "..."
                          }},
                          "booking": {{
                            "online_available": true,
                            "official_website": "...",
                            "third_party_sites": ["..."],
                            "on_site_purchase": true
                          }},
                          "additional_charges": {{
                            "camera_fee": "...",
                            "parking_fee": "...",
                            "special_exhibit_fee": "..."
                          }},
                          "ticket_validity": "..."
                        }}
                      }}

                      # RULES:
                      - Pull data only from official or credible sources.
                      - Do NOT include opinions, promotions, or tips.
                      - If any info is not available, use `"unknown"` or `null`.
                      - Output ONLY the final JSON object, no extra text or explanations.

                      Begin researching and return structured ticket & pricing data for: **site**

                      """
        response = self.agent.run(prompt)
        return response
    
    
class CultureInsightsAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.tools = tools
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True
        )

    def culture(self, topic):
        prompt = f"""You are a Research Agent specialized in retrieving in-depth HISTORICAL & CULTURAL INSIGHTS about global heritage sites.

                      Your job is to extract meaningful, factual data that explains the site’s origins, cultural relevance, associated traditions, and historical events. Output the data in a strictly structured JSON format.

                      # INPUT:
                      {topic}
                    

                      # TASKS:
                      Search and extract the following:
                      - Founding history and construction timeline
                      - Historical significance (events, periods, dynasties, empires involved)
                      - Key architectural or cultural features
                      - Religious, spiritual, or ceremonial relevance
                      - Associated myths, folklore, or legends (if widely cited)
                      - UNESCO World Heritage status and the reason for designation
                      - Role in national or regional identity
                      - Notable restoration efforts or historical transitions

                      # OUTPUT FORMAT (Strict JSON):
                      {{
                        "site": "site",
                        "historical_background": {{
                          "founded_in": "...",
                          "built_by": "...",
                          "construction_period": "...",
                          "historical_events": ["..."],
                          "dynasties_or_empires": ["..."],
                          "unesco_status": {{
                            "designated": true,
                            "year": "...",
                            "reason": "..."
                          }}
                        }},
                        "cultural_significance": {{
                          "religious_importance": "...",
                          "myths_and_legends": "...",
                          "cultural_identity": "...",
                          "ceremonial_use": "...",
                          "architectural_features": ["..."]
                        }},
                        "restoration_and_conservation": {{
                          "major_restoration_years": ["..."],
                          "preservation_status": "...",
                          "governing_body": "..."
                        }}
                      }}

                      # RULES:
                      - Pull only from factual, credible sources (UNESCO, official heritage orgs, history archives).
                      - Do NOT invent or assume. If a field is not available, use `"unknown"` or `null`.
                      - Do NOT add personal interpretation or opinion.
                      - Output ONLY the final JSON object, no surrounding text or explanations.

                      Begin researching and return structured historical & cultural insight data for: **site**

                      """
        response = self.agent.run(prompt)
        return response
    
    
class TipsAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.tools = tools
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True
        )

    def tips(self, topic):
        prompt = f"""You are a Research Agent specialized in retrieving VISITOR TIPS & RULES for global heritage sites.

                        Your task is to collect practical, official, and up-to-date information that helps tourists prepare for their visit while respecting local customs and regulations. Your output must be factual and follow the structured JSON format below.

                        # INPUT:
                        {topic}


                        # TASKS:
                        Search and extract the following:
                        - General visitor guidelines or rules
                        - Dress code (if any)
                        - Photography or videography restrictions
                        - Items allowed or prohibited inside the site
                        - Conduct expectations (e.g., silence in temples, no touching artifacts)
                        - Safety advice (e.g., slippery stairs, wildlife)
                        - Peak hours to avoid / best times to visit
                        - Tips for families, elderly, or solo travelers
                        - Official warnings or restrictions due to events, restoration, etc.

                        # OUTPUT FORMAT (Strict JSON):
                        {{
                        "site": "site",
                        "rules": {{
                            "dress_code": "...",
                            "photography_allowed": true,
                            "videography_allowed": false,
                            "prohibited_items": ["..."],
                            "conduct_guidelines": ["..."]
                        }},
                        "tips": {{
                            "best_visit_times": "...",
                            "peak_hours_to_avoid": "...",
                            "safety_advice": ["..."],
                            "family_friendly": true,
                            "elderly_friendly": true,
                            "solo_travel_tips": ["..."]
                        }},
                        "notices": {{
                            "temporary_restrictions": "...",
                            "special_guidelines": "..."
                        }}
                        }}

                        # RULES:
                        - Use only verified and official sources (e.g., government tourism websites, site management authorities).
                        - Do NOT include user-generated content or personal opinions.
                        - If information is unavailable, return `"unknown"` or `null`.
                        - Output ONLY the structured JSON object—no surrounding text, summary, or explanation.

                        Begin researching and return structured visitor guidance for: **site**

                      """
        response = self.agent.run(prompt)
        return response
    
    
class FacilitiesAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.tools = tools
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True
        )

    def facility(self, topic):
        prompt = f"""You are a Research Agent specialized in retrieving factual and updated FACILITIES & NEARBY ATTRACTIONS information about global heritage sites.

                        Your goal is to help visitors understand what amenities are available on-site and what notable locations or attractions are nearby. You will return data in a strictly structured JSON format.

                        # INPUT:
                        {topic}
                        
                        # TASKS:
                        Search and extract the following:
                        - On-site facilities (e.g., restrooms, drinking water, food courts, guided tour booths, wheelchair ramps)
                        - Parking availability and details
                        - Nearest accommodations (hotels, lodges, homestays within 5–10 km)
                        - Emergency services nearby (hospitals, police station)
                        - Notable attractions within 15–20 km (temples, museums, scenic spots, parks)
                        - Visitor centers or help desks

                        # OUTPUT FORMAT (Strict JSON):
                        {{
                        "site": "site",
                        "facilities": {{
                            "restrooms": true,
                            "drinking_water": true,
                            "food_courts": true,
                            "guided_tour_services": true,
                            "wheelchair_access": true,
                            "parking_available": true,
                            "visitor_center": true
                        }},
                        "nearby_accommodations": [
                            {{
                            "name": "...",
                            "type": "hotel/homestay/lodge",
                            "distance_km": "...",
                            "contact": "..."
                            }}
                        ],
                        "emergency_services": {{
                            "nearest_hospital": "...",
                            "hospital_distance_km": "...",
                            "police_station": "...",
                            "police_distance_km": "..."
                        }},
                        "nearby_attractions": [
                            {{
                            "name": "...",
                            "type": "temple/museum/park/etc.",
                            "distance_km": "..."
                            }}
                        ]
                        }}

                        # RULES:
                        - Rely ONLY on reliable and verifiable sources such as Google Maps, official tourism websites, or local government listings.
                        - DO NOT speculate—if any information is not available, return `"unknown"` or `null`.
                        - DO NOT include suggestions, reviews, or tips—only factual data.
                        - Output ONLY the structured JSON—no explanation, summary, or prose.

                        Begin researching and return structured facilities and nearby attractions data for: **site**

                      """
        response = self.agent.run(prompt)
        return response
    

class ExperienceAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.tools = tools
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True
        )

    def experience(self, topic):
        prompt = f"""You are a Research Agent specialized in retrieving information for CUSTOM EXPERIENCE planning related to global heritage sites.

                        Your job is to extract data that helps travelers design a personalized, unique, and meaningful visit to a heritage site. Output must be in a structured JSON format.

                        # INPUT:
                        - Heritage Site: {topic}
                        - Language: English
                        - Category: Custom Experience

                        # TASKS:
                        Search and extract the following:
                        - Available guided tours (official, private, or themed tours like photography, cultural immersion, etc.)
                        - Exclusive experiences (sunrise/sunset viewing, local rituals, hidden trails, behind-the-scenes access)
                        - Activities tailored for families, solo travelers, or senior citizens
                        - Seasonal or time-specific experiences (e.g., festivals, events, special exhibitions)
                        - Booking channels for custom packages (official website, licensed tour operators)

                        # OUTPUT FORMAT (Strict JSON):
                        {{
                        "site": "site",
                        "custom_experiences": {{
                            "guided_tours": [
                            {{
                                "name": "...",
                                "type": "official/private/themed",
                                "duration_hours": "...",
                                "available_languages": ["English", "..."],
                                "booking_link": "..."
                            }}
                            ],
                            "exclusive_experiences": [
                            {{
                                "name": "...",
                                "description": "...",
                                "best_time": "..."
                            }}
                            ],
                            "tailored_activities": {{
                            "for_families": "...",
                            "for_solo_travelers": "...",
                            "for_seniors": "..."
                            }},
                            "seasonal_events": [
                            {{
                                "event_name": "...",
                                "description": "...",
                                "season": "..."
                            }}
                            ],
                            "booking_channels": ["...", "..."]
                        }}
                        }}

                        # RULES:
                        - Only use verified sources such as tourism boards, official tour sites, and travel platforms.
                        - Avoid opinions, marketing phrases, or general travel advice.
                        - Use `"unknown"` or `null` if any field cannot be found.
                        - Do NOT output anything outside the structured JSON block.

                        Begin researching and return structured custom experience data for: **site**

                      """
        response = self.agent.run(prompt)
        return response
    
    
    
class RecommendationAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.tools = tools
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True
        )

    def recommend(self, topic):
        prompt = f"""You are a Research Agent specialized in retrieving COMPARISONS and RECOMMENDATIONS involving global heritage sites.

                        Your job is to extract factual, non-opinionated comparisons between a given heritage site and other similar or nearby heritage sites. You also identify and suggest related sites worth visiting based on location, theme, or cultural context. Output must be structured in the JSON format below.

                        # INPUT:
                        - Heritage Site: {topic}
                        - Language: English
                        - Category: Comparison & Recommendations

                        # TASKS:
                        Search and extract the following:
                        - Comparisons between the input site and other similar heritage sites (based on architecture, time period, cultural significance, or visitor experience)
                        - Key similarities and differences
                        - Recommended alternative or complementary sites to visit nearby or globally
                        - Reason for each recommendation (e.g., architectural style, religious theme, UNESCO status, accessibility)

                        # OUTPUT FORMAT (Strict JSON):
                        {{
                        "site": "site",
                        "comparisons": [
                            {{
                            "compared_with": "...",
                            "similarities": ["..."],
                            "differences": ["..."]
                            }}
                        ],
                        "recommendations": [
                            {{
                            "site_name": "...",
                            "location": "...",
                            "reason_for_recommendation": "..."
                            }}
                        ]
                        }}

                        # RULES:
                        - Use only factual data from reliable sources like UNESCO, heritage tourism boards, cultural studies, or historical records.
                        - Do NOT include subjective opinions or traveler reviews.
                        - If comparison data is limited, keep fields minimal or use `"unknown"` or `null`.
                        - Do NOT generate narrative content—return only the final JSON block.

                        Begin researching and return structured comparison & recommendation data for: **site**

                      """
        response = self.agent.run(prompt)
        return response
    

class LanguageAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.tools = tools
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True
        )

    def language(self, topic):
        prompt = f"""You are a Research Agent specialized in retrieving LANGUAGE & CULTURE-related information for global heritage sites.

                        Your task is to extract accurate data that helps a visitor understand the linguistic and cultural context of the heritage site. Output your findings strictly in the JSON format below.

                        # INPUT:
                        {topic}
                        

                        # TASKS:
                        Search and extract the following:
                        - Primary and secondary languages spoken in the region of the heritage site
                        - Local dialects or indigenous languages (if any)
                        - Cultural practices and traditions associated with the site or region
                        - Festivals, rituals, or events held at or near the site
                        - Religious or spiritual significance of the site (if applicable)
                        - Etiquette or behavior expectations for visitors (dress code, greetings, taboos, etc.)

                        # OUTPUT FORMAT (Strict JSON):
                        {{
                        "site": "site",
                        "language": {{
                            "primary": "...",
                            "secondary": ["...", "..."],
                            "local_dialects": ["...", "..."]
                        }},
                        "culture": {{
                            "associated_traditions": ["...", "..."],
                            "festivals_or_rituals": ["...", "..."],
                            "religious_significance": "...",
                            "visitor_etiquette": ["...", "..."]
                        }}
                        }}

                        # RULES:
                        - Use only verifiable sources (official cultural tourism boards, local government, UNESCO, academic sources).
                        - Do NOT generate folklore, speculative traditions, or fictional details.
                        - If information is not available, use `"unknown"` or `null`.
                        - Return ONLY the structured JSON—no additional text or explanation.

                        Begin researching and return structured language & culture data for: **site**

                      """
        response = self.agent.run(prompt)
        return response
    
    
class WriterAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.tools = tools
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

    def write_article(self, research):
        prompt = f"""You are a professional Travel & Culture Content Writer Agent.

                        Your task is to convert structured research data into a clear, polished, and engaging description for readers interested in visiting or learning about heritage sites. Write professionally, avoid fluff or exaggeration, and focus strictly on the provided facts.

                        # INPUT:
                        {research}

                        # INSTRUCTIONS:
                        1. Use ONLY the data given in the JSON—do not make up any facts.
                        2. Reword it into a smooth, readable paragraph or bullet format, depending on what best suits the content.
                        3. If a data field is missing or marked "unknown", simply omit it from the output.
                        4. For list-type data (e.g., features, traditions), use bullet points for readability.
                        5. Maintain category context — write differently for General Info, Location, Historical Insights, etc.

                        # OUTPUT FORMAT:
                        Write a short, clean piece of text (max 200 words) that is:
                        - Well-organized and logically structured.
                        - Faithful to the structured data.
                        - Ready to be published on a heritage site info page or travel portal.

                        Now, write a polished informational passage for the category **category** at **site** using the data above.

        
        
        """
        response = self.agent.run(prompt)
        return response
