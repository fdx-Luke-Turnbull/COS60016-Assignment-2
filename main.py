# Import necessary libraries and modules
import configparser
import requests
from flask import *
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer, ChatterBotCorpusTrainer
import spacy
import secrets
from sqlalchemy import create_engine, Column, Integer, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import inspect
from datetime import datetime
import logging

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)  # Generate a secure secret key

# Load spaCy model for natural language processing
nlp = spacy.load("en_core_web_md")

# SQLAlchemy configuration
engine = create_engine('sqlite:///db.sqlite3', echo=True)
Base = declarative_base()


# Define the Weather class
class Weather(Base):
    __tablename__ = 'weather'
    id = Column(Integer, primary_key=True)
    city = Column(Text)
    date = Column(DateTime, default=datetime.now)
    description = Column(Text)
    temperature = Column(Integer)
    latitude = Column(Text)
    longitude = Column(Text)
    min_temp = Column(Integer)
    max_temp = Column(Integer)
    uom = Column(Text)

    def __repr__(self):
        return (f"<Weather(id={self.id}, city={self.city}, date={self.date}, "
                f"description={self.description}, temp={self.temperature}, "
                f"latitude={self.latitude}, longitude={self.longitude}"
                f"min_temp={self.min_temp}, max_temp={self.max_temp}, uom={self.uom})>")


# Drop the existing 'weather' table to help with bug fixing
# Weather.__table__.drop(engine)

# Recreate the 'weather' table
Base.metadata.create_all(engine)

inspector = inspect(engine)
if 'weather' not in inspector.get_table_names():
    Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)


def get_weather(city_name, uom):
    """
    Get weather information for a city from the OpenWeatherMap API.

    Parameters:
    - city_name (str): Name of the city.
    - uom (str): Unit of measurement for temperature.

    Returns:
    Tuple or None: Tuple containing weather description and temperature if successful, else None.
    """
    api_key = get_api_key()

    if api_key is not None:
        api_url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&units={uom}&appid={api_key}"
        response = requests.get(api_url)
        response_dict = response.json()

        if response.status_code == 200:
            try:
                # Extract weather information from API response
                description = response_dict["weather"][0]["description"]
                temperature = round(float(response_dict["main"]["temp"]))
                latitude = response_dict["coord"]["lat"]
                longitude = response_dict["coord"]["lon"]
                max_temp = round(float(response_dict["main"]["temp_min"]))
                min_temp = round(float(response_dict["main"]["temp_max"]))
                # Store weather information in the database
                weather_entry = Weather(city=city_name,
                                        description=description,
                                        temperature=temperature,
                                        latitude=latitude,
                                        longitude=longitude,
                                        max_temp=max_temp,
                                        min_temp=min_temp,
                                        uom=uom)
                session = Session()
                session.add(weather_entry)
                session.commit()
                session.close()

                return description, temperature
            except (KeyError, IndexError):
                # Handle parsing errors
                logging.error(f"[!] Unable to parse weather response for {city_name}: {response_dict}")
                return None
        else:
            # Handle HTTP errors
            logging.error(f'[!] HTTP {response.status_code} calling [{api_url}]')
            return None
    else:
        # Handle missing API key
        return "API key not found. Please review configuration."


# Function to retrieve weather forecast responses for multiple cities
def get_weather_forecast(city_name, uom, cnt):
    """
    Get weather forecast information for a city from the OpenWeatherMap API.

    Parameters:
    - city_name (str): Name of the city.
    - uom (str): Unit of measurement for temperature.
    - cnt (int): Number of forecast data points.

    Returns:
    Dict or None: Dictionary containing max temperature per day if successful, else None.
    """
    api_key = get_api_key()

    if api_key is not None:
        forecast_url = f"data/2.5/forecast?q={city_name}&units={uom}&cnt={cnt}&appid={api_key}"
        forecast = requests.get(f"http://api.openweathermap.org/{forecast_url}")
        forecast_data = forecast.json()

        if forecast.status_code == 200:
            max_temp_per_day = {}

            for data_point in forecast_data['list']:
                ftemp = round(data_point['main']['temp_max'])
                fdate = datetime.strptime(data_point['dt_txt'], "%Y-%m-%d %H:%M:%S").strftime("%d %B %Y")

                if fdate not in max_temp_per_day or ftemp > max_temp_per_day[fdate]:
                    max_temp_per_day[fdate] = ftemp

            return max_temp_per_day

        else:
            # Handle HTTP errors
            logging.error(f'[!] HTTP {forecast.status_code} calling [{forecast_url}]')
            return None

    else:
        # Handle missing API key
        return "API key not found. Please review configuration."


# Function to get weather forecast responses for multiple cities
def get_weather_forecast_responses(cities, last_city, uom, unit, cnt):
    """
    Get weather forecast responses for multiple cities.

    Parameters:
    - cities (List[str]): List of cities for which to get forecasts.
    - last_city (str): Last city used in the session.
    - uom (str): Unit of measurement for temperature.
    - unit (str): Unit type (e.g., 'Celsius', 'Fahrenheit').
    - cnt (int): Number of forecast entries to retrieve.

    Returns:
    List[str]: List of forecast messages.
    """
    responses = []
    for city in cities or [last_city]:
        session['last_city'] = city  # Store the last city in the session
        forecast_data = get_weather_forecast(city, uom, cnt)

        if forecast_data is not None:
            forecast_messages = []
            for date, max_temp in forecast_data.items():
                max_temp_converted = round(float(max_temp))
                forecast_message = f"On {date}, expect a maximum temperature of {max_temp_converted}° {unit} in {city}."
                forecast_messages.append(forecast_message)

            response = '\n\r'.join(forecast_messages)
            responses.append(response)
            print(response)
        else:
            f"Couldn't retrieve weather forecast information for {city}."

    return response


# Initialize ChatterBot instance
my_bot = ChatBot(name="PyBot",
                 read_only=False,
                 filters=[
                     'filters.get_recent_repeated_responses'
                 ],
                 storage_adapter='chatterbot.storage.SQLStorageAdapter',
                 logic_adapters=[
                     {
                         'import_path': 'chatterbot.logic.BestMatch',
                         'default_response': 'Sorry, I am not sure how to respond.'
                     },
                 ],
                 database_uri='sqlite:///db.sqlite3',
                 preprocessors=[
                     'chatterbot.preprocessors.clean_whitespace'
                 ]
                 )

# Define small talk conversations for training
small_talk = [
    "Hello",
    "Hi there!",
    "How are you doing?",
    "I'm doing great.",
    "That is good to hear",
    "Thank you.",
    "You're welcome."
]

# Train ChatterBot with small talk conversations
list_trainer = ListTrainer(my_bot)
for item in small_talk:
    list_trainer.train(item)

# Train ChatterBot with English language corpus
corpus_trainer = ChatterBotCorpusTrainer(my_bot)
corpus_trainer.train('chatterbot.corpus.english.humor',
                     "chatterbot.corpus.english.greetings",
                     "chatterbot.corpus.english.conversations"
                     )


# Define route for home page
@app.route("/")
def home():
    return render_template("index.html")


# Define route for getting chatbot response
@app.route("/get_response", methods=['POST'])
def get_bot_response():
    user_message = request.form['user_message']

    # Check if the user is asking about the weather forecast
    forecast_keywords = ['forecast', 'tomorrow', 'week', 'weeks', 'next week']
    if any(keyword in user_message.lower() for keyword in forecast_keywords):
        # Extract cities from the user's statement
        cities = [ent.text for ent in nlp(user_message).ents if ent.label_ == "GPE"]
        last_city = session.get('last_city')

        # Check if the user mentioned "metric" or "imperial" in the message
        if any(token.text.lower() in ['metric', 'celsius'] for token in nlp(user_message)):
            unit = 'celsius'
            uom = 'metric'
        elif any(token.text.lower() in ['imperial', 'fahrenheit'] for token in nlp(user_message)):
            unit = 'fahrenheit'
            uom = 'imperial'
        else:
            # If no specific unit is mentioned, use the last known values or default to imperial
            unit = session.get('last_unit') or 'fahrenheit'
            uom = session.get('last_uom') or 'imperial'

        # Get forecast responses
        chat_response = get_weather_forecast_responses(cities, last_city, uom, unit, cnt=56)

    else:
        # If the user is not asking about the weather forecast, proceed with the regular weather information
        # Check if the user is asking about the weather
        weather_statement = nlp("Current weather in a city")
        user_message_doc = nlp(user_message)

        # Set last_unit, last_uom, and last_city before checking user's message
        last_unit = session.get('last_unit')
        last_uom = session.get('last_uom')
        last_city = session.get('last_city')

        # Check if the user mentioned "metric" or "imperial" in the message
        if any(token.text.lower() in ['metric', 'celsius'] for token in user_message_doc):
            uom = 'metric'
            unit = 'celsius'
        elif any(token.text.lower() in ['imperial', 'fahrenheit'] for token in user_message_doc):
            uom = 'imperial'
            unit = 'fahrenheit'
        else:
            # If no specific unit is mentioned, use the last known values or default to imperial
            uom = last_uom or 'imperial'
            unit = last_unit or 'fahrenheit'

        session['last_unit'] = unit  # Store the last unit in the session
        session['last_uom'] = uom  # Store the last uom in the session

        if weather_statement.similarity(user_message_doc) >= 0.75:
            # Extract cities from the user's statement
            cities = [ent.text for ent in user_message_doc.ents if ent.label_ == "GPE"]

            if not cities:
                # Use the last city if available, otherwise, ask the user to provide a city
                city = last_city or None
                if not city:
                    return "You need to tell me a city to check."

            responses = []
            for city in cities or [last_city]:
                session['last_city'] = city  # Store the last city in the session
                description, temperature = get_weather(city, uom)

                if description is not None:
                    # Generate a dynamic response based on the detected units
                    response_template = "In {city}, the current weather is {temperature}° {unit} with {description}."
                    response = response_template.format(city=city,
                                                        unit=unit,
                                                        temperature=temperature,
                                                        description=description)
                    responses.append(response)
                else:
                    responses.append(f"Couldn't retrieve weather information for {city}.")

            chat_response = "\r\n".join(responses)
        else:
            # Use the chatbot to get a response for the user's message
            chat_response = str(my_bot.get_response(user_message))

    print("user msg", user_message)
    print("bot response", chat_response)
    print(last_city)

    return str(chat_response or "Sorry, I am not sure how to respond.")


# Function to retrieve the API key from the configuration file
def get_api_key():
    """
    Retrieve the OpenWeatherMap API key from the configuration file.

    Returns:
    str or None: API key if found, else None.
    """
    try:
        # Read the API key from the configuration file
        configuration = configparser.ConfigParser()
        configuration.read('config.ini')
        return configuration['openweathermap']['api']

    except KeyError:
        logging.error("Error: API key not found in the configuration file.")
    except FileNotFoundError:
        logging.error("Error: Configuration file not found.")
    except configparser.NoSectionError or configparser.NoOptionError:
        logging.error("Error: Invalid configuration file format.")


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
