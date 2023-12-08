# Import necessary libraries and modules
import configparser
import requests
from flask import Flask, render_template, request, session
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer, ChatterBotCorpusTrainer
import spacy
import secrets
from sqlalchemy import create_engine, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import logging
import time

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

# Bind the MetaData to the engine
Base.metadata.bind = engine


# Define the Weather class
class Weather(Base):
    __tablename__ = 'weather'
    __table_args__ = {'autoload': True}

    def __repr__(self):
        return (f"<Weather(id={self.id}, "
                f"city={self.city}, "
                f"date={self.date}, "
                f"description={self.description}, "
                f"temp={self.temperature}, "
                f"latitude={self.latitude}, "
                f"longitude={self.longitude}"
                f"min_temp={self.min_temp}, "
                f"max_temp={self.max_temp}, "
                f"uom={self.uom},"
                f"feels_like={self.feels_like},"
                f"pressure={self.pressure},"
                f"humidity={self.humidity},"
                f"wind_speed={self.wind_speed},"
                f"wind_deg={self.wind_deg},"
                f"wind_gust={self.wind_gust},"
                f"cloud={self.cloud}>")


# Drop the existing 'weather' table to help with bug fixing
Weather.__table__.drop(engine)

# Recreate the 'weather' table
Base.metadata.create_all(engine)

inspector = inspect(engine)
if 'weather' not in inspector.get_table_names():
    Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)


def make_weather_request(url):
    """
    Make a weather request to the OpenWeatherMap API.

    Parameters:
    - url (str): The API endpoint URL.

    Returns:
    dict or None: JSON response if successful, else None.
    """
    api_key = get_api_key()

    if api_key is not None:
        api_url = f"http://api.openweathermap.org/{url}&appid={api_key}"
        response = requests.get(api_url)
        response_dict = response.json()

        if response.status_code == 200:
            return response_dict
        else:
            logging.error(f'[!] HTTP {response.status_code} calling [{api_url}]')
    else:
        logging.error("API key not found. Please review configuration.")
    return None


def get_weather(city_name, uom):
    """
    Get weather information for a city from the OpenWeatherMap API.

    Parameters:
    - city_name (str): Name of the city.
    - uom (str): Unit of measurement for temperature.

    Returns:
    Tuple or None: Tuple containing weather description and temperature if successful, else None.
    """
    response_dict = make_weather_request(f"data/2.5/weather?q={city_name}&units={uom}")
    if response_dict:
        print("Response Dict:", response_dict)
        try:
            print("Response Dict:", response_dict)
            # Extract weather information from API response
            description = response_dict["weather"][0]["description"]
            temperature = round(float(response_dict["main"]["temp"]))
            latitude = response_dict["coord"]["lat"]
            longitude = response_dict["coord"]["lon"]
            max_temp = round(float(response_dict["main"]["temp_min"]))
            min_temp = round(float(response_dict["main"]["temp_max"]))
            feels_like = round(float(response_dict["main"]["feels_like"]))
            pressure = str(response_dict["main"]["pressure"])
            humidity = str(response_dict["main"]["humidity"])
            wind_speed = response_dict["wind"]["speed"]
            cloud = str(response_dict["clouds"]["all"])

            # Store weather information in the database
            weather_entry = Weather(
                city=city_name,
                description=description,
                temperature=temperature,
                latitude=latitude,
                longitude=longitude,
                max_temp=max_temp,
                min_temp=min_temp,
                uom=uom,
                feels_like=feels_like,
                pressure=pressure,
                humidity=humidity,
                wind_speed=wind_speed,
                cloud=cloud
            )

            session = Session()
            session.add(weather_entry)
            session.commit()
            session.close()

            return description, temperature
        except (KeyError, IndexError):
            # Handle parsing errors
            logging.error(f"[!] Unable to parse weather response for {city_name}: {response_dict}")
            return None


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
    response_dict = make_weather_request(f"data/2.5/forecast?q={city_name}&units={uom}&cnt={cnt}")

    if response_dict:
        try:
            max_temp_per_day = {}

            for data_point in response_dict['list']:
                ftemp = round(data_point['main']['temp_max'])
                fdate = datetime.strptime(data_point['dt_txt'], "%Y-%m-%d %H:%M:%S").strftime("%d %B %Y")

                if fdate not in max_temp_per_day or ftemp > max_temp_per_day[fdate]:
                    max_temp_per_day[fdate] = ftemp

            return max_temp_per_day
        except (KeyError, IndexError):
            # Handle parsing errors
            logging.error(f"[!] Unable to parse forecast response for {city_name}: {response_dict}")
    return None


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


# Define a timing decorator
def timing_decorator(func):
    """
    Decorator to measure and print the execution time of a function.

    :param func: The function to be decorated.
    :return: The decorated function.
    """

    def wrapper(*args, **kwargs):
        # Record the start time before executing the function
        start_time = time.time()

        # Execute the original function
        result = func(*args, **kwargs)

        # Record the end time after the function execution
        end_time = time.time()

        # Print the execution time
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute.")

        # Return the result of the original function
        return result

    return wrapper


# Define route for home page
@app.route("/")
def home():
    return render_template("index.html")


# Define route for getting chatbot response
@app.route("/get_response", methods=['POST'])
@timing_decorator
def get_bot_response():
    """
    Processes the user's message and generates an appropriate response.

    :return: A response based on the user's message, either weather information or a general chatbot response.
    """

    # Retrieve the user's message from the form data
    user_message = request.form['user_message']

    # Keywords related to weather forecast
    forecast_keywords = ['forecast', 'tomorrow', 'week', 'weeks', 'next week']

    # Check if the user is asking about the weather forecast
    if any(keyword in user_message.lower() for keyword in forecast_keywords):
        return process_weather_forecast_request(user_message)
    else:
        # If not a weather forecast request, attempt to get current weather response
        weather_response = get_weather_response(user_message)

        # If the weather_response is None, use the chatbot to generate a response
        # Otherwise, return the weather_response
        return weather_response or str(my_bot.get_response(user_message))


def process_weather_forecast_request(user_message):
    """
    Processes a user's request for weather forecasts for specific cities.

    :param user_message: The message provided by the user.
    :return: A response with weather forecasts or an appropriate message if there is an issue.
    """

    # Extract cities from the user's statement
    cities = [ent.text for ent in nlp(user_message).ents if ent.label_ == "GPE"]

    # Retrieve the last known city from the session
    last_city = session.get('last_city')

    # Check if the user mentioned "metric" or "imperial" in the message
    unit, uom, speed = get_units_from_message(user_message)

    # Get forecast responses with a default count of 56 (can be adjusted)
    return get_weather_forecast_responses(cities, last_city, uom, unit, cnt=56)


def get_units_from_message(user_message):
    """
    Extracts temperature unit, unit of measurement (UOM), and speed from the user's message.

    :param user_message: The message provided by the user.
    :return: A tuple containing temperature unit, unit of measurement, and speed.
    """

    # Check if the user mentioned "metric" or "imperial" in the message
    if any(token.text.lower() in ['metric', 'celsius'] for token in nlp(user_message)):
        # Set unit to Celsius, UOM to metric, and speed to meter/sec
        unit, uom, speed = 'celsius', 'metric', 'meter/sec'
    elif any(token.text.lower() in ['imperial', 'fahrenheit'] for token in nlp(user_message)):
        # Set unit to Fahrenheit, UOM to imperial, and speed to miles/hour
        unit, uom, speed = 'fahrenheit', 'imperial', 'miles/hour'
    else:
        # If no specific unit is mentioned, use the last known values or default to imperial
        unit, uom, speed = (session.get('last_unit') or 'fahrenheit',
                            session.get('last_uom') or 'imperial',
                            session.get('last_speed') or 'miles/hour')

    # Update the session variables with the latest values
    session['last_unit'], session['last_uom'], session['last_speed'] = unit, uom, speed

    # Return the extracted unit, UOM, and speed
    return unit, uom, speed


def get_weather_response(user_message):
    """
    Generates a response about the current weather based on the user's message.

    :param user_message: The message provided by the user.
    :return: A response about the current weather or None if the message doesn't match the expected format.
    """

    # Predefined statement to detect if the user is asking about the current weather in a city
    weather_statement = nlp("Current weather in a city")

    # Tokenize the user's message
    user_message_doc = nlp(user_message)

    # Retrieve last known values for unit, UOM, and city from the session
    last_unit, last_uom, last_city = session.get('last_unit'), session.get('last_uom'), session.get('last_city')

    # Check if the user mentioned "metric" or "imperial" in the message and get the unit and UOM
    unit, uom = get_units_from_message(user_message)[:2]

    # Check if the user's message matches the predefined weather statement
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
            weather_result = get_weather(city, uom)

            if weather_result is not None:
                # Unpack the result only if it's not None
                description, temperature = weather_result

                # Generate a dynamic response based on the detected units
                response_template = "In {city}, the current weather is {temperature}° {unit} with {description}."
                response = response_template.format(
                    city=city,
                    unit=unit,
                    temperature=temperature,
                    description=description)
                responses.append(response)
            else:
                responses.append(f"Couldn't retrieve weather information for {city}.")

        return "\r\n".join(responses)
    return None


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
