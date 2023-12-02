import configparser
import requests
from flask import *

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer, ChatterBotCorpusTrainer
import spacy

app = Flask(__name__)

nlp = spacy.load("en_core_web_md")

my_bot = ChatBot(name="PyBot",
                 read_only=True,
                 logic_adapters=[
                     {
                         'import_path': 'chatterbot.logic.BestMatch',
                         'default_response': 'Sorry, I am not sure how to respond.'
                     }
                 ])

small_talk = [
    "Hello",
    "Hi there!",
    "How are you doing?",
    "I'm doing great.",
    "That is good to hear",
    "Thank you.",
    "You're welcome."
]

list_trainer = ListTrainer(my_bot)

for item in small_talk:
    list_trainer.train(item)

corpus_trainer = ChatterBotCorpusTrainer(my_bot)
corpus_trainer.train('chatterbot.corpus.english',
                     "chatterbot.corpus.english.greetings",
                     "chatterbot.corpus.english.conversations"
                     )


def get_weather(city_name):
    api_key = get_api_key()
    if api_key is not None:
        api_url = "http://api.openweathermap.org/data/2.5/weather?q={}&appid={}".format(city_name, api_key)
        response = requests.get(api_url)
        response_dict = response.json()

        if response.status_code == 200:
            description = response_dict["weather"][0]["description"]
            temperature = response_dict["main"]["temp"]
            return description, temperature

        else:
            print('[!] HTTP {0} calling [{1}]'.format(response.status_code, api_url))
            return None
    else:
        return "API key not found. Please review configuration."


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=['POST'])
def get_bot_response():
    user_message = request.form['user_message']

    # Check if the user is asking about the weather
    weather_statement = nlp("Current weather in a city")
    user_message_doc = nlp(user_message)

    if weather_statement.similarity(user_message_doc) >= 0.75:
        # Extract city from user's statement
        for ent in user_message_doc.ents:
            print("Entity:", ent.text, "Label:", ent.label_)  # Debugging line
            if ent.label_ == "GPE":  # GeoPolitical Entity
                city = ent.text
                break
            else:
                return "You need to tell me a city to check."

        description, temperature = get_weather(city)
        if description is not None:
            chat_response = (
                    "In " + city +
                    ", the current weather is "
                    + str(temperature) + " degrees with " + description + ".")
        else:
            chat_response = "Something went wrong."
    else:
        # Use the chatbot to get a response for the user's message
        chat_response = str(my_bot.get_response(user_message_doc.text))

    print("user msg", user_message)
    print("bot response", chat_response)

    return str(chat_response)


def get_api_key():
    """
    Retrieve the API key from the configuration file.

    :return: The API key for accessing the OpenWeatherMap API.

    Note:
    - The function reads the API key from the 'config.ini' configuration file.
    - If the key is not found, it prints an error message and returns None.
    """
    try:
        # Read the API key from the configuration file
        config = configparser.ConfigParser()
        config.read('config.ini')
        return config['openweathermap']['api']

    except KeyError:
        # Handle the case where the API key is not found in the configuration file
        print("Error: API key not found in the configuration file.")
        return None


if __name__ == "__main__":
    app.run(debug=True)