# COS60016-Assignment-2
COS60016 Assignment 2: Build a Weather Chatbot

## Weather Chatbot
Weather Chatbot is a simple Python application that provides weather information and forecasts using the OpenWeatherMap API. It also incorporates a chatbot powered by ChatterBot to engage in small talk and answer user queries.

## Features
* **Weather Information:** Get current weather details for a specific city, including temperature, description, and location.
* **Weather Forecast:** Retrieve a weather forecast for a city, including maximum temperatures for upcoming days.
* **Chatbot Interaction:** Engage in casual conversation and receive responses from the integrated chatbot.

## Setup
<ol>
<li>Clone the Repository:</li>
<code>git clone https://github.com/fdx-Luke-Turnbull/COS60016-Assignment-2</code><br>

<li>Open project in PyCharm</li>

<li>Install Dependencies per requirements.txt </li>
<code>pip install -r requirements.txt</code>

<li>Install spacy</li>
<code>pip install spacy</code>
<code>python -m spacy download en_core_web_md</code>

<li>Configuration:</li>
Create a configuration file named config.ini with your OpenWeatherMap API key.<br>
<code>[openweathermap]</code><br>
<code> api = your_api_key_here</code>

<li>Run the Application:</li>
<code>python main.py</code>

<li>Access the Web Interface:</li>
Open a web browser and go to http://127.0.0.1:5000/ to interact with the application.
</ol>

## Usage
* **Chatbot Interaction:** Enter messages in the chatbox to interact with the chatbot.
* **Weather Information:** Ask about the current weather in a specific city.
* **Weather Forecast:** Inquire about the weather forecast for a city, mentioning keywords like "forecast" or "tomorrow."
