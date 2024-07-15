from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datetime
import webbrowser
import subprocess

app = Flask(__name__)

# Load the model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global conversation_history

    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'response': 'No message provided'})

    user_input = user_input.strip().lower()
    
   
    if user_input == '-help':
        help_text = (
            "Here are the commands you can use:<br><br>"
            "- <strong>Tell me the day</strong>: Get the current day of the week.<br>"
            "- <strong>Search for [query]</strong>: Search Google for the specified query.<br>"
            "- <strong>Play [song]</strong>: Search YouTube for the specified song.<br>"
            "- <strong>Tell me the time</strong>: Get the current time.<br><br>"
            "Simply type the command or query, and I will assist you!"
        )
        return jsonify({'response': help_text})

    
    if "open spotify" in user_input:
        open_spotify()
        return jsonify({'response': "Opening Spotify."})

    if "search for" in user_input:
        search_query = user_input.replace("search for", "").strip()
        search(search_query)
        return jsonify({'response': f"Searching for '{search_query}'."})

    if "play" in user_input and "song" in user_input:
        song_name = user_input.replace("play", "").replace("song", "").strip()
        youtube_search(song_name)
        return jsonify({'response': f"Searching for '{song_name}' on YouTube."})

    if "tell me the time" in user_input:
        return jsonify({'response': get_current_time()})

    if "tell me the day" in user_input:
        return jsonify({'response': get_current_day()})

    
    conversation_history.append(user_input)
    
    
    history_string = "\n".join(conversation_history)

    
    combined_input = history_string + "\n" + user_input

    
    max_input_length = tokenizer.model_max_length
    if len(combined_input) > max_input_length:
        combined_input = combined_input[-max_input_length:]

    
    inputs = tokenizer.encode_plus(combined_input, return_tensors="pt", max_length=512, truncation=True)

    try:
        
        outputs = model.generate(inputs['input_ids'], max_length=1000, num_beams=5, early_stopping=True)
        
        
        bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        
        conversation_history.append(bot_response)
        
        return jsonify({'response': bot_response})

    except Exception as e:
        
        print(f"Error generating response: {e}")
        return jsonify({'response': f"Error generating response: {e}"})

def open_spotify():
    try:
        subprocess.Popen(["spotify"])
        print("Opening Spotify...")
    except Exception as e:
        print(f"Error opening Spotify: {str(e)}")

def search(query):
    try:
        search_url = f"https://www.google.com/search?q={query}"
        webbrowser.open(search_url)
        print(f"Searching for '{query}'...")
    except Exception as e:
        print(f"Error searching for '{query}': {str(e)}")

def youtube_search(query):
    try:
        
        youtube_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        webbrowser.open(youtube_url)
        print(f"Searching for '{query}' on YouTube...")
    except Exception as e:
        print(f"Error searching for '{query}' on YouTube: {str(e)}")

def get_current_time():
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    return f"The current time is {current_time}"

def get_current_day():
    current_day = datetime.datetime.now().strftime("%A")
    return f"Today is {current_day}"

if __name__ == '__main__':
    app.run(debug=True)
