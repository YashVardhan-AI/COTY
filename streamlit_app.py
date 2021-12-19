import streamlit as st
import requests
import lite



st.title("Chatbot")

st.sidebar.title("NLP Bot")
st.sidebar.markdown('This is a nlp chatbot that uses blenderbot at its backend made by me for all the lonely folk out there.(it has a tensorflow lite model running in the background to Detect the presence of suicidal thoughts.)')


API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
headers = {"Authorization": st.secrets['api_key']}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def get_text():
    input_text = st.text_input("You: ","Hello, how are you?")
    return input_text 

user_input = get_text()

if 'i' not in st.session_state:
    st.session_state['i'] = 0

if 'j' not in st.session_state:
    st.session_state['j'] = 0

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if user_input:
    output = query({
        "inputs": {
            "past_user_inputs": st.session_state.past,
            "generated_responses": st.session_state.generated,
            "text": user_input,
        },"parameters": {"repetition_penalty": 1.33},
    })

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output["generated_text"])

    

    st.text_area("Bot:", value=output['generated_text'], height=69, max_chars=None, key=None)

    level = lite.predict(user_input)
    st.session_state.i += 1

    print(level)

    if level == 'Sucide':
        st.session_state.j += 1
        st.warning('You seem to have suicidal thoughts. Please consult a doctor.')


