
import os
import pickle
import streamlit as st
import json
import tensorflow as tf
from collections import Counter
from utils import predict

#===========================================#
#        Loads Model and word_to_id         #
#===========================================#


def echo(input):
    return input


def print_dialogs(dialog):
    for i in range(len(dialog)):
        st.write(dialog[i])


dialog = None
if os.path.exists('dialog.pkl'):
    with open('dialog.pkl', 'rb') as f:
        dialog = pickle.load(f)
# dialog = pickle.load(open('dialog.pkl', 'rb'))
if dialog is None:
    dialog = []

#===========================================#
#              Streamlit Code               #
#===========================================#

st.title('Chatbot')
st.write('This is a simple chatbot that uses a seq2seq model to reply to the user.')

user_input = st.text_input('Type something')


if st.button('Generate Text'):
    dialog.append("You: " + user_input)
    generated_text = predict(user_input)
    dialog.append("Bot: " + generated_text)
    import pickle
    with open("dialog.pkl", "wb") as f:
        pickle.dump(dialog, f)
    # st.write(dialog)
    print_dialogs(dialog)


if st.button("Clear"):
    dialog = []
    if os.path.exists('dialog.pkl'):
        os.remove('dialog.pkl')
    # st.write(dialog)
    print_dialogs(dialog)
