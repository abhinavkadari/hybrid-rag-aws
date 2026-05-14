import random

import boto3
from botocore.exceptions import ClientError
import json
import time
import streamlit as st

def generate(text):
    client = boto3.client('lambda', region_name='us-east-1')
    payload = {}
    payload["text"] = text
    try:
        response = client.invoke(
                FunctionName='Generate',
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
        result = json.loads(response['Payload'].read())['body']
        result = result.replace("\\n", "\n")
    except ClientError as e:
        print(f"Client error: {e.response['Error']['Code']}")
        result = "Error: Unable to generate response at the moment."
    for chunk in result.split("\n\n"):
        yield "  \n"
        for word in chunk.split():
            yield word + " "
            time.sleep(0.05)

def streamlit_app():
    st.title("Company Internal Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("How can I help you?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            response = st.write_stream(generate(prompt))
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    streamlit_app()

if __name__ == "__main__":
    main()
