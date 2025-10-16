import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

st.title("🧠 T5 Text Transformer")
task = st.selectbox("Choose task:", ["Summarization", "Translation (en→fr)"])
text = st.text_area("Enter your text:")

if st.button("Run"):
    tokenizer, model = load_model()
    prefix = "summarize:" if task == "Summarization" else "translate English to French:"
    inputs = tokenizer(prefix + text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=60)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.success(result)
