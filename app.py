import streamlit as st
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords

ps = PorterStemmer()
def text_transform(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  y= []
  for i in text:
    if i.isalnum():
      y.append(i)
  text = y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  text = y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))
  return " ".join(y)


with open("model.pkl", "rb") as file:
  model = pickle.load(file)
with open("vectorizer.pkl", "rb") as file:
  tfidf = pickle.load(file)

st.title("Email/SMS spam classifier")
input_sms = st.text_area("Enter the message:")

if st.button("Predict"):
  transformed_text = text_transform(input_sms)

  vector_input = tfidf.transform([transformed_text])

  result = model.predict(vector_input)[0]
  if result == 1:
    st.header("Spam")
  else:
    st.header("Not spam")