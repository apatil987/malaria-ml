import streamlit as st
from joblib import dump, load
import numpy as np
import cv2


model = load("model.joblib" ) # loading up our saved model

st.title('Malaria Diagnosis') #Change the title here!
uploaded_file = st.file_uploader("Upload File")

if st.button('Diagnose'):
  st.write("Clicked")
result = False
if uploaded_file is not None:    
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, 1)
  image = cv2.resize(image,(50,50))
  st.image(image)
  #small = cv2.resize(image,2) #YOUR CODE HERE: specify image, dimensions
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray_flat = np.reshape(gray, (1,2500))
  result = model.predict(gray_flat)[0]
st.write("Has Malaria: "+ str(result))