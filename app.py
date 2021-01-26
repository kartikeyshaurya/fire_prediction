import streamlit as st
import numpy as np
from PIL import Image



st.title('Fire Prediction App ')

image = Image.open("back.jpg")
st.image(image, caption=None, width=None, use_column_width=True, clamp=False, channels='RGB', output_format='auto')

st.markdown("Forest Fires cause severe hazard in endangering vegetation , Animal and human life across the world , apart from this it can be a factor for  Airborne hazards ,  Water pollution and other Post- fire risks. Also after Post-fire  is very hard to control and even wildland fire fighters face several life-threatening hazards including heat stress, fatigue, smoke and dust, as well as the risk of other injuries such as burns, cuts and scrapes, animal bites, and even rhabdomyolysis. Between 2000–2016, more than **350** wildland firefighters died on-duty. Only Amazon forest fire cost Brazil US$957 billion to **US$3.5_trillion** over a 30-year period.")
st.markdown("Fast and efficient detection is a key factor in wildfire fighting. Early detection efforts were being made by the different goverenmet from old time, such as Fire lookout towers , Aerial and land photography.")
st.markdown("Computer vision has been quite popular in recent years.It enables computer to understand the content of images and videos.The objective of this project is to develop a model which makes a decision whether it is fire or not.Through this we can automate the process of detecting wildfire")

st.header("Model Architechture")
st.markdown("it will be updated soon")

st.header("Model Application and testing ")

if st.button('Say hello'):
    st.write('Why hello there')

genre = st.radio(
         "What's your favorite movie genre",
             ('Comedy', 'Drama', 'Documentary'))
if genre == 'Comedy':
         st.write('You selected comedy.')
else:
    st.write("You didn't select comedy.")