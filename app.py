import streamlit as st
import main
import requests
from io import BytesIO
from PIL import Image


def app():
    with st.container():
        st.sidebar.title("**Menu⚙️**")
        
        option = st.sidebar.selectbox("Choose an input method", ["Browse Image", "Image URL"])
        
# Show appropriate input form based on user selection
    if option == "Browse Image":
        
        st.title("Landmark Detection System")
        st.text_input("Enter your Name: ", key="name")
        # File uploader
        img_path = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if img_path is not None:
            # Display the uploaded image
            image = Image.open(img_path)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Make predictions
            predicted_landmarks = main.predict_landmarks(img_path, k=1)
            
            # Display the predicted landmarks
            st.write(f"Thank you {st.session_state.name}! I hope you liked it.")
            st.write("Predicted landmark :-   ", predicted_landmarks)

    #Taking inpput using Image URL
    elif option == "Image URL":
        st.title("Landmark Detection System")
        st.text_input("Enter your Name: ", key="name")
        image_url = st.text_input("Enter the image URL:") 
        #Submit button code.
        submit_button = st.button("Submit")
        if submit_button and image_url:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            st.image(img, use_column_width=True)
            # Make predictions
            predicted_landmarks = main.predict_landmarks(BytesIO(response.content), k=1)
            # Display the predicted landmarks
            st.write(f"Thank you {st.session_state.name}! I hope you liked it.")
            st.write("Predicted landmark :- ", predicted_landmarks)

        elif submit_button:
            st.write("Please enter an image URL.")  

if __name__ == '__main__':
  
    app()

