import streamlit as st
from streamlit_drawable_canvas import st_canvas
from digit_classifier.run_model import predict_digit

st.title("Digit Classifier 🤖")
st.markdown(
    """ 
    Digit Classifier 🤖 will try and predict the digit that you've drawn. :rainbow[**WOWSIES**]

    Do your best drawing of a digit between 0-9 and hit 'Predict'
    """
)

# Create a canvas component
canvas_result = st_canvas(
    stroke_width=20,
    stroke_color="#fff",
    background_color="#000",
    update_streamlit=True,
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if 'text' not in st.session_state:
    st.session_state['text'] = "🤖 patiently awaits your beautiful drawing of a digit..."

def on_button_click():
    if canvas_result.image_data is not None:
        predicted_digit, conf_percent = predict_digit(canvas_result.image_data)
        st.session_state['text'] = f"Prediction made! 🤖 thinks it's {predicted_digit} and with {conf_percent:.2f}% confidence"

st.button(label="Predict", type="primary", on_click=on_button_click)

st.write(st.session_state['text'])