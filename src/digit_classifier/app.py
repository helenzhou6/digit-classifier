import streamlit as st
from streamlit_drawable_canvas import st_canvas
from digit_classifier.run_model import predict_digit

st.title("Digit Classifier")

# Create a canvas component
canvas_result = st_canvas(
    stroke_width=20,
    stroke_color="#fff",
    background_color="#000",
    update_streamlit=True,
    height=280,
    width=280,
    drawing_mode="freedraw",
    point_display_radius=0,
    key="canvas",
)

def on_button_click():
    if canvas_result.image_data is not None:
        predict_digit(canvas_result.image_data)

st.button(label="Classify", type="primary", on_click=on_button_click)
