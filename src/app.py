import streamlit as st
from streamlit_drawable_canvas import st_canvas

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