import streamlit as st
from streamlit_drawable_canvas import st_canvas

from digit_classifier.model.run_model import predict_digit
from digit_classifier.feedback_db_commands import add_feedback_record, get_feedback_records

st.title("Digit Classifier 🤖")
st.markdown(
    """ 
    Digit Classifier 🤖 will try and predict the digit that you've drawn. :rainbow[**WOWSIES**]

    Do your best drawing of a digit between 0-9 and hit 'Classify'
    """
)

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
        st.session_state['text'] = f"Prediction made! 🤖 thinks it's {predicted_digit} and with {conf_percent:.1f}% confidence"
        st.session_state['predicted_digit'] = predicted_digit
        st.session_state['conf_percent'] = conf_percent

st.button(label="Classify", type="primary", on_click=on_button_click)

st.write(st.session_state['text'])

if 'predicted_digit' in st.session_state:
    with st.form(key="feedback-form"):
        st.write("How did 🤖 do? Give it feedback by writing in what the actual digit was")
        true_digit = st.number_input(
            label="Input the 'true' digit",
            min_value=0,
            max_value=9,
            value=0,
            step=1,
        )
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write("Thanks for the feedback! Will reward/punish 🤖 accordingly. (Joke - no robots were harmed in the making of this app)")
            add_feedback_record(st.session_state['predicted_digit'], true_digit, st.session_state['conf_percent'])
            st.session_state["feedback_records"] = get_feedback_records()

if 'feedback_records' not in st.session_state:
    st.session_state["feedback_records"] = get_feedback_records()

st.subheader("Digit Classification Feedback History")
st.dataframe(st.session_state["feedback_records"], column_config={1:"Timestamp", 2:"Predicted Digit", 3:"True Digit", 4:"Confidence %"}, hide_index=True)