import gradio as gr
import warnings
import src.utils as utils
from src.predict import prediction
from src.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")

def show_processing_text():
    return gr.update(visible=True), gr.update(visible=False)

def prediction_with_loading(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    input_data_raw = f"""
    **Input Data:**
        - Nitrogen Content (N): {nitrogen}
        - Phosphorus Content (P): {phosphorus}
        - Potassium Content (K): {potassium}
        - Temperature: {temperature} °C
        - Humidity: {humidity} %
        - Soil pH: {ph}
        - Rainfall: {rainfall} mm
    """
    try:
        logger.info("Starting prediction process...")
        response = prediction(
            nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, input_data_raw
        )
        logger.info("Prediction completed successfully.")
        return gr.update(value=response, visible=True), gr.update(visible=False)
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return "An error occurred during prediction. Please try again.", gr.update(visible=False)

with gr.Blocks(
    css=utils.css,
    js=utils.js,
    theme=gr.themes.Ocean(
        font=gr.themes.GoogleFont("Poppins"),
        primary_hue=gr.themes.colors.cyan,
        secondary_hue=gr.themes.colors.green,
    ),
    fill_width=True,
) as demo:
    gr.Markdown("## CROP RECOMMENDATION SYSTEM - GENAI", elem_classes="title")

    with gr.Row():
        with gr.Column():
            nitrogen = gr.Number(label="Nitrogen Content (N)", value=50)
        with gr.Column():
            phosphorus = gr.Number(label="Phosphorus Content (P)", value=40)
        with gr.Column():
            potassium = gr.Number(label="Potassium Content (K)", value=60)
        with gr.Column():
            temperature = gr.Number(label="Temperature (°C)", value=25)

    with gr.Row():
        with gr.Column():
            humidity = gr.Number(label="Humidity (%)", value=70)
        with gr.Column():
            ph = gr.Number(label="Soil pH", value=6.5, step=0.1)
        with gr.Column():
            rainfall = gr.Number(label="Rainfall (mm)", value=200)

    with gr.Row():
        predict_button = gr.Button("Predict", variant="primary")

    processing_text = gr.Markdown("", visible=False, height=100)
    output_text = gr.Markdown(label="LLM Generated Recommendation", container=True, show_copy_button=True, visible=False)

    predict_button.click(
        fn=show_processing_text,
        inputs=[],
        outputs=[processing_text, output_text],
        queue=False,
    )
    predict_button.click(
        fn=prediction_with_loading,
        inputs=[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall],
        outputs=[output_text, processing_text],
        queue=True,
    )

demo.launch()
