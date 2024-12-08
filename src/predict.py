import joblib
import numpy as np
from .llm import LLM
from .config import *
from .exception import log_exception, ModelLoadingError, PreprocessingError, PredictionError
from .logger import get_logger

logger = get_logger(__name__)

class DataPreprocessor:
    def __init__(self):
        try:
            self.scaler = joblib.load(f"{MODELS_DIR}/scaler_object.joblib")
            logger.info("DataPreprocessor initialized with scaler.")
        except Exception as e:
            log_exception(e, "Error loading scaler in DataPreprocessor.")
            raise ModelLoadingError("Could not load the scaler for data preprocessing.") from e

    def preprocess(self, input_data):
        try:
            scaled_data = self.scaler.transform(np.array(input_data).reshape(1, -1))
            logger.info("Data preprocessing completed successfully.")
            return scaled_data
        except Exception as e:
            log_exception(e, "Error in preprocessing data in DataPreprocessor.")
            raise PreprocessingError("Preprocessing failed. Ensure input data format is correct.") from e


class ML_Model_Predictor:
    def __init__(self):
        try:
            self.model = joblib.load(f"{MODELS_DIR}/random_forest.joblib")
            self.label_encoder = joblib.load(f"{MODELS_DIR}/label_encoder.joblib")
            logger.info("ML model and label encoder loaded successfully.")
        except Exception as e:
            log_exception(e, "Failed to load ML model and label encoder in ML_Model_Predictor.")
            raise ModelLoadingError("Could not load ML model and label encoder . Please check model path and format.") from e

    def predict(self, preprocessed_data):
        try:
            prediction = self.model.predict(preprocessed_data)
            prediction_decoded = self.label_encoder.inverse_transform(prediction)
            logger.info("ML model prediction completed successfully.")
            return prediction_decoded[0]
        except Exception as e:
            log_exception(e, "Error during ML model prediction in ML_Model_Predictor.")
            raise PredictionError("Prediction failed. Ensure input data format is correct.") from e


def prediction(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, input_data_raw):
    try:
        # Initialize classes
        preprocessor = DataPreprocessor()
        ml_predictor = ML_Model_Predictor()
        llm = LLM()
        
        # Prepare structured data input for ML model
        structured_data = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
        preprocessed_data = preprocessor.preprocess(structured_data)
        
        # Get predictions from ML model
        ml_prediction_result = ml_predictor.predict(preprocessed_data)

        result = f"""
        Crop Recommendation Report:
        
        **ML Model Prediction:** {ml_prediction_result}
        
        {input_data_raw}
        """
        
        # Generate LLM report
        report = llm.inference(result=result)
        logger.info("LLM report generated successfully.")
        return report
    
    except Exception as e:
        log_exception(e, "Error in prediction function.")
        raise PredictionError("Prediction function encountered an error. Check inputs and model paths.") from e
