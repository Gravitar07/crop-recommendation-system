from groq import Groq
from .exception import log_exception, APIError
from .config import *
from .logger import get_logger

logger = get_logger(__name__)

class LLM:
    def __init__(self):
        try:
            # Configure and initialize the Generative AI model
            self.llm = Groq(api_key=GROQ_API_KEY)
            logger.info("LLM model initialized successfully.")
        except Exception as e:
            log_exception(e, "Failed to initialize the LLM model in LLM class.")
            raise APIError("Error initializing LLM model. Check API key and model name.") from e

    def prompt_template(self, result):
        # Generates a structured prompt for the LLM based on the provided data
        prompt = f"""You are an expert agricultural advisor specializing in crop recommendations based on soil and weather data. Your role is to analyze the provided data and generate a comprehensive, structured diagnostic report that is both insightful and actionable for farmers. The report should guide farmers in making informed decisions and understanding the rationale behind the recommendation.

**Instructions:**
1. **Crop Recommendation**: Analyze the provided data and the ML-predicted crop recommendation.
2. **Key Observations**: Identify and summarize critical factors (soil measures, weather conditions) influencing the recommendation.
3. **Reasoning**: Explain why the recommended crop is suitable for the given conditions, highlighting specific input features.
4. **Practical Tips**: Suggest best practices for improving crop yield and soil health, tailored to the recommended crop and current conditions.
5. **Benefits to Farmers**: Highlight the advantages of growing the recommended crop in the given conditions.

Ensure the report is easy to understand and provides actionable advice. Avoid unnecessary technical jargon but maintain professional accuracy.

---

**Input Data**:
{result}

**Output Report**:
1. **Recommended Crop**:
   Clearly state the ML-predicted crop recommendation. For example: *"Based on the provided data, the most suitable crop is **Wheat**."*

2. **Key Observations**:
   Highlight the critical input factors and their values that contributed to the recommendation. For example:
   - Soil pH: **6.5** (optimal for wheat growth)
   - Temperature: **25°C** (ideal for wheat cultivation)
   - Rainfall: **500 mm/year** (sufficient for wheat)

3. **Reasoning**:
   Provide a detailed explanation of why this crop is the best choice for the given conditions. Consider aspects like soil compatibility, climatic suitability, and the crop's economic value.

4. **Practical Tips**:
   Offer actionable suggestions to optimize yield and maintain soil health. For example:
   - Implement **crop rotation** with legumes to enhance nitrogen content in the soil.
   - Use **drip irrigation** for water efficiency.
   - Apply **organic fertilizers** to balance soil nutrients.

5. **Benefits to Farmers**:
   Explain how growing this crop will benefit the farmer economically and ecologically. Include potential market demand, cost-effectiveness, and long-term sustainability.

---

**Example Output Report** (for illustrative purposes):

1. **Recommended Crop**:
   Based on the provided data, the most suitable crop is **Rice**.

2. **Key Observations**:
   - Soil Type: **Clayey** (excellent for water retention, required for rice)
   - Soil pH: **6.2** (slightly acidic, suitable for rice growth)
   - Rainfall: **800 mm/year** (ideal for rice cultivation)
   - Temperature: **28°C** (optimal for rice production)

3. **Reasoning**:
   Rice thrives in clayey soil due to its high water-retention capacity. The pH and temperature fall within the ideal range, ensuring healthy crop growth. Additionally, the region's rainfall is adequate for rice farming without extensive irrigation needs.

4. **Practical Tips**:
   - Use **flood irrigation** during the early growth stages.
   - Apply **potassium-based fertilizers** to improve grain quality.
   - Regularly monitor for pests like **stem borers** and use eco-friendly pest management solutions.

5. **Benefits to Farmers**:
   Growing rice ensures high market demand, providing consistent income. The compatibility with local conditions reduces input costs, and eco-friendly practices improve long-term soil fertility.

---

Use this template to create a detailed report for the user based on the input data and the recommended crop.

    """
        return prompt

    def inference(self, result):
        try:
            # Prepare and send the request to the LLM model
            refined_prompt = self.prompt_template(result)
            chat_completion = self.llm.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": refined_prompt,
                    }
                ],
                model=GROQ_MODEL_NAME,
            )
            
            response = chat_completion.choices[0].message.content

            if response:
                logger.info("LLM inference successful.")
                return response
            else:
                logger.warning("LLM did not return any response.")
                raise APIError("LLM response is empty. Please check input format and prompt.")

        except Exception as e:
            log_exception(e, "Error during LLM inference in LLM class.")
            raise APIError("Error during LLM inference. Check input data and model configuration.") from e
