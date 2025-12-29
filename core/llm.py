import google.generativeai as genai
import cv2
import logging
from PIL import Image
from config.settings import settings

logger = logging.getLogger("Eagle.LLM")

class GeminiAgent:
    def __init__(self):
        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            #cost effective + thinking
            self.model = genai.GenerativeModel('gemini-2.5-pro')
            logger.info("Gemini Agent Initialized")
        else:
            logger.warning("Gemini API Key missing! LLM features will fail.")

    def analyze(self, normal_frame, anomaly_frame):
        """
        sends two images to gemini:
        1. REFERENCE (normal)
        2. EVENT (anomaly)
        asks for specific classification of the difference.
        """
        try:
            if not settings.GEMINI_API_KEY:
                return "Config Error: No API Key"

            #convert format: opencv bgr to pil rgb
            img_normal = Image.fromarray(cv2.cvtColor(normal_frame, cv2.COLOR_BGR2RGB))
            img_anomaly = Image.fromarray(cv2.cvtColor(anomaly_frame, cv2.COLOR_BGR2RGB))

            prompt = """
            You are an industrial safety AI for a Wastewater Treatment Plant.
            
            Image 1: REFERENCE (Normal conditions)
            Image 2: EVENT (Anomaly detected)

            Compare the EVENT image to the REFERENCE. Identify exactly what foreign object or condition has appeared.
            
            Classify the anomaly into EXACTLY one of these categories:
            - Fire
            - Smoke
            - Water Leak
            - Chemical Spill
            - Corrosion
            - Human
            - Foreign Object
            - Unknown
            
            Return ONLY the category name. Do not explain.
            """

            logger.info("Sending frames to Gemini for analysis...")
            
            #now call gemini
            response = self.model.generate_content([prompt, img_normal, img_anomaly])
            result = response.text.strip()
            
            logger.info(f"Gemini Identified: {result}")
            return result

        except Exception as e:
            logger.error(f"Gemini Inference Failed: {e}")
            return "LLM Error"