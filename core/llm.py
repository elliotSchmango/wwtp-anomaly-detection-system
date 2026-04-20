import google.generativeai as genai
import cv2
import logging
import requests
import base64
from PIL import Image
from config.settings import settings

logger = logging.getLogger("Eagle.LLM")

class VisionAgent:
    concise_prompt = """
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

    detailed_prompt = """
    You are an advanced industrial safety AI for a Wastewater Treatment Plant.
    
    Image 1: REFERENCE (Normal baseline)
    Image 2: EVENT (Live feed containing an anomaly)

    First, describe exactly how the EVENT image differs from the REFERENCE image. Pay attention to changes in texture, shape, color, and newly introduced objects. Detail the relative size and location of the anomaly.

    Then, based on your visual analysis, classify the nature of the event into exactly one of these categories:
    [Fire, Smoke, Water Leak, Chemical Spill, Corrosion, Human, Foreign Object, Unknown]
    
    End your response with the exact classification category on a new line prefixed with "CATEGORY: ".
    """

    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        if self.provider == "gemini":
            if settings.GEMINI_API_KEY:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.model = genai.GenerativeModel('gemini-2.5-pro')
                logger.info("Gemini VLM Agent Initialized")
            else:
                logger.warning("Gemini API Key missing! LLM features will fail.")
        else:
            logger.info(f"Local VLM Agent Initialized ({settings.LOCAL_VLM_MODEL} via Ollama)")

    def get_prompt(self) -> str:
        if settings.LLM_PROMPT_TEMPLATE.lower() == "detailed":
            return self.detailed_prompt.strip()
        return self.concise_prompt.strip()

    def image_to_base64(self, img_pil):
        import io
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def analyze(self, normal_frame, anomaly_frame):
        """
        sends two images to the selected VLM provider:
        1. REFERENCE (normal)
        2. EVENT (anomaly)
        asks for specific classification of the difference.
        """
        try:
            prompt = self.get_prompt()
            
            #convert format: opencv bgr to pil rgb
            img_normal = Image.fromarray(cv2.cvtColor(normal_frame, cv2.COLOR_BGR2RGB))
            img_anomaly = Image.fromarray(cv2.cvtColor(anomaly_frame, cv2.COLOR_BGR2RGB))

            logger.info(f"Sending frames to {self.provider} for analysis...")

            if self.provider == "gemini":
                if not settings.GEMINI_API_KEY:
                    return "Config Error: No API Key"
                response = self.model.generate_content([prompt, img_normal, img_anomaly])
                result = response.text.strip()
            else:
                # ollama local vlm implementation
                b64_normal = self.image_to_base64(img_normal)
                b64_anomaly = self.image_to_base64(img_anomaly)
                
                payload = {
                    "model": settings.LOCAL_VLM_MODEL,
                    "prompt": prompt,
                    "images": [b64_normal, b64_anomaly],
                    "stream": False
                }
                
                resp = requests.post(settings.LOCAL_VLM_URL, json=payload, timeout=120)
                resp.raise_for_status()
                response_data = resp.json()
                
                raw_result = response_data.get("response", "").strip()
                
                if settings.LLM_PROMPT_TEMPLATE.lower() == "detailed":
                    # extract category if present
                    if "CATEGORY:" in raw_result:
                        cat = raw_result.split("CATEGORY:")[-1].strip()
                        result = f"{cat} (Detailed: {raw_result})"
                    else:
                        result = raw_result
                else:
                    result = raw_result

            logger.info(f"VLM Identified: {result}")
            return result

        except Exception as e:
            logger.error(f"VLM Inference Failed: {e}")
            return "LLM Error"
