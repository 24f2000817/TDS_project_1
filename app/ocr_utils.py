import base64
import pytesseract
from PIL import Image
import io

def extract_text_from_image(b64_string):
    image_data = base64.b64decode(b64_string)
    image = Image.open(io.BytesIO(image_data))
    text = pytesseract.image_to_string(image)
    return text
