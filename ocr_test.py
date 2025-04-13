import cv2
import pytesseract
from PIL import Image
import re


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_image(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

  
    cv2.imwrite("simple-image.png", img)
    
    return img


def extract_text_from_image(image_path):
    try:
        # Preprocess the image
        img = preprocess_image(image_path)
      
        custom_config = r"--oem 3 --psm 6"
        text = pytesseract.image_to_string(img, config=custom_config)
        
        return text
    except Exception as e:
        return f"Error during image text extraction: {e}"


def parse_bank_statement(text):

    date_pattern = r"\d{2}/\d{2}/\d{4}"
    amount_pattern = r"R\s?\d{1,3}(,\d{3})*(\.\d{2})?"
    lines = text.split("\n")
    
    extracted_data = []

    for line in lines:
        date_match = re.search(date_pattern, line)
        amount_match = re.findall(amount_pattern, line)
        
        if date_match and amount_match:
            date = date_match.group()
            amounts = " ".join([amt[0] for amt in amount_match])
            description = line.split(date)[-1].strip()
            extracted_data.append((date, description, amounts))
    
    return extracted_data

# Main function to run the OCR process
if __name__ == "__main__":
    image_path = "captec.jpg"
    text = extract_text_from_image(image_path)
    print("Extracted Text:\n", text)
    
    # Parse the extracted text
    parsed_data = parse_bank_statement(text)
    print("\nParsed Data:")
    for entry in parsed_data:
        print(entry)
