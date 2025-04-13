import pdfplumber
from docx import Document
import pandas as pd
import pytesseract
from PIL import Image
import re
import string
from transformers import pipeline
import spacy


nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file using pdfplumber.
    """
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(file_path):
    """
    Extract text from a Word document using python-docx.
    """
    doc = Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text

def extract_text_from_excel(file_path):
    """
    Extract text from an Excel file by reading all data into a string.
    """
    data = pd.read_excel(file_path, sheet_name=None)
    text = ""
    for sheet_name, sheet_data in data.items():
        text += sheet_data.to_string(index=False) + "\n"
    return text

def extract_text_from_image(image_path):
    """
    Extract text from an image using Tesseract OCR.
    """
    text = pytesseract.image_to_string(Image.open(image_path))
    return text

def preprocess_text(text):
    """
    Clean and preprocess the extracted text.
    """
    # Remove digits and special characters
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert text to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = " ".join(text.split())
    return text

def extract_entities_hf(text):
    """
    Perform NER using a pre-trained Hugging Face model.
    """
    ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    entities = ner_pipeline(text)
    return [(ent['word'], ent['entity_group']) for ent in entities]


def extract_entities_spacy(text):
    """
    Perform NER using spaCy.
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def extract_payments(text):
    """
    Extract lines mentioning payments or transfers from the text.
    """
    lines = text.split("\n")
    payments = []
    for line in lines:
        if any(keyword in line.lower() for keyword in ["payment", "transfer", "debit", "credit"]):
            payments.append(line)
    return payments

def structure_entities(entities):
    """
    Create a DataFrame for named entities.
    """
    df = pd.DataFrame(entities, columns=["Entity", "Label"])
    return df

def structure_payments(payments):
    """
    Create a DataFrame for payment information.
    """
    df = pd.DataFrame(payments, columns=["Payment Details"])
    return df

def analyze_document(file_path, file_type):
    """
    Analyze the document and return structured information.
    """
    # Extract text based on file type
    if file_type == "pdf":
        text = extract_text_from_pdf(file_path)
    elif file_type == "docx":
        text = extract_text_from_docx(file_path)
    elif file_type == "xlsx":
        text = extract_text_from_excel(file_path)
    elif file_type in ["jpg", "png", "jpeg"]:
        text = extract_text_from_image(file_path)
    else:
        raise ValueError("Unsupported file type")

    # Preprocess the text
    text = preprocess_text(text)

    # Extract entities using Hugging Face
    entities_hf = extract_entities_hf(text)

    # Extract entities using spaCy
    entities_spacy = extract_entities_spacy(text)

    # Extract payment information
    payments = extract_payments(text)

    # Structure the data
    entities_df_hf = structure_entities(entities_hf)
    entities_df_spacy = structure_entities(entities_spacy)
    payments_df = structure_payments(payments)

    return entities_df_hf, entities_df_spacy, payments_df

if __name__ == "__main__":
    file_path = input("Enter the path of the document: ")
    file_type = file_path.split(".")[-1].lower()

    # Analyze the document
    entities_df_hf, entities_df_spacy, payments_df = analyze_document(file_path, file_type)

    # Display the results
    print("\nNamed Entities (Hugging Face):")
    print(entities_df_hf)

    print("\nNamed Entities (spaCy):")
    print(entities_df_spacy)

    print("\nPayments Information:")
    print(payments_df)
