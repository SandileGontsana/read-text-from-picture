import streamlit as st
import pdfplumber
from docx import Document
import pandas as pd
import pytesseract
from PIL import Image
import re
import string
from transformers import pipeline
import spacy

# Set the path to Tesseract executable (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize Hugging Face NER pipeline
ner_pipeline = pipeline("ner", model="xlm-roberta-large-finetuned-conll03-english", aggregation_strategy="max")

# Helper functions for document parsing
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text

def extract_text_from_excel(file):
    data = pd.read_excel(file, sheet_name=None)
    text = ""
    for sheet_name, sheet_data in data.items():
        text += sheet_data.to_string(index=False) + "\n"
    return text

def extract_text_from_image(image):
    try:
        # Open the image file and perform OCR using Tesseract
        img = Image.open(image).convert("L")  # Convert to grayscale for better accuracy
        text = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
        return text
    except Exception as e:
        return f"Error during image text extraction: {e}"

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

# Text preprocessing
def preprocess_text(text):
    text = re.sub(r'[^\w\s.,R,$]', '', text)
    text = text.lower()
    text = " ".join(text.split())
    return text

# Named Entity Recognition
def extract_entities_hf(text):
    entities = ner_pipeline(text)
    return [(ent['word'], ent['entity_group']) for ent in entities]

def extract_entities_spacy(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def post_process_spacy_entities(entities):
    corrected_entities = []
    for entity, label in entities:
        if label == "DATE" and entity.isnumeric() and len(entity) == 4:
            corrected_entities.append((entity, "INVOICE"))
        else:
            corrected_entities.append((entity, label))
    return corrected_entities

# Extract payments information
def extract_payments(text):
    lines = text.split("\n")
    payments = [line.strip() for line in lines if any(keyword in line.lower() for keyword in ["payment", "transfer", "debit", "credit"])]
    return payments

# Extract currencies
def extract_currencies(text):
    # Regex to match currency patterns like R1000.00, R 1,000, ZAR1000
    currency_pattern = r'\b(R[\s]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|ZAR[\s]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)'
    matches = re.findall(currency_pattern, text)
    return matches

# Structuring Data
def structure_entities(entities):
    return pd.DataFrame(entities, columns=["Entity", "Label"])

def structure_payments(payments):
    return pd.DataFrame(payments, columns=["Payment Details"])

def structure_currencies(currencies):
    return pd.DataFrame(currencies, columns=["Currency Details"])

# Main Streamlit App
def main():
    st.title("Document Analysis Tool")
    st.write("Upload a document for analysis (PDF, Word, Excel, Text, Image)")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx", "xlsx","csv", "jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1].lower()

        # Extract text based on file type
        if file_type == "pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif file_type == "docx":
            text = extract_text_from_docx(uploaded_file)
        elif file_type == "txt":
            text = extract_text_from_txt(uploaded_file)
        elif file_type == "xlsx":
            text = extract_text_from_excel(uploaded_file)
        elif file_type == "csv":
            text = extract_text_from_excel(uploaded_file)
        elif file_type in ["jpg", "png", "jpeg"]:
            text = extract_text_from_image(uploaded_file)
        else:
            st.error("Unsupported file type.")
            return

        # Display extracted text
        st.subheader("Extracted Text")
        st.text_area("Text Content", text, height=200)

        # Preprocess the text
        preprocessed_text = preprocess_text(text)

        # Named Entity Recognition
        st.subheader("Named Entity Recognition")
        hf_entities = extract_entities_hf(preprocessed_text)
        spacy_entities = extract_entities_spacy(preprocessed_text)
        spacy_entities = post_process_spacy_entities(spacy_entities)

        hf_df = structure_entities(hf_entities)
        spacy_df = structure_entities(spacy_entities)

        st.write("Entities (Hugging Face):")
        st.dataframe(hf_df)

        st.write("Entities (spaCy):")
        st.dataframe(spacy_df)

        # Extract payments
        st.subheader("Payment Information")
        payments = extract_payments(preprocessed_text)
        payments_df = structure_payments(payments)

        if not payments_df.empty:
            st.dataframe(payments_df)
        else:
            st.write("No payment-related information found.")

        # Extract currencies
        st.subheader("Currency Information")
        currencies = extract_currencies(preprocessed_text)
        currencies_df = structure_currencies(currencies)

        if not currencies_df.empty:
            st.dataframe(currencies_df)
        else:
            st.write("No currency information found.")

if __name__ == "__main__":
    main()
