# 📄 Document Analysis Tool

A Streamlit-based app that extracts, analyzes, and displays key insights from various document types like PDFs, Word files, Excel sheets, images, and more. It supports OCR, Named Entity Recognition (NER), and financial data extraction using Hugging Face and spaCy.

---

## 🚀 Features

- 📂 Supports multiple file types: PDF, DOCX, TXT, XLSX, CSV, JPG, PNG
- 🔍 Extracts text using:
  - PDFPlumber for PDFs
  - python-docx for Word
  - pandas for Excel/CSV
  - pytesseract OCR for images
- 🧠 Named Entity Recognition with:
  - Hugging Face (`xlm-roberta-large-finetuned-conll03-english`)
  - spaCy (`en_core_web_sm`)
- 💳 Detects payment-related lines and currency values (ZAR, R, etc.)
- 🧹 Text preprocessing and structured output with DataFrames
- 💡 Fully interactive with Streamlit UI

---

## 📁 File Structure

