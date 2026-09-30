# ============================================================
# AI LEGAL DOCUMENT SIMPLIFIER
# app.py
# ============================================================

import streamlit as st
import pandas as pd
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Legal Document Simplifier",
    page_icon="⚖️",
    layout="wide"
)


# ============================================================
# 2. TITLE
# ============================================================

st.title("⚖️ AI Legal Document Simplifier")

st.write(
    "Simplify complex legal language into clear and easy-to-understand English "
    "while preserving the original legal meaning."
)

st.warning(
    "This application provides informational text simplification only. "
    "It is not a substitute for professional legal advice."
)


# ============================================================
# 3. LOAD AI MODEL
# ============================================================

MODEL_NAME = "google/flan-t5-base"


@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    return tokenizer, model


# ============================================================
# 4. SIMPLIFICATION FUNCTION
# ============================================================

def simplify_legal_text(text):

    tokenizer, model = load_model()

    prompt = f"""
You are a professional legal document simplification assistant.

Your task is to simplify the following legal text into clear,
simple, and professional English.

Rules:

1. Preserve the exact legal meaning.
2. Do not add information.
3. Do not remove important information.
4. Preserve names, dates, numbers, amounts and legal references.
5. Preserve rights and obligations.
6. Preserve conditions and exceptions.
7. Do not change "shall", "must", "may", "unless", "except",
   or similar legal conditions incorrectly.
8. Use simple English that an ordinary person can understand.
9. Do not provide legal advice.
10. Return only the simplified text.

Original legal text:

{text}

Simplified text:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            num_beams=4,
            early_stopping=True
        )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return result.strip()


# ============================================================
# 5. SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

option = st.sidebar.radio(
    "Choose an option:",
    [
        "Simplify Legal Text",
        "Process Dataset"
    ]
)


# ============================================================
# 6. SINGLE TEXT SIMPLIFIER
# ============================================================

if option == "Simplify Legal Text":

    st.header("📄 Simplify Legal Document")

    legal_text = st.text_area(
        "Enter your legal text:",
        height=300,
        placeholder="Paste your legal document or legal clause here..."
    )

    if st.button("🔍 Simplify Legal Text"):

        if not legal_text.strip():

            st.error("Please enter some legal text.")

        else:

            with st.spinner("Simplifying legal text..."):

                simplified_text = simplify_legal_text(
                    legal_text
                )

            st.subheader("Original Text")

            st.text_area(
                "Original",
                legal_text,
                height=250
            )

            st.subheader("Simplified Text")

            st.text_area(
                "Simplified",
                simplified_text,
                height=250
            )


# ============================================================
# 7. DATASET PROCESSING
# ============================================================

elif option == "Process Dataset":

    st.header("📊 Process Legal Dataset")

    st.write(
        "This section loads the Shekswess/legal-documents dataset "
        "and creates a new simplified_text column."
    )

    if st.button("📥 Load Dataset"):

        with st.spinner("Loading dataset..."):

            try:

                dataset = load_dataset(
                    "Shekswess/legal-documents"
                )

                train_data = dataset["train"]

                df = train_data.to_pandas()

                st.success(
                    f"Dataset loaded successfully! "
                    f"{len(df)} records found."
                )

                st.write("### Original Dataset")

                st.dataframe(
                    df.head(10),
                    use_container_width=True
                )

                st.session_state["df"] = df

            except Exception as e:

                st.error(
                    f"Error loading dataset: {e}"
                )


    # --------------------------------------------------------
    # PROCESS DATASET
    # --------------------------------------------------------

    if "df" in st.session_state:

        df = st.session_state["df"]

        st.write("---")

        st.subheader("Generate Simplified Text")

        number_of_records = st.number_input(
            "Number of records to process:",
            min_value=1,
            max_value=len(df),
            value=min(5, len(df)),
            step=1
        )

        st.info(
            "For testing, start with 5–10 records before processing "
            "the complete dataset."
        )

        if st.button("🚀 Generate Simplified Dataset"):

            progress_bar = st.progress(0)

            status_text = st.empty()

            # Create new column without changing original text
            simplified_results = []

            for i in range(number_of_records):

                original_text = df.iloc[i]["text"]

                status_text.write(
                    f"Processing record {i + 1} of "
                    f"{number_of_records}..."
                )

                try:

                    simplified = simplify_legal_text(
                        str(original_text)
                    )

                except Exception as e:

                    simplified = (
                        f"ERROR: {str(e)}"
                    )

                simplified_results.append(
                    simplified
                )

                progress_bar.progress(
                    (i + 1) / number_of_records
                )

            # Create a copy
            updated_df = df.copy()

            # Add new column
            updated_df["simplified_text"] = ""

            # Add generated results
            for i in range(number_of_records):

                updated_df.loc[
                    i,
                    "simplified_text"
                ] = simplified_results[i]

            # Save updated dataset
            st.session_state[
                "updated_df"
            ] = updated_df

            status_text.write(
                "Processing completed!"
            )

            st.success(
                f"{number_of_records} records simplified successfully."
            )


        # ----------------------------------------------------
        # DISPLAY UPDATED DATASET
        # ----------------------------------------------------

        if "updated_df" in st.session_state:

            updated_df = st.session_state[
                "updated_df"
            ]

            st.write("---")

            st.subheader(
                "📋 Original + Simplified Dataset"
            )

            st.dataframe(
                updated_df.head(
                    number_of_records
                ),
                use_container_width=True
            )

            # ------------------------------------------------
            # DOWNLOAD CSV
            # ------------------------------------------------

            csv_data = updated_df.to_csv(
                index=False
            )

            st.download_button(
                label="⬇️ Download Updated Dataset",
                data=csv_data,
                file_name="legal_documents_simplified.csv",
                mime="text/csv"
            )


# ============================================================
# 8. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI Legal Document Simplifier | "
    "NLP-based legal text simplification"
)
