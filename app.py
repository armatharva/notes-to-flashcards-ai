import streamlit as st
from transformers import pipeline
import torch
import re

# Initialize the summarization pipeline
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

def summarize_text(text, max_length=130, min_length=30):
    # Split text into chunks if it's too long (BART has a max input length of 1024 tokens)
    max_chunk_length = 1000
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    
    summarizer = load_summarizer()
    summaries = []
    
    for chunk in chunks:
        summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    return " ".join(summaries)

def generate_flashcards(summary):
    # Split summary into sentences
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    flashcards = []
    
    # Generate main idea flashcard
    flashcards.append({
        "question": "What is the main idea of the notes?",
        "answer": summary
    })
    
    # Generate flashcards for key points
    for i, sentence in enumerate(sentences[:3], 1):  # Take first 3 sentences for key points
        flashcards.append({
            "question": f"What is key point {i} from the notes?",
            "answer": sentence
        })
    
    return flashcards

def display_flashcard(flashcard, index):
    with st.expander(f"Flashcard {index + 1}: {flashcard['question']}"):
        st.write("**Answer:**")
        st.write(flashcard['answer'])

def main():
    st.title("Notes to Flashcards AI")
    st.write("Upload your notes to get started!")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a text file", type=['txt'])

    if uploaded_file is not None:
        # Read the file content
        content = uploaded_file.getvalue().decode("utf-8")
        
        # Display the content
        st.subheader("File Content:")
        st.text_area("Notes", content, height=300)

        # Add a summarize button
        if st.button("Summarize Notes"):
            with st.spinner("Generating summary..."):
                try:
                    summary = summarize_text(content)
                    st.subheader("Summary:")
                    st.write(summary)
                    
                    # Generate and display flashcards
                    st.subheader("Generated Flashcards")
                    flashcards = generate_flashcards(summary)
                    
                    # Create two columns for flashcards
                    col1, col2 = st.columns(2)
                    
                    # Display flashcards in a grid
                    for i, flashcard in enumerate(flashcards):
                        with col1 if i % 2 == 0 else col2:
                            display_flashcard(flashcard, i)
                    
                except Exception as e:
                    st.error(f"An error occurred while processing: {str(e)}")

if __name__ == "__main__":
    main() 