import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import torch # Explicitly import torch

# --- Page Config (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="VSS Calculator")

# --- Constants for Example ---
EXAMPLE_PROMPT = "What are the best running shoes for marathon training?"
EXAMPLE_URL = "https://www.runnersworld.com/uk/gear/shoes/a776513/best-running-shoes/" # Example URL, replace if needed

# --- Configuration & Model Loading ---
@st.cache_resource
def load_model():
    """Loads the Sentence Transformer model."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        print(f"Error loading Sentence Transformer model: {e}")
        return None

model = load_model()

# --- Helper Functions ---
def fetch_and_parse_url(url):
    """Fetches content from a URL and parses the main text."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15) # Increased timeout slightly
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        main_content = soup.find('main') or soup.find('article') or soup.body
        if main_content:
            for script_or_style in main_content(['script', 'style']):
                script_or_style.decompose()
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        text = ' '.join(text.split())
        return text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL '{url}': {e}")
        return None
    except Exception as e:
        st.error(f"Error parsing URL '{url}': {e}")
        return None

def calculate_vss(text1, text2, model_instance):
    """Calculates the Vector Similarity Score (Cosine Similarity)."""
    if not model_instance or not text1 or not text2:
        st.error("Model not loaded or text input missing.")
        return None
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_instance.to(device)
        embedding1 = model_instance.encode(text1, convert_to_tensor=True, device=device)
        embedding2 = model_instance.encode(text2, convert_to_tensor=True, device=device)
        cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
        return cosine_scores.item()
    except Exception as e:
        st.error(f"Error calculating similarity: {e}")
        return None

# --- Initialize Session State ---
# We need to store the input values in session state so the button can update them
if 'prompt_input' not in st.session_state:
    st.session_state.prompt_input = ""
if 'content_url_input' not in st.session_state:
    st.session_state.content_url_input = ""
if 'content_text_input' not in st.session_state:
    st.session_state.content_text_input = ""
if 'input_type' not in st.session_state:
    st.session_state.input_type = "URL" # Default to URL

# --- Streamlit App UI ---
st.title("📊 Vector Similarity Score (VSS) Calculator")
st.markdown("Enter a prompt and provide content (via URL or direct text) to calculate the semantic similarity.")

if model is None:
    st.error("Fatal Error: Could not load the Sentence Transformer model. Please check the terminal logs or network connection and restart.")
    st.stop()

# --- Example Button ---
if st.button("Load Example"):
    st.session_state.prompt_input = EXAMPLE_PROMPT
    st.session_state.input_type = "URL" # Set input type to URL for example
    st.session_state.content_url_input = EXAMPLE_URL
    st.session_state.content_text_input = "" # Clear text input if URL example is loaded
    # Rerun to update widgets with new session state values
    st.rerun()

# --- Inputs ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Enter Your Prompt")
    # Use session_state key to manage the value
    st.session_state.prompt_input = st.text_area(
        "Prompt:",
        value=st.session_state.prompt_input, # Use value from session state
        height=100,
        placeholder="e.g., What are the best running shoes for sore feet?",
        key="prompt_widget" # Assign a key for direct access if needed, though binding to session_state often suffices
    )


with col2:
    st.subheader("2. Provide Content")
    # Use session_state key to manage the radio button state
    st.session_state.input_type = st.radio(
        "Content Source:",
        ("URL", "Text"),
        index=["URL", "Text"].index(st.session_state.input_type), # Set index based on session state
        horizontal=True,
        key="input_type_widget"
    )

    content_input_value = None # Variable to hold the actual input value for calculation
    if st.session_state.input_type == "URL":
        # Use session_state key to manage the value
        st.session_state.content_url_input = st.text_input(
            "Enter URL:",
            value=st.session_state.content_url_input, # Use value from session state
            placeholder="https://www.example.com/page",
            key="url_widget"
        )
        content_input_value = st.session_state.content_url_input
    else:
        # Use session_state key to manage the value
        st.session_state.content_text_input = st.text_area(
            "Paste Text:",
            value=st.session_state.content_text_input, # Use value from session state
            height=200,
            placeholder="Paste your content here...",
            key="text_widget"
        )
        content_input_value = st.session_state.content_text_input


# --- Calculation Trigger ---
st.divider()
calculate_button = st.button("Calculate VSS", type="primary")

# --- Output ---
st.subheader("Results")

if calculate_button:
    content_text = None
    text_snippet = "N/A"

    # Use the values directly from session state for validation
    prompt_value = st.session_state.prompt_input
    current_input_type = st.session_state.input_type

    if not prompt_value:
        st.warning("Please enter a prompt.")
    elif not content_input_value: # Check the actual value being used
        st.warning(f"Please provide content ({current_input_type}).")
    else:
        if current_input_type == "URL":
            with st.spinner(f"Fetching and parsing URL: {content_input_value}..."):
                content_text = fetch_and_parse_url(content_input_value)
        else:
            content_text = content_input_value # Use pasted text

        if content_text:
            text_snippet = content_text[:500] + "..." if len(content_text) > 500 else content_text

            with st.spinner("Calculating Vector Similarity Score..."):
                vss_score = calculate_vss(prompt_value, content_text, model)

            if vss_score is not None:
                st.metric(label="Vector Similarity Score (VSS)", value=f"{vss_score:.4f}")
                if vss_score >= 0.8:
                    st.success("High Similarity (Score ≥ 0.8)")
                elif vss_score >= 0.6:
                    st.info("Medium Similarity (Score 0.6 - 0.79)")
                else:
                    st.warning("Low Similarity (Score < 0.6)")

            with st.expander("View Content Snippet (First 500 chars)"):
                st.text(text_snippet)
        else:
            if current_input_type == "URL" and content_text is None:
                 st.error("Failed to get content from the provided URL. Check the URL and network connection.")
            elif current_input_type == "Text" and not content_text:
                 st.warning("Pasted text content is empty.")


# --- Sidebar Instructions ---
# (Sidebar code remains the same as before)
st.sidebar.header("How to Run Locally")
st.sidebar.markdown("""
1.  **Save:** Save this code as `app.py`.
2.  **Save Requirements:** Save the requirements list as `requirements.txt` in the same folder.
3.  **Install:** Open a terminal/command prompt in that folder and run:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run:** In the same terminal, run:
    ```bash
    streamlit run app.py
    ```
5.  Your browser should open with the app.
""")

st.sidebar.header("How to Deploy (Example: Streamlit Community Cloud)")
st.sidebar.markdown("""
1.  **GitHub:** Create a GitHub repository and push `app.py` and `requirements.txt` to it.
2.  **Streamlit Cloud:** Sign up/log in to [share.streamlit.io](https://share.streamlit.io/).
3.  **Deploy:** Click "New app", connect your GitHub account, select the repository, branch (usually `main`), and the `app.py` file. Click "Deploy!".
""")