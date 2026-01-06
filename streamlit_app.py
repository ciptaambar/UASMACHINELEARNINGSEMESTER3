import streamlit as st
import time
from chatbot_engine import ChatbotEngine

# Page configuration
st.set_page_config(
    page_title="Beauty Paw Chatbot",
    page_icon="🐾",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Model Loading with Caching ---
@st.cache_resource
def get_chatbot_engine():
    # Helper to load the engine only once
    return ChatbotEngine(model_dir='models', dataset_path='datasets.json')

try:
    with st.spinner("Sedang menyiapkan Beauty Paw Chatbot..."):
        chatbot = get_chatbot_engine()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.title("🐾 Beauty Paw")
    st.markdown("Asisten virtual untuk kebutuhan skincare Anda.")
    
    st.subheader("Topik Populer:")
    quick_actions = [
        "Apa itu serum?",
        "Rekomendasi untuk kulit berjerawat",
        "Cara pemesanan",
        "Metode pembayaran",
        "Jam operasional"
    ]
    
    for action in quick_actions:
        if st.button(action, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": action})

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "confidence" in message:
             st.caption(f"Confidence: {message['confidence']}%")

# Encode Chat Logic
if prompt := st.chat_input("Ketik pesan Anda..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Check if we need to generate a response (last message is user)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_text = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        with st.spinner("Mengetik..."):
            time.sleep(0.5) # Natural delay
            
            result = chatbot.get_response(user_text)
            response_text = result["response"]
            confidence = result.get("confidence", 0) * 100
            
            st.markdown(response_text)
            st.caption(f"Confidence: {confidence:.1f}%")
            
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text,
        "confidence": f"{confidence:.1f}"
    })
