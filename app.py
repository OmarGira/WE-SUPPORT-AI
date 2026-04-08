import streamlit as st
import requests
import time
from PIL import Image
import PyPDF2
import docx
import uuid
import base64
import os
import io
from bs4 import BeautifulSoup

# ================= UI & Asset Initialization =================
# Streamlit's native st.image() is great, but to embed images inside custom HTML/CSS 
# (like our centered header), we need to convert the local image into a Base64 string.
def get_image_base64(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

logo_path = "assets/logo.png"
logo_base64 = get_image_base64(logo_path)

# Fallback mechanism: If another dev clones the repo but forgets to add the logo, 
# the app won't crash. It'll just use an emoji.
try:
    logo_img = Image.open(logo_path)
except FileNotFoundError:
    logo_img = "💜"

# Must be the very first Streamlit command. Sets the browser tab title and favicon.
st.set_page_config(page_title="WE Support AI", page_icon=logo_img, layout="wide")

# ================= Global CSS Injection =================
# Standard Streamlit looks a bit too much like a data dashboard. 
# We inject custom CSS to give it an enterprise, ChatGPT-like chat interface.
st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    /* Smooth fade-in animation for new chat bubbles */
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    
    /* Sidebar chat history item styling */
    .chat-history-item { padding: 10px; border-radius: 8px; margin-bottom: 5px; cursor: pointer; transition: 0.3s; font-size: 14px; color: #4A2558; }
    .chat-history-item:hover { background-color: #f0e6f2; }
    
    /* Chat bubbles styling (WE Brand Colors) */
    [data-testid="stChatMessage"] { border-radius: 15px; padding: 1.5rem; margin-bottom: 1rem; animation: fadeIn 0.5s ease-in; }
    [data-testid="stChatMessageAssistant"] { background: linear-gradient(135deg, #4A2558 0%, #351a40 100%) !important; color: white !important; }
    [data-testid="stChatMessageAssistant"] a { color: #f0e6f2 !important; text-decoration: underline; font-weight: bold; }
    [data-testid="stChatMessageUser"] { background-color: #ffffff !important; border: 1px solid #e0e0e0; }
    
    /* General UI overrides */
    .stButton>button { width: 100%; background-color: #4A2558 !important; color: white !important; border-radius: 20px !important; }
    .header-container { display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 5px; }
    .header-title { color: #4A2558; font-weight: 900; font-size: 2.5rem; margin: 0; }
    </style>
""", unsafe_allow_html=True)

# ================= Session State Management =================
# This is how we achieve the "Multiple Chats" feature.
# We store a dictionary of chats mapped to unique UUIDs. 
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {} 
if "current_chat_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.current_chat_id = new_id
    # Initialize the first chat with a greeting message.
    st.session_state.all_chats[new_id] = [{"role": "assistant", "content": "أهلاً بك في WE! أنا مساعدك الذكي، كيف يمكنني مساعدتك؟"}]

def start_new_chat():
    """Generates a new UUID and switches the active UI context to a fresh chat."""
    new_id = str(uuid.uuid4())
    st.session_state.current_chat_id = new_id
    st.session_state.all_chats[new_id] = [{"role": "assistant", "content": "محادثة جديدة.. أهلاً بك!"}]

# ================= Sidebar Layout =================
with st.sidebar:
    st.image(logo_img, width=120)
    if st.button("➕ محادثة جديدة"):
        start_new_chat()
        st.rerun() # Force a UI refresh to show the new empty chat.
    
    st.markdown("---")
    st.subheader("📜 المحادثات السابقة")
    
    # Render previous chat sessions as clickable buttons.
    for chat_id in list(st.session_state.all_chats.keys()):
        chat_title = "محادثة جديدة"
        # We loop through the chat history to find the user's *first* message and use it as the chat title.
        for msg in st.session_state.all_chats[chat_id]:
            if msg["role"] == "user":
                chat_title = msg["content"][:25] + "..."
                break
                
        # If the user clicks a previous chat, we update the current_chat_id and refresh the UI.
        if st.button(f"💬 {chat_title}", key=chat_id):
            st.session_state.current_chat_id = chat_id
            st.rerun()

    st.markdown("---")
    
    # 1. HTML Scraping Input
    # Useful for support agents who just want to copy-paste page source code from a live website.
    st.subheader("🌐 تحليل أكواد المواقع (HTML)")
    html_input = st.text_area("انسخ كود الموقع هنا لكي يحلله البوت:", height=100)
    
    st.markdown("---")
    st.subheader("📎 المرفقات (صور وملفات)")
    
    # Toggle for local vs cloud vision processing.
    use_local_vision = st.checkbox("Enternal Embeddings for photo (BLIP)", value=False)
    uploaded_file = st.file_uploader("ارفع ملف", type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "html", "htm"])
    
    extracted_text = ""
    image_base64_data = None 
    
    # --- Client-Side Data Parsing ---
    # We parse files here in the frontend rather than sending raw binaries to the backend. 
    # This keeps HTTP payloads extremely lightweight (just pure strings).
    
    if html_input.strip():
        try:
            # Strip out all the <div>, <script>, and CSS tags. We only want the human-readable text.
            soup = BeautifulSoup(html_input, "html.parser")
            extracted_text += "\n" + soup.get_text(separator="\n", strip=True)
            st.success("تم فلترة كود الـ HTML بنجاح ✔️")
        except:
            st.error("كود HTML غير صالح")

    if uploaded_file:
        file_ext = uploaded_file.name.lower()
        
        # Image Handling
        if file_ext.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(uploaded_file)
            st.image(img, caption="تم رفع الصورة")
            
            # CRITICAL: We downscale the image to 800x800 and drop quality to 85%.
            # Why? Because massive 4K images converted to Base64 will easily trigger a 
            # "413 Payload Too Large" error on the FastAPI server. This prevents server crashes.
            img.thumbnail((800, 800))
            buffered = io.BytesIO()
            img_format = img.format if img.format else "JPEG"
            if img_format == "MPO": img_format = "JPEG"
            img.save(buffered, format=img_format, quality=85)
            image_base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
        # HTML Handling
        elif file_ext.endswith(('.html', '.htm')):
            html_content = uploaded_file.getvalue().decode("utf-8")
            soup = BeautifulSoup(html_content, "html.parser")
            extracted_text += "\n" + soup.get_text(separator="\n", strip=True)
            
        # PDF / Docx / Txt Handlers
        elif file_ext.endswith('.pdf'):
            reader = PyPDF2.PdfReader(uploaded_file)
            extracted_text += "\n" + "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        elif file_ext.endswith('.docx'):
            doc = docx.Document(uploaded_file)
            extracted_text += "\n" + "\n".join([p.text for p in doc.paragraphs])
        elif file_ext.endswith('.txt'):
            extracted_text += "\n" + uploaded_file.getvalue().decode("utf-8")

# ================= Main Chat Interface =================
# Render the custom header
if logo_base64:
    st.markdown(f'<div class="header-container"><img src="data:image/png;base64,{logo_base64}" width="80"><h1 class="header-title">WE Support AI</h1></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="header-container"><h1 class="header-title">WE Support AI</h1></div>', unsafe_allow_html=True)

# Render the active conversation history
current_messages = st.session_state.all_chats[st.session_state.current_chat_id]
for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(f'<div dir="rtl" style="text-align: right; line-height: 1.8;">{message["content"]}</div>', unsafe_allow_html=True)

# Handle new user input
if prompt := st.chat_input("اكتب سؤالك هنا..."):
    # Immediately render the user's question to the UI so it feels responsive.
    current_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div dir="rtl" style="text-align: right; line-height: 1.8;">{prompt}</div>', unsafe_allow_html=True)

    # Show a loading spinner while we wait for the FastAPI backend.
    with st.chat_message("assistant"):
        with st.spinner("يتم تفكير"):
            try:
                # Construct the payload for our REST API.
                payload = {
                    "query": prompt,
                    "file_data": extracted_text,
                    "image_base64": image_base64_data,
                    "use_local_vision": use_local_vision
                }
                
                # We enforce a 60-second timeout. If the AI model stalls or the backend hangs,
                # the UI will throw a clean error instead of spinning indefinitely.
                response = requests.post("http://localhost:8000/ask", json=payload, timeout=60)
                response.raise_for_status() # Raises an exception if HTTP status is 4xx or 5xx.
                
                result = response.json()
                answer_text = result.get("answer", "عذراً، حدث خطأ.")
                sources = result.get("sources", [])
                
                final_response = answer_text
                
                # Dynamic Source Citation Formatting
                # We take the sources returned by the backend and format them as clean, 
                # clickable Markdown links appended to the bottom of the response.
                if sources:
                    final_response += "\n\n---\n**🔗 روابط ذات صلة للاستزادة:**\n"
                    for s in sources:
                        url = s.get('url', '#')
                        title = s.get('title', 'مصدر من WE')
                        if url.startswith("http"):
                            final_response += f"- [{title}]({url})\n"
                        else:
                            final_response += f"- {title}\n"
                
                # Render the final AI answer to the UI.
                st.markdown(f'<div dir="rtl" style="text-align: right; line-height: 1.8;">{final_response}</div>', unsafe_allow_html=True)

                # Save the assistant's reply to the session state so it persists on reload.
                current_messages.append({"role": "assistant", "content": final_response})
                st.rerun() 
                
            except requests.exceptions.RequestException as e:
                # Catch network errors (e.g., if the user forgot to start the uvicorn server).
                st.error(" لا يمكن الاتصال بالسيرفر. تأكد من تشغيل ملف main.py")