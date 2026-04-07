import streamlit as st
import time
from PIL import Image
import PyPDF2
import docx
import uuid
import base64
import os
import io
from bs4 import BeautifulSoup
from src.rag_pipeline import WEAssistant

def get_image_base64(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

logo_path = "assets/logo.png"
logo_base64 = get_image_base64(logo_path)

try:
    logo_img = Image.open(logo_path)
except FileNotFoundError:
    logo_img = "💜"

st.set_page_config(page_title="WE Support AI", page_icon=logo_img, layout="wide")

@st.cache_resource
def load_bot():
    return WEAssistant()

bot = load_bot()

# ================= CSS =================
st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    .chat-history-item { padding: 10px; border-radius: 8px; margin-bottom: 5px; cursor: pointer; transition: 0.3s; font-size: 14px; color: #4A2558; }
    .chat-history-item:hover { background-color: #f0e6f2; }
    [data-testid="stChatMessage"] { border-radius: 15px; padding: 1.5rem; margin-bottom: 1rem; animation: fadeIn 0.5s ease-in; }
    [data-testid="stChatMessageAssistant"] { background: linear-gradient(135deg, #4A2558 0%, #351a40 100%) !important; color: white !important; }
    [data-testid="stChatMessageAssistant"] a { color: #f0e6f2 !important; text-decoration: underline; font-weight: bold; }
    [data-testid="stChatMessageUser"] { background-color: #ffffff !important; border: 1px solid #e0e0e0; }
    .stButton>button { width: 100%; background-color: #4A2558 !important; color: white !important; border-radius: 20px !important; }
    .header-container { display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 5px; }
    .header-title { color: #4A2558; font-weight: 900; font-size: 2.5rem; margin: 0; }
    </style>
""", unsafe_allow_html=True)

# ================= إدارة الحالة =================
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {} 
if "current_chat_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.current_chat_id = new_id
    st.session_state.all_chats[new_id] = [{"role": "assistant", "content": "أهلاً بك في WE! أنا مساعدك الذكي، كيف يمكنني مساعدتك؟"}]

def start_new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.current_chat_id = new_id
    st.session_state.all_chats[new_id] = [{"role": "assistant", "content": "محادثة جديدة.. أهلاً بك!"}]
    bot.memory = []

# ================= الشريط الجانبي =================
with st.sidebar:
    st.image(logo_img, width=120)
    if st.button("➕ محادثة جديدة"):
        start_new_chat()
        st.rerun()
    
    st.markdown("---")
    st.subheader("📜 المحادثات السابقة")
    for chat_id in list(st.session_state.all_chats.keys()):
        chat_title = "محادثة جديدة"
        for msg in st.session_state.all_chats[chat_id]:
            if msg["role"] == "user":
                chat_title = msg["content"][:25] + "..."
                break
        if st.button(f"💬 {chat_title}", key=chat_id):
            st.session_state.current_chat_id = chat_id
            st.rerun()

    st.markdown("---")
    
    # 🔥 [الجديد] مكان مخصص لكتابة أو لصق أكواد الـ HTML
    st.subheader("🌐 لصق كود HTML مباشرة")
    html_input = st.text_area("انسخ كود الموقع هنا لكي يحلله البوت:", height=100)
    
    st.markdown("---")
    st.subheader("📎 المرفقات (صور وملفات)")
    
    use_local_vision = st.checkbox("🤖 تحليل الصورة محلياً (توفير Tokens)", value=False)
    uploaded_file = st.file_uploader("ارفع ملف", type=["pdf", "docx", "txt", "png", "jpg", "jpeg"])
    
    extracted_text = ""
    image_base64_data = None 
    
    # 1. معالجة الـ HTML المكتوب يدوياً
    if html_input.strip():
        try:
            soup = BeautifulSoup(html_input, "html.parser")
            extracted_text += "\n" + soup.get_text(separator="\n", strip=True)
            st.success("تم فلترة كود الـ HTML بنجاح ✔️")
        except:
            st.error("كود HTML غير صالح")

    # 2. معالجة الملفات المرفوعة
    if uploaded_file:
        file_ext = uploaded_file.name.lower()
        if file_ext.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(uploaded_file)
            st.image(img, caption="تم رفع الصورة")
            img.thumbnail((800, 800))
            buffered = io.BytesIO()
            img_format = img.format if img.format else "JPEG"
            if img_format == "MPO": img_format = "JPEG"
            img.save(buffered, format=img_format, quality=85)
            image_base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        elif file_ext.endswith('.pdf'):
            reader = PyPDF2.PdfReader(uploaded_file)
            extracted_text += "\n" + "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        elif file_ext.endswith('.docx'):
            doc = docx.Document(uploaded_file)
            extracted_text += "\n" + "\n".join([p.text for p in doc.paragraphs])
        elif file_ext.endswith('.txt'):
            extracted_text += "\n" + uploaded_file.getvalue().decode("utf-8")

# ================= واجهة الشات الرئيسية =================
if logo_base64:
    st.markdown(f'<div class="header-container"><img src="data:image/png;base64,{logo_base64}" width="80"><h1 class="header-title">WE Support AI</h1></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="header-container"><h1 class="header-title">WE Support AI</h1></div>', unsafe_allow_html=True)

current_messages = st.session_state.all_chats[st.session_state.current_chat_id]
for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(f'<div dir="rtl" style="text-align: right; line-height: 1.8;">{message["content"]}</div>', unsafe_allow_html=True)

if prompt := st.chat_input("اكتب سؤالك هنا..."):
    current_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div dir="rtl" style="text-align: right; line-height: 1.8;">{prompt}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث في بيانات WE... 🔍"):
            try:
                result = bot.ask(
                    query=prompt, 
                    file_data=extracted_text, 
                    image_base64=image_base64_data,
                    use_local_vision=use_local_vision
                )
                
                # 🔥 [الجديد] تجهيز الرد مدمج مع الروابط تحت بعض
                answer_text = result.get("answer", "عذراً، حدث خطأ.")
                sources = result.get("sources", [])
                
                final_response = answer_text
                
                # دمج الروابط في نهاية الإجابة مباشرة بشكل جميل
                if sources:
                    final_response += "\n\n---\n**🔗 روابط ذات صلة للاستزادة:**\n"
                    for s in sources:
                        url = s.get('url', '#')
                        title = s.get('title', 'مصدر من WE')
                        # لو الرابط حقيقي يبدأ ب http
                        if url.startswith("http"):
                            final_response += f"- [{title}]({url})\n"
                        else:
                            final_response += f"- {title}\n"
                
                # عرض الإجابة الكاملة
                st.markdown(f'<div dir="rtl" style="text-align: right; line-height: 1.8;">{final_response}</div>', unsafe_allow_html=True)

                current_messages.append({"role": "assistant", "content": final_response})
                st.rerun() 
            except Exception as e:
                st.error(f"خطأ في معالجة البيانات: {e}")