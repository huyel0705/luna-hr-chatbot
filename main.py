import os
import json
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG & ĐỊNH DẠNG
# ==========================================
st.set_page_config(page_title="LUNA - Cổng Thông Tin Nhân Sự", page_icon="🏢", layout="wide")

class HRConfig:
    # Lấy thông tin từ file secrets.toml (hoặc cấu hình trên Streamlit Cloud)
    API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
    
    # Cấu hình Email Bot (Hệ thống dùng email này để gửi cho nhân viên)
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    SENDER_EMAIL = st.secrets.get("SENDER_EMAIL", "") 
    SENDER_PASSWORD = st.secrets.get("SENDER_PASSWORD", "") 
    
    DATA_DIR = "./data"
    MODEL_NAME = "gemini-2.5-flash"
    EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
    COMPANY_DOMAIN = "@rikai.technology" # Tên miền bắt buộc
    USERS_DB = "users_db.json"     
    HISTORY_DIR = "./chat_history" 

# Tạo thư mục nếu chưa có
if not os.path.exists(HRConfig.HISTORY_DIR):
    os.makedirs(HRConfig.HISTORY_DIR)
if not os.path.exists(HRConfig.DATA_DIR):
    os.makedirs(HRConfig.DATA_DIR)

# Giao diện CSS
st.markdown("""
<style>
    .stApp { background-color: #F8FAFC; }
    .main-header { text-align: center; color: #1E3A8A; font-weight: 800; font-size: 2.5rem; margin-bottom: 1rem;}
    .stChatMessage { border-radius: 10px; padding: 10px; margin-bottom: 10px; }
    .stTabs [data-baseweb="tab-list"] { justify-content: center; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HỆ THỐNG QUẢN LÝ TÀI KHOẢN & LỊCH SỬ
# ==========================================
def load_users():
    if os.path.exists(HRConfig.USERS_DB):
        with open(HRConfig.USERS_DB, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(HRConfig.USERS_DB, "w") as f:
        json.dump(users, f)

def load_history(email):
    path = os.path.join(HRConfig.HISTORY_DIR, f"{email}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return [{"role": "assistant", "content": "Chào mừng bạn! LUNA có thể giúp gì cho bạn hôm nay?"}]

def save_history(email, messages):
    path = os.path.join(HRConfig.HISTORY_DIR, f"{email}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)

def send_otp_email(receiver_email, otp, subject):
    """Hàm gửi email. Nếu chưa điền SENDER_EMAIL, sẽ in OTP ra console để test."""
    if not HRConfig.SENDER_EMAIL or not HRConfig.SENDER_PASSWORD:
        print(f"\n[SYSTEM LOG] Giả lập gửi OTP: {otp} đến {receiver_email}\n")
        return True 
    
    try:
        msg = MIMEMultipart()
        msg['From'] = HRConfig.SENDER_EMAIL
        msg['To'] = receiver_email
        msg['Subject'] = subject
        body = f"Mã xác thực (OTP) của bạn là: {otp}\n\nVui lòng không chia sẻ mã này cho bất kỳ ai. Mã có hiệu lực trong phiên làm việc này."
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(HRConfig.SMTP_SERVER, HRConfig.SMTP_PORT)
        server.starttls()
        server.login(HRConfig.SENDER_EMAIL, HRConfig.SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Lỗi SMTP: {e}")
        return False

# Biến Session State
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "user_email" not in st.session_state: st.session_state.user_email = ""
if "messages" not in st.session_state: st.session_state.messages = []
if "reg_step" not in st.session_state: st.session_state.reg_step = 1
if "reg_email" not in st.session_state: st.session_state.reg_email = ""
if "reg_otp" not in st.session_state: st.session_state.reg_otp = ""
if "fg_step" not in st.session_state: st.session_state.fg_step = 1
if "fg_email" not in st.session_state: st.session_state.fg_email = ""
if "fg_otp" not in st.session_state: st.session_state.fg_otp = ""

# ==========================================
# 3. LỚP XỬ LÝ CHATBOT LUNA (AI Backend)
# ==========================================
class HRChatbot:
    def __init__(self):
        os.environ["GOOGLE_API_KEY"] = HRConfig.API_KEY
        self.embeddings = HuggingFaceEmbeddings(model_name=HRConfig.EMBEDDING_MODEL)
        self.vectorstore = self._init_vector_db()
        if self.vectorstore:
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 25})
            self.rag_chain = self._build_chain()
        else:
            self.rag_chain = None

    def _init_vector_db(self):
        all_files = [os.path.join(HRConfig.DATA_DIR, f) for f in os.listdir(HRConfig.DATA_DIR) 
                     if f.endswith(".docx") and not f.startswith("~$")]
        if not all_files: return None

        documents = []
        for file_path in all_files:
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500, separators=["\nĐiều ", "\nChương ", "\n\n", "\n", " "])
        chunks = text_splitter.split_documents(documents)
        return FAISS.from_documents(chunks, self.embeddings)

    def _build_chain(self):
        llm = ChatGoogleGenerativeAI(model=HRConfig.MODEL_NAME, temperature=0.3)
        system_prompt = (
            "Bạn là LUNA - Trợ lý Nhân sự chuyên nghiệp. Hãy trả lời dựa TRÊN ĐÚNG dữ liệu được cung cấp.\n\n"
            "🔴 QUY TẮC:\n"
            "1. KHÔNG nhắc đến số hiệu Điều, Khoản (Luật), (QĐ).\n"
            "2. KHÔNG tự chế số liệu. Trình bày mạch lạc bằng gạch đầu dòng.\n"
            "3. Nếu thông tin không có, báo chưa tìm thấy và đề nghị liên hệ phòng Nhân sự.\n\n"
            "DỮ LIỆU THAM KHẢO:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        return ({"context": self.retriever, "input": RunnablePassthrough()} | prompt | llm | StrOutputParser())

@st.cache_resource
def load_bot():
    return HRChatbot()

# ==========================================
# 4. GIAO DIỆN XÁC THỰC
# ==========================================
if not st.session_state.logged_in:
    st.markdown("<h1 class='main-header'>CỔNG THÔNG TIN NHÂN SỰ LUNA</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_register, tab_forgot = st.tabs(["🔐 Đăng nhập", "📝 Đăng ký", "🔑 Quên mật khẩu"])
        users = load_users()

        # ---------- TAB ĐĂNG NHẬP ----------
        with tab_login:
            st.markdown("### Đăng nhập hệ thống")
            login_input = st.text_input("Username hoặc Email", placeholder=f"VD: nguyenvan.a hoặc nguyenvan.a{HRConfig.COMPANY_DOMAIN}")
            login_pwd = st.text_input("Mật khẩu", type="password")
            
            if st.button("Đăng nhập", use_container_width=True, type="primary"):
                login_email = login_input if "@" in login_input else f"{login_input}{HRConfig.COMPANY_DOMAIN}"
                if login_email in users and users[login_email] == login_pwd:
                    st.session_state.logged_in = True
                    st.session_state.user_email = login_email
                    st.session_state.messages = load_history(login_email)
                    st.rerun()
                else:
                    st.error("Tài khoản hoặc mật khẩu không chính xác!")

        # ---------- TAB ĐĂNG KÝ ----------
        with tab_register:
            st.markdown("### Tạo tài khoản mới")
            if st.session_state.reg_step == 1:
                reg_email = st.text_input("Email công ty", placeholder=f"VD: name{HRConfig.COMPANY_DOMAIN}")
                if st.button("Nhận mã xác thực (OTP)", use_container_width=True):
                    if not reg_email.endswith(HRConfig.COMPANY_DOMAIN):
                        st.warning(f"Bắt buộc sử dụng email đuôi {HRConfig.COMPANY_DOMAIN}")
                    elif reg_email in users:
                        st.warning("Email này đã được đăng ký! Vui lòng đăng nhập.")
                    else:
                        otp = str(random.randint(100000, 999999))
                        if send_otp_email(reg_email, otp, "Mã xác thực đăng ký tài khoản LUNA"):
                            st.session_state.reg_email = reg_email
                            st.session_state.reg_otp = otp
                            st.session_state.reg_step = 2
                            if not HRConfig.SENDER_EMAIL: st.info(f"Chế độ TEST (Không có email bot). Mã OTP: **{otp}**")
                            st.rerun()
                        else:
                            st.error("Lỗi gửi email. Vui lòng thử lại.")

            elif st.session_state.reg_step == 2:
                st.info(f"Mã OTP đã được gửi đến: **{st.session_state.reg_email}**")
                entered_otp = st.text_input("Nhập mã OTP (6 số)", max_chars=6)
                c1, c2 = st.columns(2)
                if c1.button("Xác nhận", type="primary", use_container_width=True):
                    if entered_otp == st.session_state.reg_otp:
                        st.session_state.reg_step = 3
                        st.rerun()
                    else:
                        st.error("Mã OTP không chính xác!")
                if c2.button("Quay lại", use_container_width=True):
                    st.session_state.reg_step = 1
                    st.rerun()

            elif st.session_state.reg_step == 3:
                st.success("Xác thực email thành công!")
                new_pwd = st.text_input("Tạo mật khẩu", type="password")
                confirm_pwd = st.text_input("Xác nhận mật khẩu", type="password")
                
                if st.button("Hoàn tất đăng ký", type="primary", use_container_width=True):
                    if len(new_pwd) < 6:
                        st.warning("Mật khẩu phải từ 6 ký tự.")
                    elif new_pwd != confirm_pwd:
                        st.warning("Mật khẩu không khớp!")
                    else:
                        users[st.session_state.reg_email] = new_pwd
                        save_users(users)
                        st.session_state.reg_step = 1
                        st.success("Đăng ký thành công! Hãy chuyển sang tab Đăng nhập.")

        # ---------- TAB QUÊN MẬT KHẨU ----------
        with tab_forgot:
            st.markdown("### Khôi phục mật khẩu")
            if st.session_state.fg_step == 1:
                fg_input = st.text_input("Username hoặc Email", key="fg_input")
                if st.button("Gửi mã khôi phục", use_container_width=True):
                    fg_email = fg_input if "@" in fg_input else f"{fg_input}{HRConfig.COMPANY_DOMAIN}"
                    if fg_email not in users:
                        st.error("Tài khoản chưa được đăng ký!")
                    else:
                        otp = str(random.randint(100000, 999999))
                        if send_otp_email(fg_email, otp, "Khôi phục mật khẩu LUNA"):
                            st.session_state.fg_email = fg_email
                            st.session_state.fg_otp = otp
                            st.session_state.fg_step = 2
                            if not HRConfig.SENDER_EMAIL: st.info(f"Chế độ TEST. Mã OTP: **{otp}**")
                            st.rerun()
                        else:
                            st.error("Lỗi gửi email.")

            elif st.session_state.fg_step == 2:
                st.info(f"Mã đã được gửi đến: **{st.session_state.fg_email}**")
                entered_fg_otp = st.text_input("Nhập mã OTP (6 số)", key="fg_otp", max_chars=6)
                c1, c2 = st.columns(2)
                if c1.button("Xác nhận OTP", type="primary", use_container_width=True):
                    if entered_fg_otp == st.session_state.fg_otp:
                        st.session_state.fg_step = 3
                        st.rerun()
                    else:
                        st.error("Mã OTP sai!")
                if c2.button("Hủy", use_container_width=True):
                    st.session_state.fg_step = 1
                    st.rerun()

            elif st.session_state.fg_step == 3:
                st.success("Xác thực thành công. Vui lòng đặt mật khẩu mới.")
                fg_new_pwd = st.text_input("Mật khẩu mới", type="password")
                fg_confirm_pwd = st.text_input("Xác nhận mật khẩu", type="password")
                
                if st.button("Cập nhật mật khẩu", type="primary", use_container_width=True):
                    if len(fg_new_pwd) < 6:
                        st.warning("Mật khẩu phải từ 6 ký tự.")
                    elif fg_new_pwd != fg_confirm_pwd:
                        st.warning("Mật khẩu không khớp!")
                    else:
                        users[st.session_state.fg_email] = fg_new_pwd
                        save_users(users)
                        st.session_state.fg_step = 1
                        st.success("Đổi mật khẩu thành công! Vui lòng đăng nhập lại.")

# ==========================================
# 5. GIAO DIỆN CHÍNH (CHATBOT)
# ==========================================
else:
    bot = load_bot()

    # SIDEBAR
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
        display_name = st.session_state.user_email.split('@')[0]
        st.markdown(f"**Xin chào,** `{display_name}`")
        if st.button("🚪 Đăng xuất", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_email = ""
            st.rerun()
            
        st.markdown("---")
        st.markdown("### 💡 Câu hỏi thường gặp")
        with st.expander("1. Lương thử việc?"): st.write("Ít nhất bằng 85% mức lương chính thức.")
        with st.expander("2. Báo trước nghỉ việc?"): st.write("HĐ vô thời hạn: 45 ngày. HĐ 12-36T: 30 ngày. Dưới 12T: 3 ngày.")
            
        st.markdown("---")
        if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": "Lịch sử đã được xóa. Tôi có thể giúp gì mới cho bạn?"}]
            save_history(st.session_state.user_email, st.session_state.messages)
            st.rerun()

    # MAIN CHAT
    st.markdown("<h1 class='main-header'>LUNA - TRỢ LÝ NHÂN SỰ</h1>", unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        avatar = "🧑‍💼" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input("Nhập câu hỏi của bạn (VD: Quy định nghỉ phép năm)..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            if bot.rag_chain:
                with st.spinner("LUNA đang tra cứu dữ liệu..."):
                    try:
                        response = bot.rag_chain.invoke(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        save_history(st.session_state.user_email, st.session_state.messages)
                    except Exception as e:
                        st.error(f"Lỗi: {str(e)}. \nGợi ý: Kiểm tra lại API Key Google của bạn.")
            else:
                st.error("Chưa có dữ liệu. Vui lòng cho file .docx vào thư mục 'data'.")
