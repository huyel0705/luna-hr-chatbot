import os
import json
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
    API_KEY = st.secrets["GOOGLE_API_KEY"]
    DATA_DIR = "./data"
    MODEL_NAME = "gemini-2.5-flash"
    EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
    COMPANY_DOMAIN = "@rikai.technology" # Tên miền email bắt buộc để đăng ký
    USERS_DB = "users_db.json"     # File lưu tài khoản
    HISTORY_DIR = "./chat_history" # Thư mục lưu lịch sử chat

# Tạo thư mục lưu lịch sử nếu chưa có
if not os.path.exists(HRConfig.HISTORY_DIR):
    os.makedirs(HRConfig.HISTORY_DIR)

# Giao diện CSS
st.markdown("""
<style>
    .stApp { background-color: #F8FAFC; }
    .main-header { text-align: center; color: #1E3A8A; font-weight: 800; font-size: 2.5rem; }
    .faq-title { color: #0F172A; font-weight: bold; }
    .stChatMessage { border-radius: 10px; padding: 10px; margin-bottom: 10px; }
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
    return [{"role": "assistant", "content": f"Chào mừng bạn trở lại! LUNA có thể giúp gì cho bạn hôm nay?"}]

def save_history(email, messages):
    path = os.path.join(HRConfig.HISTORY_DIR, f"{email}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)

# Khởi tạo session state cho đăng nhập
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_email = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

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
        if not os.path.exists(HRConfig.DATA_DIR): return None
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
# 4. GIAO DIỆN XÁC THỰC (LOGIN / REGISTER)
# ==========================================
if not st.session_state.logged_in:
    st.markdown("<h1 class='main-header'>CỔNG THÔNG TIN NHÂN SỰ LUNA</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_register = st.tabs(["🔐 Đăng nhập", "📝 Đăng ký tài khoản"])
        users = load_users()

        with tab_login:
            login_email = st.text_input("Email công ty", key="l_email")
            login_pwd = st.text_input("Mật khẩu", type="password", key="l_pwd")
            if st.button("Đăng nhập", use_container_width=True):
                if login_email in users and users[login_email] == login_pwd:
                    st.session_state.logged_in = True
                    st.session_state.user_email = login_email
                    st.session_state.messages = load_history(login_email)
                    st.rerun()
                else:
                    st.error("Sai email hoặc mật khẩu!")

        with tab_register:
            reg_email = st.text_input("Email (Phải dùng email công ty)", key="r_email")
            reg_pwd = st.text_input("Mật khẩu mới", type="password", key="r_pwd")
            if st.button("Đăng ký", use_container_width=True):
                if not reg_email.endswith(HRConfig.COMPANY_DOMAIN):
                    st.warning(f"Vui lòng sử dụng email đuôi {HRConfig.COMPANY_DOMAIN}")
                elif reg_email in users:
                    st.warning("Email này đã được đăng ký!")
                elif len(reg_pwd) < 6:
                    st.warning("Mật khẩu phải từ 6 ký tự trở lên.")
                else:
                    users[reg_email] = reg_pwd
                    save_users(users)
                    st.success("Đăng ký thành công! Vui lòng đăng nhập.")

# ==========================================
# 5. GIAO DIỆN CHÍNH (SAU KHI ĐĂNG NHẬP)
# ==========================================
else:
    bot = load_bot()

    # SIDEBAR: Câu hỏi thường gặp & Công cụ
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
        st.markdown(f"**Xin chào,** `{st.session_state.user_email}`")
        if st.button("🚪 Đăng xuất"):
            st.session_state.logged_in = False
            st.session_state.user_email = ""
            st.rerun()
            
        st.markdown("---")
        st.markdown("### 💡 Câu hỏi thường gặp (FAQ)")
        with st.expander("1. Lương thử việc tính thế nào?"):
            st.write("Mức lương thử việc ít nhất bằng 85% mức lương của công việc đó.")
        with st.expander("2. Nghỉ việc báo trước bao nhiêu ngày?"):
            st.write("Hợp đồng vô thời hạn: 45 ngày. HĐ 12-36 tháng: 30 ngày. HĐ dưới 12 tháng: 3 ngày.")
        with st.expander("3. Lương làm thêm giờ ngày nghỉ?"):
            st.write("Ít nhất bằng 200% tiền lương thực trả của ngày bình thường.")
            
        st.markdown("---")
        if st.button("🗑️ Xóa lịch sử trò chuyện hiện tại"):
            st.session_state.messages = [{"role": "assistant", "content": "Lịch sử đã được xóa. Tôi có thể giúp gì mới cho bạn?"}]
            save_history(st.session_state.user_email, st.session_state.messages)
            st.rerun()

    # MAIN CONTENT: Khung Chat
    st.markdown("<h1 class='main-header'>LUNA - TRỢ LÝ NHÂN SỰ</h1>", unsafe_allow_html=True)
    
    # Hiển thị lịch sử chat
    for message in st.session_state.messages:
        avatar = "🧑‍💼" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Xử lý nhập câu hỏi mới
    if prompt := st.chat_input("Nhập câu hỏi của bạn (VD: Quy định nghỉ phép năm)..."):
        # Lưu câu hỏi của User
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(prompt)

        # AI trả lời
        with st.chat_message("assistant", avatar="🤖"):
            if bot.rag_chain:
                with st.spinner("LUNA đang tra cứu quy định..."):
                    try:
                        response = bot.rag_chain.invoke(prompt)
                        st.markdown(response)
                        # Lưu câu trả lời của Bot
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        # Lưu lịch sử vào file
                        save_history(st.session_state.user_email, st.session_state.messages)
                    except Exception as e:
                        st.error(f"Đã xảy ra lỗi: {str(e)}")
            else:
                st.error("Hệ thống chưa nạp dữ liệu. Vui lòng báo cho Admin.")
