import os
import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG (USER VERSION)
# ==========================================
st.set_page_config(page_title="LUNA - Trợ Lý Nhân Sự", page_icon="✨", layout="centered")

class HRConfig:
    API_KEY = st.secrets["GOOGLE_API_KEY"] 
    DATA_DIR = "./data"
    MODEL_NAME = "gemini-2.5-flash" 
    EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"

# Tùy chỉnh giao diện chuyên nghiệp hơn
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; }
    .main-header { font-size: 2.5rem; font-weight: 800; color: #1E40AF; text-align: center; margin-bottom: 20px; }
    .sub-header { font-size: 1.1rem; color: #64748B; text-align: center; margin-bottom: 40px; }
    [data-testid="stSidebar"] { background-color: #F8FAFC; border-right: 1px solid #E2E8F0; }
    .stChatMessage { border-radius: 15px; padding: 15px; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LỚP XỬ LÝ CHATBOT LUNA
# ==========================================
class HRChatbot:
    def __init__(self):
        os.environ["GOOGLE_API_KEY"] = HRConfig.API_KEY
        self.embeddings = HuggingFaceEmbeddings(model_name=HRConfig.EMBEDDING_MODEL)
        self.vectorstore = self._init_vector_db()
        if self.vectorstore:
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 20})
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

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        chunks = text_splitter.split_documents(documents)
        return FAISS.from_documents(chunks, self.embeddings)

    def _build_chain(self):
        llm = ChatGoogleGenerativeAI(model=HRConfig.MODEL_NAME, temperature=0.3)
        
        # PROMPT HOÀN TOÀN MỚI: BỎ ĐIỀU LUẬT, TRẢ LỜI MẠCH LẠC
        system_prompt = (
            "Bạn là LUNA - Chuyên gia tư vấn nhân sự thông minh. Hãy trả lời câu hỏi của nhân viên một cách hỗ trợ, chuyên nghiệp và dễ hiểu.\n\n"
            "🔴 QUY TẮC NGHIÊM NGẶT:\n"
            "1. TUYỆT ĐỐI KHÔNG nhắc đến số hiệu Điều, Khoản, hay các ký hiệu (Luật), (QĐ).\n"
            "2. Giải thích các quy định bằng ngôn ngữ đời thường, mạch lạc. Không trích dẫn nguyên văn văn bản pháp luật khô khan.\n"
            "3. Sử dụng gạch đầu dòng để trình bày các ý rõ ràng. Bôi đậm các con số quan trọng (Ví dụ: **200%**, **45 ngày**).\n"
            "4. Nếu thông tin không có trong dữ liệu, hãy lịch sự đề nghị nhân viên liên hệ phòng Nhân sự để được hỗ trợ chi tiết.\n\n"
            "DỮ LIỆU THAM KHẢO:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        return ({"context": self.retriever, "input": RunnablePassthrough()} 
                | prompt | llm | StrOutputParser())

# ==========================================
# 3. GIAO DIỆN NGƯỜI DÙNG (USER INTERFACE)
# ==========================================
@st.cache_resource
def load_bot():
    return HRChatbot()

# Sidebar ẩn dành cho Admin
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712139.png", width=100)
    st.title("Admin Panel")
    pwd = st.text_input("Mật khẩu quản trị", type="password")
    if pwd == "123": # Thay bằng mật khẩu của bạn
        if st.button("🔄 Cập nhật dữ liệu mới"):
            st.cache_resource.clear()
            st.rerun()
    st.info("Phiên bản người dùng V3.0. Dữ liệu được cập nhật từ Bộ luật Lao động mới nhất.")

# Header chính
st.markdown("<div class='main-header'>✨ Trợ Lý Nhân Sự LUNA</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Chào bạn! Tôi ở đây để hỗ trợ bạn giải đáp các thắc mắc về chính sách, lương thưởng và quyền lợi tại công ty.</div>", unsafe_allow_html=True)

# Khởi tạo bot
bot = load_bot()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Hỏi LUNA về quy định nghỉ phép, tiền lương..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if bot.rag_chain:
            with st.spinner("Đang tra cứu thông tin cho bạn..."):
                response = bot.rag_chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("Hệ thống chưa nạp dữ liệu. Vui lòng liên hệ Admin.")
