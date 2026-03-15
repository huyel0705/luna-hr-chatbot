import os
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
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
st.set_page_config(page_title="LUNA - HR Portal", page_icon="✨", layout="wide")

class HRConfig:
    API_KEY = st.secrets["GOOGLE_API_KEY"] 
    DATA_DIR = "./data"
    MODEL_NAME = "gemini-2.5-flash" 
    EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"

# Giao diện CSS
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #1E293B; }
    .stApp { background-color: #F8FAFC; }
    [data-testid="stChatMessage"] { border-radius: 12px; border: 1px solid #E2E8F0; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LỚP XỬ LÝ CHATBOT LUNA (V2 - MULTI-DOC)
# ==========================================
class HRChatbot:
    def __init__(self):
        os.environ["GOOGLE_API_KEY"] = HRConfig.API_KEY
        self.embeddings = HuggingFaceEmbeddings(model_name=HRConfig.EMBEDDING_MODEL)
        
        if not os.path.exists(HRConfig.DATA_DIR):
            os.makedirs(HRConfig.DATA_DIR)
            
        self.vectorstore = self._init_vector_db()
        if self.vectorstore:
            # Tăng k lên 25 vì số lượng file và độ dài văn bản đã tăng lên
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
            try:
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                st.error(f"Lỗi đọc file {file_path}: {e}")

        # Giữ nguyên chunk size lớn để bao quát các điều khoản phức tạp của Bộ luật
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=500,
            separators=["\nĐiều ", "\nChương ", "\n\n", "\n", " "]
        )
        chunks = text_splitter.split_documents(documents)
        return FAISS.from_documents(chunks, self.embeddings)

    def _build_chain(self):
        llm = ChatGoogleGenerativeAI(model=HRConfig.MODEL_NAME, temperature=0.0)
        
        # SYSTEM PROMPT TỐI ƯU CHO BỘ LUẬT VÀ NGHỊ ĐỊNH
        system_prompt = (
    "Bạn là LUNA - Trợ lý Pháp lý cao cấp. Hãy trả lời câu hỏi một cách chuyên nghiệp, nhẹ nhàng và mạch lạc.\n\n"
    "🔴 QUY TẮC NGHIÊM NGẶT:\n"
    "1. TUYỆT ĐỐI KHÔNG trích dẫn số hiệu Điều, Khoản, Chương hay các ký hiệu như (Luật), (QĐ), (Bộ luật) vào câu trả lời.\n"
    "2. Hãy diễn giải nội dung pháp lý thành ngôn ngữ tư vấn tự nhiên, dễ hiểu.\n"
    "3. Nếu có các con số về tiền lương, thời hạn hay phần trăm, hãy bôi đậm để người dùng dễ chú ý.\n"
    "4. Câu trả lời phải có cấu trúc rõ ràng (Dùng gạch đầu dòng), không viết thành một đoạn văn dài.\n\n"
    "LỊCH SỬ HỘI THOẠI:\n{history}\n\n"
    "DỮ LIỆU PHÁP LÝ TRA CỨU:\n{context}"
)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        return (
            {
                "context": lambda x: x["context"],
                "history": lambda x: x["history"],
                "input": lambda x: x["input"]
            }
            | prompt | llm | StrOutputParser()
        )

    def chat(self, user_query: str):
        if not self.rag_chain:
            return "Vui lòng thêm file .docx vào thư mục /data."
        try:
            chat_history_text = ""
            if "messages" in st.session_state:
                for msg in st.session_state.messages[-5:]: 
                    role = "Người dùng" if msg["role"] == "user" else "LUNA"
                    chat_history_text += f"{role}: {msg['content']}\n"

            docs = self.retriever.invoke(user_query)
            context_text = "\n\n".join(d.page_content for d in docs)

            return self.rag_chain.invoke({
                "context": context_text,
                "input": user_query, 
                "history": chat_history_text
            })
        except Exception as e:
            return f"⚠️ Lỗi hệ thống: {str(e)}"

# ==========================================
# 3. GIAO DIỆN STREAMLIT
# ==========================================
@st.cache_resource
def load_bot():
    return HRChatbot()

bot = load_bot()

with st.sidebar:
    st.markdown("### ⚙️ QUẢN LÝ HỆ THỐNG")
    if st.button("🔄 Làm mới dữ liệu", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    st.markdown("---")
    st.info("LUNA hiện đã nạp: Bộ luật Lao động 2019 và các Nghị định hướng dẫn.")

st.markdown("<h1 class='main-header'>✨ LUNA - HR LEGAL ASSISTANT (V2.5)</h1>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! LUNA đã cập nhật Bộ luật Lao động 2019. Bạn cần tra cứu quy định hay thủ tục gì?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Nhập câu hỏi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("LUNA đang đối soát Luật và Nghị định..."):
            answer = bot.chat(prompt)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
