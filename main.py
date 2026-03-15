import os
import sys
import streamlit as st

# Các thư viện LangChain
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. CẤU HÌNH & GIAO DIỆN TRANG WEB
# ==========================================
st.set_page_config(page_title="Luna - HR Assistant", page_icon="🏢", layout="wide")

# Custom CSS làm đẹp
st.markdown("""
<style>
    .main-header { text-align: center; color: #1E3A8A; font-family: sans-serif; font-weight: 700; margin-bottom: 0px; }
    .sub-header { text-align: center; color: #6B7280; font-size: 1.1rem; margin-bottom: 2rem; }
    [data-testid="stChatMessage"]:nth-child(odd) { background-color: #F3F4F6; border-radius: 15px; padding: 10px; }
    [data-testid="stChatMessage"]:nth-child(even) { background-color: #E0F2FE; border-radius: 15px; padding: 10px; border: 1px solid #BAE6FD; }
</style>
""", unsafe_allow_html=True)

class HRConfig:
    API_KEY = "AIzaSyClYmLHc_CVm7SP0sRQlrY65vO7VzdXPMs" 
    DATA_DIR = "./data"
    DB_DIR = "./hr_vector_db"
    MODEL_NAME = "models/gemini-2.5-flash"
    EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"

# ==========================================
# 2. LỚP XỬ LÝ CHATBOT HR (Backend)
# ==========================================
class HRChatbotLCEL:
    def __init__(self):
        os.environ["GOOGLE_API_KEY"] = HRConfig.API_KEY
        self.embeddings = HuggingFaceEmbeddings(model_name=HRConfig.EMBEDDING_MODEL)
        self.retriever = self._prepare_data()
        self.rag_chain = self._build_lcel_chain()

    def _prepare_data(self):
        if os.path.exists(HRConfig.DB_DIR):
            vectorstore = Chroma(persist_directory=HRConfig.DB_DIR, embedding_function=self.embeddings)
        else:
            if not os.path.exists(HRConfig.DATA_DIR):
                st.error(f"❌ Lỗi: Không tìm thấy thư mục {HRConfig.DATA_DIR}")
                st.stop()
            with st.spinner("⏳ Đang phân tích dữ liệu..."):
                loader = DirectoryLoader(HRConfig.DATA_DIR, glob="./*.docx", loader_cls=Docx2txtLoader)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
                chunks = text_splitter.split_documents(docs)
                vectorstore = Chroma.from_documents(documents=chunks, embedding=self.embeddings, persist_directory=HRConfig.DB_DIR)
        return vectorstore.as_retriever(search_kwargs={"k": 10})

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_lcel_chain(self):
        # 1. Hạ temperature xuống 0.0 để triệt tiêu sự ngẫu nhiên
        # AI sẽ trả lời giống hệt nhau 100% nếu ngữ cảnh tìm được giống nhau
        llm = ChatGoogleGenerativeAI(model=HRConfig.MODEL_NAME, temperature=0.0) 

        system_prompt = (
            "Bạn là AI-HR, một chuyên gia nhân sự chuyên nghiệp, nguyên tắc và tuân thủ quy định. "
            "Nhiệm vụ của bạn là tra cứu và cung cấp thông tin chính xác tuyệt đối từ Nội quy công ty.\n\n"
            "🔴 YÊU CẦU BẮT BUỘC VỀ CẤU TRÚC VÀ ĐỊNH DẠNG (BẮT BUỘC TUÂN THỦ):\n"
            "- ỔN ĐỊNH CẤU TRÚC: Khi nhận các câu hỏi mang ý nghĩa tương đồng như 'hành vi bị cấm', 'không được phép', 'vi phạm', 'ảnh hưởng công ty', bạn PHẢI luôn phản hồi theo một cấu trúc gom nhóm cố định dựa trên các Điều khoản.\n"
            "- PHÂN LOẠI RÕ RÀNG: Luôn sử dụng tiêu đề in đậm cho từng nhóm (Ví dụ: **1. Về tác phong và giao tiếp (Điều 15, 16)**, **2. Về trách nhiệm và thời gian (Điều 13, 14)**, **3. Các hành vi nghiêm cấm (Điều 17)**...).\n"
            "- LIỆT KÊ: BẮT BUỘC dùng gạch đầu dòng (-) để liệt kê các chi tiết bên dưới mỗi nhóm.\n"
            "- LUÔN LUÔN XUỐNG DÒNG rõ ràng giữa các mục để dễ đọc.\n\n"
            "NGUYÊN TẮC TRẢ LỜI:\n"
            "1. SỰ CHÍNH XÁC TRUYỆT ĐỐI: Bám sát văn bản, không tự ý sáng tạo hay tóm tắt quá mức làm mất đi ý nghĩa pháp lý của nội quy.\n"
            "2. BAO QUÁT THÔNG TIN: Nếu người dùng hỏi chung chung về 'các hành vi', hãy quét toàn bộ ngữ cảnh và liệt kê đầy đủ các nhóm vi phạm tìm thấy, không được bỏ sót.\n"
            "3. DẪN CHỨNG RÕ RÀNG: Luôn ghi chú (Điều X) hoặc (Chương Y) tương ứng với từng quy định.\n\n"
            "DỮ LIỆU NỘI QUY:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        return (
            {"context": self.retriever | self._format_docs, "input": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )

    def chat(self, user_query: str):
        try:
            return self.rag_chain.invoke(user_query)
        except Exception as e:
            return f"⚠️ Lỗi xử lý: {str(e)}"

# ==========================================
# 3. HIỂN THỊ GIAO DIỆN (Frontend)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.markdown("### 🏢 Cổng Thông Tin Nhân Sự")
    st.markdown("---")
    st.info("💡 Bạn có thể hỏi các vấn đề về:\n- Quy định nghỉ phép\n- Chế độ đãi ngộ\n- Nội quy công sở")
    st.markdown("---")
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown("<h1 class='main-header'>LUNA - TRỢ LÝ NHÂN SỰ</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Hệ thống AI giải đáp nội quy và chính sách công ty 24/7</p>", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_bot():
    return HRChatbotLCEL()

bot = load_bot()

if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Tôi là Luna. Tôi có thể giúp gì cho bạn hôm nay?"}]

for message in st.session_state.messages:
    avatar = "🧑‍💼" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Điểm cốt lõi sửa lỗi: Chỉ sử dụng DUY NHẤT một st.chat_input, và gán cho nó một 'key' định danh cụ thể
# 5. Xử lý Input của người dùng
if prompt := st.chat_input("Nhập câu hỏi của bạn...", key="unique_chat_input"):
    
    # In câu hỏi mới ra màn hình UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(prompt)

    # ==========================================
    # LOGIC XỬ LÝ TRÍ NHỚ HỘI THOẠI (Context Query)
    # ==========================================
    search_query = prompt
    
    # Nếu hệ thống đã có ít nhất 4 tin nhắn (Chào hỏi + Hỏi lần 1 + Đáp lần 1 + Hỏi lần 2)
    if len(st.session_state.messages) >= 4:
        # Lấy lại câu hỏi gần nhất của User (nằm ở vị trí thứ 3 từ dưới đếm lên)
        last_human_query = st.session_state.messages[-3]["content"]
        
        # Ghép câu hỏi cũ và mới lại với nhau để định hướng cho Vector DB
        search_query = f"Ngữ cảnh chủ đề đang nói đến: '{last_human_query}'. Trả lời tiếp cho câu hỏi này: '{prompt}'"

    # Bot xử lý và trả lời (Dùng search_query đã ghép ngữ cảnh thay vì prompt gốc)
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Luna đang xâu chuỗi dữ liệu..."):
            answer = bot.chat(search_query)
            st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})