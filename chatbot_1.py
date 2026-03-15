import os
import streamlit as st
import shutil

# Các thư viện LangChain
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. CẤU HÌNH & GIAO DIỆN (UI/UX)
# ==========================================
st.set_page_config(page_title="LUNA - HR Portal", page_icon="✨", layout="wide")

# CSS Hiện đại (Tối giản, bo góc, hiệu ứng nổi)
st.markdown("""
<style>
    .main-header { font-family: 'Segoe UI', sans-serif; font-weight: 800; font-size: 2.2rem; color: #1E293B; margin-top: -30px; margin-bottom: 5px;}
    .sub-header { font-family: 'Segoe UI', sans-serif; color: #64748B; font-size: 1.05rem; margin-bottom: 2rem; }
    .stApp { background-color: #F8FAFC; }
    
    /* Tùy chỉnh chat bubble */
    [data-testid="stChatMessage"] { background-color: #FFFFFF; border-radius: 12px; padding: 15px; box-shadow: 0 2px 5px rgb(0 0 0 / 0.05); border: 1px solid #E2E8F0; margin-bottom: 15px; }
    [data-testid="stChatMessage"]:nth-child(even) { background-color: #F0F9FF; border: 1px solid #BAE6FD; }
    
    /* Tùy chỉnh sidebar */
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E2E8F0; }
    .sidebar-title { font-weight: 700; color: #0F172A; font-size: 1.2rem; margin-bottom: 15px; text-align: center;}
    .faq-container { background-color: #F1F5F9; padding: 15px; border-radius: 10px; border: 1px solid #E2E8F0; margin-bottom: 10px; }
    .faq-item { color: #334155; font-size: 0.95rem; margin-bottom: 8px; font-weight: 500;}
</style>
""", unsafe_allow_html=True)

class HRConfig:
    API_KEY = "AIzaSyCViLUiTIJNQyCmiP8esYOYS6qjbuxwAY4" 
    DATA_DIR = "./data"
    DB_DIR = "./hr_vector_db"
    MODEL_NAME = "gemini-2.5-flash"
    EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"

# ==========================================
# 2. LỚP XỬ LÝ CHATBOT (Backend Cố Định)
# ==========================================
# class HRChatbot:
#     def __init__(self):
#         os.environ["GOOGLE_API_KEY"] = HRConfig.API_KEY
#         self.embeddings = HuggingFaceEmbeddings(model_name=HRConfig.EMBEDDING_MODEL)
        
#         # CHỈNH SỬA: Luôn tạo mới DB để cập nhật dữ liệu từ file Word
    
#         if os.path.exists(HRConfig.DB_DIR):
#             shutil.rmtree(HRConfig.DB_DIR) # Xóa DB cũ mỗi lần chạy để nạp file mới
            
#         self.vectorstore = self._create_vector_db()
#         self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})
#         self.rag_chain = self._build_chain()
        
#         # Kiểm tra DB tồn tại và không rỗng
#         if os.path.exists(HRConfig.DB_DIR) and os.listdir(HRConfig.DB_DIR):
#             self.vectorstore = Chroma(persist_directory=HRConfig.DB_DIR, embedding_function=self.embeddings)
#         else:
#             self.vectorstore = self._create_vector_db()
            
#         self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})
#         self.rag_chain = self._build_chain()

#     def _create_vector_db(self):
#         if not os.path.exists(HRConfig.DATA_DIR):
#             os.makedirs(HRConfig.DATA_DIR)
#             return None
            
#         all_docs = []
#         files = [f for f in os.listdir(HRConfig.DATA_DIR) if f.endswith(".docx") and not f.startswith("~$")]
        
#         for file in files:
#             file_path = os.path.join(HRConfig.DATA_DIR, file)
#             try:
#                 loader = Docx2txtLoader(file_path)
#                 all_docs.extend(loader.load())
#             except Exception as e:
#                 print(f"Bỏ qua file lỗi {file}: {e}") 
        
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1500, 
#             chunk_overlap=350,
#             separators=["\nĐiều ", "\nChương ", "\n\n", "\n", " "]
#         )
#         chunks = text_splitter.split_documents(all_docs)
#         return Chroma.from_documents(documents=chunks, embedding=self.embeddings, persist_directory=HRConfig.DB_DIR)
class HRChatbot:
    def __init__(self):
        os.environ["GOOGLE_API_KEY"] = HRConfig.API_KEY
        self.embeddings = HuggingFaceEmbeddings(model_name=HRConfig.EMBEDDING_MODEL)
        
        # Tự động tạo thư mục data nếu chưa có
        if not os.path.exists(HRConfig.DATA_DIR):
            os.makedirs(HRConfig.DATA_DIR)
            
        # Khởi tạo hoặc cập nhật Database
        self.vectorstore = self._init_or_update_db()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})
        self.rag_chain = self._build_chain()

    def _init_or_update_db(self):
        # 1. Kiểm tra xem DB đã tồn tại chưa
        if os.path.exists(HRConfig.DB_DIR) and os.listdir(HRConfig.DB_DIR):
            vectorstore = Chroma(persist_directory=HRConfig.DB_DIR, embedding_function=self.embeddings)
            # Lấy danh sách các file đã có trong DB
            existing_data = vectorstore.get()
            existing_sources = {m['source'] for m in existing_data['metadatas']} if existing_data['metadatas'] else set()
        else:
            vectorstore = None
            existing_sources = set()

        # 2. Quét thư mục data để tìm file mới
        all_files = [os.path.join(HRConfig.DATA_DIR, f) for f in os.listdir(HRConfig.DATA_DIR) 
                     if f.endswith(".docx") and not f.startswith("~$")]
        
        new_docs = []
        for file_path in all_files:
            # Chỉ nạp nếu file_path chưa tồn tại trong metadata của Chroma
            if file_path not in existing_sources:
                try:
                    loader = Docx2txtLoader(file_path)
                    new_docs.extend(loader.load())
                except Exception as e:
                    st.error(f"Lỗi khi đọc file {file_path}: {e}")

        # 3. Cập nhật nếu có dữ liệu mới
        if new_docs:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=900, chunk_overlap=150,
                separators=["\nĐiều ", "\nChương ", "\n.\n", "\n", " "]
            )
            chunks = text_splitter.split_documents(new_docs)
            
            if vectorstore:
                vectorstore.add_documents(chunks)
                st.toast(f"✅ Đã cập nhật thêm {len(new_docs)} tài liệu mới!", icon="✨")
            else:
                vectorstore = Chroma.from_documents(
                    documents=chunks, 
                    embedding=self.embeddings, 
                    persist_directory=HRConfig.DB_DIR
                )
        return vectorstore

    def _build_chain(self):
        llm = ChatGoogleGenerativeAI(model=HRConfig.MODEL_NAME, temperature=0.0)
        
        system_prompt = (
    # "Bạn là LUNA - Chuyên gia Pháp lý & Nhân sự cao cấp. "
    # "Nhiệm vụ của bạn là tra cứu và giải đáp chính xác tuyệt đối dựa trên DỮ LIỆU PHÁP LÝ TÌM ĐƯỢC.\n\n"
    "Nhiệm vụ: Bạn là LUNA. Khi người dùng hỏi về một con số (ví dụ: 7.8), bạn PHẢI tìm chính xác mục có ký hiệu đó trong {context}. Quy tắc vàng: > 1. Phân biệt rõ Điều 7.8 (Miễn giấy phép lao động) và Điều 17.8 (Hợp pháp hóa lãnh sự). 2. Nếu câu hỏi là 'Liệt kê điều kiện tại 7.8', bạn bắt buộc phải tìm nội dung có chứa cụm từ 'dưới 30 ngày' và 'không quá 03 lần'. 3. Tuyệt đối không được trả lời nội dung của Điều 17 cho câu hỏi về Điều 7."
    "Khi người dùng hỏi về nội dung của một Điều/Khoản cụ thể (Ví dụ: Điều 7.8):\n"

    "Ưu tiên tìm đoạn văn bản có chứa nội dung định nghĩa/quy định trực tiếp (Ví dụ: đoạn có chữ 'Vào Việt Nam làm việc..."

    "Loại bỏ các đoạn văn bản chỉ mang tính chất liệt kê tham chiếu (Ví dụ: 'Trường hợp quy định tại khoản 8 Điều 7...."
    "🔴 CHIẾN THUẬT SUY LUẬN (BẮT BUỘC):\n"
    "1. Bước 1: Quét toàn bộ nội dung trong {context} để tìm các từ khóa quan trọng và ký hiệu (QĐ).\n"
    "2. Bước 2: Đặc biệt chú ý đến 'Điều 7.8' hoặc 'dưới 30 ngày'. Nếu thấy, phải ưu tiên trích dẫn ngay.\n"
    "3. Bước 3: Đối chiếu câu hỏi với văn bản gốc để đảm bảo không bỏ sót các trường hợp ngoại lệ hoặc điều kiện đi kèm.\n\n"

    "🔴 QUY TẮC TRUY XUẤT NGHIÊM NGẶT:\n"
    "1. TUÂN THỦ MÃ (QĐ): Chỉ những thông tin đi kèm ký hiệu (QĐ) mới được coi là căn cứ pháp lý chính thống. Các thông tin khác chỉ mang tính chất tham khảo thứ yếu.\n"
    "2. ƯU TIÊN ĐIỀU LUẬT CỤ THỂ: Khi gặp từ khóa 'dưới 30 ngày', bạn PHẢI trích dẫn nội dung tại Điều 7.8 (QĐ). Tuyệt đối không được bỏ qua mục này nếu nó xuất hiện trong dữ liệu.\n"
    "3. NGUYÊN TẮC 'NGUYÊN VĂN': Cấm tự ý diễn giải sai lệch, thêm bớt hoặc làm tròn các con số (số ngày, số tiền, độ tuổi...). Phải giữ nguyên 100% số liệu từ văn bản gốc.\n"
    "4. CẤM PHỦ ĐỊNH THIẾU CĂN CỨ: Nếu dữ liệu có chứa nội dung liên quan đến câu hỏi (dù là một phần), bạn không được trả lời 'không tìm thấy'. Phải trích dẫn phần liên quan nhất.\n\n"
    
    "🔴 ĐỊNH DẠNG TRẢ LỜI (BẮT BUỘC):\n"
    "- DÒNG ĐẦU TIÊN: Trả lời trực diện bằng cụm từ viết hoa: 'CÓ', 'KHÔNG', hoặc 'THEO QUY ĐỊNH'.\n"
    "- PHẦN THÂN: Trình bày các ý rõ ràng bằng gạch đầu dòng.\n"
    "- TRÍCH DẪN NGUỒN: Sau mỗi ý, bắt buộc ghi nguồn trong ngoặc đơn. Ví dụ: (Nghị định 152, Điều 7.8 (QĐ)).\n"
    "- TÍNH ĐẦY ĐỦ: Nếu câu hỏi có nhiều vế, phải trả lời đầy đủ từng vế dựa trên dữ liệu.\n\n"
    
    "DỮ LIỆU PHÁP LÝ TÌM ĐƯỢC:\n{context}"
)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        return (
            {"context": self.retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
             "input": RunnablePassthrough()}
            | prompt_template | llm | StrOutputParser()
        )

    def chat(self, user_query: str):
        try:
            return self.rag_chain.invoke(user_query)
        except Exception as e:
            return f"⚠️ Lỗi xử lý từ AI: {str(e)}"

# ==========================================
# 3. HIỂN THỊ GIAO DIỆN CHÍNH
# ==========================================

@st.cache_resource(show_spinner=False)
def load_bot():
    return HRChatbot()

bot = load_bot()

# Khu vực Sidebar (Thông tin hệ thống)
with st.sidebar:
    st.markdown("<p class='sidebar-title'>✨ LUNA WORKSPACE</p>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="faq-container">
            <div style="color: #0F172A; font-weight: bold; margin-bottom: 10px;">📌 YÊU CẦU THƯỜNG GẶP</div>
            <div class="faq-item">👉 Mức lương tối thiểu vùng I là bao nhiêu?</div>
            <div class="faq-item">👉 Lộ trình tuổi nghỉ hưu của nam giới năm 2025?</div>
            <div class="faq-item">👉 Các trường hợp nào được nghỉ hưu sớm?</div>
            <div class="faq-item">👉 Mức phạt khi vi phạm quy định làm thêm giờ?</div>
        </div>
    """, unsafe_allow_html=True)
            
    st.markdown("---")
    if st.button("🔄 Cập nhật dữ liệu mới", use_container_width=True):
        with st.spinner("Đang quét thư mục dữ liệu..."):
            # Gọi lại hàm cập nhật
            bot.vectorstore = bot._init_or_update_db()
            # Làm mới retriever để nhận diện dữ liệu mới
            bot.retriever = bot.vectorstore.as_retriever(search_kwargs={"k": 10})
            bot.rag_chain = bot._build_chain()
            st.success("Cơ sở dữ liệu đã được cập nhật!")

# Khu vực hiển thị chính
st.markdown("<h1 class='main-header'>LUNA - TRỢ LÝ PHÁP LÝ NHÂN SỰ</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Hệ thống tra cứu thông minh 24/7 dành cho cán bộ Nhân sự.</p>", unsafe_allow_html=True)

if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Tôi có thể giúp bạn giải đáp các vấn đề về nội quy, đãi ngộ và pháp luật lao động hôm nay?"}]

for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "✨"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Khung nhập liệu
if prompt := st.chat_input("Nhập câu hỏi của bạn tại đây..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # LOGIC XỬ LÝ TRÍ NHỚ HỘI THOẠI
    search_query = prompt
    if len(st.session_state.messages) >= 4:
        last_human_query = st.session_state.messages[-3]["content"]
        search_query = f"Ngữ cảnh chủ đề đang nói đến: '{last_human_query}'. Trả lời tiếp cho câu hỏi này: '{prompt}'"

    # Xử lý câu trả lời
    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Đang trích xuất dữ liệu pháp lý..."):
            answer = bot.chat(search_query)
            st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})