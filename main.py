import os
import requests
import resend
import re
from dotenv import load_dotenv
from bs4 import BeautifulSoup

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ==== Khởi tạo FastAPI ====
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Load biến môi trường ====
load_dotenv()
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["USER_AGENT"] = "dongktc_bot" # Tránh bị block bởi website

# ==== Gửi email (Resend) ====
resend.api_key = os.getenv("MAIL_RESEND_API")

def send_email(subject: str, content: str):
    try:
        resend.Emails.send({
            "from": "bot@bacninhtech.com",
            "to": "contact@bacninhtech.com",
            "subject": subject,
            "html": f"<p>{content}</p>",
        })
    except Exception as e:
        print("Lỗi gửi mail:", e)

# ==== Thu thập dữ liệu từ Website ====
def get_website_docs():
    print("Bắt đầu quét dữ liệu từ website...")
    url_danh_muc = "https://bacninhtech.com/bds/"
    try:
        response = requests.get(url_danh_muc, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        links = []
        blacklist = ["/local/", "/tel:", "mailto:"]
        
        # Tìm tất cả các link trong thẻ <main>
        found_elements = soup.select('main a')
        for a in found_elements:
            link = a.get('href', '').strip()
            if link and link != "#":
                if not link.startswith('http'):
                    base_url = "https://bacninhtech.com"
                    if not link.startswith('/'):
                        link = '/' + link
                    full_link = base_url + link
                    links.append(full_link)
                else:
                    links.append(link)

        # Lọc link sạch
        clean_links = list(set([l for l in links if not any(word in l for word in blacklist)]))
        
        if not clean_links:
            print("Không tìm thấy link nào, sử dụng link gốc.")
            clean_links = [url_danh_muc]

        print(f"Tìm thấy {len(clean_links)} liên kết. Đang tải nội dung...")
        
        loader = WebBaseLoader(web_path=clean_links)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        print(f"Hoàn tất xử lý: {len(splits)} đoạn văn bản.")
        return splits
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu web: {e}")
        return []

# ==== Khởi tạo Vector Store & QA Chain ====
print("Đang khởi tạo hệ thống RAG...")
all_splits = get_website_docs()

embedding = OpenAIEmbeddings()
# Sử dụng Chroma (Lưu tạm trong /tmp để tương thích với Hugging Face/Serverless)
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=embedding,
    persist_directory="/tmp/chroma_db_web"
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0) # Temp=0 để trả lời chính xác thông tin web

# Custom Prompt để Bot trả lời chuyên nghiệp hơn
template = """Bạn là một chuyên gia bất động sản và tư vấn tại Bac Ninh Tech. 
Hãy sử dụng thông tin sau để trả lời câu hỏi của khách hàng.
Nếu thông tin không có trong tài liệu, hãy nói là bạn chưa có thông tin chính xác về vấn đề này và đề nghị khách để lại số điện thoại.
Tối đa 3 câu, viết ngắn gọn, chuyên nghiệp.

Thông tin hỗ trợ: {context}
Câu hỏi: {question}
Trả lời:"""

prompt = ChatPromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# ==== Input model ====
class ChatRequest(BaseModel):
    message: str

# ==== Helper functions ====
def extract_phone_number(text: str):
    # Regex cải tiến cho số điện thoại VN
    match = re.search(r"(?:\+84|0)\d{8,10}\b", text.replace(" ", "").replace(".", ""))
    return match.group(0) if match else None

# ==== API trả lời chat ====
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        # 1. Lấy câu trả lời từ RAG Website
        # Lưu ý: RetrievalQA dùng .run() hoặc .invoke()
        bot_answer = qa_chain.run(req.message)

        # 2. Xử lý logic thu thập thông tin khách hàng
        phone = extract_phone_number(req.message)
        msg_lower = req.message.lower()
        
        response_part = [bot_answer]

        if phone:
            send_email("Khách hàng cần tư vấn BĐS", f"Số điện thoại khách hàng: {phone}\nNội dung chat: {req.message}")
            response = "Cảm ơn bạn đã để lại số điện thoại! Chuyên viên của Bac Ninh Tech sẽ gọi lại tư vấn cho bạn ngay trong ít phút tới."
        elif any(k in msg_lower for k in ["giá", "bao nhiêu", "chi phí"]):
            response_part.append("Để nhận bảng giá chi tiết và ưu đãi mới nhất, bạn vui lòng để lại số điện thoại nhé?")
            response = "<br><br>".join(response_part)
        elif any(k in msg_lower for k in ["tư vấn", "liên hệ", "gặp"]):
            response_part.append("Bạn có thể để lại số điện thoại để chúng tôi hỗ trợ nhanh nhất không?")
            response = "<br><br>".join(response_part)
        else:
            # Nếu câu trả lời quá ngắn hoặc không có thông tin, gợi ý thêm
            if len(bot_answer) < 50:
                response_part.append("Bạn cần hỏi thêm chi tiết về dự án hay chính sách nào không?")
            response = "<br><br>".join(response_part)

        return {"answer": response}
    except Exception as e:
        print(f"Lỗi chat: {e}")
        return {"answer": "Xin lỗi, hệ thống đang bận xử lý dữ liệu website. Bạn vui lòng thử lại sau giây lát."}

@app.get("/")
def root():
    return {"message": "Bac Ninh Tech Web-RAG API is running!"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)