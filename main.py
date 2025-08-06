import os
import io
import shutil
import requests
import resend
import re
import smtplib
from email.message import EmailMessage

from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ==== Khởi tạo FastAPI ====
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các domain gọi API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Load biến môi trường ====
load_dotenv()
os.environ["CHROMA_TELEMETRY"] = "false"

# ==== Cấu hình API ====
CREDENTIALS_URL = "https://foreignervietnam.com/langchain/drive-folder.php"
CREDENTIALS_TOKEN = os.getenv("CREDENTIALS_TOKEN")
SERVICE_ACCOUNT_FILE = "/tmp/drive-folder.json"
FOLDER_ID = "1rXRIAvC4wb63WjrAaj0UUiidpL2AiZzQ"

# ==== Gửi email ====
resend.api_key = "re_DwokJ9W5_E7evBxTVZ2kVVGLPEd9puRuC"

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

# ==== Tải file credentials từ API ====
headers = {"X-Access-Token": CREDENTIALS_TOKEN}
response = requests.get(CREDENTIALS_URL, headers=headers)
if response.status_code == 200:
    with open(SERVICE_ACCOUNT_FILE, "wb") as f:
        f.write(response.content)
else:
    raise Exception(f"Không thể tải file credentials: {response.status_code}")

# ==== Google Drive functions ====
def authenticate_drive():
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
    return build("drive", "v3", credentials=creds)

def download_drive_files(drive_service):
    os.makedirs("/tmp/data", exist_ok=True)
    results = drive_service.files().list(
        q=f"'{FOLDER_ID}' in parents and trashed=false",
        fields="files(id, name)"
    ).execute()
    files = results.get("files", [])
    for file in files:
        file_path = os.path.join("/tmp/data", file["name"])
        if os.path.exists(file_path):
            continue
        request = drive_service.files().get_media(fileId=file["id"])
        with io.FileIO(file_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

# ==== Tải và xử lý tài liệu ====
def load_documents():
    docs = []
    for filename in os.listdir("/tmp/data"):
        filepath = os.path.join("/tmp/data", filename)
        if os.path.getsize(filepath) == 0:
            continue
        if filename.endswith(".pdf"):
            docs.extend(PyPDFLoader(filepath).load())
        elif filename.endswith(".txt"):
            docs.extend(TextLoader(filepath).load())
        elif filename.endswith(".docx"):
            docs.extend(Docx2txtLoader(filepath).load())
    return docs

# ==== Tạo Vectorstore từ tài liệu ====
drive_service = authenticate_drive()
download_drive_files(drive_service)
documents = load_documents()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

embedding = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory="/tmp/chroma_db"
)

# ==== Khởi tạo mô hình trả lời ====
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# ==== Input model ====
class ChatRequest(BaseModel):
    message: str

# ==== Xử lý logic ====
def extract_phone_number(text: str):
    match = re.search(r"\b(?:0|\+84)[\d\s\-\.]{8,}\b", text)
    if match:
        return match.group(0)
    return None

def extract_keyword(text: str, keyword: str):
    if keyword.lower() in text.lower():
        return keyword
    return None

# ==== API trả lời chat ====
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        response_part = [qa_chain.run(req.message)]

        phone = extract_phone_number(req.message)
        gia = extract_keyword(req.message, "giá")
        chinhsach = extract_keyword(req.message, "chính sách")
        tuvan = extract_keyword(req.message, "tư vấn")

        if phone:
            send_email("Số điện thoại từ chatbot", f"Khách hàng vừa gửi số: {phone}")
            response = "Cảm ơn bạn đã gửi số điện thoại, hãy chờ liên hệ của chúng tôi."
        elif gia:
            response_part.append("Bạn vui lòng để lại số điện thoại, chúng tôi sẽ báo giá chi tiết?")
            response = "<br><br>".join(response_part)
        elif chinhsach:
            response_part.append("Vui lòng để lại số điện thoại, chúng tôi sẽ tư vấn chi tiết chính sách.")
            response = "<br><br>".join(response_part)
        elif tuvan:
            response_part.append("Bạn có thể để lại số điện thoại, chúng tôi sẽ gọi điện và tư vấn.")
            response = "<br><br>".join(response_part)
        else:
            response_part.append("Bạn cần thêm thông tin gì nữa không?")
            response = "<br><br>".join(response_part)

        return {"answer": response}  # ✅ Trả đúng key: answer
    except Exception as e:
        return {"answer": f"Lỗi xử lý: {str(e)}"}

# ==== Kiểm tra root API ====
@app.get("/")
def root():
    return {"message": "LangChain Chatbot API is running!"}

# ==== Chạy Uvicorn nếu chạy trực tiếp ====
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
