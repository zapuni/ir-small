(Embedding model dùng sẽ là keepitreal/vietnamese-sbert, cần phải tải về rồi gọi local do model khá nhẹ nên có thể chạy bằng cpu laptop)

Dưới đây là tài liệu đặc tả kỹ thuật chi tiết cho cuộc thi/bài tập dựa trên các thông tin được cung cấp trong slide. Tài liệu được biên soạn theo chuẩn chuyên nghiệp, phù hợp để làm tài liệu hướng dẫn (Guidelines/Specs) cho các lập trình viên tham gia.

---

# ĐẶC TẢ KỸ THUẬT: HỆ THỐNG OFFLINE RAG COMPETITION

**Phiên bản:** 1.0
**Mục tiêu:** Xây dựng hệ thống cung cấp API Endpoint để truy vấn tài liệu sử dụng mô hình ngôn ngữ lớn (LLM) và kỹ thuật Retrieval-Augmented Generation (RAG) với framework FastAPI.

---

## 1. TỔNG QUAN HỆ THỐNG
Cuộc thi tổ chức đánh giá năng lực xây dựng hệ thống RAG của sinh viên. Môi trường triển khai là **Offline RAG Competition**, trong đó máy chủ của sinh viên (Student Server) được đặt trong mạng nội bộ (LAN) và **hoàn toàn không có kết nối Internet (No WAN connection)**.

Hệ thống bao gồm 3 thành phần chính:
1.  **Teacher Proxy Server (Teacher Server):** Đóng vai trò là API Gateway trung gian, quản lý cuộc thi, gửi tài liệu, đặt câu hỏi, chấm điểm tự động, quản lý timeout/rate-limit và điều phối request tới Public LLM. (Có kết nối Internet).
2.  **PTIT LLM Server:** Máy chủ cung cấp LLM.
3.  **Student Server (Máy sinh viên):** Ứng dụng do sinh viên phát triển (sử dụng FastAPI). Chịu trách nhiệm nhận tài liệu, xử lý Chunking, lưu trữ VectorDB, thực hiện RAG và trả lời câu hỏi trắc nghiệm.

---

## 2. QUY TRÌNH LUỒNG THI (COMPETITION FLOW)
1.  **Đăng ký:** Sinh viên khởi động máy chủ (Local Server) của mình và gọi API đến Teacher Server để đăng ký địa chỉ IP/Port.
2.  **Kích hoạt thi:** Sinh viên gọi API `/evaluate` đến Teacher Server để báo hiệu sẵn sàng.
3.  **Nhận tài liệu (Upload Phase):** Teacher Server chủ động gọi API `/upload` xuống Student Server, gửi kèm dữ liệu gốc. Student Server có tối đa **120 giây** để xử lý chunking và lưu vào VectorDB.
4.  **Hỏi đáp (QA Phase):** Teacher Server sẽ "bơm" lần lượt 10 câu hỏi trắc nghiệm xuống Student Server thông qua API `/ask`. Student Server có tối đa **60 giây/câu hỏi** để retrieve ngữ cảnh, gọi LLM qua Proxy và trả về đáp án.
5.  **Kết thúc:** Teacher Server tính tổng điểm và trả kết quả.

---

## 3. CẤU HÌNH KẾT NỐI LLM (PROXY LLM)
Do Student Server không có mạng Internet, việc sinh text bằng LLM bắt buộc phải đi qua Proxy của Teacher Server. Student Server sẽ coi Proxy này như một endpoint của OpenAI.

*   **Thư viện:** `openai` (Python)
*   **Base URL:** `http://192.168.50.218:8000/api/v1/proxy`
*   **API Key:** `<Mã Sinh viên viết hoa>` (Ví dụ: `B21DCCN629`)
*   **Model hỗ trợ:** `gpt-4o-mini`

**Đoạn code tích hợp mẫu:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.50.218:8000/api/v1/proxy",
    api_key="B21DCCN629" # Sử dụng MSSV làm API KEY
)

res = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "..."}]
)
```

---

## 4. API ĐẶC TẢ YÊU CẦU ĐỐI VỚI STUDENT SERVER (Bắt buộc sinh viên tự viết)

Sinh viên **BẮT BUỘC** phải xây dựng 2 API Endpoints trên máy chủ của mình với schema được thống nhất dưới đây.

### 4.1. Nhận tài liệu (Upload)
*   **Endpoint:** `POST /upload`
*   **Chức năng:** Nhận document từ Teacher Server. Thực hiện Chunking, Embedding và lưu vào Vector Database.
*   **Timeout cho phép:** Tối đa 120 giây.
*   **Request Schema (Teacher gửi tới):**
    ```json
    {
      "doc_id": "none",
      "text": "Nội dung tài liệu RAG..."
    }
    ```
*   **Response Schema (Student trả về):**
    ```json
    {
      "status": "success",
      "doc_id": "abc_doc",
      "chunks": 42
    }
    ```

### 4.2. Trả lời truy vấn (Ask)
*   **Endpoint:** `POST /ask`
*   **Chức năng:** Nhận câu hỏi, retrieve context từ VectorDB, gửi prompt chứa context cho LLM (qua Proxy) và trả ra đáp án trắc nghiệm.
*   **Timeout cho phép:** Tối đa 60 giây / 1 câu hỏi.
*   **Request Schema (Teacher gửi tới):**
    ```json
    {
      "question": "RAG là gì? A.xxx B.xxx C.xxx D.xxx"
    }
    ```
*   **Response Schema (Student trả về):**
    ```json
    {
      "answer": "B",
      "sources": [
        "chunk_1_content_...",
        "chunk_2_content_..."
      ]
    }
    ```
*   **⚠️ Ràng buộc cực kỳ quan trọng (Quy định chấm điểm):** Giá trị của field `answer` **BẮT BUỘC** chỉ được là **1 ký tự duy nhất** tương ứng với đáp án trắc nghiệm: `A`, `B`, `C`, hoặc `D`.

---

## 5. API ĐẶC TẢ TỪ TEACHER SERVER (Sinh viên gọi đến)

*   **Base URL:** `http://192.168.50.218:8000/api/v1`
*   **Xác thực:** Tất cả các request đều **BẮT BUỘC** truyền Header định danh:
    `X-Student-ID: <Mã Sinh viên viết hoa>` (VD: `X-Student-ID: B21DCCN629`)

### 5.1. Đăng ký máy chủ thi
*   **Endpoint:** `POST /competition/register`
*   **Chức năng:** Đăng ký địa chỉ IP và Port của Student Server để Teacher Server biết đường dẫn gọi ngược lại.
*   **Request Body:**
    ```json
    {
      "server_url": "http://192.168.1.15:5000"
    }
    ```
*   **Response:**
    ```json
    {
      "message": "Đăng ký thành công!",
      "student_id": "B21DCCN629",
      "server_url": "http://192.168.1.15:5000"
    }
    ```

### 5.2. Kích hoạt quá trình thi
*   **Endpoint:** `POST /competition/evaluate`
*   **Chức năng:** Bắt đầu quá trình thi. Ngay sau khi gọi, Teacher Server sẽ bắt đầu gọi ngược lại `/upload` và 10 lần `/ask` vào máy sinh viên.
*   **Request Body:** *(Trống)*
*   **Response (Sau khi quá trình thi kết thúc):**
    ```json
    {
      "student_id": "B21DCCN629",
      "score": 8.0,
      "status": "completed",
      "detail": [ ... ]
    }
    ```

### 5.3. Reset trạng thái
*   **Endpoint:** `POST /competition/reset`
*   **Chức năng:** Xóa bỏ điểm cũ, reset toàn bộ trạng thái thi để bắt đầu lại từ đầu (hữu ích trong trường hợp code của sinh viên bị crash giữa chừng).
*   **Request Body:** *(Trống)*
*   **Response:**
    ```json
    {
      "status": "success",
      "message": "Đã reset trạng thái cho sinh viên B21DCCN629"
    }
    ```

### 5.4. Tra cứu kết quả hiện tại
*   **Endpoint:** `GET /competition/result`
*   **Chức năng:** Kiểm tra trạng thái hiện tại của phiên thi và điểm số tạm thời.
*   **Request Body:** *(Trống)*
*   **Response:**
    ```json
    {
      "student_id": "B21DCCN629",
      "score": 5.0,
      "status": "evaluating",
      "current_question": 6
    }
    ```

---
*Tài liệu này được trích xuất và hệ thống hóa từ bản thuyết trình hướng dẫn bài tập môn học. Các lập trình viên/sinh viên cần tuân thủ nghiêm ngặt các Schema, rào cản thời gian (timeout) và định dạng kết quả trả về để hệ thống chấm điểm tự động (Auto-grader) có thể hoạt động chính xác.*