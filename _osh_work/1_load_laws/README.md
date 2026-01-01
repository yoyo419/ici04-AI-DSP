# 法律文件前處理器 (loader.py)

本專案提供一個用於處理台灣職業安全相關法律文件的前處理腳本 `loader.py`，功能包含：PDF 文字擷取（PyMuPDF + Tesseract OCR）、表格 / 公式偵測與轉換、法律結構 (章/節/條) 解析，以及切分成可供後續 NLP/embedding 使用的 chunk。

**主要功能**
- 自動從 PDF 擷取文字，必要時使用 Tesseract OCR。
- 偵測並轉換以方框字元表示的表格為可閱讀文字描述（可呼叫 OpenAI API）。
- 偵測並說明數學公式（可呼叫 OpenAI API）。
- 對法律文本解析章、節、條，並切分成句子友好的 chunk，估算 token 數。

**需求 (主要套件)**
- Python 3.8+
- pymupdf (fitz)
- pytesseract
- opencv-python
- pillow
- numpy
- openai (或最新 OpenAI SDK)
- tiktoken

範例安裝指令：
```bash
pip install pymupdf pytesseract opencv-python pillow numpy openai tiktoken
```

**Tesseract（OCR）注意事項**
- 在 Windows 上請先安裝 Tesseract 可執行檔，並將安裝路徑加入系統環境變數 `PATH`，或在程式中指定 `pytesseract.pytesseract.tesseract_cmd`。
- 若需辨識繁體中文，請確保 Tesseract 已安裝 `chi_tra` 語言包（tessdata）。

**環境變數**
- `OPENAI_API_KEY`：必須設定，用於呼叫 OpenAI API（`loader.py` 會檢查此變數）。
- `INPUT_FOLDER`（選用）：輸入 PDF 的資料夾路徑，預設 `./input`。
- `OUTPUT_FOLDER`（選用）：處理結果輸出資料夾，預設 `./processed_output`。

**執行方式**
1. 設定環境變數（範例）：
```powershell
setx OPENAI_API_KEY "sk-..."
setx INPUT_FOLDER "C:\path\to\input"
setx OUTPUT_FOLDER "C:\path\to\processed_output"
```
2. 執行：
```bash
python loader.py
```

處理完成後，輸出檔案會存放於 `OUTPUT_FOLDER`，每個處理過的 PDF 會有對應的 `_processed.json`，並且在輸出資料夾會生成 `all_documents.json`。

**範例 & 測試**
- 建議先準備少量 PDF 做測試，確認 Tesseract 與 OpenAI 金鑰可用後再批次處理大量文件。

**注意事項**
- `loader.py` 內部會直接呼叫 OpenAI，請留意 API 使用與費用。
- 若不想呼叫 OpenAI，可暫時 mock 或修改 `OpenAIConverter` 的行為。

若需要，我可以幫您：
- 產生 `requirements.txt`
- 將程式包裝為可重複執行的 CLI
- 或協助把 OpenAI 呼叫改為可選模式（例如 `--dry-run`）
