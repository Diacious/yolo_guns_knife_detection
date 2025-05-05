import os
import cv2
import json
import uuid
import time
import psutil
import numpy as np
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from ultralytics import YOLO
from dotenv import load_dotenv
import yaml

# Загружаем имена классов из файла data.yaml
with open('data.yaml', 'r') as f:
    data_yaml = yaml.safe_load(f)
CLASS_NAMES = data_yaml['names']  # например, ['pistol', 'knife']

# Загружаем переменные окружения из файла .env
load_dotenv()
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
HISTORY_FILE = os.getenv("HISTORY_FILE", "request_history.json")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

app = FastAPI()

# --- Загружаем модель YOLO с использованием Ultralytics ---
def load_model(model_path=MODEL_PATH):
    return YOLO(model_path)

model = load_model()

def process_image_with_yolo(img):
    """
    Обрабатывает изображение с помощью YOLO и возвращает обнаружения в формате:
    {"detections": [{"bbox": [x1, y1, x2, y2], "confidence": conf, "class": int_cls}, ...]}
    """
    results = model(img)
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class": cls
            })
    return {"detections": detections}

def draw_boxes(image, detections):
    """
    Отрисовывает рамки (с метками и уровнем уверенности) на изображении.
    """
    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = list(map(int, bbox))
        conf = det["confidence"]
        cls_id = det.get("class", 0)
        class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        label = f"{class_name}: {conf:.2f}"
        color = (255, 0, 0)  # Синий цвет для рамки
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

def save_request_history(entry: dict):
    data = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
    data.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=4)

def compute_label_stats(detections):
    """
    Вычисляет статистику по меткам для списка обнаружений.
    Возвращает словарь, сопоставляющий имя метки со статистикой (количество и средняя уверенность).
    """
    stats = {}
    for det in detections:
        cls_id = det.get("class", 0)
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        if label not in stats:
            stats[label] = {"count": 0, "total_confidence": 0.0}
        stats[label]["count"] += 1
        stats[label]["total_confidence"] += det.get("confidence", 0)
    # Вычисляем среднюю уверенность для каждой метки
    for label, values in stats.items():
        count = values["count"]
        values["avg_confidence"] = values["total_confidence"] / count if count > 0 else 0
    return stats

# --- Эндпоинт для обработки изображений ---
@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Неверный формат изображения")
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")
    
    # Измеряем время начала и использование памяти до обработки
    start_time = time.time()
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss

    # Обрабатываем изображение с помощью YOLO
    detection_result = process_image_with_yolo(img)
    detections = detection_result.get("detections", [])
    
    # Вычисляем статистику по меткам
    label_stats = compute_label_stats(detections)
    
    # Отрисовываем рамки на изображении
    processed_img = draw_boxes(img, detections)
    
    # Измеряем время окончания обработки и использование памяти после обработки
    end_time = time.time()
    mem_after = proc.memory_info().rss
    processing_time = end_time - start_time
    memory_used = (mem_after - mem_before) / (1024 * 1024)  # в МБ

    # Сохраняем обработанное изображение
    processed_filename = f"processed_{uuid.uuid4().hex}.jpg"
    processed_filepath = os.path.join(OUTPUT_DIR, processed_filename)
    cv2.imwrite(processed_filepath, processed_img)
    
    # Сохраняем запись в истории с дополнительной статистикой
    history_entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "file_name": file.filename,
        "processed_file": processed_filename,
        "result": detection_result,
        "processing_time": processing_time,
        "memory_used": memory_used,
        "label_stats": label_stats
    }
    save_request_history(history_entry)
    
    return FileResponse(processed_filepath, media_type="image/jpeg", filename=processed_filename)

# --- Эндпоинт для обработки видео ---
@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    if file.content_type not in ["video/mp4", "video/mov", "video/avi"]:
        raise HTTPException(status_code=400, detail="Неверный формат видео")
    
    # Временно сохраняем загруженное видео
    temp_video_path = os.path.join(OUTPUT_DIR, f"temp_{file.filename}")
    with open(temp_video_path, "wb") as f:
        f.write(await file.read())
    
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        os.remove(temp_video_path)
        raise HTTPException(status_code=400, detail="Неверный формат видео или ошибка открытия файла")
    
    # Получаем параметры видео
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # или 'H264'
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    processed_filename = f"processed_{uuid.uuid4().hex}.mp4"
    processed_filepath = os.path.join(OUTPUT_DIR, processed_filename)
    out = cv2.VideoWriter(processed_filepath, fourcc, fps, (width, height))
    
    # Измеряем время начала и использование памяти для обработки видео
    start_time = time.time()
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss

    frame_idx = 0
    all_detections = []  # Список для хранения обнаружений для каждого кадра
    aggregate_detections = []  # Для накопления всех обнаружений для статистики по меткам

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = process_image_with_yolo(frame)
        detections = result.get("detections", [])
        processed_frame = draw_boxes(frame, detections)
        out.write(processed_frame)
        
        all_detections.append({
            "frame": frame_idx,
            "detections": detections
        })
        aggregate_detections.extend(detections)
        frame_idx += 1

    cap.release()
    out.release()
    
    # Удаляем временный видеофайл
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    
    end_time = time.time()
    mem_after = proc.memory_info().rss
    processing_time = end_time - start_time
    memory_used = (mem_after - mem_before) / (1024 * 1024)  # в МБ

    # Вычисляем статистику по меткам для всего видео
    label_stats = compute_label_stats(aggregate_detections)
    
    # Сохраняем запись в истории для обработки видео
    history_entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "file_name": file.filename,
        "processed_file": processed_filename,
        "result": {"video_detections": all_detections},
        "processing_time": processing_time,
        "memory_used": memory_used,
        "label_stats": label_stats
    }
    save_request_history(history_entry)
    
    return FileResponse(processed_filepath, media_type="video/mp4", filename=processed_filename)

# --- Эндпоинт для получения истории запросов ---
@app.get("/history/")
def get_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []
    return data

# --- Эндпоинт для генерации отчета ---
@app.get("/report/")
def generate_report(report_type: str = "pdf"):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []
    
    # Для Excel преобразуем необходимые поля в DataFrame
    if report_type.lower() == "excel":
        import pandas as pd
        # Готовим список словарей с нужными полями
        rows = []
        for entry in data:
            row = {
                "ID": entry.get("id"),
                "Timestamp": entry.get("timestamp"),
                "File": entry.get("file_name"),
                "Processed File": entry.get("processed_file"),
                "Processing Time (sec)": entry.get("processing_time"),
                "Memory Used (MB)": entry.get("memory_used"),
                "Label Stats": entry.get("label_stats")
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        report_file = os.path.join(OUTPUT_DIR, "report.xlsx")
        df.to_excel(report_file, index=False)
    else:
        report_file = os.path.join(OUTPUT_DIR, "report.pdf")
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        # Для каждой записи из истории добавляем поля в PDF
        for entry in data:
            pdf.cell(200, 10, txt=f"ID: {entry.get('id')}", ln=1)
            pdf.cell(200, 10, txt=f"Timestamp: {entry.get('timestamp')}", ln=1)
            pdf.cell(200, 10, txt=f"File: {entry.get('file_name')}", ln=1)
            pdf.cell(200, 10, txt=f"Processed File: {entry.get('processed_file')}", ln=1)
            pdf.cell(200, 10, txt=f"Processing Time: {entry.get('processing_time'):.2f} sec", ln=1)
            pdf.cell(200, 10, txt=f"Memory Used: {entry.get('memory_used'):.2f} MB", ln=1)
            pdf.cell(200, 10, txt=f"Label Stats: {entry.get('label_stats')}", ln=1)
            pdf.ln(10)
        pdf.output(report_file)
    
    return FileResponse(report_file, media_type="application/octet-stream", filename=os.path.basename(report_file))

# --- Запуск приложения FastAPI с помощью uvicorn ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host=HOST, port=PORT, reload=True)
