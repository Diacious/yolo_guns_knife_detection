import streamlit as st
import requests
import os
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import json
import numpy as np

# Загружаем переменные окружения из файла .env
load_dotenv()
PORT = int(os.getenv("PORT", "8001"))
BACKEND_URL = os.getenv("BACKEND_URL", f"http://localhost:{PORT}/")

st.set_page_config(page_title="Обнаружение объектов YOLOv11", layout="centered")

st.title("🔍 Приложение обнаружения объектов YOLOv11")
st.markdown("Выберите действие для обработки медиа или просмотра статистики и отчетов.")

# Выбор действия на боковой панели
option = st.sidebar.radio("Выберите действие", ("Изображение", "Видео", "Камера", "Статистика", "Сгенерировать отчет"))

# ------------------- ОБРАБОТКА ИЗОБРАЖЕНИЙ -------------------
if option == "Изображение":
    st.header("Обработка изображения")
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Загруженное изображение", use_column_width=True)
        if st.button("Обработать изображение"):
            with st.spinner("Обработка изображения..."):
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(f"{BACKEND_URL}/process_image/", files=files)
                if response.status_code == 200:
                    # Ответ содержит обработанное изображение с нарисованными рамками обнаружения.
                    image = Image.open(BytesIO(response.content))
                    st.image(image, caption="Обработанное изображение", use_column_width=True)
                    temp_filename = f"processed_{uploaded_file.name}"
                    with open(temp_filename, "wb") as f:
                        f.write(response.content)
                    st.download_button(
                        label="Скачать обработанное изображение",
                        data=response.content,
                        file_name=temp_filename,
                        mime="image/jpeg"
                    )
                    os.remove(temp_filename)
                else:
                    st.error("Ошибка обработки изображения.")

# ------------------- ОБРАБОТКА ВИДЕО -------------------
elif option == "Видео":
    st.header("Обработка видео")
    uploaded_file = st.file_uploader("Загрузите видео", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        st.video(uploaded_file)
        if st.button("Обработать видео"):
            with st.spinner("Обработка видео..."):
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(f"{BACKEND_URL}/process_video/", files=files)
                if response.status_code == 200:
                    temp_filename = f"processed_{uploaded_file.name}"
                    with open(temp_filename, "wb") as f:
                        f.write(response.content)
                    st.video(temp_filename)
                    st.download_button(
                        label="Скачать обработанное видео",
                        data=response.content,
                        file_name=temp_filename,
                        mime="video/mp4"
                    )
                    os.remove(temp_filename)
                else:
                    st.error("Ошибка обработки видео.")

# ------------------- ЗАХВАТ ИЗ КАМЕРЫ -------------------
elif option == "Камера":
    st.header("Захват с камеры")
    # st.camera_input доступна в новых версиях Streamlit.
    captured_image = st.camera_input("Сделайте снимок")
    if captured_image is not None:
        st.image(captured_image, caption="Снимок", use_column_width=True)
        if st.button("Обработать снимок"):
            with st.spinner("Обработка снимка..."):
                # Преобразуем полученное изображение в байты
                bytes_data = captured_image.getvalue()
                files = {"file": ("captured.jpg", bytes_data, "image/jpeg")}
                response = requests.post(f"{BACKEND_URL}/process_image/", files=files)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    st.image(image, caption="Обработанное изображение", use_column_width=True)
                    temp_filename = "processed_captured.jpg"
                    with open(temp_filename, "wb") as f:
                        f.write(response.content)
                    st.download_button(
                        label="Скачать обработанное изображение",
                        data=response.content,
                        file_name=temp_filename,
                        mime="image/jpeg"
                    )
                    os.remove(temp_filename)
                else:
                    st.error("Ошибка обработки изображения.")

# ------------------- СТАТИСТИКА -------------------
elif option == "Статистика":
    st.header("Статистика обнаружения")
    with st.spinner("Получение истории..."):
        response = requests.get(f"{BACKEND_URL}/history/")
        if response.status_code == 200:
            history = response.json()
            if not history:
                st.info("История обработки отсутствует.")
            else:
                total_requests = len(history)
                total_images = sum(1 for entry in history if "detections" in entry.get("result", {}))
                total_videos = sum(1 for entry in history if "video_detections" in entry.get("result", {}))

                st.write(f"**Всего запросов:** {total_requests}")
                st.write(f"**Обработано изображений:** {total_images}")
                st.write(f"**Обработано видео:** {total_videos}")

                # Выводим среднее время обработки и расход памяти, если указаны
                total_processing_time = sum(entry.get("processing_time", 0) for entry in history)
                avg_processing_time = total_processing_time / total_requests if total_requests > 0 else 0
                st.write(f"**Среднее время обработки:** {avg_processing_time:.2f} сек")

                total_memory_used = sum(entry.get("memory_used", 0) for entry in history)
                avg_memory_used = total_memory_used / total_requests if total_requests > 0 else 0
                st.write(f"**Средний расход памяти:** {avg_memory_used:.2f} МБ")

                # Выводим статистику по обнаруженным объектам для каждого класса
                all_label_stats = {}
                for entry in history:
                    label_stats = entry.get("label_stats", {})
                    for label, stats in label_stats.items():
                        if label not in all_label_stats:
                            all_label_stats[label] = {"count": 0, "total_confidence": 0}
                        all_label_stats[label]["count"] += stats.get("count", 0)
                        all_label_stats[label]["total_confidence"] += stats.get("avg_confidence", 0) * stats.get("count", 0)

                if all_label_stats:
                    st.subheader("Обнаружения по меткам")
                    for label, stats in all_label_stats.items():
                        count = stats["count"]
                        avg_conf = stats["total_confidence"] / count if count > 0 else 0
                        st.write(f"**Метка {label}:** {count} обнаружений (Средняя уверенность: {avg_conf:.2f})")
        else:
            st.error("Не удалось получить историю обработки с сервера.")

# ------------------- ГЕНЕРАЦИЯ ОТЧЕТА -------------------
elif option == "Сгенерировать отчет":
    st.header("Генерация отчета")
    report_type = st.radio("Выберите тип отчета", ("PDF", "Excel"))
    if st.button("Сгенерировать отчет"):
        with st.spinner("Генерация отчета..."):
            response = requests.get(f"{BACKEND_URL}/report/?report_type={report_type.lower()}")
            if response.status_code == 200:
                report_filename = f"report.{report_type.lower()}"
                st.success("Отчет сгенерирован!")
                st.download_button(
                    label="Скачать отчет",
                    data=response.content,
                    file_name=report_filename,
                    mime="application/octet-stream"
                )
            else:
                st.error("Не удалось сгенерировать отчет.")
