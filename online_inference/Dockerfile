FROM python:3.8-slim
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY __init__.py __init__.py
COPY heart_desease_app.py heart_desease_app.py
COPY src /src

WORKDIR .

ENV MODEL_URL="https://disk.yandex.ru/d/qYi-7WjL_fHDuw"
ENV MODEL_PATH="model/model.7z"

CMD ["uvicorn", "heart_desease_app:app", "--host", "0.0.0.0", "--port", "5757"]
#CMD ["gunicorn", "-w 1", "app", "-b", "0.0.0.0:5757"]