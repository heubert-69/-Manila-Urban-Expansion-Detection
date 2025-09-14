FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

#Exposing Streamlit's Port
EXPOSE 8501

CMD ["streamlit", "run", "final_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false"]
