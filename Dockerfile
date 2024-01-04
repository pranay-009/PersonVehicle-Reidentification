FROM python:3.10.10-slim-buster
COPY . /app
WORKDIR /app 
RUN apt-get update && apt-get install -y build-essential libopencv-dev
RUN pip install --upgrade pip && pip install -r requirements.txt
EXPOSE 8080
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]