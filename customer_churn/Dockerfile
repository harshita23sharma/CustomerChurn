FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["airflow", "standalone"]
# Change this to run the scheduler instead of the standalone
# CMD ["airflow", "scheduler"]