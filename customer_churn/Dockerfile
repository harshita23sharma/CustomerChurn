FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --upgrade pip \
    && pip3 install --upgrade setuptools \
    && pip3 install numpy --upgrade \
    && pip3 install -r requirements.txt

#Need all files
COPY . .

CMD ["airflow", "standalone"]
# Change this to run the scheduler instead of the standalone
# CMD ["airflow", "scheduler"]