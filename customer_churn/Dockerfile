FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN apt-get update -y && apt-get install -y gcc  python3-dev

RUN pip3 install --upgrade pip\
    && pip3 install psutil \
    && pip3 install --upgrade setuptools \
    && pip3 install numpy --upgrade \
    && pip3 install -r requirements.txt

#Need all files
COPY . .

CMD ["airflow", "standalone"]
# Change this to run the scheduler instead of the standalone
# CMD ["airflow", "scheduler"]