FROM python

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY score.py .
COPY Support_Vector_Machine.pkl .

EXPOSE 5000

CMD ["python", "app.py"]