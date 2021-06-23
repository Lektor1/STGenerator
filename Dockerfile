FROM python:3.8.6
COPY ./app python-flask
WORKDIR python-flask
RUN pip install -r requirements.txt
CMD python app.py
