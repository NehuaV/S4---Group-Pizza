FROM python:3.10

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8501

COPY ./project_datasets/data_D.csv ./project_datasets/
COPY ./project_datasets/data_I.csv ./project_datasets/
COPY ./project_datasets/data_K.csv ./project_datasets/
COPY ./project_datasets/data_S.csv ./project_datasets/
COPY ./models ./models/

COPY ./site.py ./

ENTRYPOINT ["streamlit", "run"]

CMD ["site.py"]