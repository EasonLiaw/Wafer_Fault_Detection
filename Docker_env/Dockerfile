FROM python:3.10
COPY . /app
WORKDIR /app
EXPOSE 8501
RUN pip install -r requirements.txt
RUN mv _tree.py ../usr/local/lib/python3.10/site-packages/shap/explainers
RUN mv BorutaShap.py ../usr/local/lib/python3.10/site-packages
CMD streamlit run main.py --server.port $PORT
