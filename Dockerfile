FROM python :3.8 -slim-buster
WORKDIR / service
COPY requirements.txt .
COPY . ./
RUN pip install -r requriments.txt
ENTRYPOINT [ "python3" ,"app.py"]