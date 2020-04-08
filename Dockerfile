FROM python:3

ADD main.py /
ADD persistence_diagram.py /
ADD utils.py /

RUN apt-get update && apt-get -y install cmake protobuf-compiler
RUN apt-get install libboost-all-dev -y
RUN pip install tensorflow dionysus matplotlib scipy

CMD ["python", "./main.py"]