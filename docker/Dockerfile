FROM python:3.10.10

WORKDIR app/

RUN pip3 install setuptools==59.6.0 pip==23.0.1
COPY docker/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN apt-get update && apt-get install -y liblaszip8 \
    && rm -rf /var/lib/apt/lists/*

RUN touch README.rst
COPY setup.py setup.py
COPY py3dtiles py3dtiles
RUN pip3 install .

RUN mkdir data/
VOLUME data/
WORKDIR data/

ENTRYPOINT ["py3dtiles"]
