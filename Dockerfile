FROM python:3.7-buster

WORKDIR /opt/app-root
ENV PATH=/opt/app-root/bin:$PATH

# Create python virtual environment for installing required packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    /usr/local/bin/python -m venv /opt/app-root/ && \
    /opt/app-root/bin/pip install -U pip wheel && \
    useradd -m -N -u 1001 -s /bin/bash -g 0 user && \
    chown -R 1001:0 /opt/app-root && \
    chmod -R og+rx /opt/app-root && \
    mkdir /opt/app-root/src

COPY . src/.

RUN pip install --no-cache-dir -r src/requirements.txt \
    && chown -R 1001:0 /opt/app-root

USER 1001
