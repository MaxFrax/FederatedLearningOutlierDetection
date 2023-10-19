#FROM jupyter/base-notebook:python-3.11
FROM python:3.11-bullseye

# Install Python packages
COPY requirements.txt /tmp/pip-tmp/
RUN pip install --requirement /tmp/pip-tmp/requirements.txt

RUN pip install jupyterlab

CMD ["python", "-v"]