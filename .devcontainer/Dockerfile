FROM python:3.11-bullseye

# Install Python packages
COPY requirements.txt /tmp/pip-tmp/
RUN pip install --requirement /tmp/pip-tmp/requirements.txt

RUN pip install scikit-learn==1.2.2
RUN pip install jupyterlab pandas numpy matplotlib gurobipy neptune python-dotenv

COPY gurobi.lic /root/gurobi.lic

CMD ["python", "-v"]