FROM python:3.11-bullseye

# Install Python packages
COPY requirements.txt /tmp/pip-tmp/
RUN pip install --requirement /tmp/pip-tmp/requirements.txt

RUN pip install jupyterlab pandas numpy matplotlib scikit-learn gurobipy

COPY gurobi.lic /root/gurobi.lic

CMD ["python", "-v"]