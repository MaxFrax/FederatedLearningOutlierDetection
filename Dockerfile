FROM python:3.11-bullseye

RUN pip install scikit-learn==1.2.2
RUN pip install jupyterlab pandas numpy matplotlib gurobipy neptune python-dotenv

COPY .devcontainer/gurobi.lic /root/gurobi.lic

WORKDIR /FederatedLearningOutlierDetection

COPY ./ ./

CMD ["python", "__main__.py", "baseline_sklearn", "kdd99", "1", "1", "iid", "--njobs", "1"]