FROM python:3.11-bullseye

RUN pip install scikit-learn==1.2.2
RUN pip install pandas numpy matplotlib gurobipy neptune python-dotenv

COPY .devcontainer/gurobi.lic /root/gurobi.lic

WORKDIR /FederatedLearningOutlierDetection

COPY ./ ./

CMD ["sh", "experiments_penlocal.sh"]