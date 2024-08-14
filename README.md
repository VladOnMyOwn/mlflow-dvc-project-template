# Project Usage Guide

### Prepare Environment
1. In case of restricted access: ```$ pip config set global.trusted-host "files.pythonhosted.org pypi.org pypi.python.org"```  
2. For Windows systems: ```$ set MLFLOW_TRACKING_URI=http://<host>:<port>```  
For Unix systems: ```$ export MLFLOW_TRACKING_URI=http://<host>:<port>```  
3. In case of using remote storage to store artifacts: ```$ export MLFLOW_S3_ENDPOINT_URL=https://<domain-name>```  
For Windows: ```$ set MLFLOW_S3_ENDPOINT_URL=https://<domain-name>```  

### Run Project
##### 1. Run MLflow server
Save artifacts locally: ```$ mlflow ui --host <host> --port <port>```  
Save artifacts into remote storage: ```$ mlflow ui --host <host> --port <port> --artifacts-destination <remote-uri>```  
Save artifacts into remote storage w/o proxying access through tracking server: ```$ mlflow ui --host <host> --port <port> --no-serve-artifacts --default-artifact-root <remote-uri>```  

##### 2. Run step
To see available options run: ```$ mlflow run --help```  
To run particular project step: ```$ mlflow run . --entry-point <entry-point-name> -P <param1>=<value1> -P <param2>=<value2> --experiment-name <Experiment_Name> --run-name <Run_Name> --env-manager <env-manager>```
