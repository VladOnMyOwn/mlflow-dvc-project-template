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
> NOTE: To see available options run: ```$ mlflow run --help```  

To run particular project step: ```$ mlflow run . --entry-point <entry-point-name> -P <param1>=<value1> -P <param2>=<value2> --experiment-name <Experiment_Name> --run-name <Run_Name> --env-manager <env-manager>```

### Serve/deploy the model
##### 1. Prepare environment
Install *pyenv* requirements: ```$ sudo apt-get install zlib1g-dev libssl-dev libbz2-dev libncursesw5-dev libffi-dev libreadline-dev libsqlite3-dev tk-dev liblzma-dev```  
Install *pyenv* for Python version management: ```$ curl https://pyenv.run | bash```  
Add *pyenv* to PATH: ```$ export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"```  
Install *virtualenv* for Python environment management: ```$ python -m  pip install virtualenv```

##### 2. Server the model locally
Serve via REST API: ```$ mlflow models serve -m "models:/<model-name>@<model-alias>" --port <port>``` (can also use ```<model-name>/<model-version>``` instead of alias)  

Build docker image: ```$ mlflow models build-docker -m "models:/<model-name>@<model-alias>" -n "<image-name>"``` (can also use ```<model-name>/<model-version>``` instead of alias)  
Serve model via docker container: ```$ docker run -p <serving-port>:8080 <image-name>```  

Test model API: ```$ curl -d '<input-data>' -H 'Content-Type: application/json' -X POST <host>:<port>/invocations```  

### Get model inference w/o serving
```$ mlflow models predict -m models:/<model-name>@<model-alias> -i <inputs-path> -o <predictions-path>```  
