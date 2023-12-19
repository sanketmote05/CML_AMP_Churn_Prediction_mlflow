# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

# Part 8B: Model Training

# This script is used to pick the latest version of the model from the Model Registry and deploy the model as API 
# end point. The assumption is that you already have stored the model in the model registry.
#
# If you haven't yet, run through the initialization steps in the README file and Part 1 and Part4 and Part 8a.In Part 1, 
# the data is imported into the table you specified in Hive. All data accesses fetch from Hive.
# In Part 4, the model is trained with different parameters and the outputs are logged as MLFlow Experiments
# In Part 8A, we will save the model to the model registry
# Finally, in this step, we will get the latest version of the model from Model Registry and deploy it as an API endpoint
# 
#
# There is  1 other way of running this script. Which is to make an MLOps pipeline. You could add this as a job and 
# schedule it as dependent on Register Model Job ( see 8A_register_model.py) , If this has been created earlier
# ***Scheduled Jobs***
#
# The **[Jobs](https://docs.cloudera.com/machine-learning/cloud/jobs-pipelines/topics/ml-creating-a-pipeline.html)**
# feature allows for adhoc, recurring and depend jobs to run specific scripts. To run this model registration and
# deployment process , create a new job by going to the Project window and clicking _Jobs >
# New Job_ and entering the following settings:
# * **Name** : Register and Deploy Model
# * **Script** : 8B_deploy_registered_model.py
# * **Arguments** : _Leave blank_
# * **Kernel** : Python 3
# * **Schedule** : Manual
# * **Engine Profile** : 1 vCPU / 2 GiB
# The rest can be left as is. Once the job has been created, click **Run** to start a manual
# run for that job.
#
#
# *** WORKING WITH CML APIv2 ( SDK)***
# The code here demonstrates using CML APIv2 to automate different MLOps tasks. In 8A_register_model we used
# mlflow and API v2 to register the model 
# Here we will query the model registry to get the latest version and deploy it as an End point
# In many scenarios the team tasked with deployment of model from Registry may not be involved in Development of the model
# This script assumes that the model has been succesfully registered in the Model Registry using a UI or through the script
# in 8A_register_model
#
#
# *** TESTING THE RESULTS OF THIS SCRIPT
# This script will deploy the latest version of a model as an API End point. To test that the deployed model inferencing there
# are 2 options. 
# OPTION 1 : Find the Deployed model called Customer Churn Model MLOps API Endpoint on the UI and provide the following inputs 
#           and test 
# {"inputs": [[0.0, 0.0, 1.0, 1.0, 58.0, 1.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 1.0, 20.5, 1191.4], [0.0, 0.0, 1.0, 0.0, 50.0, 1.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 1.0, 0.0, 0.0, 75.7, 3876.2], [0.0, 0.0, 1.0, 0.0, 55.0, 1.0, 2.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 90.15, 4916.95], [0.0, 0.0, 0.0, 0.0, 16.0, 1.0, 2.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 88.45, 1422.1], [0.0, 0.0, 1.0, 0.0, 8.0, 1.0, 0.0, 1.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 101.15, 842.9]]}
# Expected UI Response : [ "No", "No", "No", "No", "No"]

# OPTION 2 : Run the 8C_test_deployed_model.py file in a session. 

from __future__ import print_function
import cmlapi
from cmlapi.rest import ApiException
from pprint import pprint
import json, secrets, os, time, string,sys
from datetime import datetime
import random
import mlflow
from src.api import ApiUtility
import cdsw


cml_client = cmlapi.default_client()

# This name depends on 8A_registermodel.py. If you are testing model registry and deployments, use the name from the UI instead of the below
model_name =  os.getenv("REGISTERED_MODEL_NAME") or "Customer Churn Model MLOps API Endpoint"
print("Registered Model : {0} ".format(model_name))
#List all the experiments in a given project of a certain name
search_filter= { "model_name" : model_name}

#We are currently using our Project for Deployments/ change this for cross project deployments
project_id = os.environ["CDSW_PROJECT_ID"]
search = json.dumps(search_filter)

#session id : primarily used in testing for creating non-unique model names
session_id = secrets.token_hex(nbytes=4)

try : 

    registered_model_list = cml_client.list_registered_models( search_filter = search)
    pprint(registered_model_list)
    
except ApiException as e:
    print("Exception when calling CMLServiceApi->list_registered_models: %s\n" % e)    
    raise
    
model_id = registered_model_list.models[0].model_id
print(f"Model Id : {model_id}")
# sort by -version_number gets us a list of models with version number descending first.

try : 
    registered_model = cml_client.get_registered_model(model_id, sort="-version_number")
    pprint(registered_model)
    
except ApiException as e:
    print("Exception when calling CMLServiceApi->get_registered_model: %s\n" % e)    
    raise
    
# lets us create 2 model variables #deploy_model_Id and #deploy_version_id
deploy_model_id =registered_model.model_id
deploy_version_id = registered_model.model_versions[0].model_version_id
print(f"Deployment Model Id : {deploy_model_id} Deployment Version Id:  {deploy_version_id}")

"""
Function : create_model_for_deployment
Parameters: 
client : cml api client
projectId: Id of project where model needs to be deployed
modelName : Name per Model Registry
modelId : identifier
description: Model Endpoint Description

Purpose: 
Create a model request to be deployed as an API endpoint from Model Registry or Fetch a
reference to the already deployed model

"""

def create_model_for_deployment(client, projectId, modelName, modelId, description = "Churn Model via API"):
    """
    Method to create a model  for deployment from Model Registry, if a model exists with the same name it wirl 
    """

    # first check if the Model with that name exists
    search_filter = {"name": modelName}
    search = json.dumps(search_filter)    

    api_response=client.list_all_models(search_filter=search)
    

    if len(api_response.models) != 0 : 
        print("Model with this name already deployed")
        pprint(api_response.models[0])
        modelObj = api_response.models[0]
    
    else :
        project_id = os.environ["CDSW_PROJECT_ID"]
        CreateModelRequest = {
                                "project_id": projectId, 
                                "name" : modelName,
                                "description": description, 
                                "disable_authentication": True,
                                "registered_model_id": modelId
                             }

        try:
            # Create a model.
            api_response = client.create_model(CreateModelRequest, projectId)
            pprint(api_response)
            modelObj = api_response
        except ApiException as e:
            print("Exception when calling CMLServiceApi->create_model: %s\n" % e)
            raise
        
    return modelObj



# create a model request
#model_body = cmlapi.CreateModelRequest(project_id=project_id, name="Churn Model API Endpoint - API", description="Deploy Churn Model with the API Endpoint")
model_description = "Churn Model API Endpoint - via CML API"  
model = create_model_for_deployment(client = cml_client , projectId = project_id, description=model_description,modelName = model_name,modelId=deploy_model_id )



"""
Function : create_modelBuild_for_deployment

Parameters: 
client : cml api client
projectId: Id of project where model needs to be deployed
modelName : Name per Model Registry
modelId : identifier
description: Model Endpoint Description

Purpose : 
Method to create a model Build for deployment 
"""

def create_modelBuild_for_deployment(client, projectId, modelVersionId, modelCreationId):
    """
    Method to create a Model build
    """
    
    # Create Model Build
    CreateModelBuildRequest = {
                                "registered_model_version_id": modelVersionId, 
                                "runtime_identifier": "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-workbench-python3.9-standard:2023.08.2-b8",
                                "comment": "invoking model build",
                                "model_id": modelCreationId
                              }

    try:
        # Create a model build.
        api_response = client.create_model_build(CreateModelBuildRequest, projectId, modelCreationId)
        pprint(api_response)
    except ApiException as e:
        print("Exception when calling CMLServiceApi->create_model_build: %s\n" % e)

    return api_response


#take the model creation id from the model value above
model_build = create_modelBuild_for_deployment(client = cml_client, projectId = project_id, modelVersionId=deploy_version_id, modelCreationId =model.id)

start_time = datetime.now()
print(start_time.strftime("%H:%M:%S"))

#Let us track the status of the model build
while model_build.status not in ["built", "build failed"]:
    print("waiting for model to build...")
    time.sleep(10)
    model_build = cml_client.get_model_build(project_id ,model.id, model_build.id)
    if model_build.status == "build failed" :
        print("model build failed, see UI for more information")
        sys.exit(1)
        
build_time = datetime.now()   
print(f"Time required for building model (sec): {(build_time - start_time).seconds}")
print("model built successfully!")


# Now let us create Deployment request for the model
model_deployment_body = cmlapi.CreateModelDeploymentRequest(project_id=project_id, model_id=model.id, build_id=model_build.id, cpu=1, memory=2)
model_deployment = cml_client.create_model_deployment(model_deployment_body, project_id, model.id, model_build.id)

while model_deployment.status not in ["stopped", "failed", "deployed"]:
    print("waiting for model to deploy...")
    time.sleep(10)
    model_deployment = cml_client.get_model_deployment(project_id, model.id, model_build.id, model_deployment.id)

curr_time = datetime.now()

if model_deployment.status != "deployed":
    print("model deployment failed, see UI for more information")
    sys.exit(1)

if model_deployment.status == "deployed" :
    print(f"Time required for deploying model (sec): {(curr_time - start_time).seconds}")
print("model deployed successfully!")