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

# Part 8A: Saving Model to Model Registry using API

# This script is used to move  the experiment with the best performance ( Target. KPIs)  to the Model Registry 
# in CML and then subsequently (See 8B_deploy_registered_model) take the latest version of the model from Model
# registry  and deploy it as an API Endpoint

# If you haven't yet, run through the initialization steps in the README file and Part 1 and Part4.In Part 1, 
# the data is imported into the table you specified in Hive. All data accesses fetch from Hive.
# In Part 4, the model is trained with different parameters and the outputs are logged as MLFlow Experiments
# We will take the best performing experiment run as a starting point for building our pipeline
# Filtering runs we use MLFlow API , and can include nuanced filtering such as below 
# filter_string="metric.test_score < 0.71 and metric.test_score > 0.6", 
#
#
# There is  1 other way of running this script, which is as a deployment pipeline. You could add this as a job 
# and schedule it as dependent on Train Model Job , If this has been created earlier
# ***Scheduled Jobs***
#
# The **[Jobs](https://docs.cloudera.com/machine-learning/cloud/jobs-pipelines/topics/ml-creating-a-pipeline.html)**
# feature allows for adhoc, recurring and depend jobs to run specific scripts. To run this model registration and
# deployment process , create a new job by going to the Project window and clicking _Jobs >
# New Job_ and entering the following settings:
# * **Name** : Register and Deploy Model
# * **Script** : 8A_register_model.py
# * **Arguments** : _Leave blank_
# * **Kernel** : Python 3
# * **Schedule** : Manual
# * **Engine Profile** : 1 vCPU / 2 GiB
# The rest can be left as is. Once the job has been created, click **Run** to start a manual
# run for that job.

# *** working with CML APIv2 ( SDK)***
# The code here demonstrates using CML APIv2 to automate different MLOps tasks
# Step 1 : We will query CML Experiments with a filter criteria to obtain the best performing experiment run above 
#          our threshold values of target performance metric
# Step 2 : We will register this experiment run in CML Model Registry

from __future__ import print_function
import cmlapi
from cmlapi.rest import ApiException
from pprint import pprint
import json, secrets, os, time
from datetime import datetime
import random
import mlflow

cml_client = cmlapi.default_client()
username = os.environ["PROJECT_OWNER"]
session_id = secrets.token_hex(nbytes=4)

try : 
    #List all the experiments in a given project with a project name
    search_filter= { "name" : "Churn Model Tuning"}
    project_id = os.environ["CDSW_PROJECT_ID"]
    search = json.dumps(search_filter)
    api_response = cml_client.list_experiments( project_id = project_id , search_filter = search)
    
    assert len(api_response.experiments) == 1
    experiment = api_response.experiments[0]
    pprint(experiment)

except ApiException as e :
    print("Exception when calling CMLServiceapi -> list experiments: $s\n" %e)

try : 
   
    # Let us set a Threshold to get the best experiments above a threshold
    threshold = 0.71
    # we use now MLFlow api to filter the runs for metrics ,since they are not implemented in V2 yet
    exp_run_list =mlflow.search_runs(
        experiment.id,  

        filter_string=f"metric.test_score > {threshold}",
        order_by=["metrics.training_score DESC", "start_time DESC"]
    )

    # lets take the first run with the highest training score above threshold
    best_run=exp_run_list.iloc[0]
    pprint(best_run)
    
except Exception as err:
    print(f"Error in Mlflow Experiment search_runs {err=}, {type(err)=}")
    raise

    
#Let us now Register the model in model registry
#model_name = 'churn_model-' + username + "-" + session_id # uncomment  only for testing and modifications and change corresponding names in 8B_deploy_registered_model
model_name = "Customer Churn Model MLOps API Endpoint" #use this as the final name, comment this and uncomment previous line if you are testing
model_path = '{0}/model'.format(best_run.artifact_uri)
print(model_path)
CreateRegisteredModelRequest = {
                                "project_id": project_id, 
                                "experiment_id" : best_run.experiment_id,
                                "run_id": best_run.run_id, 
                                "model_name": model_name, 
                                "model_path": model_path, 
                                "visibility": "PUBLIC"
                                }
                               
print(CreateRegisteredModelRequest)
try : 
    #Register a model
    api_response = cml_client.create_registered_model(CreateRegisteredModelRequest)
    
    # This prints the metadata after we register the model including the model version details
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CMLServiceApi->create_registered_model: %s\n" % e)