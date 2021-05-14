import Algorithmia
import json
import os.path
from pathlib import Path
import joblib
import hashlib

import shap
import pandas as pd
import uuid
import sklearn
import joblib
import pickle
from arize.api import Client as ArizeClient
from arize.types import ModelTypes
import os

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

algoClient = Algorithmia.client()

def load_model_manifest(rel_path="model_manifest.json"):
    """Loads the model manifest file as a dict.
    A manifest file has the following structure:
    {
      "model_filepath": Uploaded model path on Algorithmia data collection
      "model_md5_hash": MD5 hash of the uploaded model file
      "model_origin_repo": Model development repository with the Github CI workflow
      "model_origin_ref": Branch of the model development repository related to the trigger of the CI workflow,
      "model_origin_commit_SHA": Commit SHA related to the trigger of the CI workflow
      "model_origin_commit_msg": Commit message related to the trigger of the CI workflow
      "model_uploaded_utc": UTC timestamp of the automated model upload
    }
    """
    manifest = []
    manifest_path = "{}/{}".format(Path(__file__).parents[1], rel_path)
    if os.path.exists(manifest_path):
        with open(manifest_path) as json_file:
            manifest = json.load(json_file)
    return manifest


def load_model(manifest):
    """Loads the model object from the file at model_filepath key in config dict"""
    model_path = manifest["model_filepath"]
    if __name__ == "__main__":
        model_file = model_path
    else:
        model_file = algoClient.file(model_path).getFile().name
        assert_model_md5(model_file)
    model_obj = joblib.load(model_file)
    return model_obj


def assert_model_md5(model_file):
    """
    Calculates the loaded model file's MD5 and compares the actual file hash with the hash on the model manifest
    """
    md5_hash = None
    DIGEST_BLOCK_SIZE = 128 * 64
    with open(model_file, "rb") as f:
        hasher = hashlib.md5()
        buf = f.read(DIGEST_BLOCK_SIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(DIGEST_BLOCK_SIZE)
        md5_hash = hasher.hexdigest()
    assert manifest["model_md5_hash"] == md5_hash
    print("Model file's runtime MD5 hash equals to the upload time hash, great!")

# Load the model manifest and model file
manifest = load_model_manifest()
model = load_model(manifest)

# Setting up Arize client
arize_org_key = 'fMXkCnzL'
arize_api_key = 'FDOJvy1Fu4Nk/HDDJrrG'
arize_client = ArizeClient(organization_key=arize_org_key, api_key=arize_api_key)

# API calls will begin at the apply() method, with the request body passed as 'input'
# For more details, see algorithmia.com/developers/algorithm-development/languages
def apply(input):

    # Generate new predictions in production
    X_data = pd.read_json(input)
    y_pred = model.predict(X_data)

    shap_values = shap.Explainer(model, X_data).shap_values(X_data)
    shap_values = pd.DataFrame(shap_values, columns=X_data.columns)
    
    ids = pd.Series([str(uuid.uuid4()) for _ in range(len(X_data))])

    # Log the data to Arize after making prediction
    log_responses = arize_client.log_bulk_predictions(
        model_id="Algorithmia_Tutorial_Model", 
        model_version="1.0",
        model_type=ModelTypes.BINARY,
        features=X_data,
        prediction_ids=ids,
        prediction_labels=pd.Series(y_pred))
    
    # Log the data to Arize after making prediction
    shap_responses = arize_client.log_bulk_shap_values(
        model_id=f"Algorithmia_Tutorial_{manifest['model_md5_hash']}",
        prediction_ids=ids, # Again, pass in the same IDs to match the predictions & actuals. 
        shap_values=shap_values
    )

    # Return to the caller a json object containing:
    # - prediction id and prediction value for each
    # - metadata associated with the model used to make the prediction
    res = pd.DataFrame(y_pred)
    res.index = ids
    res.index.rename("pred_id", inplace=True)
    resObj = res.to_json()
    # resObj["predictions"] = res.to_json()
    # resObj["model_metadata"] = {
    #     "model_file": manifest["model_filepath"],
    #     "origin_repo": manifest["model_origin_repo"],
    #     "origin_commit_SHA": manifest["model_origin_commit_SHA"],
    #     "origin_commit_msg": manifest["model_origin_commit_msg"]
    # }
    return resObj

# Facilitate local testing
if __name__ == "__main__":
    
    # 1 Load data and split data
    data = datasets.load_breast_cancer()
    X, y = datasets.load_breast_cancer(return_X_y=True)
    X, y = pd.DataFrame(X.astype(np.float32), columns=data['feature_names']), pd.Series(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    print ('Test data loaded..')

    algo_result = apply(X_test.to_json())
    print(f"\nPredictions:")
    print(json.dumps(algo_result, indent=2))
    # print(json.dumps(algo_result['predictions'], indent=2))
    # print(f"\n\nModel metadata:")
    # print(json.dumps(algo_result['model_metadata'], indent=2))
