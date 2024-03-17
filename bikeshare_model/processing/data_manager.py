import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import typing as t
import re
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from bikeshare_model import __version__ as _version
from bikeshare_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, ROOT, config


##  Pre-Pipeline Preparation

# 1. Extract the year and month from the datetime column
def extract_yr_mnth(data):
  df = data.copy()
  df[config.model_config.dteday] = pd.to_datetime(df[config.model_config.dteday],format=config.model_config.format)
  df[config.model_config.year] = df[config.model_config.dteday].dt.year.astype(str)
  df[config.model_config.month] = df[config.model_config.dteday].dt.month_name()
  return df
    
# 2. processing numerical and categorical columns

def num_cat_vars(data):
  numerical = []
  categorical = []
  for col in data.columns:
    print(data[col].dtypes)
    if(data[col].dtypes == 'float64' or data[col].dtypes == 'int64'):
      numerical.append(col)
    else:
      categorical.append(col)
  print(numerical)
  print(categorical)

def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = extract_yr_mnth(dataframe)
  
    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    print("save_file_name: " + save_file_name)
    
    save_path = TRAINED_MODEL_DIR / save_file_name
    print("save_path: " + str(save_path))
    sys.stdout.flush()

    remove_old_pipelines(files_to_keep=[save_file_name])
    print("dumping pipeline:"+str(pipeline_to_persist))
    joblib.dump(pipeline_to_persist, save_path)
    


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    print("root: " + str(ROOT))
    print("package root: " + str(TRAINED_MODEL_DIR))
    file_path = TRAINED_MODEL_DIR / file_name
    print("file_path: ", str(file_path))
    
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
