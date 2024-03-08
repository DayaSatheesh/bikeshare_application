import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer
from bikeshare_model.processing.features import Mapper
from bikeshare_model.processing.features import WeathersitImputer
from bikeshare_model.processing.features import OutlierHandler
from bikeshare_model.processing.features import WeekdayOneHotEncoder

bikeshare_pipe = Pipeline([

    ######### Imputation ###########
    ('weekday_imputation', WeekdayImputer(variable=config.model_config.weekday, day=config.model_config.day, dteday=config.model_config.dteday)),
    ('weathersit_imputation', WeathersitImputer(variable=config.model_config.weathersit)),

    ######### Mapper ###########
    ('map_year', Mapper(config.model_config.year, config.model_config.yearmapping)),
    ('map_month', Mapper(config.model_config.month, config.model_config.monthmapping)),
    ('map_season', Mapper(config.model_config.season, config.model_config.seasonmapping)),
    ('map_weathersit', Mapper(config.model_config.weathersit, config.model_config.weathersitmapping)),
    ('map_holiday', Mapper(config.model_config.holiday, config.model_config.holmapping)),
    ('map_workingday', Mapper(config.model_config.workingday, config.model_config.workingdaymapping)),
    ('map_hr', Mapper(config.model_config.hr, config.model_config.hourmapping)),

    ######## Handle outliers ########
    ('handle_outliers_temp', OutlierHandler(config.model_config.temp)),
    ('handle_outliers_atemp', OutlierHandler(config.model_config.atemp)),
    ('handle_outliers_hum', OutlierHandler(config.model_config.hum)),
    ('handle_outliers_windspeed', OutlierHandler(config.model_config.windspeed)),

    ######## One-hot encoding ########
    ('encode_weekday', WeekdayOneHotEncoder(variable = config.model_config.weekday)),

    # Scale features
    ('scaler', StandardScaler()),

    # Regressor
    ('model_rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
])