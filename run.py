# -*- coding: utf-8 -*-


from src.data_collecting import collect_data
from src.data_processing import process_data
from src.feature_engineering import engineer_features
from src.model_building import build_regional_models
from src.reporting import create_dashboard_data

collect_data()
process_data()
engineer_features()
build_regional_models()
create_dashboard_data()