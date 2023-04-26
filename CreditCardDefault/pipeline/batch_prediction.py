from CreditCardDefault.exception import CreditCardDefaultException
from CreditCardDefault.logger import logging

import pandas as pd
from CreditCardDefault.utils import load_object
import os, sys
from datetime import datetime


PREDICTION_DIR = "prediction"
INPUT_DIR = "input"
OUTPUT_DIR = "output"








def start_batch_prediction(input_file,transformer_object_path,model_path):
    try:
        os.makedirs(PREDICTION_DIR, exist_ok=True)


        logging.info(f"Reading file :{input_file}")
        df = pd.read_csv(input_file)


        # saving input file:
        logging.info(f"Saving the input file")
        input_dir_path = os.path.join(PREDICTION_DIR,INPUT_DIR)
        output_dir_path = os.path.join(PREDICTION_DIR,OUTPUT_DIR)
        os.makedirs(input_dir_path,exist_ok=True)
        os.makedirs(output_dir_path,exist_ok=True)
        input_file_path = os.path.join(input_dir_path, f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        df.to_csv(input_file_path, index=False, header=True)

        # validation

        logging.info(f"Loading transformer to transform dataset")

        transformer = load_object(file_path= transformer_object_path)
        input_feature_names = list(transformer.feature_names_in_)
        input_arr = transformer.transform(df[input_feature_names])

        logging.info(f"Loading model to make prediction")

        model = load_object(file_path=model_path)
        prediction = model.predict(input_arr)

        output_df = df

        output_df["Prediction"] = prediction

        prediction_file_name = os.path.basename(input_file_path).replace(".csv",
                                                                         f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(output_dir_path, f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        output_df.to_csv(prediction_file_path, index=False, header=True)

        return prediction_file_path
    except Exception as e:
        raise CreditCardDefaultException(e, sys)
