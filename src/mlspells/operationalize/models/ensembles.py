from mlflow.pyfunc import PythonModel, PythonModelContext
from sklearn.compose import ColumnTransformer
import pandas as pd
import mlflow

class PartitionedModelEnsemble(PythonModel):
    """
    A custom model that implements a paritioned inferrencing strategy over an ensamble of models.
    """
    def __init__(self, key: str, prediction_col: str, transformations: ColumnTransformer):
        """
        Creates a new instance of the mode.

        Parameters
        ----------
        key: str
            The name of the column the data will be partitioned on.
        prediction_col: str
            The name of the column the model generates predictions on.
        transformations: ColumnTransformer
            Any given transformation that needs to be applied to the data before sending to the model.
        """
        self.pred_col = prediction_col
        self.key = key
        self.transformations = transformations

    def load_context(self, context: PythonModelContext):
        """
        Loads all the models for the given partitions. This method assumes the models were logged with the
        different of the column `key` as artifact key.
        """
        self.models = { key: mlflow.pyfunc.load_model(model_path) for key, model_path in context.artifacts.items() }
        
    def predict(self, context: PythonModelContext, data: pd.DataFrame) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            predictions = pd.DataFrame(0, index=data.index, columns=[self.pred_col])
            
            # Get all the unique partition's value in the input data
            key_ids = data[self.key].unique()

            # We will run 1 predict call per each partition
            for key_id in key_ids:
                input_data_idx = data[self.key] == key_id

                if self.transformations:
                    columns = [name.split('__')[1] for name in self.transformations.get_feature_names_out()]
                    transformed_data = pd.DataFrame(self.transformations.transform(data[input_data_idx]), columns=columns)
                else:
                    transformed_data = data[input_data_idx]
                predictions[input_data_idx] = self.models[key_id].predict(transformed_data.loc[:, transformed_data.columns != self.key]).reshape(-1,1)

            return predictions
        
        raise TypeError("This implementation can only take pandas DataFrame as inputs")