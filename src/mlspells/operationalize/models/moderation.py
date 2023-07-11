import logging
from typing import Dict, List, Any, Optional, Union
import collections
import pandas as pd
import numpy as np
import mlflow
import aiometer
import functools

from mlflow.pyfunc import PythonModel, PythonModelContext
from mlflow.models.signature import ModelSignature
from mlflow.utils.environment import _mlflow_conda_env

from azure.core.exceptions import HttpResponseError
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory

from mlspells.base import StringEnum
from mlspells.base.async_tasks import run_async, prepare_async
from mlspells.operationalize import ResourceAccessMode, ValueSource, ValueSourceType

class ModerationActionBehavior(StringEnum):
    """
    Indicates the moderation behavior that the model should follow is a moderation violation is encountered.

    - ModerationActionBehavior.Append: Appends a moderation message at the end of the text
    - ModerationActionBehavior.Drop: Drop the entire row and replace it with a warning.
    """

    Append = "append"
    Drop = "drop"

class ModerationErrorBehavior(StringEnum):
    Drop = "drop"
    Ignore = "ignore"

class ModerationAction():
    def __init__(self, action: ModerationActionBehavior, message: Union[str, None] = None):
        assert action == ModerationActionBehavior.Drop or message, f"Action {str(action)} requires indicating a message."

        self.action = action
        self.message = message
        self.template = "Moderator: {message}. Violation {category} with severity {severity}"

class AzureContentSafetyGuardrailModel(PythonModel):
    def __init__(self, model_uri: str, column_name: Optional[str] = None, endpoint_uri: Optional[str] = None, 
                 access_mode: Optional[ResourceAccessMode] = None, categories: Optional[List[TextCategory]] = None,
                 action: Optional[ModerationAction] = None, config: Optional[Dict[str, ValueSource]] = None):
        """Creates a Python Model for MLflow that can apply content moderation to a base model.

        Parameters
        ----------
        model_uri : str
            The URI of the model you want to moderate. You can use a path to a local MLflow model
            or a URI like `models:/mymodel/latest.
        column_name : Optional[str], optional
            The name of the column you want to moderate, in case the model returns multiple
            columns, by default None. Needs to be indicated for models that has a ColSpec
            signature.
        endpoint_uri : Optional[str], optional
            The URL of the Azure Content Safety service, by default None. If not provided,
            a configuration with key `endpoint_uri` needs to be configured.
        access_mode : Optional[ResourceAccessMode], optional
            The access configuration for the Azure Content Safety service. If not configured,
            Key-based access will be used. When `ResourceAccessMode.Key` is used, a configuration
            with key `key` needs to be added to provide the given key.
        categories : Optional[List[TextCategory]], optional
            The moderation categories you want to moderate for, by default None
        action : Optional[ModerationAction], optional
            The action that has to be taken if the moderation detects unsuitable content, by
            default, ModerationActionBehavior.Drop is used.
        config : Optional[Dict[str, ValueSource]], optional
            A dictionary with the configuration of the model, by default None. You can use this
            configuration to control most of the properties of the model either by providing a
            value or by configuring environment variables to read the configuration from.

            For instance, use the following configuration to pull information from environment
            variables:

            config = { 
                "endpoint_uri": ValueSource(ValueSourceType.Environment, "ACS_ENDPOINT_URI"),
                "key": ValueSource(ValueSourceType.Environment, "ACS_ENDPOINT_KEY"),
            }

            You can also use Azure Key Value to protect values:

            config = { 
                "endpoint_uri": ValueSource(ValueSourceType.Environment, "ACS_ENDPOINT_URI"),
                "key": ValueSource(
                    ValueSourceType.AzureKeyVault, 
                    "ACS_ENDPOINT_KEY", 
                    connection_string="<akv connection string>"
                ),
            }            
        """
        self._model_artifact = "model"
        self._conf_key = "key"
        self._conf_column_name = "column_name"
        self._conf_access_mode = "access_mode"
        self._conf_endpoint_uri = "endpoint_uri"
        self._conf_severity_limit = "severity_limit"
        self._conf_acs_rps = "acs_rps"
        self.on_error = ModerationErrorBehavior.Drop
        self.on_error_message = "An error has occurred when moderating this data item and it has been removed from results."

        assert endpoint_uri or (config and self._conf_endpoint_uri in config.keys()), f"``{self._conf_endpoint_uri}`` has to be provided either as a parameter or as a configuration value"
        assert access_mode or (config and self._conf_access_mode in config.keys()), f"``{self._conf_access_mode}`` has to be provided either as a parameter or as a configuration value"
        assert access_mode == None or access_mode == ResourceAccessMode.RBAC or (config and self._conf_key in config.keys()), f"A configuration ``{self._conf_key}`` needs to be provided when ``ResourceAccess.Key``."

        self.model_uri = model_uri
        self.column_name = column_name
        self.categories = categories or [TextCategory.HATE, TextCategory.SELF_HARM]
        self.moderation = action or ModerationAction(ModerationActionBehavior.Drop, message="Content has been blocked by Azure Content Safety.")

        self.config = config or {}
        if access_mode:
            self.config[self._conf_access_mode] = ValueSource(ValueSourceType.Literal, default=str(access_mode))
        if endpoint_uri:
            self.config[self._conf_endpoint_uri] = ValueSource(ValueSourceType.Literal, default=endpoint_uri)
        if isinstance(self.config[self._conf_key], str):
            self.config[self._conf_key] = ValueSource(ValueSourceType.Literal, default=self.config[self._conf_key])
        if self._conf_severity_limit not in self.config.keys():
            self.config[self._conf_severity_limit] = ValueSource(ValueSourceType.Literal, default=0)
        if self._conf_acs_rps not in self.config.keys():
            self.config[self._conf_acs_rps] = ValueSource(ValueSourceType.Literal, default=1000)

    def load(self):
        self.load_context(None)
        
    def load_context(self, context: PythonModelContext):
        if context:
            self._model = mlflow.pyfunc.load_model(context.artifacts[self._model_artifact])
        else:
            self._model = mlflow.pyfunc.load_model(self.model_uri)
        
        self.endpoint_uri = self.config[self._conf_endpoint_uri].get_value()
        self.access_mode = self.config[self._conf_access_mode].get_value()
        self.max_rps = self.config[self._conf_acs_rps].get_value()
        self.severity_limit = self.config[self._conf_severity_limit].get_value()
        prepare_async()

    def _get_secret(self) -> str:
        if self.access_mode == ResourceAccessMode.Key:
            key = self.config[self._conf_key].get_value(not_null=True)
        elif self.access_mode == ResourceAccessMode.RBAC:
            raise NotImplementedError(ResourceAccessMode.RBAC)
        else:
            raise ValueError(self.access_mode)

        assert isinstance(key, str), f"Unable to get a valid secret from {self.config[self._conf_key].source}"
        return key
        
    def _moderate(self, contents, moderations):
        for idx in range(len(contents)):
            if moderations[idx]:
                for category, result in moderations[idx].items():
                    if result and result["severity"] > self.severity_limit:
                        if self.moderation.action == ModerationActionBehavior.Append:
                            contents[idx] = contents[idx] + self.moderation.template.format(
                                message=self.moderation.message,
                                category=category,
                                severity=result["severity"]
                            )
                        elif self.moderation.action == ModerationActionBehavior.Drop:
                            contents[idx] = self.moderation.template.format(
                                message=self.moderation.message,
                                category=category,
                                severity=result["severity"]
                            )
                        else:
                            raise ValueError(self.moderation.action)
            else:
                if self.on_error == ModerationErrorBehavior.Drop:
                    logging.warning(f"Item in row {idx} hasn't been moderated.")
                    contents[idx] = self.on_error_message

        return contents

    async def _analyze_content_async(self, contents: np.ndarray):
        async def analyze_content_request_async(client, request, idx):
            try:
                response = client.analyze_text(request)
            except HttpResponseError as e:
                logging.error("Error at sending request to Azure Content Safety service.")
                if e.error:
                    logging.error(f"Error code: {e.error.code}. Error message: {e.error.message}")
                
                errors[idx] = e
                response = None

            results[idx] = response

        results = np.arange(len(contents), dtype=object)
        pending = range(len(contents))
        for retry in range(1, 4):
            client = ContentSafetyClient(self.endpoint_uri, AzureKeyCredential(self._get_secret()))

            errors = {}
            requests = [
                functools.partial(analyze_content_request_async, client, AnalyzeTextOptions(text=content, categories=self.categories), idx)
                for idx, content in enumerate(contents[pending])
            ]
            await aiometer.run_all(
                requests, 
                max_per_second=self.max_rps,
            )

            if errors:
                pending = list(errors.keys())
                logging.error(f"{len(pending)} moderation requests have failed. Retry {retry} with a max of 3 times.")
            else:
                break
        
        return results

    def predict(self, context, input_data):
        predictions = self._model.predict(input_data)

        if self.column_name:
            if isinstance(predictions, pd.DataFrame):
                if self.column_name not in predictions.columns:
                    raise ValueError(f"Column {self.column_name} is not in the returned data")
                contents = predictions[self.column_name]
            elif isinstance(predictions, dict):
                if self.column_name not in predictions.keys():
                    raise ValueError(f"Key {self.column_name} is not in the returned data")
                if isinstance(predictions[self.column_name], np.ndarray):
                    contents = predictions[self.column_name]
                else:
                    contents = np.asarray([predictions[self.column_name]])
            else:
                raise TypeError("Indicating a column name requires predictions to be pandas or dictionary.")
        elif isinstance(predictions, collections.abc.Iterable):
            contents = np.asarray(predictions)
        else:
            raise ValueError(f"Model returned a type {str(type(predictions))} but we don't know how to read it")
            
        moderation = run_async(self._analyze_content_async(contents))
        self._moderate(contents, moderation)

        return contents

    def save_model(self, path: str):
        pip_requirements = mlflow.utils.requirements_utils._infer_requirements(self.model_uri, "pyfunc")
        pip_requirements.extend(["aiometer", "azure-ai-contentsafety"])

        if any([ conf.source == ValueSourceType.AzureKeyVault for _, conf in self.config.items() ]):
            pip_requirements.append("azure-keyvault-secrets")

        custom_env =_mlflow_conda_env(
            additional_conda_deps=None,
            additional_pip_deps=pip_requirements,
            additional_conda_channels=None,
        )

        mlflow.pyfunc.save_model(
            path,
            python_model=self,
            artifacts={
                "model": self.model_uri
            },
            conda_env=custom_env,
        ) 

    def log_model(self, artifact_path: str, registered_model_name: Optional[str]=None):
        pip_requirements = mlflow.utils.requirements_utils._infer_requirements(self.model_uri, "pyfunc")
        pip_requirements.extend(["aiometer", "azure-ai-contentsafety"])

        if any([ conf.source == ValueSourceType.AzureKeyVault for _, conf in self.config.items() ]):
            pip_requirements.append("azure-keyvault-secrets")

        custom_env =_mlflow_conda_env(
            additional_conda_deps=None,
            additional_pip_deps=pip_requirements,
            additional_conda_channels=None,
        )

        return mlflow.pyfunc.log_model(
            artifact_path, 
            python_model=self,
            artifacts={
                "model": self.model_uri
            },
            pip_requirements=pip_requirements,
            conda_env=custom_env,
            registered_model_name=registered_model_name)