from typing import Any, Optional
from mlspells.base import StringEnum
import os

class ResourceAccessMode(StringEnum):
    Key = "key"
    RBAC = "rbac"

class ValueSourceType(StringEnum):
    AzureKeyVault = "akv"
    Environment = "environ"
    Literal = "literal"

class ValueSource():
    def __init__(self, source: ValueSourceType, name: Optional[str] = None, default: Any = None, connection_string: Optional[str] = None):
        self.source = source
        self.name = name
        self.connection_string = connection_string
        self.default = default

        assert source != ValueSourceType.Literal or name is None, "``name`` is not valid for ``ValueSourceType.Literal``"
        assert source != ValueSourceType.Literal or default is not None, "``default`` has to be indicated for ``ValueSourceType.Literal``"
        assert source != ValueSourceType.AzureKeyVault or connection_string, "``connection_string`` is required for ``ValueSourceType.AzureKeyVault``"

    def get_value(self, not_null: bool = False) -> Any:
        if self.source == ValueSourceType.Environment:
            value = os.environ.get(self.name, self.default) # type: ignore
            if value == None and not_null:
                raise ValueError(f"The environment variable {self.name} is not correctly configured.")
            return value
        elif self.source == ValueSourceType.AzureKeyVault:
            try:
                from azure.keyvault.secrets import SecretClient
                from azure.identity import DefaultAzureCredential
            except ImportError as ex:
                raise ImportError("azure-keyvault-secrets") from ex

            if self.connection_string and self.name:
                client = SecretClient(vault_url=self.connection_string, credential=DefaultAzureCredential())

                try:
                    retrieved_secret = client.get_secret(self.name)
                    value = retrieved_secret.value
                except:
                    value = None

                if value == None and not_null:
                    raise ValueError(f"The azure key vault doesn't have a secret for {self.name}.")
                return value
            else:
                raise ValueError("``connection_string`` or ``name`` are not configured")
        elif self.source == ValueSourceType.Literal:
            return self.default
