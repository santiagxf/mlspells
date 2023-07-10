from enum import Enum

class StringEnum(Enum):
    """
    Represents enums with string associated values.
    """
    def __str__(self):
        return str(self.value)
    
    def __eq__(self, other):
        return str(self) == str(other)
        