from enum import Enum

class AggregationFunc(Enum): 
    MAX = 'max'
    MIN = 'min'
    MEAN = 'mean'
    MODE = lambda x:x.value_counts().index[0]
    SUM = 'sum'
    VAR = 'var'
    CONCAT = lambda x: ''.join(x)

    ANY = lambda x: any(x)
    ALL = lambda x: all(x)
    NOT_ALL = lambda x: not all(x)
    NONE = lambda x: not any(x)

    FIRST = lambda x: x.iloc[0]
    LAST = lambda x: x.iloc[-1]
    COUNT = lambda x:x.count()

    MAX_DIFF = lambda x: max(x) - min(x)

