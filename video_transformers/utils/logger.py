from argparse import Namespace
from typing import Any, Dict, Generator, List, MutableMapping, Optional

import numpy as np
from torch import Tensor

# modified from lightning/src/pytorch_lightning/utilities/logger.py


def _flatten_dict(params: Dict[Any, Any], delimiter: str = "/") -> Dict[str, Any]:
    """Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.
    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.
    Returns:
        Flattened dict.
    Examples:
        >>> _flatten_dict({'a': {'b': 'c'}})
        {'a/b': 'c'}
        >>> _flatten_dict({'a': {'b': 123}})
        {'a/b': 123}
        >>> _flatten_dict({5: {'a': 123}})
        {'5/a': 123}
    """

    def _dict_generator(
        input_dict: Any, prefixes: List[Optional[str]] = None
    ) -> Generator[Any, Optional[List[str]], List[Any]]:
        prefixes = prefixes[:] if prefixes else []
        if isinstance(input_dict, MutableMapping):
            for key, value in input_dict.items():
                key = str(key)
                if isinstance(value, (MutableMapping, Namespace)):
                    value = vars(value) if isinstance(value, Namespace) else value
                    yield from _dict_generator(value, prefixes + [key])
                else:
                    yield prefixes + [key, value if value is not None else str(None)]
        else:
            yield prefixes + [input_dict if input_dict is None else str(input_dict)]

    return {delimiter.join(keys): val for *keys, val in _dict_generator(params)}


def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Returns params with non-primitvies converted to strings for logging.
    >>> params = {"float": 0.3,
    ...           "int": 1,
    ...           "string": "abc",
    ...           "bool": True,
    ...           "list": [1, 2, 3],
    ...           "namespace": Namespace(foo=3),
    ...           "layer": torch.nn.BatchNorm1d}
    >>> import pprint
    >>> pprint.pprint(_sanitize_params(params))  # doctest: +NORMALIZE_WHITESPACE
    {'bool': True,
        'float': 0.3,
        'int': 1,
        'layer': "<class 'torch.nn.modules.batchnorm.BatchNorm1d'>",
        'list': '[1, 2, 3]',
        'namespace': 'Namespace(foo=3)',
        'string': 'abc'}
    """
    for k in params.keys():
        # convert relevant np scalars to python types first (instead of str)
        if isinstance(params[k], (np.bool_, np.integer, np.floating)):
            params[k] = params[k].item()
        elif type(params[k]) not in [bool, int, float, str, Tensor]:
            params[k] = str(params[k])
    return params
