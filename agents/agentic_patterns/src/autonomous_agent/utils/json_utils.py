import json
from datetime import datetime
from typing import Any

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def json_serialize(obj: Any) -> str:
    """Helper function to serialize objects to JSON with datetime support."""
    return json.dumps(obj, cls=CustomJSONEncoder)

def json_deserialize(json_str: str) -> Any:
    """Helper function to deserialize JSON with datetime parsing."""
    def datetime_parser(dct):
        for k, v in dct.items():
            if isinstance(v, str):
                try:
                    dct[k] = datetime.fromisoformat(v)
                except ValueError:
                    pass
        return dct
    
    return json.loads(json_str, object_hook=datetime_parser)