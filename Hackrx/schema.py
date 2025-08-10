import uuid
from typing import List, Union

from pydantic import BaseModel


class QueryIn(BaseModel):
    documents : Union[List[str], str]
    questions : List[str]
