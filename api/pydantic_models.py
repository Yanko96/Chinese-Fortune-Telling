from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class ModelName(str, Enum):
    MOONSHOT_V1_8K = "moonshot-v1-8k"
    MOONSHOT_V1_32K = "moonshot-v1-32k"
    MOONSHOT_V1_128K = "moonshot-v1-128k"

class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.MOONSHOT_V1_8K)

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime

class DeleteFileRequest(BaseModel):
    file_id: int