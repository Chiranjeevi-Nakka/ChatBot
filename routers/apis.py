from typing import List, Optional
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, Depends
from services import api_service, api_service
from pydantic import BaseModel
from sqlalchemy.orm import Session
# import model

router = APIRouter(
    responses = {404:{"description": 'Not Found'}},
    tags = ["Employee Assistance"],
    prefix='/harvey'
)

class ChatBotRequest(BaseModel):
    query_string: str
    memory_messages: Optional[List[dict]] = []

@router.post("/harvey-chatbot")
def get_answer_from_harvey(
    userRequest: ChatBotRequest
):
    # response =  api_services.pdf_service(query_string)

    query_string= userRequest.query_string
    memory_messages= userRequest.memory_messages
    
    return StreamingResponse(api_service.pdf_service(query_string, memory_messages), headers={"Content-Encoding": "none"}, media_type='text/event-stream')
