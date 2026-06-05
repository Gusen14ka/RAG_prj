from pydantic import BaseModel

class Question(BaseModel):
    question_id: int
    term: str
    question: str
    answers: list[str]
    chunk_ids: list[str]



class ChunkMetaData(BaseModel):
    section: str
    subsection: str


class Chunk(BaseModel):
    chunk_id: str
    clean_text: str
    raw_text: str
    metadata: ChunkMetaData


class ResponseRagSearch(BaseModel):
    chunk: Chunk
    score: float


class ResponseRagSearchLlm(BaseModel):
    chunks_response: list[ResponseRagSearch]
    llm_answer: str

class ResponseCheckAnswer(BaseModel):
    is_correct: bool
    score: float
    gold_answers: list[str]
    source_chunks: list[Chunk]