from dataclasses import asdict

from entities import (
    Question as DQuestion,
    ResponseCheckAnswer as DResponseCheckAnswer,
    ChunkMetaData as DChunkMetaData,
    Chunk as DChunk,
    ResponseRagSearch as DResponseRagSearch,
    ResponseRagSearchLlm as DResponseRagSearchLlm,
)

from api_schemas import (
    Question,
    ResponseCheckAnswer,
    ChunkMetaData,
    Chunk,
    ResponseRagSearch,
    ResponseRagSearchLlm,
)


def convert_chunk_metadata(obj: DChunkMetaData) -> ChunkMetaData:
    out = ChunkMetaData.model_validate(asdict(obj))
    return out


def convert_chunk(obj: DChunk) -> Chunk:
    data = asdict(obj)
    data["metadata"] = convert_chunk_metadata(obj.metadata)
    out = Chunk.model_validate(data)
    return out


def convert_rag_search(obj: DResponseRagSearch) -> ResponseRagSearch:
    out = ResponseRagSearch(
        chunk=convert_chunk(obj.chunk),
        score=float(obj.score),
    )
    return out


def convert_rag_search_llm(obj: DResponseRagSearchLlm) -> ResponseRagSearchLlm:
    out = ResponseRagSearchLlm(
        chunks_response=[convert_rag_search(x) for x in obj.chunks_response],
        llm_answer=obj.llm_answer,
    )
    return out


def convert_question(obj: DQuestion) -> Question:
    out = Question.model_validate(asdict(obj))
    return out


def convert_check_answer(obj: DResponseCheckAnswer) -> ResponseCheckAnswer:
    out = ResponseCheckAnswer.model_validate(asdict(obj))
    return out