from entities import Chunk

"""
Утилита reciprocal rank fusion
results_list: список списков output чанков,
                уже отсортированных по релевантности.
weight_list: список соответствующих весов, если
                без веса, то оставить пустым
return: список chunk_id
"""
def rrf_scoring(results_lists: list[list[Chunk]], weight_list = [], k=60):
    if len(weight_list) == 0 or len(weight_list) != len(results_lists):
        weight_list = [1 for i in range(len(results_lists))]
    scores: dict[str, float] = {}
    for i in range(len(results_lists)):
        for rank, chunk in enumerate(results_lists[i]):
            chunk_id = chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0) + weight_list[i] / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)