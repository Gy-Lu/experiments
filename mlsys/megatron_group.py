DP_rank = []
PP_rank = []
TP_rank = []
CP_rank = []

tensor_model_parallel_size = 4
data_parallel_size = 4
context_parallel_size = 2
pipeline_model_parallel_size = 4

world_size = tensor_model_parallel_size * data_parallel_size * pipeline_model_parallel_size

num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size
num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size

for i in range(pipeline_model_parallel_size):
    start_rank = i * num_pipeline_model_parallel_groups
    end_rank = (i + 1) * num_pipeline_model_parallel_groups
    for j in range(tensor_model_parallel_size):
        ranks = range(start_rank + j, end_rank, tensor_model_parallel_size)
        DP_rank.append(list(ranks))
        for k in range(data_parallel_size // context_parallel_size):
            ranks = range(
                start_rank + j + k * (tensor_model_parallel_size * context_parallel_size),
                start_rank + j + (k + 1) * (tensor_model_parallel_size * context_parallel_size),
                tensor_model_parallel_size,
            )
            CP_rank.append(list(ranks))
            

for i in range(num_tensor_model_parallel_groups):
    ranks = range(i * tensor_model_parallel_size,
                    (i + 1) * tensor_model_parallel_size)
    TP_rank.append(list(ranks))

for i in range(num_pipeline_model_parallel_groups):
    ranks = range(i, world_size, num_pipeline_model_parallel_groups)
    PP_rank.append(list(ranks))

print(f"TP rank: {TP_rank}")
print(f"DP rank: {DP_rank}")
print(f"CP rank: {CP_rank}")
print(f"PP rank: {PP_rank}")
