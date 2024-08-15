DP_rank = []
PP_rank = []
TP_rank = []
CP_rank = []
EP_rank = []
EP_and_TP_rank = []

tensor_model_parallel_size = 4
data_parallel_size = 4
context_parallel_size = 2
pipeline_model_parallel_size = 4
expert_model_parallel_size = 4

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

tensor_and_data_group_size: int = tensor_model_parallel_size * data_parallel_size
num_tensor_and_data_groups: int = world_size // tensor_and_data_group_size
tensor_and_expert_group_size: int = tensor_model_parallel_size * expert_model_parallel_size
num_expert_groups: int = data_parallel_size // expert_model_parallel_size
for i in range(num_tensor_and_data_groups):
    for j in range(num_expert_groups):
        # TPxEP Group
        start_rank = i * tensor_and_data_group_size + j * tensor_and_expert_group_size
        end_rank = i * tensor_and_data_group_size + (j + 1) * tensor_and_expert_group_size
        ranks = range(start_rank, end_rank)
        EP_and_TP_rank.append(list(ranks))
        for k in range(tensor_model_parallel_size):
            ranks = range(
                start_rank + k, end_rank, tensor_model_parallel_size
            )
            EP_rank.append(list(ranks))

print(f"TP rank: {TP_rank}")
print(f"DP rank: {DP_rank}")
print(f"CP rank: {CP_rank}")
print(f"PP rank: {PP_rank}")
print(f"EP and TP rank: {EP_and_TP_rank}")
print(f"EP rank: {EP_rank}")
