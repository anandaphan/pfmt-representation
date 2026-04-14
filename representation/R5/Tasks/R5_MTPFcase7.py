# Multitasing Problem for Case 7
# 076 and 109
import FingeringTask as FTask

task_1 = FTask.FingeringTask("./PIG/076-1_fingering.txt", decoding_type="latent", model_path="./models/VAE_l1_checkpoint.pt")
task_2 = FTask.FingeringTask("./PIG/109-1_fingering.txt", decoding_type="latent", model_path="./models/VAE_l1_checkpoint.pt")

def task_fit(vec, i):
    if i == 1:
        vars_, nRawFinger = task_1.decode(vec)
        fitness, penalty = task_1.fingering_difficulty(vars_, nRawFinger)
        return fitness, penalty
    elif i == 2:
        vars_, nRawFinger= task_2.decode(vec)   
        fitness, penalty = task_2.fingering_difficulty(vars_, nRawFinger)
        return fitness, penalty

