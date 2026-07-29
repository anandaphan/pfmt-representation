# Multitasing Problem for Case 3
# 110 and 145
import FingeringTask as FTask

task_1 = FTask.FingeringTask("./PIG/110-1_fingering.txt", decoder_type="topk_truncation_mapping")
task_2 = FTask.FingeringTask("./PIG/145-1_fingering.txt", decoder_type="topk_truncation_mapping")

def task_fit(vec, i):
    if i == 1:
        vars_, nRawFinger = task_1.decode(vec)
        fitness, penalty = task_1.fingering_difficulty(vars_, nRawFinger)
        return fitness, penalty
    elif i == 2:
        vars_, nRawFinger= task_2.decode(vec)   
        fitness, penalty = task_2.fingering_difficulty(vars_, nRawFinger)
        return fitness, penalty