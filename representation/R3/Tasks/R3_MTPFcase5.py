# Multitasing Problem for Case 5
# 040 and 109
import FingeringTask as FTask

task_1 = FTask.FingeringTask("./PIG/040-1_fingering.txt", decoder_type="controlled_softmax_decoder")
task_2 = FTask.FingeringTask("./PIG/109-1_fingering.txt", decoder_type="controlled_softmax_decoder")

def task_fit(vec, i, tau):
    if i == 1:
        vars_, nRawFinger = task_1.decode(vec, tau)
        fitness, penalty = task_1.fingering_difficulty(vars_, nRawFinger)
        return fitness, penalty
    elif i == 2:
        vars_, nRawFinger= task_2.decode(vec, tau)   
        fitness, penalty = task_2.fingering_difficulty(vars_, nRawFinger)
        return fitness, penalty