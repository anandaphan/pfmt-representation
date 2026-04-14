# Multitasing Problem for Case 6
# 074 and 109
import FingeringTask as FTask

task_1 = FTask.FingeringTask("./PIG/074-1_fingering.txt")
task_2 = FTask.FingeringTask("./PIG/109-1_fingering.txt")

def task_fit(vec, i):
    if i == 1:
        vars = task_1.decode(vec)
        return task_1.fingering_difficulty(vars)
    elif i == 2:
        vars = task_2.decode(vec)
        return task_2.fingering_difficulty(vars)
