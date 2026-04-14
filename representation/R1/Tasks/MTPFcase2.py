# Multitasing Problem for Case 2
# 002 and 145
import FingeringTask as FTask

task_1 = FTask.FingeringTask("./PIG/002-1_fingering.txt")
task_2 = FTask.FingeringTask("./PIG/145-1_fingering.txt")

def task_fit(vec, i):
    if i == 1:
        vars = task_1.decode(vec)
        return task_1.fingering_difficulty(vars)
    elif i == 2:
        vars = task_2.decode(vec)
        return task_2.fingering_difficulty(vars)
