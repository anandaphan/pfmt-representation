classdef R3_PF_MTSO7_076_109 < Problem
% <Multi-task> <Single-objective> <None>

methods
    function Prob = R3_PF_MTSO7_076_109(varargin)
        % 调用父类构造函数
        Prob = Prob@Problem(varargin);
        Prob.maxFE = 2 * 50 * 1000;
        Prob.N = 50;
    end

    function setTasks(Prob)
        Prob.T = 2;
        Prob.M = [1, 1];
        Prob.D = [155, 341];
        Prob.Lb{1} = zeros(1, Prob.D(1));
        Prob.Ub{1} = ones(1, Prob.D(1));
        Prob.Lb{2} = zeros(1, Prob.D(2));
        Prob.Ub{2} = ones(1, Prob.D(2));
        Prob.Fnc{1} = @(x)TaskWrapper(x, 1);
        Prob.Fnc{2} = @(x)TaskWrapper(x, 2);
    end
end
end


function [Obj, Con] = TaskWrapper(x, taskIdx)
    % （保持原有包装函数不变）
    Obj = zeros(size(x, 1), 1);
    Con = zeros(size(x, 1), 1);
    py.sys.path().append('Problems/PianoFingering/R3/Tasks');
    R3_MTSO7 = py.importlib.import_module('R3_MTPFcase7');
    py.importlib.reload(R3_MTSO7);

    for i = 1:size(x, 1);
        y = py.numpy.array(x(i, :), dtype = py.numpy.float32);
        out = R3_MTSO7.task_fit(y, taskIdx);
        Obj(i) = double(out(1));
        Con(i) = double(out(2));
    end
    py.gc.collect();
    clear py;
end
