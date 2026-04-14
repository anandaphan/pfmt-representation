classdef R5_PF_MTSO4_046_145 < Problem
% <Multi-task> <Single-objective> <None>

methods
    function Prob = R5_PF_MTSO4_046_145(varargin)
        % 调用父类构造函数
        Prob = Prob@Problem(varargin);
        Prob.maxFE = 2 * 50 * 1000;
        Prob.N = 50;
    end

    function setTasks(Prob)
        Prob.T = 2;
        Prob.M = [1, 1];
        Prob.D = [145, 179];
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
    py.sys.path().append('Problems/PianoFingering/R5/Tasks'); 
    R5_MTSO4 = py.importlib.import_module('R5_MTPFcase4'); 
    py.importlib.reload(R5_MTSO4);
    for i = 1:size(x, 1)
        y = py.numpy.array(x(i, :), dtype = py.numpy.float32);
        out = R5_MTSO4.task_fit(y, taskIdx);
        Obj(i) = double(out(1));
        Con(i) = double(out(2));
    end
    py.gc.collect();
    clear py;
end    