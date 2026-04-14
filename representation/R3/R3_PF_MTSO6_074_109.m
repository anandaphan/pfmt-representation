classdef R3_PF_MTSO6_074_109 < Problem
% <Multi-task> <Single-objective> <None>

properties
    Tau = 1;
    Tau_min = 0.1;
    Alpha = 0.99;
end 

methods
    function Prob = R3_PF_MTSO6_074_109(varargin)
        % 调用父类构造函数
        Prob = Prob@Problem(varargin);
        Prob.maxFE = 2 * 50 * 1000;
        Prob.N = 50;
    end

    function parameter = getParameter(Prob)
        parameter = {'Tau_0: Initial Softmax Temperature', num2str(Prob.Tau), 'Tau_min: Minimum Tau', num2str(Prob.Tau_min), 'Alpha: Annealing Rate', num2str(Prob.Alpha)}
        parent_para = Prob.getRunParameter();
        parameter = [parent_para, parameter];       
    end

    function Prob = setParameter(Prob, Parameter)
        Prob.N = str2double(Parameter{1});
        Prob.maxFE = str2double(Parameter{2});
        Prob.Tau = str2double(Parameter{3});
        Prob.Tau_min = str2double(Parameter{4});
        Prob.Alpha = str2double(Parameter{5});
    end

    function setTasks(Prob)
        Prob.T = 2;
        Prob.M = [1, 1];
        Prob.D = [555*5, 170*5];
        Prob.Lb{1} = zeros(1, Prob.D(1));
        Prob.Ub{1} = ones(1, Prob.D(1));
        Prob.Lb{2} = zeros(1, Prob.D(2));
        Prob.Ub{2} = ones(1, Prob.D(2));
        Prob.Fnc{1} = @(x)TaskWrapper(x, 1, Prob);
        Prob.Fnc{2} = @(x)TaskWrapper(x, 2, Prob);
    end 

    function checkUpdateTau(Prob, taskIdx)
        if(taskIdx == Prob.T)
            Prob.Tau = max(Prob.Tau * Prob.Alpha, Prob.Tau_min)
        end
    end
end
end


function [Obj, Con] = TaskWrapper(x, taskIdx, Prob)
    % （保持原有包装函数不变）
    Obj = zeros(size(x, 1), 1);
    Con = zeros(size(x, 1), 1);
    py.sys.path().append('Problems/PianoFingering/R3/Tasks'); 
    R3_MTSO6 = py.importlib.import_module('R3_MTPFcase6'); 
    py.importlib.reload(R3_MTSO6);

    for i = 1:size(x, 1);
        y = py.numpy.array(x(i, :), dtype = py.numpy.float32);
        out = R3_MTSO6.task_fit(y, taskIdx, Prob.Tau);
        Obj(i) = double(out(1));
        Con(i) = double(out(2));
    end
    py.gc.collect();
    clear py;

    Prob.checkUpdateTau(taskIdx);
end    