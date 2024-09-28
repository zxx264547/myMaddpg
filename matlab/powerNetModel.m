% 加载 MATPOWER 案例
mpc = loadcase('case14');

% 执行潮流计算
results = runpf(mpc);

% 查看结果
disp(results.bus);
