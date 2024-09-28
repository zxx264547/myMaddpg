import pandapower as pp
import pandapower.networks as pn
import pandapower.plotting as plot

# 创建一个测试网络
net = pn.case30()

# 运行潮流计算
pp.runpp(net)

# 使用 simple_plot 进行可视化
plot.simple_plot(net)

# 打印出每个节点的电压
for bus, voltage in zip(net.bus.index, net.res_bus.vm_pu):
    print(f"Bus {bus}: Voltage = {voltage:.4f} pu")
