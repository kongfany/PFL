# 根据在pfl_model文件中设置的job参数
# 从server包中导入了FLStandaloneServer，再从strategy中导入的联邦学习策略算法FederateStrategy
from gfl.core.server import FLStandaloneServer
from gfl.core.strategy import FederateStrategy

FEDERATE_STRATEGY = FederateStrategy.FED_AVG

if __name__ == "__main__":
    # 定义好server的联邦学习方法策略为FederateStrategy,并启动server
    FLStandaloneServer(FEDERATE_STRATEGY).start()

# 运行后日志提示聚合开始