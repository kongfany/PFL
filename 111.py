
# 使用单例模式进行演示
# 服务端的pfl_model,pfl_server
# 和客户端的pfl_client

# pfl_model:模型文件，定义训练时使用的模型，并且生成相应的联邦学习任务
from torch import nn
import torch.nn.functional as F
# pfl的核心策略包，包含pfl的工作模式策略，联邦学习算法策略和训练策略
import gfl.core.strategy as strategy
# pfl的任务管理包
from gfl.core.job_manager import JobManager


# 定义了一个简单的模型类net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    # 实例化模型model，和任务管理器job_manager
    model = Net()
    job_manager = JobManager()
    # 定义一个联邦学习任务job，工作模式设为单例模式，（也可以使用集群模式），联邦学习算法策略为联邦平均法，（也可以使用联邦蒸馏算法），epoch为3，模型类型为net
    job = job_manager.generate_job(work_mode=strategy.WorkModeStrategy.WORKMODE_STANDALONE,
                                   fed_strategy=strategy.FederateStrategy.FED_AVG, epoch=3, model=Net)
    # 向job_manager传入定义好的job和model
    job_manager.submit_job(job, model)

# 运行pfl_model后可以看到，job已经生成并添加成功了。同时在日志文件中也有了相应的记录。
# 编写pfl_server以及pfl_client
# 先运行pfl_server，运行后日志提示聚合开始
# 然后运行pfl_client,可以看到client已经获取到了job，载入模型开始进行训练
# 等client完成一轮训练后，可以在server端看到第一轮的参数聚合在server端完成了。
# 训练完成后，会产生一个模型文件
#
# 当然pfl框架也可以使用集群模式进行训练
# 只需要在client和server文件中声明地址，并选择工作模式为集群模式即可
# https://galaxylearning.github.io/quickstart/