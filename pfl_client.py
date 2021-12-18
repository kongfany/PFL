import torch
from torchvision import datasets, transforms
# 导入pfl的相关包，包括客户端包，策略包，训练控制器
from gfl.core.client import FLClient
from gfl.core.strategy import WorkModeStrategy, TrainStrategy, LossStrategy
from gfl.core.trainer_controller import TrainerController

CLIENT_ID = 0

if __name__ == "__main__":
    # CLIENT_ID = int(sys.argv[1])
    # 数据集mnist的导入
    mnist_data = datasets.MNIST("D:\Python\PycharmProjects\pflttt\mnist_data", download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.13066062,), (0.30810776,))
    ]))
    # 实例化一个客户端
    client = FLClient()
    # 获得服务器端的模型pfl
    gfl_models = client.get_remote_gfl_models()

    # 由于一个flclient可以参与多个job，所以获得的pflmodels也有可能是多个
    # 循环遍历这些模型
    for gfl_model in gfl_models:
        # 对于每个模型，要定义好优化集，optimize，和训练策略，train_strategy
        optimizer = torch.optim.SGD(gfl_model.get_model().parameters(), lr=0.01, momentum=0.5)
        train_strategy = TrainStrategy(optimizer=optimizer, batch_size=32, loss_function=LossStrategy.NLL_LOSS)
        # 并将策略传入到对应的pfl_model中，(不同的模型也可以设置不同的训练策略)
        gfl_model.set_train_strategy(train_strategy)

    # 调用TrainerController设置工作模式为单例模式，模型为pfl_models,数据为mnist数据集
    # 客户端id为0，curve设置本地是否显示训练曲线，concurrent_num为本地线程数
    # 启动TrainerController
    TrainerController(work_mode=WorkModeStrategy.WORKMODE_STANDALONE, models=gfl_models, data=mnist_data, client_id=CLIENT_ID,
                      curve=True, concurrent_num=3).start()