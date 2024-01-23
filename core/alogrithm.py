from abc import ABC, abstractmethod


# 定义了调度算法的接口，用户自定义的调度算法必须实现这一接口，是实现策略模式的关键
class Algorithm(ABC):
    @abstractmethod
    def __call__(self, cluster, clock):
        pass
