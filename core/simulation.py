from core.monitor import Monitor


# simulation 是对一次仿真的建模，一次仿真必须构造一个集群 Cluster 实例；
# 构造一系列作业配置 JobConfig 实例，利用这些作业配置实例构造一个 Broker 实例；
# 构造一个调度器 Scheduler 实例。在一次仿真可以选择开是否使用一个 Monitor 实例进行仿真过程的监测
class Simulation(object):
    def __init__(self, env, cluster, task_broker, scheduler, event_file):
        self.env = env
        self.cluster = cluster
        self.task_broker = task_broker
        self.scheduler = scheduler
        self.event_file = event_file
        if event_file is not None:
            self.monitor = Monitor(self)

        self.task_broker.attach(self)
        self.scheduler.attach(self)

    def run(self):
        # Starting monitor process before task_broker process
        # and scheduler process is necessary for log records integrity.
        if self.event_file is not None:
            self.env.process(self.monitor.run())
        self.env.process(self.task_broker.run())
        self.env.process(self.scheduler.run())

    @property
    def finished(self):
        return self.task_broker.destroyed \
               and len(self.cluster.unfinished_jobs) == 0
