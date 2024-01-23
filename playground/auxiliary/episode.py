import simpy
from core.cluster import Cluster
from core.scheduler import Scheduler
from core.broker import Broker
from core.simulation import Simulation


# episode 中的 Episode 类用于 episodic 方式的仿真实验
class Episode(object):
    broker_cls = Broker

    def __init__(self, machine_configs, task_configs, algorithm, event_file):
        self.env = simpy.Environment()
        cluster = Cluster()
        cluster.add_machines(machine_configs)

        task_broker = Episode.broker_cls(self.env, task_configs)

        scheduler = Scheduler(self.env, algorithm)

        self.simulation = Simulation(self.env, cluster, task_broker, scheduler, event_file)

    def run(self):
        self.simulation.run()
        self.env.run()
