from core.job import Job


# 实现了类 Broker，Broker 代替用户对计算集群提交作业
class Broker(object):
    job_cls = Job

    def __init__(self, env, job_configs):
        self.env = env
        self.simulation = None
        self.cluster = None
        self.destroyed = False
        self.job_configs = job_configs

    def attach(self, simulation):
        self.simulation = simulation
        self.cluster = simulation.cluster

    def run(self):
        for job_config in self.job_configs:
            # job_config.submit_time和self.env.now均为0
            job_config.submit_time = self.env.now
            assert job_config.submit_time >= self.env.now
            yield self.env.timeout(job_config.submit_time - self.env.now)
            job = Broker.job_cls(self.env, job_config)
            # print('a task arrived at time %f' % self.env.now)
            self.cluster.add_job(job)
        self.destroyed = True
