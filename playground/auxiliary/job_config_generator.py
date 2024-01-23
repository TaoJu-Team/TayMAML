import random

from DAG_generator import plot_DAG
import DAG_generator as DAG
import csv
import numpy as np
import pandas as pd


class Generator_Of_job:
    def __init__(self, num_of_nodes, max_out, alpha, beta):
        self.num_of_nodes = num_of_nodes
        self.max_out = max_out
        self.alpha = alpha
        self.beta = beta
        self.dag = None
        self.edges_array = None
        self.in_degree = None
        self.out_degree = None
        self.positions = None
        self.num_of_jobs = 10000
        # self.dag_plot = plot_DAG(self.edges_array, self.positions)

    def adjust_prior(self, node_num, prior):
        if (node_num + 1) in prior:
            return False
        else:
            return True

    def adjust_precursor_node(self, node, array):
        if node in array:
            return False
        else:
            return True

    def get_prior(self):
        # 前序节点序列，初始只有'Start'
        prior = []
        precursor_node = ['Start']
        while len(prior) < len(self.in_degree):
            head_position = len(precursor_node)
            temp_precursor_node = precursor_node
            for i in self.edges_array:
                if (i[0] in temp_precursor_node) and (i[1] != 'Exit'):
                    if self.adjust_precursor_node(i[1], precursor_node):
                        precursor_node.append(i[1])
                        self.in_degree[i[1] - 1] = self.in_degree[i[1] - 1] - 1
                    else:
                        self.in_degree[i[1] - 1] = self.in_degree[i[1] - 1] - 1
            end_position = len(precursor_node)
            precursor_node = precursor_node[head_position:end_position]
            for i in range(len(self.in_degree)):
                if self.in_degree[i] == 0 and self.adjust_prior(i, prior):
                    prior.append(i + 1)
        return prior

    def generate_job_config(self):
        """
        依据edges的值进行作业配置的生成：
        在edges当中，
        'start' 为第一个job
        ‘1’ 为第二个job
        ...
        ‘n’ 为第n+1个job
        ...
        ‘exit’ 为最后一个job
        [(1, 6),
        (2, 8), (2, 5),
        (3, 8), (3, 6), (3, 5),
        (4, 5),
        ('Start', 1), ('Start', 2), ('Start', 3), ('Start', 4), ('Start', 7),
        (5, 'Exit'), (6, 'Exit'), (7, 'Exit'), (8, 'Exit')]
        """
        job_index = []
        task_id = []
        instances_num = []
        task_type = []
        job_id = []
        status = []
        start_time = []
        end_time = []
        cpu = []
        memory = []
        duration = []
        disk = []
        submit_time = []

        time = random.randint(1, 20)
        for index_job in range(self.num_of_jobs):
            self.dag = DAG.DAGs_generate(self.num_of_nodes, self.max_out, self.alpha, self.beta)
            self.edges_array = self.dag[0]
            self.in_degree = self.dag[1]
            self.out_degree = self.dag[2]
            self.positions = self.dag[3]

            edges = self.edges_array
            edges = np.array(edges)
            for edge in edges:
                if edge[0] == 'Start':
                    edge[0] = 1
                else:
                    edge[0] = int(edge[0]) + 1
                if edge[1] == 'Exit':
                    edge[1] = self.num_of_nodes + 2
                else:
                    edge[1] = int(edge[1]) + 1
            temp = []
            for edge in edges:
                array = []
                data1 = int(edge[0])
                array.append(data1)
                data2 = int(edge[1])
                array.append(data2)
                temp.append(tuple(array))
            edges = temp

            # job_index内容的写入
            job_index_temp = []
            for i in range(self.num_of_nodes + 2):
                job_index_temp.append(i)
            job_index = job_index + job_index_temp

            # task_id内容的写入
            task_id_temp = []
            for i in range(self.num_of_nodes + 2):
                task_id_str = str(i + 1)
                for edge in edges:
                    if edge[1] == i + 1:
                        task_id_str = task_id_str + '_' + str(edge[0])
                task_id_temp.append(task_id_str)
            task_id = task_id + task_id_temp
            # task_id = pd.DataFrame(data=task_id)
            # task_id.to_csv('C:/Users/hasee/Desktop/job.csv', index=False, header=['task_id'])

            # instances_num内容的写入
            instances_num_temp = []
            for i in range(self.num_of_nodes + 2):
                instances_num_temp.append(random.randint(1, 25))
            instances_num = instances_num + instances_num_temp

            # task_type内容的写入
            task_type_temp = []
            for i in range(self.num_of_nodes + 2):
                task_type_temp.append('A')
            task_type = task_type + task_type_temp

            # job_id内容的写入
            job_id_temp = []
            for i in range(self.num_of_nodes + 2):
                job_id_temp.append(index_job + 1)
            job_id = job_id + job_id_temp

            # status内容的写入
            status_temp = []
            for i in range(self.num_of_nodes + 2):
                status_temp.append('Terminated')
            status = status + status_temp

            # start_time、end_time内容的写入
            start_time_temp = []
            end_time_temp = []
            time = time
            for i in range(self.num_of_nodes + 2):
                start_time_temp.append(time)
                end_time_temp.append(time + random.randint(1, 10))
                time += random.randint(1, 100)
            start_time = start_time + start_time_temp
            end_time = end_time + end_time_temp

            # cpu内容的写入
            cpu_array = [0.5, 1.0]
            cpu_temp = []
            for i in range(self.num_of_nodes + 2):
                cpu_temp.append(random.choice(cpu_array))
            cpu = cpu + cpu_temp

            # memory内容的写入
            memory_temp = []
            for i in range(self.num_of_nodes + 2):
                memory_temp.append(random.random())
            memory = memory + memory_temp

            # duration内容的写入
            duration_temp = []
            for i in range(self.num_of_nodes + 2):
                duration_temp.append(random.uniform(1.0, 100.0))
            duration = duration + duration_temp

            # disk内容的写入
            disk_temp = []
            for i in range(self.num_of_nodes + 2):
                disk_temp.append(random.random())
            disk = disk + disk_temp

            # submit_time内容的写入
            submit_time_temp = []
            for i in range(self.num_of_nodes + 2):
                submit_time_temp.append(index_job + 1)
            submit_time = submit_time + submit_time_temp

        job_index = pd.DataFrame(data=job_index)
        job_index.to_csv('C:/Users/hasee/Desktop/job.csv', index=False, header=['index'])
        temp = pd.read_csv('C:/Users/hasee/Desktop/job.csv')
        temp['task_id'] = task_id
        temp['instances_num'] = instances_num
        temp['task_type'] = task_type
        temp['job_id'] = job_id
        temp['status'] = status
        temp['start_time'] = start_time
        temp['end_time'] = end_time
        temp['cpu'] = cpu
        temp['memory'] = memory
        temp['duration'] = duration
        temp['disk'] = disk
        temp['submit_time'] = submit_time
        temp.to_csv('C:/Users/hasee/Desktop/job.csv', index=False)

        # # edges存储DAG图中的有向边
        # # 对其中内容进行调整：‘Start’变为1，‘Exit’变为self.num_of_nodes + 2，数字加1
        # edges = self.edges_array
        # edges = np.array(edges)
        # for edge in edges:
        #     if edge[0] == 'Start':
        #         edge[0] = 1
        #     else:
        #         edge[0] = int(edge[0]) + 1
        #     if edge[1] == 'Exit':
        #         edge[1] = self.num_of_nodes + 2
        #     else:
        #         edge[1] = int(edge[1]) + 1
        # temp = []
        # for edge in edges:
        #     array = []
        #     data1 = int(edge[0])
        #     array.append(data1)
        #     data2 = int(edge[1])
        #     array.append(data2)
        #     temp.append(tuple(array))
        # edges = temp
        #
        # # job_index内容的写入
        # for i in range(self.num_of_nodes + 2):
        #     job_index.append(i)
        # job_index = np.array(job_index)
        # job_index = pd.DataFrame(data=job_index)
        # job_index.to_csv('C:/Users/hasee/Desktop/job.csv', index=False, header=['index'])
        #
        # # task_id内容的写入
        # for i in range(self.num_of_nodes + 2):
        #     task_id_str = str(i + 1)
        #     for edge in edges:
        #         if edge[1] == i + 1:
        #             task_id_str = task_id_str + '_' + str(edge[0])
        #     task_id.append(task_id_str)
        # task_id = np.array(task_id)
        # # task_id = pd.DataFrame(data=task_id)
        # # task_id.to_csv('C:/Users/hasee/Desktop/job.csv', index=False, header=['task_id'])
        # temp = pd.read_csv('C:/Users/hasee/Desktop/job.csv')
        # temp['task_id'] = task_id
        # temp.to_csv('C:/Users/hasee/Desktop/job.csv', index=False)
        #
        # # instances_num内容的写入
        # for i in range(self.num_of_nodes + 2):
        #     instances_num.append(random.randint(1, 25))
        # temp = pd.read_csv('C:/Users/hasee/Desktop/job.csv')
        # temp['instances_num'] = instances_num
        # temp.to_csv('C:/Users/hasee/Desktop/job.csv', index=False)
        #
        # # task_type内容的写入
        # for i in range(self.num_of_nodes + 2):
        #     task_type.append('A')
        # temp = pd.read_csv('C:/Users/hasee/Desktop/job.csv')
        # temp['task_type'] = task_type
        # temp.to_csv('C:/Users/hasee/Desktop/job.csv', index=False)
        #
        # # job_id内容的写入
        # for i in range(self.num_of_nodes + 2):
        #     job_id.append(1)
        # temp = pd.read_csv('C:/Users/hasee/Desktop/job.csv')
        # temp['job_id'] = job_id
        # temp.to_csv('C:/Users/hasee/Desktop/job.csv', index=False)
        #
        # # status内容的写入
        # for i in range(self.num_of_nodes + 2):
        #     status.append('Terminated')
        # temp = pd.read_csv('C:/Users/hasee/Desktop/job.csv')
        # temp['status'] = status
        # temp.to_csv('C:/Users/hasee/Desktop/job.csv', index=False)
        #
        # # start_time、end_time内容的写入
        # time = random.randint(1, 20)
        # for i in range(self.num_of_nodes + 2):
        #     start_time.append(time)
        #     end_time.append(time + random.randint(1, 10))
        #     time += random.randint(1, 100)
        # temp = pd.read_csv('C:/Users/hasee/Desktop/job.csv')
        # temp['start_time'] = start_time
        # temp['end_time'] = end_time
        # temp.to_csv('C:/Users/hasee/Desktop/job.csv', index=False)
        #
        # # cpu内容的写入
        # cpu_array = [0.5, 1.0]
        # for i in range(self.num_of_nodes + 2):
        #     cpu.append(random.choice(cpu_array))
        # temp = pd.read_csv('C:/Users/hasee/Desktop/job.csv')
        # temp['cpu'] = cpu
        # temp.to_csv('C:/Users/hasee/Desktop/job.csv', index=False)
        #
        # # memory内容的写入
        # for i in range(self.num_of_nodes + 2):
        #     memory.append(random.random())
        # temp = pd.read_csv('C:/Users/hasee/Desktop/job.csv')
        # temp['memory'] = memory
        # temp.to_csv('C:/Users/hasee/Desktop/job.csv', index=False)
        #
        # # duration内容的写入
        # for i in range(self.num_of_nodes + 2):
        #     duration.append(random.uniform(1.0, 100.0))
        # temp = pd.read_csv('C:/Users/hasee/Desktop/job.csv')
        # temp['duration'] = duration
        # temp.to_csv('C:/Users/hasee/Desktop/job.csv', index=False)
        #
        # # disk内容的写入
        # for i in range(self.num_of_nodes + 2):
        #     disk.append(random.random())
        # temp = pd.read_csv('C:/Users/hasee/Desktop/job.csv')
        # temp['disk'] = disk
        # temp.to_csv('C:/Users/hasee/Desktop/job.csv', index=False)
        #
        # # submit_time内容的写入
        # for i in range(self.num_of_nodes + 2):
        #     submit_time.append(0)
        # temp = pd.read_csv('C:/Users/hasee/Desktop/job.csv')
        # temp['submit_time'] = submit_time
        # temp.to_csv('C:/Users/hasee/Desktop/job.csv', index=False)
