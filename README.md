该项目为元强化学习算法TayMAML的代码实现，使用语言为python，开发框架为tensorflow v1.0，代码编辑器为pycharm。其中包括一个深度强化学习算法代码实现，一个元学习算法代码实现，两部分代码的用途和组成如下。
主要代码组成：
1.基于D3DQN算法改进的边缘计算任务调度算法D3DQN-CAA。算法为深度强化学习算法，主要用于解决任务类型单一的边缘计算任务调度问题，代码组成包括：playgroud/DAG/algorithm/D3DQN/(agent.py, brain.py, D3DQN.py), playground/DAG/launch_scripts/D3DQN_Run.py。
2.基于MAML算法改进的边缘计算任务调度算法TayMAML。算法为元强化学习，由两部分组成，元学习算法和强化学习算法，其中强化学习算法为D3DQN-CAA，元学习算法为TayMAML，主要用于解决任务异构的边缘计算任务调度问题。代码组成包括：playgroud/DAG/algorithm/D3DQN/(agent.py, brain.py, D3DQN.py), playground/DAG/launch_scripts/TayMAML_Run.py。
第2部分算法中，由D3DQN-CAA作为元强化学习算法内循环的学习算法，TayMAML作为外循环的学习算法。
