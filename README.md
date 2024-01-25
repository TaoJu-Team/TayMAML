该项目为元强化学习算法TayMAML的代码实现，使用语言为python，开发框架为tensorflow v1.0，代码编辑器为pycharm。其中包括一个深度强化学习算法代码实现，一个元学习算法代码实现，两部分代码的用途和组成如下。
1.深度强化学习算法D3DQN-CAA：主要用于解决任务类型单一的边缘计算任务调度问题，主要代码组成包括：agent.py, brain.py, D3DQN.py, D3DQN_Run.py。
2.元强化学习算法TayMAML：由两部分组成，元学习算法和强化学习算法，其中强化学习算法为D3DQN-CAA，元学习算法为TayMAML，主要用于解决任务异构的边缘计算任务调度问题。主要代码组成包括：agent.py, brain.py, D3DQN.py, TayMAML_Run.py。
