import uuid
from multiprocessing import Process

from vllm.interfaces.launcher import Launcher, SubClassOfDistributedTask


class MPLauncher(Launcher):
    # this is intended to work in single node
    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks

    def launch(self, task_type: SubClassOfDistributedTask):
        launch_id = str(uuid.uuid4())
        envs = [{} for _ in range(self.n_tasks)]
        for i, env in enumerate(envs):
            env['LAUNCH_ID'] = launch_id
            env['WORLD_SIZE'] = str(self.n_tasks)
            env['RANK'] = str(i)
            env['LOCAL_WORLD_SIZE'] = str(self.n_tasks)
            env['LOCAL_RANK'] = str(i)
            env['MASTER_ADDR'] = 'localhost'
            env['MASTER_PORT'] = '29500'
        tasks = []
        for i in range(self.n_tasks):
            p = Process(target=task_type, args=(envs[i], (), {}))
            p.start()
            tasks.append(p)
        for task in tasks:
            task.join()
