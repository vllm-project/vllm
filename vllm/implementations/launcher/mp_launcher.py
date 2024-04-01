import uuid
from multiprocessing import Process

from vllm.interfaces.launcher import Launcher, SubClassOfDistributedTask
from vllm.utils import get_open_port


class MPLauncher(Launcher):
    # this is intended to work in single node
    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks

    def launch(self, *, task_type: SubClassOfDistributedTask, **kwargs):
        # be cautious that `kwargs` might well be serialized
        # and deserialized before being passed to tasks
        launch_id = str(uuid.uuid4())
        envs = [{} for _ in range(self.n_tasks)]
        port = str(get_open_port())
        for i, env in enumerate(envs):
            env['LAUNCH_ID'] = launch_id
            env['WORLD_SIZE'] = str(self.n_tasks)
            env['RANK'] = str(i)
            env['LOCAL_WORLD_SIZE'] = str(self.n_tasks)
            env['LOCAL_RANK'] = str(i)
            env['MASTER_ADDR'] = 'localhost'
            env['MASTER_PORT'] = port
        tasks = []
        for i in range(self.n_tasks):
            p = Process(target=task_type, args=(envs[i], (), kwargs))
            p.start()
            tasks.append(p)
        for task in tasks:
            task.join()
        msg = ""
        for i, task in enumerate(tasks):
            if task.exitcode != 0:
                msg += f"Task {i} exited with code {task.exitcode}"
        # if no error, `msg` should be empty
        # if there is an error, `msg` should contain the error message
        assert msg == "", msg
