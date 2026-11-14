import vllm_simulator.hook as vllm_simulator_hook


# define hook
class C_DemoHook(vllm_simulator_hook.BaseHook):
    # if running with pytest, the module name is "test_hook_demo"
    HOOK_MODULE_NAME = __name__
    HOOK_CLASS_NAME = "Demo"

    @classmethod
    def hook(cls, target_class):
        original_run = target_class.run

        def wrapped_run(self, msg: str):
            _ = original_run(self, msg)
            # ignore the original result
            return f"{msg} from Hook"

        # replace the target function
        target_class.run = wrapped_run
        return target_class


vllm_simulator_hook.install_class_hooks(C_DemoHook)


# the target class should be defined after installing hook
class Demo:
    def run(self, msg: str):
        return f"{msg} from Demo"


def test_hook_demo():
    demo = Demo()
    msg = "Message"
    res = demo.run(msg)
    # the message will be returned form hook
    assert res == f"{msg} from Hook"


if __name__ == "__main__":
    test_hook_demo()
