"""CPU utilities also work without the optional Prime-RL runtime installed."""

__all__ = ["FinpipeTaskset"]


def __getattr__(name):
    if name == "FinpipeTaskset":
        from finpipe_rl.taskset import FinpipeTaskset

        return FinpipeTaskset
    raise AttributeError(name)
