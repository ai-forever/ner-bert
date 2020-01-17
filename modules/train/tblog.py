from tensorboardX import SummaryWriter
import os
import numpy as np


class TensorboardLog(object):
    """Log to tensorboard."""

    def __init__(self, tensorboard_dir):
        self.tensorboard_dir = tensorboard_dir
        self._writers = dict()

    def get_writer(self, key):
        if key not in self._writers:
            self._writers[key] = SummaryWriter(
                os.path.join(self.tensorboard_dir, key),
            )
        return self._writers[key]

    def log(self, stats, epoch, tag, step=None, pr=None):
        """Log intermediate stats to tensorboard."""
        writer = self.get_writer(tag)
        values = []
        keys = []
        for key in stats.keys():
            writer.add_scalar(key, stats[key], step)
            values.append(np.round(stats[key], 3))
            keys.append(key + " {} |")
        if pr is not None:
            pr.set_description("{} | epoch {} | {} ".format(tag, epoch, " ".join(keys).format(*values)))

    def __exit__(self, *exc):
        for writer in getattr(self, '_writers', {}).values():
            writer.close()
        return False
