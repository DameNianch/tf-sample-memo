import tensorflow as tf


class TrainExecutor(object):
    def __init__(self, model, trainer):
        self._model = model
        self._trainer = trainer

    def train_once(self, x, label_diff):
        x_real, x_target = tf.unstack(x, axis=-1)
        with tf.GradientTape() as gt:
            x_tuned = self._model(x_real, label_diff)
            loss = self._trainer.calc_loss(x_target, x_tuned)
        self._trainer.optimize(gt)
        return loss


class GPUTrainExecutor(TrainExecutor):
    def __init__(self, model, trainer):
        super(GPUTrainExecutor, self).__init__(model, trainer)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None, 3, 2], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None], dtype=tf.float32)
        ],
        experimental_follow_type_hints=True
    )
    def __call__(self, x, label_diff):
        return self.train_once(x, label_diff)


class TPUTrainExecutor(TrainExecutor):
    def __init__(self, model, trainer, strategy):
        super(TPUTrainExecutor, self).__init__(model, trainer)
        self._strategy = strategy

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None, 3, 2]),
        tf.TensorSpec(shape=[None, None])
    ])
    def __call__(self, x, label_diff):
        replica_loss = self._strategy.run(self.train_once, args=(x, label_diff))
        loss = {}
        for k, v in replica_loss.items():
            loss[k] = self._strategy.reduce(tf.distribute.ReduceOp.SUM, v, axis=None)
        return loss
