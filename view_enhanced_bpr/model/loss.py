from torch.nn import LogSigmoid


def bpr_loss(x):
    return -LogSigmoid()(x[0] - x[1]).mean()


def view_enhanced_bpr_loss(x, alpha=1.0):
    return (-LogSigmoid()(x[0] - x[2]) - alpha * LogSigmoid()(x[0] - x[1]) - (1 - alpha) * LogSigmoid()(x[1] - x[2])).mean()
    # return (-LogSigmoid()(x[0] - x[2])).mean()