from torch import einsum

from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.weights = kwargs.get('weights')
        self.gamma = kwargs.get('gamma', 0)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        p = pred_softmax[:, self.idk, ...]
        log_p = (p + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()
        if self.weights is not None:
            mask = mask * self.weights[self.idk, None, None]
        norm = mask.sum() + 1e-10
        if self.gamma:
            mask = mask * (1 - p) ** self.gamma

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= norm

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


class DiceLoss():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        p = pred_softmax[:, self.idk, ...]
        g = weak_target[:, self.idk, ...].float()

        inter = einsum("bkwh,bkwh->k", p, g)
        total = einsum("bkwh->k", p) + einsum("bkwh->k", g)
        dice = (2 * inter + 1) / (total + 1)

        return 1 - dice.mean()


class DiceCE():
    def __init__(self, **kwargs):
        self.ce = CrossEntropy(**kwargs)
        self.dice = DiceLoss(**kwargs)

    def __call__(self, pred_softmax, weak_target):
        return self.ce(pred_softmax, weak_target) + self.dice(pred_softmax, weak_target)
