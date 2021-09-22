from torch import nn
from torch import Tensor


class AbstractTrainer:
    def __init__(self, *args, **kwargs):
        self._net = kwargs["net"]
        self._opt_config = kwargs["opt_config"]
        self._opt = kwargs["opt"](self._net.parameters(), **self._opt_config)
        self._loss = nn.CrossEntropyLoss()

    def train_step(self, *args, **kwargs):
        raise NotImplementedError

    def valid_step(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def _calculate_loss(self, data: Tensor, target: Tensor):
        """
        standard loss
        :param data: predicted
        :param target: true
        :return:
        """
        output = self._net(data)
        loss = self._loss(output, target)
        loss.backward()
        self._opt.step()
        return output, loss


class ClientTrainer(AbstractTrainer):
    def __init__(self, *args, **kwargs):
        super(ClientTrainer, self).__init__(*args, **kwargs)

    def train_step(self, *args, **kwargs):
        self._net.train()

        train_loader = kwargs["train_loader"]
        train_loader_size = len(train_loader.dataset)
        n_batches = len(train_loader)
        device = kwargs["device"]
        c_round = kwargs["c_round"]
        n_round = kwargs["n_round"]
        epoch = kwargs["epoch"]
        epochs = kwargs["epochs"]

        total_correct = 0.0
        total_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            self._opt.zero_grad()
            y_hat, loss = self._calculate_loss(x, y)

            pred = y_hat.data.max(1, keepdim=True)[1]
            total_correct += pred.eq(y.data.view_as(pred)).sum()
            total_loss += loss
            curr_acc = 100.0 * (total_correct / float(train_loader_size))
            curr_loss = (total_loss / float(batch_idx))

            print(
                f"Round:[{c_round + 1}/{n_round}] | Epoch:[{epoch + 1}/{epochs}] | Batch:[{batch_idx + 1}/{n_batches}] Acc: {curr_acc}, Loss: {curr_loss}")

    def valid_step(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        epochs = kwargs["epochs"]

        for epoch in range(epochs):
            self.train_step(epoch=epoch, **kwargs)
