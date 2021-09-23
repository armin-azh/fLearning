from torch import nn
from torch import Tensor
import torch

import numpy as np


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
        epoch = kwargs["epoch"]
        epochs = kwargs["epochs"]

        total_acc = []
        total_loss = []

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            self._opt.zero_grad()
            y_hat, loss = self._calculate_loss(x, y)

            pred = y_hat.data.max(1, keepdim=True)[1]
            total_correct = pred.eq(y.data.view_as(pred)).sum()
            curr_acc = total_correct / float(len(x))

            total_acc.append(curr_acc)
            total_loss.append(loss)

            if batch_idx % 100 == 0:
                print(
                    f"Epoch:[{epoch + 1}/{epochs}] | Batch:[{batch_idx + 1}/{n_batches}] Acc: {curr_acc}, Loss: {loss}")

    def valid_step(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        epochs = kwargs["epochs"]

        for epoch in range(epochs):
            self.train_step(epoch=epoch, epochs=kwargs["epochs"], train_loader=kwargs["train_loader"],
                            device=kwargs["device"])

    @property
    def get_net(self):
        return self._net


def eval_global_model(net, test_loader):
    net.eval()
    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        test_correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            loss = criterion(output, labels)

            prediction = torch.max(output, 1)
            total += labels.size(0)
            test_correct += np.sum(prediction[1].cpu().numpy() == labels.cpu().numpy())

        acc = (float(test_correct) / total) * 100

        return acc, loss
