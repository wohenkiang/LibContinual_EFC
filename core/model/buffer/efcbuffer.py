import torch
from random import sample, shuffle


class PRACEBuffer:

    def __init__(self,batch_size) -> None:
        self.seen_classes = 0
        self.current_classes = 0
        self.previous_batch_samples = None
        self.previous_batch_labels = None
        self.batch_size = batch_size
        self.total_classes = 0
        self.buffer_size = 0

    def add_data(self, current_samples, current_targets):
        self.previous_batch_labels = current_targets
        self.previous_batch_samples = current_samples

    def sample(self):
        if self.seen_classes + self.current_classes < self.batch_size:
            diff = self.batch_size - (self.seen_classes + self.current_classes)
            samples = list(range(0, self.seen_classes + self.current_classes))
            shuffle(samples)
            samples = torch.tensor(samples)
            samples = torch.cat([samples, samples[:diff]])
        else:
            samples = torch.tensor(sample(range(0, self.seen_classes + self.current_classes), self.batch_size))

        n = torch.sum(samples < self.seen_classes)
        m = torch.sum(samples >= self.seen_classes)

        return n, self.previous_batch_samples[:m.item(), :], self.previous_batch_labels[:m.item()]

    def get_seen_and_current_class_num(self,task_id,task_dict):
        self.seen_classes = sum([len(task_dict[t]) for t in range(task_id)])
        self.current_classes = len(task_dict[task_id])
