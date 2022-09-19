import torch


class MTLLoss(torch.nn.Module):
    """Args:
            losses: a list of task specific loss terms
            num_tasks: number of tasks
    """

    def __init__(self, num_tasks=2):
        super(MTLLoss, self).__init__()
        self.num_tasks = num_tasks
        self.log_sigma = torch.nn.Parameter(torch.zeros((num_tasks)))

    def get_precisions(self):
        return 0.5 * torch.exp(- self.log_sigma) ** 2

    def forward(self, *loss_terms):
        assert len(loss_terms) == self.num_tasks

        total_loss = 0
        self.precisions = self.get_precisions()

        for task in range(self.num_tasks):
            total_loss += self.precisions[task] * loss_terms[task] + self.log_sigma[task]

        return total_loss
