import torch

class GradPruneDefense:
    def __init__(self, prune_ratio):
        """
        Args:
            prune_ratio (float): the ratio of gradients to be pruned
        """
        super().__init__()
        self.prune_ratio = prune_ratio

    def apply(self, input_grads):
        """
        Args:
            model (nn.Module): the original model
        Returns:
            nn.Module: the model with pruned gradients
        """
        # parameters = model.parameters()

        # if isinstance(parameters, torch.Tensor):
        #     parameters = [parameters]
        #     parameters = list(
        #         filter(lambda p: p.grad is not None, parameters))

        # input_grads = [p.grad.data for p in parameters]

        threshold = [
            torch.quantile(torch.abs(input_grads[i]), self.prune_ratio)
            for i in range(len(input_grads))
        ]
        import ipdb; ipdb.set_trace()
        for i, p in enumerate(input_grads):
            p[torch.abs(p) < threshold[i]] = 0
        return input_grads
    
    def model_apply(self, model):
        """
        Args:
            model (nn.Module): the original model
        Returns:
            nn.Module: the model with pruned gradients
        """
        parameters = model.parameters()

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
            parameters = list(
                filter(lambda p: p.grad is not None, parameters))

        input_grads = [p.grad.data for p in parameters]

        threshold = [
            torch.quantile(torch.abs(input_grads[i]), self.prune_ratio)
            for i in range(len(input_grads))
        ]

        for i, p in enumerate(model.parameters()):
            p.grad[torch.abs(p.grad) < threshold[i]] = 0
        return model
