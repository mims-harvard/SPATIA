import torch


class SinkhornDistance(torch.nn.Module):
    def __init__(self, eps: float = 1e-2, max_iter: int = 100, reduction: str = "none"):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, c: torch.Tensor):
        C = -c
        x_points = C.shape[-2]
        batch_size = C.shape[0]

        mu = (
            torch.empty(
                batch_size,
                x_points,
                dtype=C.dtype,
                requires_grad=False,
                device=C.device,
            )
            .fill_(1.0 / x_points)
            .squeeze()
        )
        nu = (
            torch.empty(
                batch_size,
                x_points,
                dtype=C.dtype,
                requires_grad=False,
                device=C.device,
            )
            .fill_(1.0 / x_points)
            .squeeze()
        )
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        thresh = 1e-12

        for i in range(self.max_iter):
            if i % 2 == 0:
                u1 = u
                u = (
                    self.eps
                    * (torch.log(mu) - torch.logsumexp(self.M(C, u, v), dim=-1))
                    + u
                )
                err = (u - u1).abs().sum(-1).mean()
            else:
                v = (
                    self.eps
                    * (
                        torch.log(nu)
                        - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)
                    )
                    + v
                )
                v = v.detach().requires_grad_(False)
                v[v > 9 * 1e8] = 0.0
                v = v.detach().requires_grad_(True)

            if err.item() < thresh:
                break

        U, V = u, v
        pi = torch.exp(self.M(C, U, V))


        return pi, C, U, V

    def M(self, C, u, v):
        """$M_{ij} = (-c_{ij} + u_i + v_j) / epsilon$"""
        return (-C + u.unsqueeze(-1) + v.unsqueeze(1)) / self.eps

    @staticmethod
    def ave(u, u1, tau):
        return tau * u + (1 - tau) * u1
