import torch
import numpy as np


class DPM_Solver:
    def __init__(self, noise_schedule, algorithm_type="dpmsolver++"):
        self.noise_schedule = noise_schedule
        assert algorithm_type == "dpmsolver++"
        self.algorithm_type = algorithm_type

    def data_prediction_fn(self, x, t):
        return self.model(x, t)

    def model_fn(self, x, t):
        return self.data_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        lambda_T = self.noise_schedule.marginal_lambda(t_T)
        lambda_0 = self.noise_schedule.marginal_lambda(t_0)
        # logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
        logSNR_steps = torch.linspace(0, 1, N + 1).to(device)
        logSNR_steps = logSNR_steps * (lambda_0 - lambda_T) + lambda_T
        return self.noise_schedule.inverse_lambda(logSNR_steps)

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [
                    3,
                ] * (
                    K - 2
                ) + [2, 1]
            elif steps % 3 == 1:
                orders = [
                    3,
                ] * (
                    K - 1
                ) + [1]
            else:
                orders = [
                    3,
                ] * (
                    K - 1
                ) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [
                    2,
                ] * K
            else:
                K = steps // 2 + 1
                orders = [
                    2,
                ] * (
                    K - 1
                ) + [1]
        elif order == 1:
            K = 1
            orders = [
                1,
            ] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        return timesteps_outer, orders

    def dpm_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False):
        ns = self.noise_schedule
        dims = x.dim()
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        phi_1 = torch.expm1(-h)
        if model_s is None:
            model_s = self.model_fn(x, s)
        x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s
        if return_intermediate:
            return x_t, {"model_s": model_s}
        else:
            return x_t

    def singlestep_dpm_solver_second_update(
        self, x, s, t, r1=0.5, model_s=None, return_intermediate=False, solver_type="dpmsolver"
    ):
        if solver_type not in ["dpmsolver", "taylor"]:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = (
            ns.marginal_log_mean_coeff(s),
            ns.marginal_log_mean_coeff(s1),
            ns.marginal_log_mean_coeff(t),
        )
        sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
        alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)

        phi_11 = torch.expm1(-r1 * h)
        phi_1 = torch.expm1(-h)

        if model_s is None:
            model_s = self.model_fn(x, s)
        x_s1 = (sigma_s1 / sigma_s) * x - (alpha_s1 * phi_11) * model_s
        model_s1 = self.model_fn(x_s1, s1)
        if solver_type == "dpmsolver":
            x_t = (
                (sigma_t / sigma_s) * x
                - (alpha_t * phi_1) * model_s
                - (0.5 / r1) * (alpha_t * phi_1) * (model_s1 - model_s)
            )
        elif solver_type == "taylor":
            x_t = (
                (sigma_t / sigma_s) * x
                - (alpha_t * phi_1) * model_s
                + (1.0 / r1) * (alpha_t * (phi_1 / h + 1.0)) * (model_s1 - model_s)
            )

        if return_intermediate:
            return x_t, {"model_s": model_s, "model_s1": model_s1}
        else:
            return x_t

    def singlestep_dpm_solver_third_update(
        self,
        x,
        s,
        t,
        r1=1.0 / 3.0,
        r2=2.0 / 3.0,
        model_s=None,
        model_s1=None,
        return_intermediate=False,
        solver_type="dpmsolver",
    ):
        if solver_type not in ["dpmsolver", "taylor"]:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 1.0 / 3.0
        if r2 is None:
            r2 = 2.0 / 3.0
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = (
            ns.marginal_log_mean_coeff(s),
            ns.marginal_log_mean_coeff(s1),
            ns.marginal_log_mean_coeff(s2),
            ns.marginal_log_mean_coeff(t),
        )
        sigma_s, sigma_s1, sigma_s2, sigma_t = (
            ns.marginal_std(s),
            ns.marginal_std(s1),
            ns.marginal_std(s2),
            ns.marginal_std(t),
        )
        alpha_s1, alpha_s2, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_s2), torch.exp(log_alpha_t)

        phi_11 = torch.expm1(-r1 * h)
        phi_12 = torch.expm1(-r2 * h)
        phi_1 = torch.expm1(-h)
        phi_22 = torch.expm1(-r2 * h) / (r2 * h) + 1.0
        phi_2 = phi_1 / h + 1.0
        phi_3 = phi_2 / h - 0.5

        if model_s is None:
            model_s = self.model_fn(x, s)
        if model_s1 is None:
            x_s1 = (sigma_s1 / sigma_s) * x - (alpha_s1 * phi_11) * model_s
            model_s1 = self.model_fn(x_s1, s1)
        x_s2 = (
            (sigma_s2 / sigma_s) * x
            - (alpha_s2 * phi_12) * model_s
            + r2 / r1 * (alpha_s2 * phi_22) * (model_s1 - model_s)
        )
        model_s2 = self.model_fn(x_s2, s2)
        if solver_type == "dpmsolver":
            x_t = (
                (sigma_t / sigma_s) * x
                - (alpha_t * phi_1) * model_s
                + (1.0 / r2) * (alpha_t * phi_2) * (model_s2 - model_s)
            )
        elif solver_type == "taylor":
            D1_0 = (1.0 / r1) * (model_s1 - model_s)
            D1_1 = (1.0 / r2) * (model_s2 - model_s)
            D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
            D2 = 2.0 * (D1_1 - D1_0) / (r2 - r1)
            x_t = (
                (sigma_t / sigma_s) * x
                - (alpha_t * phi_1) * model_s
                + (alpha_t * phi_2) * D1
                - (alpha_t * phi_3) * D2
            )

        if return_intermediate:
            return x_t, {"model_s": model_s, "model_s1": model_s1, "model_s2": model_s2}
        else:
            return x_t

    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpmsolver"):
        if solver_type not in ["dpmsolver", "taylor"]:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = (
            ns.marginal_lambda(t_prev_1),
            ns.marginal_lambda(t_prev_0),
            ns.marginal_lambda(t),
        )
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)

        phi_1 = torch.expm1(-h)
        if solver_type == "dpmsolver":
            x_t = (sigma_t / sigma_prev_0) * x - (alpha_t * phi_1) * model_prev_0 - 0.5 * (alpha_t * phi_1) * D1_0
        elif solver_type == "taylor":
            x_t = (
                (sigma_t / sigma_prev_0) * x
                - (alpha_t * phi_1) * model_prev_0
                + (alpha_t * (phi_1 / h + 1.0)) * D1_0
            )
        
        return x_t

    def multistep_dpm_solver_third_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpmsolver"):
        ns = self.noise_schedule
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = (
            ns.marginal_lambda(t_prev_2),
            ns.marginal_lambda(t_prev_1),
            ns.marginal_lambda(t_prev_0),
            ns.marginal_lambda(t),
        )
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        D1_1 = (1.0 / r1) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)

        phi_1 = torch.expm1(-h)
        phi_2 = phi_1 / h + 1.0
        phi_3 = phi_2 / h - 0.5
        x_t = (
            (sigma_t / sigma_prev_0) * x
            - (alpha_t * phi_1) * model_prev_0
            + (alpha_t * phi_2) * D1
            - (alpha_t * phi_3) * D2
        )
        return x_t

    def singlestep_dpm_solver_update(
        self, x, s, t, order, return_intermediate=False, solver_type="dpmsolver", r1=None, r2=None
    ):
        if order == 1:
            return self.dpm_solver_first_update(x, s, t, return_intermediate=return_intermediate)
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(
                x, s, t, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1
            )
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(
                x, s, t, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1, r2=r2
            )
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order, solver_type="dpmsolver"):
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        elif order == 3:
            return self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def dpm_solver_adaptive(
        self, x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-5, solver_type="dpmsolver"
    ):
        ns = self.noise_schedule
        s = t_T * torch.ones((1,)).to(x)
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.dpm_solver_first_update(x, s, t, return_intermediate=True)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_second_update(
                x, s, t, r1=r1, solver_type=solver_type, **kwargs
            )
        elif order == 3:
            r1, r2 = 1.0 / 3.0, 2.0 / 3.0
            lower_update = lambda x, s, t: self.singlestep_dpm_solver_second_update(
                x, s, t, r1=r1, return_intermediate=True, solver_type=solver_type
            )
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_third_update(
                x, s, t, r1=r1, r2=r2, solver_type=solver_type, **kwargs
            )
        else:
            raise ValueError("For adaptive step size solver, order must be 2 or 3, got {}".format(order))
        while torch.abs((s - t_0)).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(torch.ones_like(x).to(x) * atol, rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(E <= 1.0):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            h = torch.min(theta * h * torch.float_power(E, -1.0 / order).float(), lambda_0 - lambda_s)
            nfe += order
        print("adaptive solver nfe", nfe)
        return x

    def sample(
        self,
        model_fn,
        x,
        steps=3,
        t_start=None,
        t_end=None,
        order=3,
        skip_type="logSNR",
        method="singlestep",
        lower_order_final=True,
        solver_type="dpmsolver",
        atol=0.0078,
        rtol=0.05,
        return_intermediate=False,
    ):
        self.model = lambda x, t: model_fn(x, t)
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        if return_intermediate:
            assert method in [
                "multistep",
                "singlestep",
                "singlestep_fixed",
            ], "Cannot use adaptive solver when saving intermediate values"
        device = x.device
        intermediates = []
        # print(t_0.shape, t_T.shape) # torch.Size([128, 1, 1, 1]) torch.Size([128, 1, 1, 1])
        with torch.no_grad():
            if method == "adaptive":
                x = self.dpm_solver_adaptive(
                    x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol, solver_type=solver_type
                )
            elif method == "multistep":
                assert steps >= order
                timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
                assert timesteps.shape[-1] - 1 == steps
                # Init the initial values.
                step = 0
                t = timesteps[:, :, :, step : step + 1]
                t_prev_list = [t]
                model_prev_list = [self.model_fn(x, t)]
                if return_intermediate:
                    intermediates.append(x)
                # Init the first `order` values by lower order multistep DPM-Solver.
                for step in range(1, order):
                    t = timesteps[:, :, :, step : step + 1]
                    x = self.multistep_dpm_solver_update(
                        x, model_prev_list, t_prev_list, t, step, solver_type=solver_type
                    )
                    if return_intermediate:
                        intermediates.append(x)
                    t_prev_list.append(t)
                    model_prev_list.append(self.model_fn(x, t))
                # Compute the remaining values by `order`-th order multistep DPM-Solver.
                for step in range(order, steps + 1):
                    t = timesteps[:, :, :, step : step + 1]
                    # We only use lower order for steps < 10
                    # [CHANGE] remove the above restriction
                    if lower_order_final:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    x = self.multistep_dpm_solver_update(
                        x, model_prev_list, t_prev_list, t, step_order, solver_type=solver_type
                    )
                    if return_intermediate:
                        intermediates.append(x)
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = t
                    # We do not need to evaluate the final model value.
                    if step < steps:
                        model_prev_list[-1] = self.model_fn(x, t)
            elif method in ["singlestep", "singlestep_fixed"]:
                if method == "singlestep":
                    timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(
                        steps=steps, order=order, skip_type=skip_type, t_T=t_T, t_0=t_0, device=device
                    )
                elif method == "singlestep_fixed":
                    K = steps // order
                    orders = [
                        order,
                    ] * K
                    timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
                for step, order in enumerate(orders):
                    # print(timesteps_outer.shape) # torch.Size([128, 1, 1, steps])
                    s, t = timesteps_outer[:, :, :, step : step + 1], timesteps_outer[:, :, :, step + 1 : step + 2]
                    # print("shape check 0", s.shape, t.shape, timesteps_outer.shape)
                    # timesteps_inner = self.get_time_steps(
                    #     skip_type=skip_type, t_T=s, t_0=t, N=order, device=device
                    # )
                    # lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
                    # h = lambda_inner[-1] - lambda_inner[0]
                    # r1 = None if order <= 1 else (lambda_inner[0][1] - lambda_inner[0][0]) / h
                    # r2 = None if order <= 2 else (lambda_inner[0][2] - lambda_inner[1][0]) / h
                    r1 = None
                    r2 = None
                    # print(timesteps_outer.shape, s.shape, t.shape)
                    x = self.singlestep_dpm_solver_update(x, s, t, order, solver_type=solver_type, r1=r1, r2=r2)
                    if return_intermediate:
                        intermediates.append(x)
            else:
                raise ValueError("Got wrong method {}".format(method))
        if return_intermediate:
            return x, intermediates
        else:
            return x


class NoiseScheduleEDM:
    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        return torch.zeros_like(t)#.to(torch.float64)

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.ones_like(t)#.to(torch.float64)

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return t#.to(torch.float64)

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """

        return -torch.log(t)#.to(torch.float64)

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        return torch.exp(-lamb)#.to(torch.float64)
