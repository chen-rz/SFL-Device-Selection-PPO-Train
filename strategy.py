import math
import random
from logging import DEBUG, WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr.common
import pandas
import torchvision
from flwr.common import (
    EvaluateIns, EvaluateRes, FitIns, FitRes, GetParametersIns,
    ndarrays_to_parameters, parameters_to_ndarrays,
    MetricsAggregationFn, NDArrays, Parameters, Scalar,
)
from flwr.common.logger import log
from flwr.server import SimpleClientManager
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

import client as clt
from constants import *
from dataset_utils import cifar10Transformation


class PPO_ClientManager(SimpleClientManager): # TODO
    def sample(self, num_clients: int, server_round=0, time_constr=0):
        # For model initialization
        if num_clients == 1:
            # return [self.clients[str(random.randint(0, pool_size - 1))]]
            return [self.clients[str(0)]]

        # For evaluation, use the same devices as in the fit round
        elif num_clients == -1:
            with open(
                    "./output/fit_server/round_{}.txt".format(server_round),
                    mode='r'
            ) as inputFile:
                cids_in_fit = eval(inputFile.readline())["clients_selected"]
            return [self.clients[str(cid)] for cid in cids_in_fit]

        ##########################################################################################
        # Get info of previous round
        if server_round == 1:
            init_parameters = self.clients["0"].get_parameters(
                ins=GetParametersIns(config={}),
                timeout=None
            ).parameters
            init_param_ndarrays = parameters_to_ndarrays(init_parameters)
            init_eval_func = clt.get_evaluate_fn(
                torchvision.datasets.CIFAR10(
                    root="./data", train=False, transform=cifar10Transformation()
                )
            )
            eval_res = init_eval_func(0, init_param_ndarrays, {})
            avg_loss_of_prev_round = eval_res[0]

        else:
            with open("./output/fit_server/round_{}.txt".format(server_round - 1)) as inputFile:
                cids_in_prev_round = eval(inputFile.readline())["clients_selected"]

            valid_loss_sum, valid_n_num = 0.0, 0
            losses_of_prev_round = []

            for n in range(pool_size):
                with open("./output/val_loss_per_cli/client_{}.txt".format(n)) as inputFile:
                    losses_of_prev_round.append(eval(inputFile.readlines()[-1]))
                if n in cids_in_prev_round:
                    assert losses_of_prev_round[-1] > 0
                    valid_loss_sum += losses_of_prev_round[-1]
                    valid_n_num += 1
                else:
                    assert losses_of_prev_round[-1] == -1

            avg_loss_of_prev_round = valid_loss_sum / valid_n_num

        with open(
                "./output/avg_loss_per_round/avg_loss_round_{}.txt".format(server_round - 1), mode='w'
        ) as outputFile:
            outputFile.write(str(avg_loss_of_prev_round))
        ##########################################################################################

        param_dicts = []
        available_cids = []

        C_record = []
        updateTimeList = []

        for n in range(pool_size):
            # Get each client's parameters
            param_dicts.append(
                self.clients[str(n)].get_properties(
                    flwr.common.GetPropertiesIns(config={}), 68400
                ).properties.copy()
            )

            param_dicts[n]["isSelected"] = False
            updateTimeList.append(param_dicts[n]["updateTime"])

        C_min = min(updateTimeList)
        C_max = max(updateTimeList)
        for n in range(pool_size):
            param_dicts[n]["C"] = (param_dicts[n]["updateTime"] - C_min) / (C_max - C_min)
            C_record.append(param_dicts[n]["C"])
        
        with open("./output/C_records/round_{}.txt".format(server_round), mode='w') as outputFile:
            outputFile.write(str(C_record))

        # 1st iteration: random 1st step
        if server_round == 1:
            for i in range(pool_size):
                param_dicts[i]["sigma"] = 1
                param_dicts[i]["g"] = 0

            # Volatility
            active_cids = list(range(pool_size))
            for _ in range(num_on_strike):
                pop_idx = random.randint(0, len(active_cids) - 1)
                active_cids.pop(pop_idx)

            # TODO !!!
            # available_cids = sorted(
            #     active_cids, key=lambda i: param_dicts[i]["dataSize"], reverse=True # TODO C?
            # )[:num_to_choose]

        # Common cases
        else:
            with open("./output/involvement_history.txt", mode='r') as inputFile:
                involvement_history = eval(inputFile.readline())
            for i in range(pool_size):
                param_dicts[i]["involvement_history"] = involvement_history[i]

            log(DEBUG, "Involvement history: " + str(involvement_history))

            with open("./output/fit_server/round_{}.txt".format(server_round - 1)) as inputFile:
                cids_in_prev_round = eval(inputFile.readline())["clients_selected"]

            # log(DEBUG, "Cids in previous round: " + str(cids_in_prev_round))

            with open("./output/avg_loss_per_round/avg_loss_round_{}.txt".format(server_round - 2)) as inputFile:
                L_2 = eval(inputFile.readline())

            losses_of_prev_round = []
            sum_of_loss_of_prev = 0.0
            for n in range(pool_size):
                with open("./output/val_loss_per_cli/client_{}.txt".format(n)) as inputFile:
                    losses_of_prev_round.append(eval(inputFile.readlines()[-1]))
                if n in cids_in_prev_round:
                    assert losses_of_prev_round[-1] > 0
                    sum_of_loss_of_prev += losses_of_prev_round[-1]
                else:
                    assert losses_of_prev_round[-1] == -1
            for i in range(pool_size):
                if i not in cids_in_prev_round:
                    assert losses_of_prev_round[i] == -1
                    # For those did not involve in previous rounds, loss should be the average
                    losses_of_prev_round[i] = sum_of_loss_of_prev / len(cids_in_prev_round)
                    # losses_of_prev_round[i] = L_2

            # log(DEBUG, "Loss in previous round: " + str(losses_of_prev_round))

            delta = [0 for _ in range(pool_size)]
            for i in range(pool_size):
                delta[i] = L_2 - losses_of_prev_round[i]

            delta_min = min(delta)
            delta_max = max(delta)
            for i in range(pool_size):
                # param_dicts[i]["sigma"] = 1 / (1 + math.pow(math.e, - delta))
                param_dicts[i]["sigma"] = (delta[i] - delta_min) / (delta_max - delta_min)
                
            sum_of_prev_C = [0 for _ in range(pool_size)]
            for t in range(1, server_round):
                with open("./output/fit_server/round_{}.txt".format(t)) as inputFile:
                    cids_in_t_round = eval(inputFile.readline())["clients_selected"]
                with open("./output/C_records/round_{}.txt".format(t)) as inputFile:
                    C_in_t_round = eval(inputFile.readline())
                for _ in range(pool_size):
                    if _ in cids_in_t_round:
                        sum_of_prev_C[_] += C_in_t_round[_]

            # log(DEBUG, "Sum of C in previous round: " + str(sum_of_prev_C))
            
            UCB_mu = []
            for i in range(pool_size):
                if involvement_history[i] == 0:
                    UCB_mu.append(0)
                else:
                    UCB_mu.append(sum_of_prev_C[i] / involvement_history[i])
                param_dicts[i]["UCB_mu"] = UCB_mu[i]

            # log(DEBUG, "UCB_mu: " + str(UCB_mu))

            UCB_u = []
            for i in range(pool_size):
                if involvement_history[i] == 0:
                    UCB_u.append(0)
                else:
                    UCB_u.append(
                        UCB_mu[i] + math.sqrt(
                            (num_to_choose + 1) * math.log(server_round) / involvement_history[i]
                        )
                    )
                param_dicts[i]["UCB_U"] = UCB_u[i]
                
            # log(DEBUG, "UCB_U: " + str(UCB_u))

            denominator = 0.0
            for k in cids_in_prev_round:
                denominator += param_dicts[k]["dataSize"] * param_dicts[k]["sigma"]

            for i in range(pool_size):
                sgn = (num_to_choose * param_dicts[i]["dataSize"] * param_dicts[i]["sigma"] / denominator) \
                    - (involvement_history[i] / server_round)
                if sgn > 0:
                    sgn = 1
                elif sgn < 0:
                    sgn = -1
                
                param_dicts[i]["g"] = sgn * math.pow(
                    abs(
                        (num_to_choose * param_dicts[i]["dataSize"] * param_dicts[i]["sigma"] / denominator) \
                        - (involvement_history[i] / server_round)
                    ),
                    alpha
                )

            # TODO UCB_omega = [(- UCB_u[i] + beta * param_dicts[i]["g"]) for i in range(pool_size)]
            UCB_omega = [(- UCB_u[i]) for i in range(pool_size)]
            for i in range(pool_size):
                param_dicts[i]["UCB_omega"] = UCB_omega[i]

            # log(DEBUG, "UCB_omega: " + str(UCB_omega))

            # Volatility
            active_cids = list(range(pool_size))
            for _ in range(num_on_strike):
                pop_idx = random.randint(0, len(active_cids) - 1)
                active_cids.pop(pop_idx)

            # TODO !!!
            # available_cids = sorted(
            #     active_cids, key=lambda i: UCB_omega[i], reverse=True
            # )[:num_to_choose]

        # TODO RANDOM!!!
        # --------------------------------------------------------------------------------
        cids_tbd = active_cids.copy()
        for _ in range(pool_size - num_on_strike - num_to_choose):
            pop_idx = random.randint(0, len(cids_tbd) - 1)
            cids_tbd.pop(pop_idx)

        available_cids = cids_tbd.copy()
        assert len(available_cids) == num_to_choose
        # --------------------------------------------------------------------------------

        # Record client parameters
        fit_round_time = 0
        for _ in available_cids:
            param_dicts[_]["isSelected"] = True
            if param_dicts[_]["updateTime"] > fit_round_time:
                fit_round_time = param_dicts[_]["updateTime"]

        # Record reward
        reward = 0
        for k in available_cids:
            reward += (- param_dicts[k]["C"] + beta * param_dicts[k]["g"])
            # TODO reward += (- param_dicts[k]["C"])
        reward *= (1 / num_to_choose)
        with open("./output/reward.txt", mode='a') as outputFile:
            outputFile.write(str(reward) + "\n")

        # Calculate regret
        best_cids = sorted(
            active_cids, key=lambda i: (- param_dicts[i]["C"] + beta * param_dicts[i]["g"]),
            # TODO active_cids, key=lambda i: (- param_dicts[i]["C"]),
            reverse=True
        )[:num_to_choose]
        best_reward = 0
        for k in best_cids:
            best_reward += (- param_dicts[k]["C"] + beta * param_dicts[k]["g"])
            # TODO best_reward += (- param_dicts[k]["C"])
        best_reward *= (1 / num_to_choose)
        regret_of_round = best_reward - reward
        with open("./output/regret.txt", mode='r') as inputFile:
            lines = inputFile.readlines()
            if lines:
                last_regret = eval(lines[-1])
            else:
                last_regret = 0.0
        with open("./output/regret.txt", mode='a') as outputFile:
            outputFile.write(str(last_regret + regret_of_round) + "\n")

        log(DEBUG, "Round " + str(server_round) + " selected cids " + str(available_cids))
        log(DEBUG, "Round " + str(server_round) + " best cids: " + str(best_cids))
        log(DEBUG, "Round " + str(server_round) + " reward: " + str(reward))
        log(DEBUG, "Round " + str(server_round) + " best reward: " + str(best_reward))

        return [self.clients[str(cid)] for cid in available_cids], \
            {
                "clients_selected": available_cids,
                "time_elapsed": fit_round_time,
                "time_constraint": time_constr
            }, \
            param_dicts


class Random_ClientManager(SimpleClientManager):
    def sample(self, num_clients: int, server_round=0, time_constr=0):
        # For model initialization
        if num_clients == 1:
            return [self.clients[str(random.randint(0, pool_size - 1))]]

        # For evaluation
        elif num_clients == -1:
            with open(
                    "./output/fit_server/round_{}.txt".format(server_round),
                    mode='r'
            ) as inputFile:
                cids_in_fit = eval(inputFile.readline())["clients_selected"]
            return [self.clients[str(cid)] for cid in cids_in_fit]

        # Sample clients in a random way
        param_dicts = []
        updateTimeList = []

        # ------------------------------------------------------
        # Volatility
        active_cids = list(range(pool_size))
        for _ in range(num_on_strike):
            pop_idx = random.randint(0, len(active_cids) - 1)
            active_cids.pop(pop_idx)
        # ------------------------------------------------------

        cids_tbd = active_cids.copy()
        for _ in range(pool_size - num_on_strike - num_to_choose):
            pop_idx = random.randint(0, len(cids_tbd) - 1)
            cids_tbd.pop(pop_idx)

        available_cids = cids_tbd.copy()
        assert len(available_cids) == num_to_choose

        for n in range(pool_size):
            # Get each client's parameters
            param_dicts.append(
                self.clients[str(n)].get_properties(
                    flwr.common.GetPropertiesIns(config={}), 68400
                ).properties.copy()
            )

            param_dicts[n]["isSelected"] = False
            updateTimeList.append(param_dicts[n]["updateTime"])

        C_min = min(updateTimeList)
        C_max = max(updateTimeList)
        for n in range(pool_size):
            param_dicts[n]["C"] = (param_dicts[n]["updateTime"] - C_min) / (C_max - C_min)

        fit_round_time = 0
        for _ in available_cids:
            param_dicts[_]["isSelected"] = True
            if param_dicts[_]["updateTime"] > fit_round_time:
                fit_round_time = param_dicts[_]["updateTime"]

        # -----------------------------------------------------------------------------------------------------
        # Record reward
        reward = 0
        for k in available_cids:
            # TODO reward += (- param_dicts[k]["C"] + beta * param_dicts[k]["g"])
            reward += (- param_dicts[k]["C"])
        reward *= (1 / num_to_choose)
        with open("./output/reward.txt", mode='a') as outputFile:
            outputFile.write(str(reward) + "\n")

        # Calculate regret
        best_cids = sorted(
            # TODO active_cids, key=lambda i: (- param_dicts[i]["C"] + beta * param_dicts[i]["g"]),
            active_cids, key=lambda i: (- param_dicts[i]["C"]),
            reverse=True
        )[:num_to_choose]
        best_reward = 0
        for k in best_cids:
            # TODO best_reward += (- param_dicts[k]["C"] + beta * param_dicts[k]["g"])
            best_reward += (- param_dicts[k]["C"])
        best_reward *= (1 / num_to_choose)
        regret_of_round = best_reward - reward
        with open("./output/regret.txt", mode='r') as inputFile:
            lines = inputFile.readlines()
            if lines:
                last_regret = eval(lines[-1])
            else:
                last_regret = 0.0
        with open("./output/regret.txt", mode='a') as outputFile:
            outputFile.write(str(last_regret + regret_of_round) + "\n")

        log(DEBUG, "Round " + str(server_round) + " selected cids " + str(available_cids))
        log(DEBUG, "Round " + str(server_round) + " best cids: " + str(best_cids))
        log(DEBUG, "Round " + str(server_round) + " reward: " + str(reward))
        log(DEBUG, "Round " + str(server_round) + " best reward: " + str(best_reward))
        # -----------------------------------------------------------------------------------------------------

        return [self.clients[str(cid)] for cid in available_cids], \
            {
                "clients_selected": available_cids,
                "time_elapsed": fit_round_time,
                "time_constraint": time_constr
            }, \
            param_dicts


class SFL(Strategy):
    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
            self,
            *,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:

        super().__init__()

        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def __repr__(self) -> str:
        rep = f"TCS (accept_failures={self.accept_failures})"
        return rep

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
            self, server_round: int, parameters: Parameters,
            client_manager: Union[PPO_ClientManager, Random_ClientManager]
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit_clients config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Check records of previous round
        if server_round > 1:
            with open(
                    "./output/fit_server/round_{}.txt".format(server_round - 1),
                    mode='r'
            ) as inputFile:
                clients_of_prev_round = eval(inputFile.readline())["clients_selected"]

            for _ in range(pool_size):
                # If the client was not selected in the previous round,
                # help it complete the records
                if _ not in clients_of_prev_round:
                    with open(
                            "./output/train_loss_per_cli/client_{}.txt".format(_),
                            mode='a'
                    ) as outputFile:
                        outputFile.write("-1" + "\n")
                    with open(
                            "./output/val_accu_per_cli/client_{}.txt".format(_),
                            mode='a'
                    ) as outputFile:
                        outputFile.write("-1" + "\n")
                    with open(
                            "./output/val_loss_per_cli/client_{}.txt".format(_),
                            mode='a'
                    ) as outputFile:
                        outputFile.write("-1" + "\n")

            # Record historic involvements
            with open("./output/involvement_history.txt", mode='r') as inputFile:
                fileLine = inputFile.readline()
                if not fileLine:
                    involvement_history = [0 for _ in range(pool_size)]
                else:
                    involvement_history = eval(fileLine)
            for _ in range(pool_size):
                if _ in clients_of_prev_round:
                    involvement_history[_] += 1
            with open("./output/involvement_history.txt", mode='w') as outputFile:
                outputFile.write(str(involvement_history))

        # Sample clients
        clients, fit_round_dict, param_dicts = client_manager.sample(
            num_clients=0, server_round=server_round, time_constr=None
        )

        # Record information of clients
        pandas.DataFrame.from_records(param_dicts).to_excel(
            "./output/fit_clients/fit_round_{}.xlsx".format(server_round)
        )

        # Record information of server
        with open(
                "./output/fit_server/round_{}.txt".format(server_round),
                mode='w'
        ) as outputFile:
            outputFile.write(str(fit_round_dict))

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
            self, server_round: int, parameters: Parameters,
            client_manager: Union[PPO_ClientManager, Random_ClientManager]
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients: use same clients as in fit
        clients = client_manager.sample(
            num_clients=-1, server_round=server_round, time_constr=None
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit_clients results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
