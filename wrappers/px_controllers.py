from typing import Dict, Any

from gym import spaces
import numpy as np
from numpy import typing as npt

from wrappers.controllers import Controller


class PixelController(Controller):
    def __init__(self, env_cfg, version=0) -> None:
        """
        For the robot unit
        - 4 cardinal direction movement (4 dims)
        - a move center no-op action (1 dim)
        - transfer action just for transferring ice in 4 cardinal directions or center (5)
        - pickup action for power (1 dims)
        - dig action (1 dim)
        - no op action (1 dim) - equivalent to not submitting an action queue which costs power

        It does not include
        - self destruct action
        - recharge action
        - planning (via actions executing multiple times or repeating actions)
        - transferring power or resources other than ice

        For a factory:
        - Build a heavy robot
        - Build a light robot
        - Grow lichen
        - No-op
        """
        self.env_cfg = env_cfg

        self.move_act_dims = 4
        self.transfer_act_dims = 5
        self.pickup_act_dims = 1
        self.dig_act_dims = 1
        self.no_op_dims = 1

        self.move_dim_high = self.move_act_dims
        self.transfer_dim_high = self.move_dim_high + self.transfer_act_dims
        self.pickup_dim_high = self.transfer_dim_high + self.pickup_act_dims
        self.dig_dim_high = self.pickup_dim_high + self.dig_act_dims
        self.no_op_dim_high = self.dig_dim_high + self.no_op_dims

        self.total_robot_actions = self.no_op_dim_high
        self.total_factory_actions = 4

        map_size = env_cfg.map_size
        action_space = spaces.Dict({
            'factory': spaces.MultiDiscrete(np.zeros((map_size, map_size), dtype=int) + self.total_factory_actions),
            'heavy': spaces.MultiDiscrete(np.zeros((map_size, map_size), dtype=int) + self.total_robot_actions),
            'light': spaces.MultiDiscrete(np.zeros((map_size, map_size), dtype=int) + self.total_robot_actions),
        })

        self.version = version
        super().__init__(action_space)

    def _is_move_action(self, action_id):
        return action_id < self.move_dim_high

    def _get_move_action(self, action_id):
        # Move direction is id + 1 since we don't allow move center here
        return np.array([0, action_id + 1, 0, 0, 0, 1])

    def _is_transfer_action(self, action_id):
        return action_id < self.transfer_dim_high

    def _get_transfer_action(self, action_id):
        action_id = action_id - self.move_dim_high
        transfer_dir = action_id % 5
        return np.array([1, transfer_dir, 0, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_pickup_action(self, action_id):
        return action_id < self.pickup_dim_high

    def _get_pickup_action(self, action_id):
        return np.array([2, 0, 4, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_dig_action(self, action_id):
        return action_id < self.dig_dim_high

    def _get_dig_action(self, action_id):
        return np.array([3, 0, 0, 0, 0, 1])

    def action_to_lux_action(self, agent: str, obs: Dict[str, Any], action: npt.NDArray):
        shared_obs = obs["player_0"]
        lux_action = dict()

        units = shared_obs["units"][agent]
        for unit_id, unit in units.items():
            action_id = action[0][tuple(unit['pos'])]
            action_queue = []
            no_op = False
            if self._is_move_action(action_id):
                action_queue = [self._get_move_action(action_id)]
            elif self._is_transfer_action(action_id):
                action_queue = [self._get_transfer_action(action_id)]
            elif self._is_pickup_action(action_id):
                action_queue = [self._get_pickup_action(action_id)]
            elif self._is_dig_action(action_id):
                action_queue = [self._get_dig_action(action_id)]
            else:
                no_op = True

            if len(unit["action_queue"]) > 0 and len(action_queue) > 0:
                same_actions = (unit["action_queue"][0] == action_queue[0]).all()
                if same_actions:
                    no_op = True
            if not no_op:
                lux_action[unit_id] = action_queue

        factories = shared_obs["factories"][agent]
        for factory_id, factory in factories.items():
            action_id = action[1][tuple(factory['pos'])]
            if action_id < 3:
                lux_action[factory_id] = action_id

        return lux_action

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        shared_obs = obs[agent]
        units_action_mask = self._units_action_masks(agent, shared_obs)
        factories_action_mask = self._factory_action_masks(agent, shared_obs)
        return {
            'units': units_action_mask,
            'factories': factories_action_mask
        }

    def _factory_action_masks(self, agent, shared_obs):
        map_size = self.env_cfg.map_size
        action_mask = np.zeros((self.total_factory_actions, map_size, map_size), dtype=bool)

        for factory_id,  factory in shared_obs['factories'][agent]:
            indexer = tuple(factory['pos'])
            if factory['power'] >= self.env_cfg['ROBOTS']['HEAVY'].POWER_COST and \
                    factory['cargo']['metal'] >= self.env_cfg['ROBOTS']['HEAVY'].METAL_COST:
                action_mask[indexer][0] = True
            if factory['power'] >= self.env_cfg['ROBOTS']['LIGHT'].POWER_COST and \
                    factory['cargo']['metal'] >= self.env_cfg['ROBOTS']['LIGHT'].METAL_COST:
                action_mask[indexer][1] = True
            # TODO: check water cost to grow lichen
            action_mask[indexer][2:3] = True

        return action_mask

    def _units_action_masks(self, agent, shared_obs):
        # TODO: check if has enough power
        map_size = self.env_cfg.map_size
        rubble_map = shared_obs["board"]["rubble"]
        factory_occupancy_map = (
                np.ones_like(rubble_map, dtype=int) * -1
        )
        factories = dict()
        for player in shared_obs["factories"]:
            factories[player] = dict()
            for unit_id in shared_obs["factories"][player]:
                f_data = shared_obs["factories"][player][unit_id]
                f_pos = f_data["pos"]
                factory_occupancy_map[(f_pos[0] - 1):(f_pos[0] + 2), (f_pos[1] - 1):(f_pos[1] + 2)] = f_data[
                    "strain_id"]
        units = shared_obs["units"][agent]
        action_mask = np.zeros((self.total_robot_actions, map_size, map_size), dtype=bool)
        for unit_id, unit in units.items():
            action_mask = np.zeros(self.total_robot_actions)
            pos = np.array(unit["pos"])
            indexer = tuple(pos)
            # Movement is always valid
            action_mask[indexer][:4] = True

            # Transferring is valid only if the target exists
            # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
            move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
            for i, move_delta in enumerate(move_deltas):
                transfer_pos = np.array(
                    [pos[0] + move_delta[0], pos[1] + move_delta[1]]
                )
                # check if theres a factory tile there
                if (
                        transfer_pos[0] < 0
                        or transfer_pos[1] < 0
                        or transfer_pos[0] >= len(factory_occupancy_map)
                        or transfer_pos[1] >= len(factory_occupancy_map[0])
                ):
                    continue
                factory_there = factory_occupancy_map[transfer_pos[0], transfer_pos[1]]
                if factory_there in shared_obs["teams"][agent]["factory_strains"]:
                    action_mask[indexer][self.transfer_dim_high - self.transfer_act_dims + i] = True

            factory_there = factory_occupancy_map[pos[0], pos[1]]
            on_top_of_factory = factory_there in shared_obs["teams"][agent]["factory_strains"]

            # Dig is valid only if on top of tile with rubble or resources or lichen
            board_sum = (
                    shared_obs["board"]["ice"][pos[0], pos[1]]
                    + shared_obs["board"]["ore"][pos[0], pos[1]]
                    + shared_obs["board"]["rubble"][pos[0], pos[1]]
                    + shared_obs["board"]["lichen"][pos[0], pos[1]]
            )
            if board_sum > 0 and not on_top_of_factory:
                action_mask[indexer][(self.dig_dim_high - self.dig_act_dims):self.dig_dim_high] = True

            # Pickup is valid only if on top of factory tile
            if on_top_of_factory:
                action_mask[indexer][
                (self.pickup_dim_high - self.pickup_act_dims):self.pickup_dim_high] = True
                action_mask[indexer][(self.dig_dim_high - self.dig_act_dims):self.dig_dim_high] = False

            # No-op is always valid
            action_mask[indexer][-1] = True
        return action_mask
