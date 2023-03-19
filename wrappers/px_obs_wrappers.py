from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces


class PixelObservationWrapper(gym.ObservationWrapper):
    """
    Features vector for each pixel:

    * Global information
    - Is day
    - Own total factories
    - Enemy total factories
    - Own total heavy robots
    - Enemy total heavy robots
    - Own total light robots
    - Enemy total light robots
    - Own total water
    - Enemy total water
    - Own average water per factory
    - Enemy average water per factory
    - Own total power
    - Enemy total power
    - Own average power per factory
    - Enemy total power per factory
    - Own total metal
    - Enemy total metal
    - Own average metal per factory
    - Enemy average metal per factory
    - Own total lichen
    - Enemy total licen
    - Own average lichen per factory
    - Enemy average lichen per factory
    - Power cost per heavy robot
    - Power cost per light robot
    - Metal cost per heavy robot
    - Metal cost per light robot
    - Heavy robot battery cap
    - Light robot battery cap
    - Heavy robot cargo cap
    - Light robot cargo cap
    - Heavy robot charging rate
    - Light robot charging rate
    - Factory power charging rate
    - Factory ice processing rate
    - Factory ore processing rate
    - Factory ice processing ratio
    - Factory ore processing ratio
    - Factory size
    # - Power costs
    # - Water costs
    # - Lichen strains

    * Image features
    - If no factory
    - If own factory
    - If enemy factory
    - If no robot
    - If own heavy robot
    - If enemy heavy robot
    - If own light robot
    - If enemy light robot
    - Rubble level
    - If it is resource
    - Ore amount
    - Ice amount
    - Lichen amount
    # - Lichen strains
    - Factory power
    - Factory ice cargo
    - Factory ore cargo
    - Factory water cargo
    - Factory metal cargo
    - Robot power
    - Robot ice cargo
    - Robot ore cargo
    - Robot water cargo
    - Robot metal cargo
    - X normalized position
    - Y normalized position
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        env_cfg = env.env_cfg
        self.observation_space = spaces.Dict({
            'global': spaces.Box(-999, 999, shape=(39,)),
            'img': spaces.Box(-999, 999, shape=(45, env_cfg.map_size, env_cfg.map_size))
        })

    def observation(self, obs):
        return PixelObservationWrapper.convert_obs(obs, self.env.state.env_cfg)

    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any, player_id=None) -> Dict[str, npt.NDArray]:
        observation = dict()

        for agent in obs.keys():
            if player_id is not None and agent != player_id:
                continue
            own_player = agent
            enemy_player = "player_1" if own_player == "player_0" else "player_0"
            shared_obs = obs[own_player]
            map_size = env_cfg.map_size

            # Generate global features
            is_day = int(shared_obs['real_env_steps'] % env_cfg.CYCLE_LENGTH < env_cfg.DAY_LENGTH)
            heavy_power_cost = env_cfg.ROBOTS['HEAVY'].POWER_COST
            heavy_metal_cost = env_cfg.ROBOTS['HEAVY'].METAL_COST
            heavy_cargo_cap = env_cfg.ROBOTS['HEAVY'].CARGO_SPACE
            heavy_battery_cap = env_cfg.ROBOTS['HEAVY'].BATTERY_CAPACITY
            heavy_charging_rate = env_cfg.ROBOTS['HEAVY'].CHARGE
            light_power_cost = env_cfg.ROBOTS['LIGHT'].POWER_COST
            light_metal_cost = env_cfg.ROBOTS['LIGHT'].METAL_COST
            light_cargo_cap = env_cfg.ROBOTS['LIGHT'].CARGO_SPACE
            light_battery_cap = env_cfg.ROBOTS['LIGHT'].BATTERY_CAPACITY
            light_charging_rate = env_cfg.ROBOTS['LIGHT'].CHARGE
            factory_charging_rate = env_cfg.FACTORY_CHARGE
            factory_ice_proc_rate = env_cfg.FACTORY_PROCESSING_RATE_WATER
            factory_ore_proc_rate = env_cfg.FACTORY_PROCESSING_RATE_METAL
            factory_ice_proc_ratio = env_cfg.ICE_WATER_RATIO
            factory_ore_proc_ratio = env_cfg.ORE_METAL_RATIO
            factory_size = 3/map_size
            global_obs = [[
                is_day, heavy_power_cost, heavy_metal_cost, heavy_cargo_cap, heavy_battery_cap, heavy_charging_rate,
                light_power_cost, light_metal_cost, light_cargo_cap, light_battery_cap, light_charging_rate,
                factory_charging_rate, factory_ice_proc_rate, factory_ore_proc_rate, factory_ice_proc_ratio,
                factory_ore_proc_ratio, factory_size
            ]]

            for player in [own_player, enemy_player]:
                factories = list(shared_obs['factories'][player].values())
                units = list(shared_obs['units'][player].values())
                player_lichens = _mask_lichen_strains(shared_obs['board']['lichen'],
                                                      shared_obs['teams'][player]['factory_strains'])

                num_factories = len(factories)
                num_heavy_robots = sum([1 if u['unit_type'] == 'HEAVY' else 0 for u in units])
                num_light_robots = sum([1 if u['unit_type'] == 'LIGHT' else 0 for u in units])
                total_factory_power = sum([f['power'] for f in factories])
                avg_factory_power = total_factory_power/num_factories
                total_factory_water = sum([f['cargo']['water'] for f in factories])
                avg_factory_water = total_factory_water/num_factories
                total_factory_metal = sum([f['cargo']['metal'] for f in factories])
                avg_factory_metal = total_factory_metal/num_factories
                total_lichens = np.sum(player_lichens)
                avg_factory_lichen = total_lichens/num_factories

                player_global_info = [
                    num_factories, num_heavy_robots, num_light_robots, total_factory_power, avg_factory_power,
                    total_factory_water, avg_factory_water, total_factory_metal, avg_factory_metal, total_lichens,
                    avg_factory_lichen
                ]
                global_obs.append(player_global_info)

            global_obs = np.concatenate(global_obs, axis=-1)

            # Generate per pixel features
            img_obs = []
            num_factories = np.zeros((map_size, map_size))
            num_units = np.zeros((map_size, map_size))

            for player in [own_player, enemy_player]:
                has_factory = np.zeros((map_size, map_size))
                factory_power = np.zeros((map_size, map_size))
                factory_water = np.zeros((map_size, map_size))
                factory_metal = np.zeros((map_size, map_size))
                factory_ice = np.zeros((map_size, map_size))
                factory_ore = np.zeros((map_size, map_size))

                factories = list(shared_obs['factories'][player].values())
                if factories:
                    factories_pos = np.stack([f['pos'] for f in factories])
                    factories_indexer = tuple(factories_pos.T)

                    has_factory[factories_indexer] = 1
                    num_factories += has_factory

                    factory_power[factories_indexer] = [f['power'] for f in factories]
                    factory_water[factories_indexer] = [f['cargo']['water'] for f in factories]
                    factory_metal[factories_indexer] = [f['cargo']['metal'] for f in factories]
                    factory_ice[factories_indexer] = [f['cargo']['ice'] for f in factories]
                    factory_ore[factories_indexer] = [f['cargo']['ore'] for f in factories]

                img_obs.append(has_factory)
                img_obs.append(factory_power)
                img_obs.append(factory_water)
                img_obs.append(factory_metal)
                img_obs.append(factory_ice)
                img_obs.append(factory_ore)

                units = list(shared_obs['units'][player].values())
                for unit_type in ['HEAVY', 'LIGHT']:
                    has_unit = np.zeros((map_size, map_size))
                    unit_power = np.zeros((map_size, map_size))
                    unit_water = np.zeros((map_size, map_size))
                    unit_metal = np.zeros((map_size, map_size))
                    unit_ice = np.zeros((map_size, map_size))
                    unit_ore = np.zeros((map_size, map_size))

                    units_of_type = [u for u in units if u['unit_type'] == unit_type]
                    if units_of_type:
                        units_pos = np.stack([u['pos'] for u in units_of_type])
                        units_indexer = tuple(units_pos.T)

                        has_unit[units_indexer] = 1
                        num_units += has_unit
                        unit_power[units_indexer] = [u['power'] for u in units_of_type]
                        unit_water[units_indexer] = [u['cargo']['water'] for u in units_of_type]
                        unit_metal[units_indexer] = [u['cargo']['metal'] for u in units_of_type]
                        unit_ice[units_indexer] = [u['cargo']['ice'] for u in units_of_type]
                        unit_ore[units_indexer] = [u['cargo']['ore'] for u in units_of_type]

                    img_obs.append(has_unit)
                    img_obs.append(unit_power)
                    img_obs.append(unit_water)
                    img_obs.append(unit_metal)
                    img_obs.append(unit_ice)
                    img_obs.append(unit_ore)

            img_obs.append((num_factories > 0).astype(int))
            img_obs.append((num_units > 0).astype(int))
            img_obs.append(shared_obs['board']['rubble'])
            img_obs.append(shared_obs['board']['ice'])
            img_obs.append(shared_obs['board']['ore'])
            img_obs.append((shared_obs['board']['ice'] + shared_obs['board']['ore'] > 0).astype(int))
            img_obs.append(shared_obs['board']['lichen'])
            img_obs.extend(np.meshgrid(np.arange(map_size)/map_size, np.arange(map_size)/map_size, indexing='xy'))
            img_obs = np.stack(img_obs, axis=0)

            observation[player] = {
                'global': global_obs,
                'img': img_obs
            }

        return observation


def _mask_lichen_strains(lichen_board: npt.NDArray, lichen_strains: list[int]):
    is_selected_strains = np.isin(lichen_board, lichen_strains)
    res = lichen_board.copy()
    res[~is_selected_strains] = 0
    return res
