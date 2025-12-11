from pyswarms.backend.topology import Star
from pyswarms.backend.handlers import VelocityHandler
from pyswarms.backend.swarms import Swarm
from pyswarms.backend.handlers import BoundaryHandler
import numpy as np
import math
import pdb
import traceback

# match the format as pso_per_case_with_nan_control
def pso_with_nan_control(objective_fn, bounds, n_particles, n_iterations, param_names, crash_sentinel=None, max_retry_per_particle=5, prev_init_pos=None, prev_best_cost=None, custom_initializer=None, prev_swarm=None, verbose=True):
    lower_bounds, upper_bounds = np.array(bounds[0]), np.array(bounds[1])
    dim = len(lower_bounds)
    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
    if prev_swarm is not None: # continue prev swarm
        swarm = prev_swarm
    else: # initialize new swarm
        if custom_initializer is not None:
            if prev_init_pos is not None:
                assert prev_init_pos.shape == (dim,), "prev_init_pos must match dimensionality"
                init_pos = np.stack([custom_initializer() for _ in range(n_particles)])
                init_pos[0] = prev_init_pos  # overwrite first particle
                # print(f"[INFO] Using previous best position for particle 0: {init_pos[0]}")
                print(f"[INFO] Using custom initializer for particles, with prev best position for particle 0")
            else:
                init_pos = np.stack([custom_initializer() for _ in range(n_particles)])
        else:  # random initialization
            if prev_init_pos is not None: # prev best initial position provided
                assert prev_init_pos.shape == (dim,), "prev_init_pos must match dimensionality"
                init_pos = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(n_particles, dim))
                init_pos[0] = prev_init_pos  # overwrite first particle
                # print(f"[INFO] Using previous best position for particle 0: {init_pos[0]}")
                print(f"[INFO] Using previous best position for particle 0")
            else:
                init_pos = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(n_particles, dim))
        init_vel = np.zeros_like(init_pos)
        swarm = Swarm(position=init_pos, velocity=init_vel, options=options)

        swarm.pbest_pos = np.copy(init_pos)
        swarm.pbest_cost = np.full(n_particles, np.inf)

    topology = Star()
    velocity_handler = VelocityHandler(strategy="unmodified")

    gbest_cost = np.inf
    gbest_pos = None

    cost_history = []
    gbest_z01_history = []
    gbest_penalty_history = []

    current_z01 = np.full(n_particles, np.nan)
    current_penalty = np.full(n_particles, np.nan)
    pbest_z01 = np.full(n_particles, np.nan)
    pbest_penalty = np.full(n_particles, np.nan)

    gbest_iteration_record = None

    for iter_idx in range(n_iterations):
        if verbose:
            print(f"Iteration {iter_idx + 1}/{n_iterations}-----")
        cost = np.zeros(n_particles)
        z01_losses = []

        for idx in range(n_particles):
            if verbose:
                print(f"Processing particle {idx}")
            retry_count = 0
            fom, z01_loss = None, None

            if idx == 0 and iter_idx == 0 and prev_best_cost is not None:
                cost[idx] = prev_best_cost
                z01_losses.append(None)
                if verbose:
                    print(f"[SKIP] Particle 0 loaded with prev_best_cost: {prev_best_cost}")
                continue

            original_particle = swarm.position[idx]
            original_velocity = swarm.velocity[idx]

            while retry_count < max_retry_per_particle:
                particle = swarm.position[idx]
                particle_dict = dict(zip(param_names, particle)) if param_names else {}

                try:
                    result = objective_fn(inner_iter=iter_idx, particle_idx=idx, **particle_dict)
                except Exception as e:
                    print(f"[EXCEPTION] Particle {idx} crashed: {e}")
                    traceback.print_exc()
                    result = crash_sentinel

                if result is None or result == crash_sentinel:
                    print(f"[NaN] Particle {idx} produced NaN. Retrying... ({retry_count+1})")
                    # OPTION 3: Retry with random position within bounds
                    # swarm.position[idx] = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(dim,))

                    # OPTION 2: Retry near the original particle by scaling the velocity
                    scale_factor = 0.5 ** retry_count
                    new_position = original_particle + scale_factor * original_velocity
                    new_position = np.clip(new_position, lower_bounds, upper_bounds)
                    swarm.position[idx] = new_position

                    # OPTION 1: Retry near the original particle by adding Gaussian noise
                    # original_particle = particle
                    # noise_scale = 0.10  # 10% noise
                    # noise = np.random.normal(loc=0.0, scale=noise_scale, size=(dim,))
                    # perturbed = original_particle + noise * (upper_bounds - lower_bounds)
                    # perturbed = np.clip(perturbed, lower_bounds, upper_bounds)
                    # swarm.position[idx] = perturbed

                    retry_count += 1
                else:
                    fom, z01_loss, penalty = result
                    if math.isnan(fom):
                        print(f"[NaN] Particle {idx} produced NaN. Retrying... ({retry_count+1})")
                        # swarm.position[idx] = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(dim,))
                        scale_factor = 0.5 ** retry_count
                        new_position = original_particle + scale_factor * original_velocity
                        new_position = np.clip(new_position, lower_bounds, upper_bounds)
                        swarm.position[idx] = new_position
                        retry_count += 1
                    else:
                        break

            if fom is None or z01_loss is None:
                print(f"[FAIL] Particle {idx} failed after {max_retry_per_particle} retries.")
                cost[idx] = 1e6
                z01_losses.append(None)
                current_z01[idx] = np.nan
                current_penalty[idx] = np.nan
            else:
                cost[idx] = - (fom - z01_loss - penalty)
                z01_losses.append(z01_loss)
                current_z01[idx] = z01_loss
                current_penalty[idx] = penalty

            if verbose:
                print(f"Particle {idx} cost: {cost[idx]}, fom:{fom}, z01_loss: {z01_losses[-1]}, penalty: {penalty}")

            # JUST FOR VERIFICATION #############################################################################################################
            # if iter_idx == 0 and idx == 0:
            #     if prev_best_cost is not None and abs(cost[idx] - prev_best_cost) > 1e-4:
            #         print(f"[WARNING] Cost for loaded prev particle 0 changed significantly from previous best: {prev_best_cost} to {cost[idx]}")
            #####################################################################################################################################
                        
        # Update personal best
        better_mask = cost < swarm.pbest_cost
        swarm.pbest_cost[better_mask] = cost[better_mask]
        swarm.pbest_pos[better_mask] = swarm.position[better_mask]
        pbest_z01[better_mask] = current_z01[better_mask]
        pbest_penalty[better_mask] = current_penalty[better_mask]

        if verbose:
            print(f"Personal best costs updated: {swarm.pbest_cost}")

        # Update global best
        gbest_idx = np.argmin(swarm.pbest_cost)
        if swarm.pbest_cost[gbest_idx] < gbest_cost:
            swarm.best_cost = swarm.pbest_cost[gbest_idx]
            swarm.best_pos = swarm.pbest_pos[gbest_idx]
            gbest_cost = swarm.best_cost
            gbest_pos = swarm.best_pos
            gbest_iteration_record = (iter_idx, gbest_idx)

        if verbose:
            print(f"Global best cost: {gbest_cost}")

        cost_history.append(gbest_cost)
        gbest_z01_history.append(pbest_z01[gbest_idx])
        gbest_penalty_history.append(pbest_penalty[gbest_idx])

        # Update velocity and position
        swarm.velocity = topology.compute_velocity(
            swarm=swarm,
            clamp=None,
            vh=velocity_handler,
            bounds=(lower_bounds, upper_bounds)
        )

        bh=BoundaryHandler(strategy="nearest")
        swarm.position = topology.compute_position(swarm, bounds=(lower_bounds, upper_bounds), bh=bh)

        if np.any(swarm.position < lower_bounds) or np.any(swarm.position > upper_bounds):
            print(f"[WARNING] Particle out of bounds!")
        
    return gbest_cost, gbest_pos, cost_history, gbest_z01_history, gbest_penalty_history, gbest_iteration_record, swarm

def pso_with_nan_control_penalty(objective_fn, bounds, n_particles, n_iterations, param_names, crash_sentinel=None, max_retry_per_particle=5, prev_init_pos=None, prev_best_cost=None, custom_initializer=None, prev_swarm=None, verbose=True):
    lower_bounds, upper_bounds = np.array(bounds[0]), np.array(bounds[1])
    dim = len(lower_bounds)
    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
    if prev_swarm is not None: # continue prev swarm
        swarm = prev_swarm
    else: # initialize new swarm
        if custom_initializer is not None:
            if prev_init_pos is not None:
                assert prev_init_pos.shape == (dim,), "prev_init_pos must match dimensionality"
                init_pos = np.stack([custom_initializer() for _ in range(n_particles)])
                init_pos[0] = prev_init_pos  # overwrite first particle
                # print(f"[INFO] Using previous best position for particle 0: {init_pos[0]}")
                print(f"[INFO] Using custom initializer for particles, with prev best position for particle 0")
            else:
                init_pos = np.stack([custom_initializer() for _ in range(n_particles)])
        else:  # random initialization
            if prev_init_pos is not None: # prev best initial position provided
                assert prev_init_pos.shape == (dim,), "prev_init_pos must match dimensionality"
                init_pos = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(n_particles, dim))
                init_pos[0] = prev_init_pos  # overwrite first particle
                # print(f"[INFO] Using previous best position for particle 0: {init_pos[0]}")
                print(f"[INFO] Using previous best position for particle 0")
            else:
                init_pos = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(n_particles, dim))
        init_vel = np.zeros_like(init_pos)
        swarm = Swarm(position=init_pos, velocity=init_vel, options=options)

        swarm.pbest_pos = np.copy(init_pos)
        swarm.pbest_cost = np.full(n_particles, np.inf)

    topology = Star()
    velocity_handler = VelocityHandler(strategy="unmodified")

    gbest_cost = np.inf
    gbest_pos = None

    cost_history = []
    # z01_loss_history = []
    # penalty_history = []
    gbest_z01_history = []
    gbest_penalty_history = []

    current_z01 = np.full(n_particles, np.nan)
    current_penalty = np.full(n_particles, np.nan)
    pbest_z01 = np.full(n_particles, np.nan)
    pbest_penalty = np.full(n_particles, np.nan)

    gbest_iteration_record = None

    for iter_idx in range(n_iterations):
        if verbose:
            print(f"Iteration {iter_idx + 1}/{n_iterations}-----")
        cost = np.zeros(n_particles)
        z01_losses = []

        for idx in range(n_particles):
            if verbose:
                print(f"Processing particle {idx}")
            retry_count = 0
            fom, z01_loss = None, None

            if idx == 0 and iter_idx == 0 and prev_best_cost is not None:
                cost[idx] = prev_best_cost
                z01_losses.append(None)
                if verbose:
                    print(f"[SKIP] Particle 0 loaded with prev_best_cost: {prev_best_cost}")
                continue

            original_particle = swarm.position[idx]
            original_velocity = swarm.velocity[idx]

            while retry_count < max_retry_per_particle:
                particle = swarm.position[idx]
                particle_dict = dict(zip(param_names, particle)) if param_names else {}

                try:
                    result = objective_fn(inner_iter=iter_idx, particle_idx=idx, **particle_dict)
                except Exception as e:
                    print(f"[EXCEPTION] Particle {idx} crashed: {e}")
                    traceback.print_exc()
                    result = crash_sentinel

                if result is None or result == crash_sentinel:
                    print(f"[NaN] Particle {idx} produced NaN. Retrying... ({retry_count+1})")
                    # OPTION 3: Retry with random position within bounds
                    # swarm.position[idx] = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(dim,))

                    # OPTION 2: Retry near the original particle by scaling the velocity
                    scale_factor = 0.5 ** retry_count
                    new_position = original_particle + scale_factor * original_velocity
                    new_position = np.clip(new_position, lower_bounds, upper_bounds)
                    swarm.position[idx] = new_position

                    # OPTION 1: Retry near the original particle by adding Gaussian noise
                    # original_particle = particle
                    # noise_scale = 0.10  # 10% noise
                    # noise = np.random.normal(loc=0.0, scale=noise_scale, size=(dim,))
                    # perturbed = original_particle + noise * (upper_bounds - lower_bounds)
                    # perturbed = np.clip(perturbed, lower_bounds, upper_bounds)
                    # swarm.position[idx] = perturbed

                    retry_count += 1
                else:
                    fom, z01_loss, penalty = result
                    if math.isnan(fom):
                        print(f"[NaN] Particle {idx} produced NaN. Retrying... ({retry_count+1})")
                        # swarm.position[idx] = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(dim,))
                        scale_factor = 0.5 ** retry_count
                        new_position = original_particle + scale_factor * original_velocity
                        new_position = np.clip(new_position, lower_bounds, upper_bounds)
                        swarm.position[idx] = new_position
                        retry_count += 1
                    else:
                        break

            if fom is None or z01_loss is None:
                print(f"[FAIL] Particle {idx} failed after {max_retry_per_particle} retries.")
                cost[idx] = 1e6
                z01_losses.append(None)
                current_z01[idx] = np.nan
                current_penalty[idx] = np.nan
            else:
                cost[idx] = - (fom - z01_loss - penalty)
                z01_losses.append(z01_loss)
                current_z01[idx] = z01_loss
                current_penalty[idx] = penalty

            if verbose:
                print(f"Particle {idx} cost: {cost[idx]}, fom:{fom}, z01_loss: {z01_losses[-1]}, penalty: {penalty}")

            # JUST FOR VERIFICATION #############################################################################################################
            # if iter_idx == 0 and idx == 0:
            #     if prev_best_cost is not None and abs(cost[idx] - prev_best_cost) > 1e-4:
            #         print(f"[WARNING] Cost for loaded prev particle 0 changed significantly from previous best: {prev_best_cost} to {cost[idx]}")
            #####################################################################################################################################
                        
        # Update personal best
        better_mask = cost < swarm.pbest_cost
        swarm.pbest_cost[better_mask] = cost[better_mask]
        swarm.pbest_pos[better_mask] = swarm.position[better_mask]
        pbest_z01[better_mask] = current_z01[better_mask]
        pbest_penalty[better_mask] = current_penalty[better_mask]

        if verbose:
            print(f"Personal best costs updated: {swarm.pbest_cost}")

        # Update global best
        gbest_idx = np.argmin(swarm.pbest_cost)
        if swarm.pbest_cost[gbest_idx] < gbest_cost:
            swarm.best_cost = swarm.pbest_cost[gbest_idx]
            swarm.best_pos = swarm.pbest_pos[gbest_idx]
            gbest_cost = swarm.best_cost
            gbest_pos = swarm.best_pos
            gbest_iteration_record = (iter_idx, gbest_idx)

        if verbose:
            print(f"Global best cost: {gbest_cost}")

        cost_history.append(gbest_cost)
        gbest_z01_history.append(pbest_z01[gbest_idx])
        gbest_penalty_history.append(pbest_penalty[gbest_idx])
        # z01_loss_history.append(z01_losses)

        # Update velocity and position
        swarm.velocity = topology.compute_velocity(
            swarm=swarm,
            clamp=None,
            vh=velocity_handler,
            bounds=(lower_bounds, upper_bounds)
        )

        bh=BoundaryHandler(strategy="nearest")
        swarm.position = topology.compute_position(swarm, bounds=(lower_bounds, upper_bounds), bh=bh)

        if np.any(swarm.position < lower_bounds) or np.any(swarm.position > upper_bounds):
            print(f"[WARNING] Particle out of bounds!")

    return gbest_cost, gbest_pos, cost_history, gbest_z01_history, gbest_penalty_history, gbest_iteration_record, swarm

####### REVISED 0829 - penalty included #######
def pso_per_case_with_nan_control(objective_fn, n_particles, n_iterations, 
                                  case1_bounds, case2_bounds, case3_bounds,
                                  case1_param_names, case2_param_names, case3_param_names,
                                  neg_z_fom, nan_z_fom,
                                  crash_sentinel=None, # crashed not because of Neg Z or NaN
                                  max_retry_per_particle=5, 
                                  case1_prev_init_pos=None, case2_prev_init_pos=None, case3_prev_init_pos=None,
                                  prev_best_cost=None, # list [case1_cost, case2_cost, case3_cost]
                                  custom_initializer=None, # should skip per_case
                                  prev_swarm=None, # list [case1_swarm, case2_swarm, case3_swarm]
                                  verbose=True):

    def resimulate_case(case_num, valid_particle_dict, idx, iter_idx): # when Neg Z or NaN occurred due to other cases
        try:
            result = objective_fn(inner_iter=iter_idx, particle_idx=idx, **valid_particle_dict)
        except Exception as e:
            print(f"[Resimulating] [EXCEPTION] [CASE{case_num}] Particle {idx} crashed during resimulation: {e}")
            traceback.print_exc()
            return None

        if result is None or result == crash_sentinel:
            print(f"[Resimulating] [CASE{case_num}] Resimulation produced NaN.")
            return None

        return result

    case1_lower_bounds, case1_upper_bounds = np.array(case1_bounds[0]), np.array(case1_bounds[1])
    case2_lower_bounds, case2_upper_bounds = np.array(case2_bounds[0]), np.array(case2_bounds[1])
    case3_lower_bounds, case3_upper_bounds = np.array(case3_bounds[0]), np.array(case3_bounds[1])

    case1_dim = len(case1_lower_bounds)
    case2_dim = len(case2_lower_bounds)
    case3_dim = len(case3_lower_bounds)

    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}

    case1_swarm, case2_swarm, case3_swarm = prev_swarm

    # initialize new swarm
    if case1_swarm is None:
        if custom_initializer is not None:
            if case1_prev_init_pos is not None:
                assert case1_prev_init_pos.shape == (case1_dim,), "prev_init_pos must match dimensionality for case 1"
                case1_init_pos = np.stack([custom_initializer("case1") for _ in range(n_particles)])
                case1_init_pos[0] = case1_prev_init_pos  # overwrite first particle                
                print(f"[CASE1] Using custom initializer for particles, with prev best position for particle 0")
            else:
                case1_init_pos = np.stack([custom_initializer("case1") for _ in range(n_particles)])
        else:
            if case1_prev_init_pos is not None:
                assert case1_prev_init_pos.shape == (case1_dim,), "prev_init_pos must match dimensionality for case 1"
                case1_init_pos = np.random.uniform(low=case1_lower_bounds, high=case1_upper_bounds, size=(n_particles, case1_dim))
                case1_init_pos[0] = case1_prev_init_pos  # overwrite first particle
                print(f"[CASE1] Using previous best position for particle 0")
            else:
                case1_init_pos = np.random.uniform(low=case1_lower_bounds, high=case1_upper_bounds, size=(n_particles, case1_dim))
        case1_init_vel = np.zeros_like(case1_init_pos)
        case1_swarm = Swarm(position=case1_init_pos, velocity=case1_init_vel, options=options)
        case1_swarm.pbest_pos = np.copy(case1_init_pos)
        case1_swarm.pbest_cost = np.full(n_particles, np.inf)

    if case2_swarm is None:
        if custom_initializer is not None:
            if case2_prev_init_pos is not None:
                assert case2_prev_init_pos.shape == (case2_dim,), "prev_init_pos must match dimensionality for case 2"
                case2_init_pos = np.stack([custom_initializer("case2") for _ in range(n_particles)])
                case2_init_pos[0] = case2_prev_init_pos  # overwrite first particle
                print(f"[CASE2] Using custom initializer for particles, with prev best position for particle 0")
            else:
                case2_init_pos = np.stack([custom_initializer("case2") for _ in range(n_particles)])
        else:
            if case2_prev_init_pos is not None:  # prev best initial position provided
                assert case2_prev_init_pos.shape == (case2_dim,), "prev_init_pos must match dimensionality for case 2"
                case2_init_pos = np.random.uniform(low=case2_lower_bounds, high=case2_upper_bounds, size=(n_particles, case2_dim))
                case2_init_pos[0] = case2_prev_init_pos  # overwrite first particle
                print(f"[CASE2] Using previous best position for particle 0")
            
            else:
                case2_init_pos = np.random.uniform(low=case2_lower_bounds, high=case2_upper_bounds, size=(n_particles, case2_dim))
        
        case2_init_vel = np.zeros_like(case2_init_pos)
        case2_swarm = Swarm(position=case2_init_pos, velocity=case2_init_vel, options=options)
        case2_swarm.pbest_pos = np.copy(case2_init_pos)
        case2_swarm.pbest_cost = np.full(n_particles, np.inf)

    if case3_swarm is None:
        if custom_initializer is not None:
            if case3_prev_init_pos is not None:
                assert case3_prev_init_pos.shape == (case3_dim,), "prev_init_pos must match dimensionality for case 3"
                case3_init_pos = np.stack([custom_initializer("case3") for _ in range(n_particles)])
                case3_init_pos[0] = case3_prev_init_pos  # overwrite first particle
                print(f"[CASE3] Using custom initializer for particles, with prev best position for particle 0")
            else:
                case3_init_pos = np.stack([custom_initializer("case3") for _ in range(n_particles)])
        else:  # random initialization
            if case3_prev_init_pos is not None: # prev best initial position provided
                assert case3_prev_init_pos.shape == (case3_dim,), "prev_init_pos must match dimensionality for case 3"
                case3_init_pos = np.random.uniform(low=case3_lower_bounds, high=case3_upper_bounds, size=(n_particles, case3_dim))
                case3_init_pos[0] = case3_prev_init_pos  # overwrite first particle
                print(f"[CASE3] Using previous best position for particle 0")
            else:
                case3_init_pos = np.random.uniform(low=case3_lower_bounds, high=case3_upper_bounds, size=(n_particles, case3_dim))

        case3_init_vel = np.zeros_like(case3_init_pos)
        case3_swarm = Swarm(position=case3_init_pos, velocity=case3_init_vel, options=options)
        case3_swarm.pbest_pos = np.copy(case3_init_pos)
        case3_swarm.pbest_cost = np.full(n_particles, np.inf)
    
    case1_prev_best_cost, case2_prev_best_cost, case3_prev_best_cost = prev_best_cost

    topology = Star()
    velocity_handler = VelocityHandler(strategy="unmodified")

    case1_gbest_cost = np.inf
    case1_gbest_pos = None
    case2_gbest_cost = np.inf
    case2_gbest_pos = None
    case3_gbest_cost = np.inf
    case3_gbest_pos = None

    case1_cost_history = []
    case1_gbest_iteration_record = None
    case2_cost_history = []
    case2_gbest_iteration_record = None
    case3_cost_history = []
    case3_gbest_iteration_record = None

    case1_gbest_z01_history = []
    case1_gbest_penalty_history = []
    case2_gbest_z01_history = []
    case2_gbest_penalty_history = []
    case3_gbest_z01_history = []
    case3_gbest_penalty_history = []

    case1_current_z01 = np.full(n_particles, np.nan)
    case1_current_penalty = np.full(n_particles, np.nan)
    case1_pbest_z01 = np.full(n_particles, np.nan)
    case1_pbest_penalty = np.full(n_particles, np.nan)

    case2_current_z01 = np.full(n_particles, np.nan)
    case2_current_penalty = np.full(n_particles, np.nan)
    case2_pbest_z01 = np.full(n_particles, np.nan)
    case2_pbest_penalty = np.full(n_particles, np.nan)

    case3_current_z01 = np.full(n_particles, np.nan)
    case3_current_penalty = np.full(n_particles, np.nan)
    case3_pbest_z01 = np.full(n_particles, np.nan)
    case3_pbest_penalty = np.full(n_particles, np.nan)

    for iter_idx in range(n_iterations):
        if verbose:
            print(f"Iteration {iter_idx + 1}/{n_iterations}------------------------------------------------------")
        case1_cost = np.zeros(n_particles)
        case2_cost = np.zeros(n_particles)
        case3_cost = np.zeros(n_particles)

        for idx in range(n_particles):
            if verbose:
                print(f"Processing particle {idx}")
            retry_count = 0
            case1_fom, case1_z01_loss = None, None
            case2_fom, case2_z01_loss = None, None
            case3_fom, case3_z01_loss = None, None

            if idx == 0 and iter_idx == 0:
                if case1_prev_best_cost is not None and case2_prev_best_cost is not None and case3_prev_best_cost is not None:
                    case1_cost[idx] = case1_prev_best_cost
                    case2_cost[idx] = case2_prev_best_cost
                    case3_cost[idx] = case3_prev_best_cost
                    if verbose:
                        print(f"[SKIP] [ALL CASES] Particle 0 loaded with prev best costs: {case1_prev_best_cost}, {case2_prev_best_cost}, {case3_prev_best_cost}")
                    continue

            original_case1_particle = case1_swarm.position[idx]
            original_case2_particle = case2_swarm.position[idx]
            original_case3_particle = case3_swarm.position[idx]

            original_case1_velocity = case1_swarm.velocity[idx]
            original_case2_velocity = case2_swarm.velocity[idx]
            original_case3_velocity = case3_swarm.velocity[idx]

            while retry_count < max_retry_per_particle:
                case1_particle = case1_swarm.position[idx]
                case2_particle = case2_swarm.position[idx]
                case3_particle = case3_swarm.position[idx]
                
                case1_particle_dict = dict(zip(case1_param_names, case1_particle)) if case1_param_names else {}
                case2_particle_dict = dict(zip(case2_param_names, case2_particle)) if case2_param_names else {}
                case3_particle_dict = dict(zip(case3_param_names, case3_particle)) if case3_param_names else {}

                particle_dict = {**case1_particle_dict, **case2_particle_dict, **case3_particle_dict}

                try:
                    result = objective_fn(inner_iter=iter_idx, particle_idx=idx, **particle_dict)
                except Exception as e:
                    print(f"[EXCEPTION] Particle {idx} crashed: {e}")
                    traceback.print_exc()
                    result = crash_sentinel
                    raise
                
                retry_needed = False

                if result == crash_sentinel: # crashed not because of Neg Z or NaN
                    print(f"[EXCEPTION] Particle {idx} crashed. Retrying... ({retry_count+1})")
                    retry_needed = True

                else:
                    case1_fom, case2_fom, case3_fom, case1_weighted_z01_loss, case2_weighted_z01_loss, case3_weighted_z01_loss, case1_weighted_penalty, case2_weighted_penalty, case3_weighted_penalty = result
                    case_foms = [case1_fom, case2_fom, case3_fom]
                    case_losses = [case1_weighted_z01_loss, case2_weighted_z01_loss, case3_weighted_z01_loss]
                    case_penalties = [case1_weighted_penalty, case2_weighted_penalty, case3_weighted_penalty]
                    case_dicts = [case1_particle_dict, case2_particle_dict, case3_particle_dict]
                    case_dicts_dims = [case1_dim, case2_dim, case3_dim]
                    retry_succeeded_dicts = []

                    if any(fom == nan_z_fom for fom in case_foms): # at least one of the cases produced NaN
                        print(f"[NaN] Particle {idx} produced NaN data. Retrying... ({retry_count+1}/{max_retry_per_particle})")
                        retry_needed = True

                    elif any(fom is None for fom in case_foms): # at least one of the cases produced Neg Z
                            for i, fom in enumerate(case_foms):
                                # Resimulate the case that is not responsible for Neg Z
                                if fom is None:
                                    print(f"[NegZ] [CASE{i+1}] Resimulating possible VALID particle {idx}...")
                                    valid_particle_dict = case_dicts[i].copy()
                                    if len(valid_particle_dict) != case_dicts_dims[i]:
                                        print(f"Using incorrect particle_dict for case {i+1}, expected {case_dicts_dims[i]} keys, got {len(valid_particle_dict)} keys.")
                                        pdb.set_trace()
                                    if retry_succeeded_dicts:
                                        dicts = retry_succeeded_dicts[-1]
                                        valid_particle_dict= {**valid_particle_dict, **dicts}
                                        print(f"[NegZ] [CASE{i+1}] Merging VALID particle dicts from previous retries") ### JUST FOR DEBUGGING
                                    result = resimulate_case(i+1, valid_particle_dict, idx, iter_idx)
                                    if result is None:
                                        retry_needed = True
                                        print(f"[NegZ] [CASE{i+1}] Resimulation crashed, retrying all cases...")
                                        break
                                    retry_foms = result[:3]
                                    retry_losses = result[3:6]
                                    retry_penalties = result[6:]

                                    if retry_foms[i] is None or retry_foms[i] == nan_z_fom:
                                        retry_needed = True
                                        print(f"[NegZ] [CASE{i+1}] Resimulation produced NaN, retrying all cases...")
                                        break
                                    case_foms[i] = retry_foms[i]
                                    case_losses[i] = retry_losses[i]
                                    case_penalties[i] = retry_penalties[i]
                                    retry_succeeded_dicts.append(valid_particle_dict)

                    elif any(math.isnan(fom) for fom in case_foms):
                        print(f"[NaN] Particle {idx} produced FoM of NaN. Retrying... ({retry_count+1}/{max_retry_per_particle})")
                        retry_needed = True

                if retry_needed:
                    if iter_idx == 0: # if first iteration, randomly generate new point for simulation
                        case1_swarm.position[idx] = np.random.uniform(low=case1_lower_bounds, high=case1_upper_bounds, size=(case1_dim,))
                        case2_swarm.position[idx] = np.random.uniform(low=case2_lower_bounds, high=case2_upper_bounds, size=(case2_dim,))
                        case3_swarm.position[idx] = np.random.uniform(low=case3_lower_bounds, high=case3_upper_bounds, size=(case3_dim,))
                    else:
                        scale_factor = 0.5 ** retry_count
                        new_case1_position = original_case1_particle + scale_factor * original_case1_velocity
                        new_case1_position = np.clip(new_case1_position, case1_lower_bounds, case1_upper_bounds)
                        case1_swarm.position[idx] = new_case1_position

                        new_case2_position = original_case2_particle + scale_factor * original_case2_velocity
                        new_case2_position = np.clip(new_case2_position, case2_lower_bounds, case2_upper_bounds)
                        case2_swarm.position[idx] = new_case2_position

                        new_case3_position = original_case3_particle + scale_factor * original_case3_velocity
                        new_case3_position = np.clip(new_case3_position, case3_lower_bounds, case3_upper_bounds)
                        case3_swarm.position[idx] = new_case3_position

                    print(f"[Retry] Particle {idx} retrying with new positions")
                    retry_count += 1
                else:
                    case1_fom, case2_fom, case3_fom = case_foms
                    case1_weighted_z01_loss, case2_weighted_z01_loss, case3_weighted_z01_loss = case_losses
                    case1_weighted_penalty, case2_weighted_penalty, case3_weighted_penalty = case_penalties
                    break

            # If ended with no succeeded resimulation within max_retry_per_particle
            # Retry cases that are not responsible for NaN
            final_retry_succeeded_dicts = []
            if any(fom is None for fom in case_foms) and nan_z_fom in case_foms:
                print(f"[FAIL] Particle {idx} failed after {max_retry_per_particle} retries.")
                for i, fom in enumerate(case_foms):
                    if fom is None:
                        print(f"[NaN] [CASE{i+1}] Resimulating possible VALID particle {idx}...")
                        final_valid_particle_dict = case_dicts[i].copy()
                        if len(final_valid_particle_dict) != case_dicts_dims[i]:
                            print(f"Using incorrect particle_dict for case {i+1}, expected {case_dicts_dims[i]} keys, got {len(final_valid_particle_dict)} keys.")
                            pdb.set_trace()
                        if final_retry_succeeded_dicts:
                            final_dicts = final_retry_succeeded_dicts[-1]
                            final_valid_particle_dict= {**final_valid_particle_dict, **final_dicts}
                            print(f"[NaN] [CASE{i+1}] Merging VALID particle dicts from previous retries") ### JUST FOR DEBUGGING
                        result = resimulate_case(i+1, final_valid_particle_dict, idx, iter_idx)
                        if result is None:
                            print(f"[NaN] [CASE{i+1}] Resimulation crashed...")
                            break
                        retry_foms = result[:3]
                        retry_losses = result[3:6]
                        retry_penalties = result[6:]

                        if retry_foms[i] is None or retry_foms[i] == nan_z_fom:
                            retry_needed = True
                            print(f"[NaN] [CASE{i+1}] Resimulation produced NaN...")
                            break
                        case_foms[i] = retry_foms[i]
                        case_losses[i] = retry_losses[i]
                        case_penalties[i] = retry_penalties[i]
                        final_retry_succeeded_dicts.append(final_valid_particle_dict)
                case1_fom, case2_fom, case3_fom = case_foms
                case1_weighted_z01_loss, case2_weighted_z01_loss, case3_weighted_z01_loss = case_losses
                case1_weighted_penalty, case2_weighted_penalty, case3_weighted_penalty = case_penalties

            if case1_fom is None or case1_weighted_z01_loss is None:
                print(f"[FAIL][CASE1] Particle {idx} failed after {max_retry_per_particle} retries.")
                case1_cost[idx] = - nan_z_fom
                case1_current_z01[idx] = np.nan
                case1_current_penalty[idx] = np.nan
            else:
                case1_cost[idx] = - (case1_fom - case1_weighted_z01_loss - case1_weighted_penalty)
                case1_current_z01[idx] = case1_weighted_z01_loss
                case1_current_penalty[idx] = case1_weighted_penalty

            if case2_fom is None or case2_weighted_z01_loss is None:
                print(f"[FAIL][CASE2] Particle {idx} failed after {max_retry_per_particle} retries.")
                case2_cost[idx] = - nan_z_fom
                case2_current_z01[idx] = np.nan
                case2_current_penalty[idx] = np.nan
            else:
                case2_cost[idx] = - (case2_fom - case2_weighted_z01_loss - case2_weighted_penalty)
                case2_current_z01[idx] = case2_weighted_z01_loss
                case2_current_penalty[idx] = case2_weighted_penalty

            if case3_fom is None or case3_weighted_z01_loss is None:
                print(f"[FAIL][CASE3] Particle {idx} failed after {max_retry_per_particle} retries.")
                case3_cost[idx] = - nan_z_fom
                case3_current_z01[idx] = np.nan
                case3_current_penalty[idx] = np.nan
            else:
                case3_cost[idx] = - (case3_fom - case3_weighted_z01_loss - case3_weighted_penalty)
                case3_current_z01[idx] = case3_weighted_z01_loss
                case3_current_penalty[idx] = case3_weighted_penalty

            if verbose:
                print(f"Case1 Particle {idx} cost: {case1_cost[idx]}, fom:{case1_fom}, z01_loss: {case1_weighted_z01_loss}, penalty: {case1_weighted_penalty}")
                print(f"Case2 Particle {idx} cost: {case2_cost[idx]}, fom:{case2_fom}, z01_loss: {case2_weighted_z01_loss}, penalty: {case2_weighted_penalty}")
                print(f"Case3 Particle {idx} cost: {case3_cost[idx]}, fom:{case3_fom}, z01_loss: {case3_weighted_z01_loss}, penalty: {case3_weighted_penalty}")

            if iter_idx == 0 and idx == 0:
                if prev_best_cost is not None:
                    case1_prev_best_cost, case2_prev_best_cost, case3_prev_best_cost = prev_best_cost
                    if case1_prev_best_cost is not None and abs(case1_cost[idx] - case1_prev_best_cost) > 1e-4:
                        print(f"[WARNING] Case1 Cost for loaded prev particle 0 changed significantly from previous best: {case1_prev_best_cost} to {case1_cost[idx]}")
                    if case2_prev_best_cost is not None and abs(case2_cost[idx] - case2_prev_best_cost) > 1e-4:
                        print(f"[WARNING] Case2 Cost for loaded prev particle 0 changed significantly from previous best: {case2_prev_best_cost} to {case2_cost[idx]}")
                    if case3_prev_best_cost is not None and abs(case3_cost[idx] - case3_prev_best_cost) > 1e-4:
                        print(f"[WARNING] Case3 Cost for loaded prev particle 0 changed significantly from previous best: {case3_prev_best_cost} to {case3_cost[idx]}")


        # Update personal best
        case1_better_mask = case1_cost < case1_swarm.pbest_cost
        case1_swarm.pbest_cost[case1_better_mask] = case1_cost[case1_better_mask]
        case1_swarm.pbest_pos[case1_better_mask] = case1_swarm.position[case1_better_mask]
        case1_pbest_z01[case1_better_mask] = case1_current_z01[case1_better_mask]
        case1_pbest_penalty[case1_better_mask] = case1_current_penalty[case1_better_mask]

        case2_better_mask = case2_cost < case2_swarm.pbest_cost
        case2_swarm.pbest_cost[case2_better_mask] = case2_cost[case2_better_mask]
        case2_swarm.pbest_pos[case2_better_mask] = case2_swarm.position[case2_better_mask]
        case2_pbest_z01[case2_better_mask] = case2_current_z01[case2_better_mask]
        case2_pbest_penalty[case2_better_mask] = case2_current_penalty[case2_better_mask]

        case3_better_mask = case3_cost < case3_swarm.pbest_cost
        case3_swarm.pbest_cost[case3_better_mask] = case3_cost[case3_better_mask]
        case3_swarm.pbest_pos[case3_better_mask] = case3_swarm.position[case3_better_mask]
        case3_pbest_z01[case3_better_mask] = case3_current_z01[case3_better_mask]
        case3_pbest_penalty[case3_better_mask] = case3_current_penalty[case3_better_mask]

        if verbose:
            print(f"Case1 Personal best costs updated: {case1_swarm.pbest_cost}")
            print(f"Case2 Personal best costs updated: {case2_swarm.pbest_cost}")
            print(f"Case3 Personal best costs updated: {case3_swarm.pbest_cost}")

        # Update global best
        case1_gbest_idx = np.argmin(case1_swarm.pbest_cost)
        if case1_swarm.pbest_cost[case1_gbest_idx] < case1_gbest_cost:
            case1_swarm.best_cost = case1_swarm.pbest_cost[case1_gbest_idx]
            case1_swarm.best_pos = case1_swarm.pbest_pos[case1_gbest_idx]
            case1_gbest_cost = case1_swarm.best_cost
            case1_gbest_pos = case1_swarm.best_pos
            case1_gbest_iteration_record = (iter_idx, case1_gbest_idx)

            if verbose:
                print(f"Case1 Global best cost updated: {case1_gbest_cost}")
        
        case2_gbest_idx = np.argmin(case2_swarm.pbest_cost)
        if case2_swarm.pbest_cost[case2_gbest_idx] < case2_gbest_cost:
            case2_swarm.best_cost = case2_swarm.pbest_cost[case2_gbest_idx]
            case2_swarm.best_pos = case2_swarm.pbest_pos[case2_gbest_idx]
            case2_gbest_cost = case2_swarm.best_cost
            case2_gbest_pos = case2_swarm.best_pos
            case2_gbest_iteration_record = (iter_idx, case2_gbest_idx)

            if verbose:
                print(f"Case2 Global best cost updated: {case2_gbest_cost}")
        
        case3_gbest_idx = np.argmin(case3_swarm.pbest_cost)
        if case3_swarm.pbest_cost[case3_gbest_idx] < case3_gbest_cost:
            case3_swarm.best_cost = case3_swarm.pbest_cost[case3_gbest_idx]
            case3_swarm.best_pos = case3_swarm.pbest_pos[case3_gbest_idx]
            case3_gbest_cost = case3_swarm.best_cost
            case3_gbest_pos = case3_swarm.best_pos
            case3_gbest_iteration_record = (iter_idx, case3_gbest_idx)

            if verbose:
                print(f"Case3 Global best cost updated: {case3_gbest_cost}")

        case1_cost_history.append(case1_gbest_cost)
        case2_cost_history.append(case2_gbest_cost)
        case3_cost_history.append(case3_gbest_cost)
        case1_gbest_z01_history.append(case1_pbest_z01[case1_gbest_idx])
        case1_gbest_penalty_history.append(case1_pbest_penalty[case1_gbest_idx])
        case2_gbest_z01_history.append(case2_pbest_z01[case2_gbest_idx])
        case2_gbest_penalty_history.append(case2_pbest_penalty[case2_gbest_idx])
        case3_gbest_z01_history.append(case3_pbest_z01[case3_gbest_idx])
        case3_gbest_penalty_history.append(case3_pbest_penalty[case3_gbest_idx])

        bh=BoundaryHandler(strategy="nearest")

        # Update velocity and position using backend API
        case1_swarm.velocity = topology.compute_velocity(
            swarm=case1_swarm,
            clamp=None,
            vh=velocity_handler,
            bounds=(case1_lower_bounds, case1_upper_bounds)
        )
        case1_swarm.position = topology.compute_position(case1_swarm, bounds=(case1_lower_bounds, case1_upper_bounds), bh=bh)

        if np.any(case1_swarm.position < case1_lower_bounds) or np.any(case1_swarm.position > case1_upper_bounds):
            print(f"[WARNING] Case1 Next Particle out of bounds!")

        case2_swarm.velocity = topology.compute_velocity(
            swarm=case2_swarm,
            clamp=None,
            vh=velocity_handler,
            bounds=(case2_lower_bounds, case2_upper_bounds)
        )
        case2_swarm.position = topology.compute_position(case2_swarm, bounds=(case2_lower_bounds, case2_upper_bounds), bh=bh)

        if np.any(case2_swarm.position < case2_lower_bounds) or np.any(case2_swarm.position > case2_upper_bounds):
            print(f"[WARNING] Case2 Next Particle out of bounds!")

        case3_swarm.velocity = topology.compute_velocity(
            swarm=case3_swarm,
            clamp=None,
            vh=velocity_handler,
            bounds=(case3_lower_bounds, case3_upper_bounds)
        )
        case3_swarm.position = topology.compute_position(case3_swarm, bounds=(case3_lower_bounds, case3_upper_bounds), bh=bh)

        if np.any(case3_swarm.position < case3_lower_bounds) or np.any(case3_swarm.position > case3_upper_bounds):
            print(f"[WARNING] Case3 Next Particle out of bounds!")

    gbest_cost = [case1_gbest_cost, case2_gbest_cost, case3_gbest_cost]

    gbest_pos = {
        'case1': case1_gbest_pos,
        'case2': case2_gbest_pos,
        'case3': case3_gbest_pos
    }

    gbest_iteration_record = {
        'case1': case1_gbest_iteration_record,
        'case2': case2_gbest_iteration_record,
        'case3': case3_gbest_iteration_record
    }

    cost_history = {
        'case1': case1_cost_history,
        'case2': case2_cost_history,
        'case3': case3_cost_history
    }

    gbest_z01_history = {
        'case1': case1_gbest_z01_history,
        'case2': case2_gbest_z01_history,
        'case3': case3_gbest_z01_history
    }

    gbest_penalty_history= {
        'case1': case1_gbest_penalty_history,
        'case2': case2_gbest_penalty_history,
        'case3': case3_gbest_penalty_history
    }

    swarm = [case1_swarm, case2_swarm, case3_swarm]

    return gbest_cost, gbest_pos, cost_history, gbest_z01_history, gbest_penalty_history, gbest_iteration_record, swarm

def bothOpt_pso_per_case_with_nan_control(
                                  objective_fn, n_particles, n_iterations, 
                                  case1_alpha_bounds, case2_alpha_bounds, case3_alpha_bounds, 
                                  case1_omega_bounds, case2_omega_bounds, case3_omega_bounds,
                                  case1_alpha_names, case2_alpha_names, case3_alpha_names,
                                  case1_omega_names, case2_omega_names, case3_omega_names,
                                  neg_z_fom, nan_z_fom,
                                  crash_sentinel=None, # crashed not because of Neg Z or NaN
                                  max_retry_per_particle=5, 
                                  custom_alpha_initializer=None, 
                                  verbose=True):

    def resimulate_case(case_num, valid_particle_dict, idx, iter_idx): # when Neg Z or NaN occurred due to other cases
        alpha_particle_dict = {k:v for k,v in valid_particle_dict.items() if k in (case1_alpha_names + case2_alpha_names + case3_alpha_names)}
        omega_particle_dict = {k:v for k,v in valid_particle_dict.items() if k in (case1_omega_names + case2_omega_names + case3_omega_names)}
        try:
            result = objective_fn(inner_iter=iter_idx, particle_idx=idx, alpha_vars=alpha_particle_dict, omega_vars=omega_particle_dict)
        except Exception as e:
            print(f"[Resimulating] [EXCEPTION] [CASE{case_num}] Particle {idx} crashed during resimulation: {e}")
            traceback.print_exc()
            return None

        if result is None or result == crash_sentinel:
            print(f"[Resimulating] [CASE{case_num}] Resimulation produced NaN.")
            return None

        return result

    case1_alpha_lower_bounds, case1_alpha_upper_bounds = np.array(case1_alpha_bounds[0]), np.array(case1_alpha_bounds[1])
    case2_alpha_lower_bounds, case2_alpha_upper_bounds = np.array(case2_alpha_bounds[0]), np.array(case2_alpha_bounds[1])
    case3_alpha_lower_bounds, case3_alpha_upper_bounds = np.array(case3_alpha_bounds[0]), np.array(case3_alpha_bounds[1])

    case1_omega_lower_bounds, case1_omega_upper_bounds = np.array(case1_omega_bounds[0]), np.array(case1_omega_bounds[1])
    case2_omega_lower_bounds, case2_omega_upper_bounds = np.array(case2_omega_bounds[0]), np.array(case2_omega_bounds[1])
    case3_omega_lower_bounds, case3_omega_upper_bounds = np.array(case3_omega_bounds[0]), np.array(case3_omega_bounds[1])

    case1_alpha_dim = len(case1_alpha_lower_bounds)
    case2_alpha_dim = len(case2_alpha_lower_bounds)
    case3_alpha_dim = len(case3_alpha_lower_bounds)

    case1_omega_dim = len(case1_omega_lower_bounds)
    case2_omega_dim = len(case2_omega_lower_bounds)
    case3_omega_dim = len(case3_omega_lower_bounds)

    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}

    # initialize new swarm
    if custom_alpha_initializer is not None:
        case1_init_alpha_pos = np.stack([custom_alpha_initializer("case1") for _ in range(n_particles)])
    else:
        case1_init_alpha_pos = np.random.uniform(low=case1_alpha_lower_bounds, high=case1_alpha_upper_bounds, size=(n_particles, case1_alpha_dim))

    case1_init_omega_pos = np.random.uniform(low=case1_omega_lower_bounds, high=case1_omega_upper_bounds, size=(n_particles, case1_omega_dim))
    case1_init_pos =  np.hstack((case1_init_alpha_pos, case1_init_omega_pos))
    case1_init_vel = np.zeros_like(case1_init_pos)
    case1_swarm = Swarm(position=case1_init_pos, velocity=case1_init_vel, options=options)
    case1_swarm.pbest_pos = np.copy(case1_init_pos)
    case1_swarm.pbest_cost = np.full(n_particles, np.inf)

    case1_param_names = case1_alpha_names + case1_omega_names # pos: alpha + omega

    if custom_alpha_initializer is not None:
        case2_init_alpha_pos = np.stack([custom_alpha_initializer("case2") for _ in range(n_particles)])
    else:
        case2_init_alpha_pos = np.random.uniform(low=case2_alpha_lower_bounds, high=case2_alpha_upper_bounds, size=(n_particles, case2_alpha_dim))

    case2_init_omega_pos = np.random.uniform(low=case2_omega_lower_bounds, high=case2_omega_upper_bounds, size=(n_particles, case2_omega_dim))
    case2_init_pos =  np.hstack((case2_init_alpha_pos, case2_init_omega_pos))
    case2_init_vel = np.zeros_like(case2_init_pos)
    case2_swarm = Swarm(position=case2_init_pos, velocity=case2_init_vel, options=options)
    case2_swarm.pbest_pos = np.copy(case2_init_pos)
    case2_swarm.pbest_cost = np.full(n_particles, np.inf)

    case2_param_names = case2_alpha_names + case2_omega_names # pos: alpha + omega

    if custom_alpha_initializer is not None:
        case3_init_alpha_pos = np.stack([custom_alpha_initializer("case3") for _ in range(n_particles)])
    else:
        case3_init_alpha_pos = np.random.uniform(low=case3_alpha_lower_bounds, high=case3_alpha_upper_bounds, size=(n_particles, case3_alpha_dim))
    case3_init_omega_pos = np.random.uniform(low=case3_omega_lower_bounds, high=case3_omega_upper_bounds, size=(n_particles, case3_omega_dim))
    case3_init_pos =  np.hstack((case3_init_alpha_pos, case3_init_omega_pos))
    case3_init_vel = np.zeros_like(case3_init_pos)
    case3_swarm = Swarm(position=case3_init_pos, velocity=case3_init_vel, options=options)
    case3_swarm.pbest_pos = np.copy(case3_init_pos)
    case3_swarm.pbest_cost = np.full(n_particles, np.inf)

    case3_param_names = case3_alpha_names + case3_omega_names # pos: alpha + omega

    topology = Star()
    velocity_handler = VelocityHandler(strategy="unmodified")

    case1_gbest_cost = np.inf
    case1_gbest_pos = None
    case2_gbest_cost = np.inf
    case2_gbest_pos = None
    case3_gbest_cost = np.inf
    case3_gbest_pos = None

    case1_cost_history = []
    case1_gbest_iteration_record = None
    case2_cost_history = []
    case2_gbest_iteration_record = None
    case3_cost_history = []
    case3_gbest_iteration_record = None

    case1_gbest_z01_history = []
    case1_gbest_penalty_history = []
    case2_gbest_z01_history = []
    case2_gbest_penalty_history = []
    case3_gbest_z01_history = []
    case3_gbest_penalty_history = []

    case1_current_z01 = np.full(n_particles, np.nan)
    case1_current_penalty = np.full(n_particles, np.nan)
    case1_pbest_z01 = np.full(n_particles, np.nan)
    case1_pbest_penalty = np.full(n_particles, np.nan)

    case2_current_z01 = np.full(n_particles, np.nan)
    case2_current_penalty = np.full(n_particles, np.nan)
    case2_pbest_z01 = np.full(n_particles, np.nan)
    case2_pbest_penalty = np.full(n_particles, np.nan)

    case3_current_z01 = np.full(n_particles, np.nan)
    case3_current_penalty = np.full(n_particles, np.nan)
    case3_pbest_z01 = np.full(n_particles, np.nan)
    case3_pbest_penalty = np.full(n_particles, np.nan)

    case1_lower_bounds = np.hstack((case1_alpha_lower_bounds, case1_omega_lower_bounds))
    case1_upper_bounds = np.hstack((case1_alpha_upper_bounds, case1_omega_upper_bounds))
    case1_dim = len(case1_lower_bounds)

    case2_lower_bounds = np.hstack((case2_alpha_lower_bounds, case2_omega_lower_bounds))
    case2_upper_bounds = np.hstack((case2_alpha_upper_bounds, case2_omega_upper_bounds))
    case2_dim = len(case2_lower_bounds)

    case3_lower_bounds = np.hstack((case3_alpha_lower_bounds, case3_omega_lower_bounds))
    case3_upper_bounds = np.hstack((case3_alpha_upper_bounds, case3_omega_upper_bounds))
    case3_dim = len(case3_lower_bounds)

    for iter_idx in range(n_iterations):
        if verbose:
            print(f"Iteration {iter_idx + 1}/{n_iterations}------------------------------------------------------")
        case1_cost = np.zeros(n_particles)
        case2_cost = np.zeros(n_particles)
        case3_cost = np.zeros(n_particles)

        for idx in range(n_particles):
            if verbose:
                print(f"Processing particle {idx}")
            retry_count = 0
            case1_fom, case1_z01_loss = None, None
            case2_fom, case2_z01_loss = None, None
            case3_fom, case3_z01_loss = None, None

            original_case1_particle = case1_swarm.position[idx]
            original_case2_particle = case2_swarm.position[idx]
            original_case3_particle = case3_swarm.position[idx]

            original_case1_velocity = case1_swarm.velocity[idx]
            original_case2_velocity = case2_swarm.velocity[idx]
            original_case3_velocity = case3_swarm.velocity[idx]

            while retry_count < max_retry_per_particle:
                case1_particle = case1_swarm.position[idx]
                case2_particle = case2_swarm.position[idx]
                case3_particle = case3_swarm.position[idx]

                if len(case1_particle) != len(case1_param_names):
                    print(f"[ERROR] Case1 Particle length {len(case1_particle)} does not match param names length {len(case1_param_names)}")
                    pdb.set_trace()
                if len(case2_particle) != len(case2_param_names):
                    print(f"[ERROR] Case2 Particle length {len(case2_particle)} does not match param names length {len(case2_param_names)}")
                    pdb.set_trace()
                if len(case3_particle) != len(case3_param_names):
                    print(f"[ERROR] Case3 Particle length {len(case3_particle)} does not match param names length {len(case3_param_names)}")
                    pdb.set_trace()
                
                case1_particle_dict = dict(zip(case1_param_names, case1_particle)) if case1_param_names else {}
                case2_particle_dict = dict(zip(case2_param_names, case2_particle)) if case2_param_names else {}
                case3_particle_dict = dict(zip(case3_param_names, case3_particle)) if case3_param_names else {}

                case1_alpha_dict = {k: v for k, v in case1_particle_dict.items() if k in case1_alpha_names}
                case2_alpha_dict = {k: v for k, v in case2_particle_dict.items() if k in case2_alpha_names}
                case3_alpha_dict = {k: v for k, v in case3_particle_dict.items() if k in case3_alpha_names}

                case1_omega_dict = {k: v for k, v in case1_particle_dict.items() if k in case1_omega_names}
                case2_omega_dict = {k: v for k, v in case2_particle_dict.items() if k in case2_omega_names}
                case3_omega_dict = {k: v for k, v in case3_particle_dict.items() if k in case3_omega_names}

                alpha_particle_dict = {**case1_alpha_dict, **case2_alpha_dict, **case3_alpha_dict}
                omega_particle_dict = {**case1_omega_dict, **case2_omega_dict, **case3_omega_dict}

                try:
                    result = objective_fn(inner_iter=iter_idx, particle_idx=idx, alpha_vars=alpha_particle_dict, omega_vars=omega_particle_dict)
                except Exception as e:
                    print(f"[EXCEPTION] Particle {idx} crashed: {e}")
                    traceback.print_exc()
                    result = crash_sentinel
                    raise
                
                retry_needed = False

                if result == crash_sentinel: # crashed not because of Neg Z or NaN
                    print(f"[EXCEPTION] Particle {idx} crashed. Retrying... ({retry_count+1})")
                    retry_needed = True

                else:
                    case1_fom, case2_fom, case3_fom, case1_weighted_z01_loss, case2_weighted_z01_loss, case3_weighted_z01_loss, case1_weighted_penalty, case2_weighted_penalty, case3_weighted_penalty = result
                    case_foms = [case1_fom, case2_fom, case3_fom]
                    case_losses = [case1_weighted_z01_loss, case2_weighted_z01_loss, case3_weighted_z01_loss]
                    case_penalties = [case1_weighted_penalty, case2_weighted_penalty, case3_weighted_penalty]
                    case_dicts = [case1_particle_dict, case2_particle_dict, case3_particle_dict]
                    case_dicts_dims = [case1_dim, case2_dim, case3_dim]
                    retry_succeeded_dicts = []

                    if any(fom == nan_z_fom for fom in case_foms): # at least one of the cases produced NaN
                        print(f"[NaN] Particle {idx} produced NaN data. Retrying... ({retry_count+1}/{max_retry_per_particle})")
                        retry_needed = True

                    elif any(fom is None for fom in case_foms): # at least one of the cases produced Neg Z
                            for i, fom in enumerate(case_foms):
                                # Resimulate the case that is not responsible for Neg Z
                                if fom is None:
                                    print(f"[NegZ] [CASE{i+1}] Resimulating possible VALID particle {idx}...")
                                    valid_particle_dict = case_dicts[i].copy()
                                    if len(valid_particle_dict) != case_dicts_dims[i]:
                                        print(f"Using incorrect particle_dict for case {i+1}, expected {case_dicts_dims[i]} keys, got {len(valid_particle_dict)} keys.")
                                        pdb.set_trace()
                                    if retry_succeeded_dicts:
                                        dicts = retry_succeeded_dicts[-1]
                                        valid_particle_dict= {**valid_particle_dict, **dicts}
                                        print(f"[NegZ] [CASE{i+1}] Merging VALID particle dicts from previous retries") ### JUST FOR DEBUGGING
                                    result = resimulate_case(i+1, valid_particle_dict, idx, iter_idx)
                                    if result is None:
                                        retry_needed = True
                                        print(f"[NegZ] [CASE{i+1}] Resimulation crashed, retrying all cases...")
                                        break
                                    retry_foms = result[:3]
                                    retry_losses = result[3:6]
                                    retry_penalties = result[6:]

                                    if retry_foms[i] is None or retry_foms[i] == nan_z_fom:
                                        retry_needed = True
                                        print(f"[NegZ] [CASE{i+1}] Resimulation produced NaN, retrying all cases...")
                                        break
                                    case_foms[i] = retry_foms[i]
                                    case_losses[i] = retry_losses[i]
                                    case_penalties[i] = retry_penalties[i]
                                    retry_succeeded_dicts.append(valid_particle_dict)

                    elif any(math.isnan(fom) for fom in case_foms):
                        print(f"[NaN] Particle {idx} produced FoM of NaN. Retrying... ({retry_count+1}/{max_retry_per_particle})")
                        retry_needed = True

                if retry_needed:
                    if iter_idx == 0: # if first iteration, randomly generate new point for simulation
                        case1_swarm.position[idx] = np.random.uniform(low=case1_lower_bounds, high=case1_upper_bounds, size=(case1_dim,))
                        case2_swarm.position[idx] = np.random.uniform(low=case2_lower_bounds, high=case2_upper_bounds, size=(case2_dim,))
                        case3_swarm.position[idx] = np.random.uniform(low=case3_lower_bounds, high=case3_upper_bounds, size=(case3_dim,))
                    else:
                        scale_factor = 0.5 ** retry_count
                        new_case1_position = original_case1_particle + scale_factor * original_case1_velocity
                        new_case1_position = np.clip(new_case1_position, case1_lower_bounds, case1_upper_bounds)
                        case1_swarm.position[idx] = new_case1_position

                        new_case2_position = original_case2_particle + scale_factor * original_case2_velocity
                        new_case2_position = np.clip(new_case2_position, case2_lower_bounds, case2_upper_bounds)
                        case2_swarm.position[idx] = new_case2_position

                        new_case3_position = original_case3_particle + scale_factor * original_case3_velocity
                        new_case3_position = np.clip(new_case3_position, case3_lower_bounds, case3_upper_bounds)
                        case3_swarm.position[idx] = new_case3_position

                    print(f"[Retry] Particle {idx} retrying with new positions")
                    retry_count += 1
                else:
                    case1_fom, case2_fom, case3_fom = case_foms
                    case1_weighted_z01_loss, case2_weighted_z01_loss, case3_weighted_z01_loss = case_losses
                    case1_weighted_penalty, case2_weighted_penalty, case3_weighted_penalty = case_penalties
                    break

            # If ended with no succeeded resimulation within max_retry_per_particle
            # Retry cases that are not responsible for NaN
            final_retry_succeeded_dicts = []
            if any(fom is None for fom in case_foms) and nan_z_fom in case_foms:
                print(f"[FAIL] Particle {idx} failed after {max_retry_per_particle} retries.")
                for i, fom in enumerate(case_foms):
                    if fom is None:
                        print(f"[NaN] [CASE{i+1}] Resimulating possible VALID particle {idx}...")
                        final_valid_particle_dict = case_dicts[i].copy()
                        if len(final_valid_particle_dict) != case_dicts_dims[i]:
                            print(f"Using incorrect particle_dict for case {i+1}, expected {case_dicts_dims[i]} keys, got {len(final_valid_particle_dict)} keys.")
                            pdb.set_trace()
                        if final_retry_succeeded_dicts:
                            final_dicts = final_retry_succeeded_dicts[-1]
                            final_valid_particle_dict= {**final_valid_particle_dict, **final_dicts}
                            print(f"[NaN] [CASE{i+1}] Merging VALID particle dicts from previous retries") ### JUST FOR DEBUGGING
                        result = resimulate_case(i+1, final_valid_particle_dict, idx, iter_idx)
                        if result is None:
                            print(f"[NaN] [CASE{i+1}] Resimulation crashed...")
                            break
                        retry_foms = result[:3]
                        retry_losses = result[3:6]
                        retry_penalties = result[6:]

                        if retry_foms[i] is None or retry_foms[i] == nan_z_fom:
                            retry_needed = True
                            print(f"[NaN] [CASE{i+1}] Resimulation produced NaN...")
                            break
                        case_foms[i] = retry_foms[i]
                        case_losses[i] = retry_losses[i]
                        case_penalties[i] = retry_penalties[i]
                        final_retry_succeeded_dicts.append(final_valid_particle_dict)
                case1_fom, case2_fom, case3_fom = case_foms
                case1_weighted_z01_loss, case2_weighted_z01_loss, case3_weighted_z01_loss = case_losses
                case1_weighted_penalty, case2_weighted_penalty, case3_weighted_penalty = case_penalties

            if case1_fom is None or case1_weighted_z01_loss is None:
                print(f"[FAIL][CASE1] Particle {idx} failed after {max_retry_per_particle} retries.")
                case1_cost[idx] = - nan_z_fom
                case1_current_z01[idx] = np.nan
                case1_current_penalty[idx] = np.nan
            else:
                case1_cost[idx] = - (case1_fom - case1_weighted_z01_loss - case1_weighted_penalty)
                case1_current_z01[idx] = case1_weighted_z01_loss
                case1_current_penalty[idx] = case1_weighted_penalty

            if case2_fom is None or case2_weighted_z01_loss is None:
                print(f"[FAIL][CASE2] Particle {idx} failed after {max_retry_per_particle} retries.")
                case2_cost[idx] = - nan_z_fom
                case2_current_z01[idx] = np.nan
                case2_current_penalty[idx] = np.nan
            else:
                case2_cost[idx] = - (case2_fom - case2_weighted_z01_loss - case2_weighted_penalty)
                case2_current_z01[idx] = case2_weighted_z01_loss
                case2_current_penalty[idx] = case2_weighted_penalty

            if case3_fom is None or case3_weighted_z01_loss is None:
                print(f"[FAIL][CASE3] Particle {idx} failed after {max_retry_per_particle} retries.")
                case3_cost[idx] = - nan_z_fom
                case3_current_z01[idx] = np.nan
                case3_current_penalty[idx] = np.nan
            else:
                case3_cost[idx] = - (case3_fom - case3_weighted_z01_loss - case3_weighted_penalty)
                case3_current_z01[idx] = case3_weighted_z01_loss
                case3_current_penalty[idx] = case3_weighted_penalty

            if verbose:
                print(f"Case1 Particle {idx} cost: {case1_cost[idx]}, fom:{case1_fom}, z01_loss: {case1_weighted_z01_loss}, penalty: {case1_weighted_penalty}")
                print(f"Case2 Particle {idx} cost: {case2_cost[idx]}, fom:{case2_fom}, z01_loss: {case2_weighted_z01_loss}, penalty: {case2_weighted_penalty}")
                print(f"Case3 Particle {idx} cost: {case3_cost[idx]}, fom:{case3_fom}, z01_loss: {case3_weighted_z01_loss}, penalty: {case3_weighted_penalty}")

        # Update personal best
        case1_better_mask = case1_cost < case1_swarm.pbest_cost
        case1_swarm.pbest_cost[case1_better_mask] = case1_cost[case1_better_mask]
        case1_swarm.pbest_pos[case1_better_mask] = case1_swarm.position[case1_better_mask]
        case1_pbest_z01[case1_better_mask] = case1_current_z01[case1_better_mask]
        case1_pbest_penalty[case1_better_mask] = case1_current_penalty[case1_better_mask]

        case2_better_mask = case2_cost < case2_swarm.pbest_cost
        case2_swarm.pbest_cost[case2_better_mask] = case2_cost[case2_better_mask]
        case2_swarm.pbest_pos[case2_better_mask] = case2_swarm.position[case2_better_mask]
        case2_pbest_z01[case2_better_mask] = case2_current_z01[case2_better_mask]
        case2_pbest_penalty[case2_better_mask] = case2_current_penalty[case2_better_mask]

        case3_better_mask = case3_cost < case3_swarm.pbest_cost
        case3_swarm.pbest_cost[case3_better_mask] = case3_cost[case3_better_mask]
        case3_swarm.pbest_pos[case3_better_mask] = case3_swarm.position[case3_better_mask]
        case3_pbest_z01[case3_better_mask] = case3_current_z01[case3_better_mask]
        case3_pbest_penalty[case3_better_mask] = case3_current_penalty[case3_better_mask]

        if verbose:
            print(f"Case1 Personal best costs updated: {case1_swarm.pbest_cost}")
            print(f"Case2 Personal best costs updated: {case2_swarm.pbest_cost}")
            print(f"Case3 Personal best costs updated: {case3_swarm.pbest_cost}")

        # Update global best
        case1_gbest_idx = np.argmin(case1_swarm.pbest_cost)
        if case1_swarm.pbest_cost[case1_gbest_idx] < case1_gbest_cost:
            case1_swarm.best_cost = case1_swarm.pbest_cost[case1_gbest_idx]
            case1_swarm.best_pos = case1_swarm.pbest_pos[case1_gbest_idx]
            case1_gbest_cost = case1_swarm.best_cost
            case1_gbest_pos = case1_swarm.best_pos
            case1_gbest_iteration_record = (iter_idx, case1_gbest_idx)

            if verbose:
                print(f"Case1 Global best cost updated: {case1_gbest_cost}")
        
        case2_gbest_idx = np.argmin(case2_swarm.pbest_cost)
        if case2_swarm.pbest_cost[case2_gbest_idx] < case2_gbest_cost:
            case2_swarm.best_cost = case2_swarm.pbest_cost[case2_gbest_idx]
            case2_swarm.best_pos = case2_swarm.pbest_pos[case2_gbest_idx]
            case2_gbest_cost = case2_swarm.best_cost
            case2_gbest_pos = case2_swarm.best_pos
            case2_gbest_iteration_record = (iter_idx, case2_gbest_idx)

            if verbose:
                print(f"Case2 Global best cost updated: {case2_gbest_cost}")
        
        case3_gbest_idx = np.argmin(case3_swarm.pbest_cost)
        if case3_swarm.pbest_cost[case3_gbest_idx] < case3_gbest_cost:
            case3_swarm.best_cost = case3_swarm.pbest_cost[case3_gbest_idx]
            case3_swarm.best_pos = case3_swarm.pbest_pos[case3_gbest_idx]
            case3_gbest_cost = case3_swarm.best_cost
            case3_gbest_pos = case3_swarm.best_pos
            case3_gbest_iteration_record = (iter_idx, case3_gbest_idx)

            if verbose:
                print(f"Case3 Global best cost updated: {case3_gbest_cost}")

        case1_cost_history.append(case1_gbest_cost)
        case2_cost_history.append(case2_gbest_cost)
        case3_cost_history.append(case3_gbest_cost)
        case1_gbest_z01_history.append(case1_pbest_z01[case1_gbest_idx])
        case1_gbest_penalty_history.append(case1_pbest_penalty[case1_gbest_idx])
        case2_gbest_z01_history.append(case2_pbest_z01[case2_gbest_idx])
        case2_gbest_penalty_history.append(case2_pbest_penalty[case2_gbest_idx])
        case3_gbest_z01_history.append(case3_pbest_z01[case3_gbest_idx])
        case3_gbest_penalty_history.append(case3_pbest_penalty[case3_gbest_idx])

        bh=BoundaryHandler(strategy="nearest")

        # Update velocity and position using backend API
        case1_swarm.velocity = topology.compute_velocity(
            swarm=case1_swarm,
            clamp=None,
            vh=velocity_handler,
            bounds=(case1_lower_bounds, case1_upper_bounds)
        )
        case1_swarm.position = topology.compute_position(case1_swarm, bounds=(case1_lower_bounds, case1_upper_bounds), bh=bh)

        if np.any(case1_swarm.position < case1_lower_bounds) or np.any(case1_swarm.position > case1_upper_bounds):
            print(f"[WARNING] Case1 Next Particle out of bounds!")

        case2_swarm.velocity = topology.compute_velocity(
            swarm=case2_swarm,
            clamp=None,
            vh=velocity_handler,
            bounds=(case2_lower_bounds, case2_upper_bounds)
        )
        case2_swarm.position = topology.compute_position(case2_swarm, bounds=(case2_lower_bounds, case2_upper_bounds), bh=bh)

        if np.any(case2_swarm.position < case2_lower_bounds) or np.any(case2_swarm.position > case2_upper_bounds):
            print(f"[WARNING] Case2 Next Particle out of bounds!")

        case3_swarm.velocity = topology.compute_velocity(
            swarm=case3_swarm,
            clamp=None,
            vh=velocity_handler,
            bounds=(case3_lower_bounds, case3_upper_bounds)
        )
        case3_swarm.position = topology.compute_position(case3_swarm, bounds=(case3_lower_bounds, case3_upper_bounds), bh=bh)

        if np.any(case3_swarm.position < case3_lower_bounds) or np.any(case3_swarm.position > case3_upper_bounds):
            print(f"[WARNING] Case3 Next Particle out of bounds!")

    gbest_cost = [case1_gbest_cost, case2_gbest_cost, case3_gbest_cost]

    gbest_pos = {
        'case1': case1_gbest_pos,
        'case2': case2_gbest_pos,
        'case3': case3_gbest_pos
    }

    gbest_iteration_record = {
        'case1': case1_gbest_iteration_record,
        'case2': case2_gbest_iteration_record,
        'case3': case3_gbest_iteration_record
    }

    cost_history = {
        'case1': case1_cost_history,
        'case2': case2_cost_history,
        'case3': case3_cost_history
    }

    gbest_z01_history = {
        'case1': case1_gbest_z01_history,
        'case2': case2_gbest_z01_history,
        'case3': case3_gbest_z01_history
    }

    gbest_penalty_history= {
        'case1': case1_gbest_penalty_history,
        'case2': case2_gbest_penalty_history,
        'case3': case3_gbest_penalty_history
    }

    swarm = [case1_swarm, case2_swarm, case3_swarm]

    return gbest_cost, gbest_pos, cost_history, gbest_z01_history, gbest_penalty_history, gbest_iteration_record, swarm
