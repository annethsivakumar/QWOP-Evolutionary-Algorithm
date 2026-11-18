"""
======================================
Evolves 60-second Q/W/O/P input sequences for the pymunk/pyglet QWOP clone.
--------------------------------------
Put this file in the repo root (next to character.py, game_class.py, assets/).
    python qwop_ea.py train            # run the EA headless (fast, multi-core)
    python qwop_ea.py play             # open the game window, press S to auto-play the best runner
    python qwop_ea.py baseline         # random-search baseline
--------------------------------------
  Genotype  : (N, 4) bit array, N = MAX_TIME / DT; each row = [Q, W, O, P]     -> random_genome()
  Phenotype : the timed key presses replayed in the game                       -> simulate() / AutoPlayGame
  Fitness   : < 100 m maximise distance; >= 100 m minimise finish time         -> fitness_from_result()
  EA loop   : initialise -> evaluate -> select -> vary -> replace              -> run_ea()
======================================
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from types import SimpleNamespace
import numpy as np

# =====================================================================
# Settings (everything user may want to tune lives here)
# =====================================================================
# ---- Encoding -------------------------------------------------------
# seconds each 4-bit input is held
DT = 0.1     
# seconds per run                      
MAX_TIME = 60.0
# genome length N = 600 steps
N_STEPS = int(round(MAX_TIME / DT))

# ---- Simulation (mirrors game_class.py / qwop.py exactly) -----------
# game.update() is scheduled every 0.01 s of sim time
TICK = 0.01  
# space.step(1/50/10/2) is called 10x per tick               
SUBSTEPS = 10             
SUBSTEP_DT = 1 / 50 / 10 / 2
TICKS_PER_STEP = int(round(DT / TICK))
# default pyglet window width -> start x = WORLD_W // 2
WORLD_W = 640
# the game's "meters" conversion               
PX_TO_M = 1.25 / 200     
GOAL_M = 100.0

# Which body point counts as "distance". Measuring the torso lets a runner score by
# lunging forward and falling, so by default the average of both feet is used:
#   "torso"     - the game's own meter (torso x position)
#   "feet_mean" - average x of the two feet (default)
#   "feet_min"  - the rearmost foot (strictest: both feet must advance)
PROGRESS_MEASURE = "feet_mean"

# Early stop: if the best distance has not improved by STALL_GAIN_M 
# for STALL_SECONDS the runner is "stuck" and the run ends. 
# Set STALL_SECONDS = None to always run the full MAX_TIME.
STALL_SECONDS = 3.0
STALL_GAIN_M = 0.1

# ---- Anti-exploit rules (no airtime) --------------------------------
# The runner must keep at least one leg part (thigh, calf or foot) touching the
# floor. If every leg part is off the floor for longer than AIR_GRACE_SECONDS the
# run ends, and distance is only credited on ticks where the runner is touching
# the floor. This stops the "launch yourself through the air" exploit.
NO_AIRTIME = True
AIR_GRACE_SECONDS = 0.25
# End the run if the measured forward speed exceeds
# the maximum permitted speed over one DT interval.
MAX_SPEED_MPS = 12.0

# ---- Evolutionary algorithm ----------------------------------------
POP_SIZE = 500
GENERATIONS = 500
# best individuals copied unchanged into the next generation
ELITES = 10
# selection pressure (same value as generate_samples.py)                  
TOURNAMENT_SIZE = 3 
# probability a child is made by crossover (else clone of a parent)        
CROSSOVER_RATE = 0.3    
# per-step probability a whole 4-bit input is re-randomised   
MUTATION_RATE = 0.03 
# per-bit flip probability (finer-grained mutation       
BIT_FLIP_RATE = 0.01
# only mutate/cross) inside the part of the genome that was actually       
ACTIVE_ONLY = True          
# played before the runner fell (genes after the fall are never used)
# ... plus this many extra steps after the fall point
ACTIVE_MARGIN = 10          
# set an int for reproducible runs
SEED = 16                 
# fraction of every new generation replaced by brand-new random runners (keeps diversity)
IMMIGRANT_FRACTION = 0.25
WORKERS = max(1, cpu_count() - 1)

# ---- Files ----------------------------------------------------------
OUT_DIR = "ea_output"
BEST_FILE = os.path.join(OUT_DIR, "best_genome.npy")
HISTORY_FILE = os.path.join(OUT_DIR, "best_per_generation.npz")
LOG_FILE = os.path.join(OUT_DIR, "fitness_log.csv")
PLOT_FILE = os.path.join(OUT_DIR, "fitness_vs_generation.png")
VIDEO_DIR = os.path.join(OUT_DIR, "videos")      
VIDEO_FPS = 25                                    
RECORD_TAIL_SECONDS = 1.0                         


# =====================================================================
# Headless simulator (same physics as the game, no window needed)
# =====================================================================
class RunMonitor:
    """
    Track floor contact, airtime and speed for one run (headless or visual).

    Parameters:
        None.

    Returns:
        None.
    """

    def __init__(self):
        """
        Create an empty monitor.

        Parameters:
            None.

        Returns:
            None.
        """
        self.contacts = 0
        self.ever_grounded = False
        self.air_ticks = 0
        self.dist_history = []

    def attach(self, space):
        """
        Count contacts between the floor (type 100) and leg parts (type 2).

        Parameters:
            space (pymunk.Space): Physics space to watch.

        Returns:
            None.
        """
        space.on_collision(collision_type_a=100, collision_type_b=2,
                           begin=self.on_touch, separate=self.on_leave)

    def on_touch(self, arbiter, space, data):
        """
        Record that a leg part started touching the floor.

        Parameters:
            arbiter (pymunk.Arbiter): Collision information.
            space (pymunk.Space): Physics space.
            data (dict): Collision handler data.

        Returns:
            None.
        """
        self.contacts += 1
        self.ever_grounded = True

    def on_leave(self, arbiter, space, data):
        """
        Record that a leg part stopped touching the floor.

        Parameters:
            arbiter (pymunk.Arbiter): Collision information.
            space (pymunk.Space): Physics space.
            data (dict): Collision handler data.

        Returns:
            None.
        """
        self.contacts = max(0, self.contacts - 1)

    @property
    def grounded(self):
        """
        Whether any leg part is currently touching the floor.

        Parameters:
            None.

        Returns:
            bool: True if at least one leg part touches the floor.
        """
        return self.contacts > 0

    def check(self, dist):
        """
        Update airtime and speed after one tick and report any rule violation.

        Parameters:
            dist (float): Current distance in metres.

        Returns:
            str | None: "airborne", "too_fast", or None if the run is still legal.
        """
        self.dist_history.append(dist)
        if len(self.dist_history) > TICKS_PER_STEP + 1:
            self.dist_history.pop(0)

        if NO_AIRTIME:
            # airtime only counts once the runner has landed for the first time
            if self.ever_grounded and not self.grounded:
                self.air_ticks += 1
            else:
                self.air_ticks = 0
            if self.air_ticks * TICK > AIR_GRACE_SECONDS:
                return "airborne"

        if MAX_SPEED_MPS and len(self.dist_history) > TICKS_PER_STEP:
            speed = (self.dist_history[-1] - self.dist_history[0]) / (TICKS_PER_STEP * TICK)
            if speed > MAX_SPEED_MPS:
                return "too_fast"
        return None


def _build_headless_world(render=False):
    """
    Create the QWOP physics world without opening a window.

    Parameters:
        render (bool): Whether to create graphical sprites.
    
    Returns:
        tuple[pymunk.Space, Character, SimpleNamespace, int]: Physics space, runner, fall state, and starting x-position.
    """
    import pymunk
    from pymunk.vec2d import Vec2d
    import character as character_module        

    state = SimpleNamespace(fallen=False, monitor=RunMonitor())

    space = pymunk.Space()
    space.gravity = (0, -9820)
    space.damping = 0.99

    def hit_ground(arbiter, space_, data):
        """
        Mark the runner as fallen when its body contacts the floor.
        
        Parameters:
            arbiter (pymunk.Arbiter): Collision information.
            space_ (pymunk.Space): Physics space.
            data (dict): Collision handler data.

        Returns:
            None.
        """
        state.fallen = True

    # torso/head have collision_type 1, floor has 100 (same as the game)
    space.on_collision(collision_type_a=100, collision_type_b=1, begin=hit_ground)
    state.monitor.attach(space)

    floor_height = 10
    floor = pymunk.Segment(
        space.static_body,
        Vec2d(-WORLD_W * 100, floor_height),
        Vec2d(WORLD_W * 100, floor_height),
        1,
    )
    floor.friction = 10.3
    floor.collision_type = 100
    space.add(floor)

    body_w, body_h = 100, 200
    body_x = WORLD_W // 2
    body_y = floor_height + body_h + body_h / 8 + 10

    # Character.__init__ loads sprites through character.load_sprite, which needs an
    # OpenGL window. Swap in a stub just for this call (and restore it afterwards so
    # the visual game in the same process is unaffected).
    original = character_module.load_sprite
    if not render:
        character_module.load_sprite = lambda name, body, offset=(0, 0): SimpleNamespace(
            body=body, offset=offset
        )
    try:
        char = character_module.Character(space, body_x, body_y, body_w, body_h)
    finally:
        character_module.load_sprite = original

    return space, char, state, body_x


def progress_x(char):
    """
    Return the x-position (pixels) used to measure how far the runner has travelled.

    Parameters:
        char (Character): Runner to measure.

    Returns:
        float: Torso x, mean foot x, or rearmost foot x, depending on PROGRESS_MEASURE.
    """
    if PROGRESS_MEASURE == "feet_mean":
        return (char.footL.position.x + char.footR.position.x) / 2.0
    if PROGRESS_MEASURE == "feet_min":
        return min(char.footL.position.x, char.footR.position.x)
    return char.get_position()[0]


def apply_keys(char, keys):
    """
    Apply the Q, W, O, and P movements indicated by four bits.
    
    Parameters:
        char (Character): Runner to control.
        keys (np.ndarray): Four binary keyboard inputs.

    Returns:
        None.
    """
    if keys[0]:
        char.move_thighL()
    if keys[1]:
        char.move_thighR()
    if keys[2]:
        char.move_calfL()
    if keys[3]:
        char.move_calfR()


def simulation_ticks(genome, render=False):
    """
    Simulate a genome and measure the runner’s performance.
    
    Parameters:
        genome (np.ndarray): Sequence of four-bit QWOP inputs.
        render (bool): Whether to create graphical sprites.
    
    Returns:
        dict[str, float | int | bool | None]: Distance, elapsed time, finish and fall status, steps used, and why the run ended.
    """
    space, char, state, x0 = _build_headless_world(render=render)
    monitor = state.monitor
    # distance is measured from the runner's start with the same measure used below
    x0 = progress_x(char)

    max_ticks = int(round(MAX_TIME / TICK))
    # best distance reached while touching the floor
    furthest = -1e9
    # reference point for stall detection
    ref_dist = -1e9
    ref_tick = 0
    finish_time = None
    steps_used = 0
    end_reason = "timeout"
    tick = 0

    yield space, char, None

    for tick in range(max_ticks):
        step_idx = min(tick // TICKS_PER_STEP, len(genome) - 1)
        apply_keys(char, genome[step_idx])
        for _ in range(SUBSTEPS):
            space.step(SUBSTEP_DT)

        steps_used = step_idx + 1
        dist = (progress_x(char) - x0) * PX_TO_M

        violation = monitor.check(dist)
        if violation:
            end_reason = violation
        else:
            if monitor.grounded or not monitor.ever_grounded:
                furthest = max(furthest, dist)
            if dist > ref_dist + STALL_GAIN_M:
                ref_dist, ref_tick = dist, tick
            if dist >= GOAL_M and monitor.grounded:
                finish_time = (tick + 1) * TICK
                end_reason = "finished"
            elif state.fallen:
                end_reason = "fell"
            elif STALL_SECONDS and (tick - ref_tick) * TICK > STALL_SECONDS:
                end_reason = "stalled"

        done = end_reason != "timeout" or tick + 1 == max_ticks
        result = {
            "distance": float(furthest if furthest >= -1e8 else 0.0),
            "time": float((tick + 1) * TICK),
            "finished": finish_time is not None,
            "finish_time": finish_time,
            "fell": state.fallen,
            "steps_used": int(steps_used),
            "end_reason": end_reason,
            "airborne": end_reason in ("airborne", "too_fast"),
        }
        yield space, char, (result, done)
        if done:
            return


def simulate(genome):
    """Evaluate a genome using the shared physics simulation."""
    for _, _, frame in simulation_ticks(genome):
        if frame is not None and frame[1]:
            return frame[0]


# =====================================================================
# Fitness: before 100 m maximise distance, at 100 m minimise time
# =====================================================================
def fitness_from_result(res):
    """
    Score a run by distance, or by completion time after reaching the goal.
    
    Parameters:
        res (dict): Simulation result from simulate().
    
    Returns:
        float: Fitness score to maximize.
    """
    if res["finished"]:
        return GOAL_M + (MAX_TIME - res["finish_time"])

    distance = max(0.0, res["distance"])
    progress = min(distance / GOAL_M, 1.0)
    survival = min(res["time"] / MAX_TIME, 1.0)

    fitness = GOAL_M * progress * (1 - 0.1 * (1 - survival))

    return min(fitness, GOAL_M - 1e-6)


def evaluate(genome):
    """
    Simulate a genome and calculate its fitness.
    
    Parameters:
        genome (np.ndarray): QWOP input sequence.
    
    Returns:
        tuple[float, dict]: Fitness score and simulation result.
    """
    res = simulate(genome)
    return fitness_from_result(res), res


# =====================================================================
# Evolutionary operators
# =====================================================================
def random_genome(rng):
    """
    Generate a random sequence of four-bit QWOP inputs.
    
    Parameters:
        rng (np.random.Generator): Random number generator.
    
    Returns:
        np.ndarray: Binary genome with shape (N_STEPS, 4).
    """
    return rng.integers(0, 2, size=(N_STEPS, 4), dtype=np.uint8)


def tournament_select(pop, fits, rng):
    """
    Select the fittest individual from a random tournament.
    
    Parameters:
        pop (list[np.ndarray]): Population of genomes.
        fits (np.ndarray): Fitness scores for the population.
        rng (np.random.Generator): Random number generator.
    
    Returns:
        int: Index of the selected individual.
    """
    idx = rng.integers(0, len(pop), size=TOURNAMENT_SIZE)
    winner = idx[np.argmax([fits[i] for i in idx])]
    return winner


def crossover(a, b, limit, rng):
    """
    Combine two genomes using one-point crossover.
    
    Parameters:
        a (np.ndarray): First parent genome.
        b (np.ndarray): Second parent genome.
        limit (int): Upper bound for the crossover point.
        rng (np.random.Generator): Random number generator.
    
    Returns:
        np.ndarray: Child genome formed from both parents.
    """
    limit = int(np.clip(limit, 2, len(a)))
    cut = int(rng.integers(1, limit))
    child = np.concatenate([a[:cut], b[cut:]], axis=0)
    return child


def mutate(g, limit, rng):
    """
    Randomize input steps and flip individual key bits within a limit.
    
    Parameters:
        g (np.ndarray): Genome to mutate.
        limit (int): Number of initial steps eligible for mutation.
        rng (np.random.Generator): Random number generator.
    
    Returns:
        np.ndarray: Mutated copy of the genome.
    """
    g = g.copy()
    limit = int(np.clip(limit, 1, len(g)))

    if rng.random() < 0.8:
        start = max(0, limit - 20)
    else:
        start = 0

    for i in range(start, limit):
        if rng.random() < MUTATION_RATE:
            g[i] = rng.integers(0, 2, size=4)

        for j in range(4):
            if rng.random() < BIT_FLIP_RATE:
                g[i, j] ^= 1

    return g


def active_limit(res):
    """
    Determine how many genome steps are eligible for variation.
    
    Parameters:
        res (dict): Simulation result containing steps_used.
    
    Returns:
        int: Active step limit, including the configured margin.
    """
    if not ACTIVE_ONLY:
        return N_STEPS
    return min(N_STEPS, res["steps_used"] + ACTIVE_MARGIN)


# =====================================================================
# Parallel evaluation
# =====================================================================
def evaluate_population(pool, genomes):
    """
    Evaluate genomes sequentially or using a worker pool.
    
    Parameters:
        pool (multiprocessing.pool.Pool | None): Optional parallel worker pool.
        genomes (list[np.ndarray]): Genomes to evaluate.
    
    Returns:
        list[tuple[float, dict]]: Fitness and simulation result for each genome.
    """
    if pool is None:
        return [evaluate(g) for g in genomes]
    return pool.map(evaluate, genomes, chunksize=max(1, len(genomes) // (WORKERS * 4)))


# =====================================================================
# The EA
# =====================================================================
def run_ea(generations=GENERATIONS, pop_size=POP_SIZE, workers=WORKERS, seed=SEED,
           evaluator=evaluate_population, verbose=True, out_dir=OUT_DIR):
    """
    Evolve QWOP input sequences and save training results.
    
    Parameters:
        generations (int): Number of evolutionary generations.
        pop_size (int): Number of genomes per generation.
        workers (int): Number of parallel workers.
        seed (int | None): Optional random seed.
        evaluator (callable): Function for evaluating the population.
        verbose (bool): Whether to print generation progress.
        out_dir (str): Folder for this run's genome, history, log and plot files.
    
    Returns:
        tuple[tuple[float, np.ndarray, dict], list[dict]]: Best fitness/genome/result and per-generation logs.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 1. Initialise
    pop = [random_genome(rng) for _ in range(pop_size)]

    if os.path.exists(BEST_FILE):
        best_genome = np.load(BEST_FILE)

        if best_genome.shape == (N_STEPS, 4):
            pop[0] = best_genome.copy()

            best_result = simulate(best_genome)

            print("Previous best genome loaded successfully!")
            print("Previous distance:", best_result["distance"])
            print("Previous end reason:", best_result["end_reason"])

            for i in range(1, pop_size // 2):
                pop[i] = mutate(
                    best_genome,
                    active_limit(best_result),
                    rng
                )
        else:
            print("Genome shape mismatch!")
    else:
        print("No previous genome found. Starting from scratch.")

    pool = Pool(workers) if (workers and workers > 1) else None
    log_rows = []
    history_best = []          # best genome of every generation (for the replay viewer)
    best_ever = (-1e9, None, None)

    try:
        for gen in range(generations + 1):
            t0 = time.time()

            # 2. Evaluate
            evaluated = evaluator(pool, pop)
            fits = np.array([e[0] for e in evaluated])
            results = [e[1] for e in evaluated]

            order = np.argsort(-fits)
            b = int(order[0])
            history_best.append(pop[b].copy())
            if fits[b] > best_ever[0]:
                best_ever = (float(fits[b]), pop[b].copy(), results[b])
                np.save(os.path.join(out_dir, os.path.basename(BEST_FILE)), best_ever[1])

            row = {
                "generation": gen,
                "best_fitness": float(fits[b]),
                "best_distance_m": results[b]["distance"],
                "mean_fitness": float(fits.mean()),
                "median_fitness": float(np.median(fits)),
                "n_finished": int(sum(r["finished"] for r in results)),
                "n_airtime_cut": int(sum(r["airborne"] for r in results)),
                "best_ever_fitness": best_ever[0],
                
                "best_survival_time": results[b]["time"],
                "best_end_reason": results[b]["end_reason"],
                "max_distance_m": max(r["distance"] for r in results),
                "mean_distance_m": float(np.mean([r["distance"] for r in results])),
                "max_survival_time": max(r["time"] for r in results),

                "seconds": round(time.time() - t0, 2),
            }
            log_rows.append(row)
            if verbose:
                tag = "  <- initial population" if gen == 0 else ""
                print(f"gen {gen:3d} | best {row['best_distance_m']:7.2f} m "
                      f"| mean {row['mean_fitness']:6.2f} | finished {row['n_finished']:3d} "
                      f"| airtime-cut {row['n_airtime_cut']:3d} "
                      f"| {row['seconds']:5.1f}s{tag}", flush=True)

            _write_log(log_rows, os.path.join(out_dir, os.path.basename(LOG_FILE)))
            np.savez_compressed(os.path.join(out_dir, os.path.basename(HISTORY_FILE)), genomes=np.array(history_best),
                                fitness=np.array([r["best_fitness"] for r in log_rows]))

            if gen == generations:
                break

            # 3-5. Select, vary, replace (generational with elitism)
            nxt = [pop[int(i)].copy() for i in order[:ELITES]]
            n_immigrants = int(round(IMMIGRANT_FRACTION * pop_size))
            while len(nxt) < pop_size - n_immigrants:
                ia = tournament_select(pop, fits, rng)
                ib = tournament_select(pop, fits, rng)
                a, b_ = pop[ia], pop[ib]
                if rng.random() < CROSSOVER_RATE:
                    limit = min(active_limit(results[ia]), active_limit(results[ib]))
                    child = crossover(a, b_, limit, rng)
                else:
                    child = a
                # mutate inside the region the first parent actually played
                child = mutate(child, active_limit(results[ia]), rng)
                nxt.append(child)
            # fresh random runners keep diversity high (helps escape local optima)
            nxt.extend(random_genome(rng) for _ in range(pop_size - len(nxt)))
            pop = nxt
    finally:
        if pool:
            pool.close()
            pool.join()

    _plot_log(log_rows, os.path.join(out_dir, os.path.basename(PLOT_FILE)))
    return best_ever, log_rows


def _write_log(rows, path=None):
    """
    Write generation statistics to a CSV file.
    
    Parameters:
        rows (list[dict]): Generation statistics to save.
        path (str | None): Output file; defaults to LOG_FILE.

    Returns:
        None.
    """
    with open(path or LOG_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _plot_log(rows, path=None):
    """
    Plot training fitness and distance across generations.
    
    Parameters:
        rows (list[dict]): Generation statistics to plot.
        path (str | None): Output image; defaults to PLOT_FILE.
    
    Returns:
        None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping the plot (pip install matplotlib)")
        return
    g = [r["generation"] for r in rows]
    plt.figure(figsize=(8, 4.5))
    plt.plot(g, [r["best_distance_m"] for r in rows], label="best distance (m)", linewidth=2)
    plt.plot(g, [r["mean_fitness"] for r in rows], label="population mean fitness", linewidth=1.5)
    plt.axhline(rows[0]["best_distance_m"], color="gray", linestyle="--",
                label="initial population (generation 0 best)")
    plt.xlabel("Generation")
    plt.ylabel("Fitness / distance (m)")
    plt.title("QWOP EA: fitness vs generation")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig((path or PLOT_FILE), dpi=150)
    plt.close()
    print(f"Saved plot -> {(path or PLOT_FILE)}")


# =====================================================================
# Random-search baseline
# =====================================================================
def run_baseline(n_samples=POP_SIZE * (GENERATIONS + 1), workers=WORKERS, seed=SEED):
    """
    Measure the best distance achieved through random genome search.
    
    Parameters:
        n_samples (int): Number of random genomes to evaluate.
        workers (int): Number of parallel workers.
        seed (int | None): Optional random seed.
    
    Returns:
        float: Greatest distance reached, in metres.
    """
    rng = np.random.default_rng(seed)
    pool = Pool(workers) if workers > 1 else None
    best = -1.0
    try:
        done = 0
        while done < n_samples:
            batch = [random_genome(rng) for _ in range(min(POP_SIZE, n_samples - done))]
            out = evaluate_population(pool, batch)
            best = max(best, max(r[1]["distance"] for r in out))
            done += len(batch)
        print(f"Random search: {n_samples} evaluations, best distance {best:.2f} m")
    finally:
        if pool:
            pool.close()
            pool.join()
    return best


# =====================================================================
# Visual auto-play (press S to start - no keyboard input from you)
# =====================================================================
def make_viewer_class():
    """
    Define the visual QWOP replay class only when needed.

    Parameters:
        None.
    
    Returns:
        type: AutoPlayGame class used to replay evolved genomes.
    """
    import pyglet
    from pyglet.window import key
    from game_class import Game           

    class AutoPlayGame(Game):
        """
        Reuses Game's world + drawing. Differences:
           - waits for S, then presses Q/W/O/P by itself from the genome
           - script is driven by simulation ticks (not wall-clock dt), so what you see
             is exactly what the headless fitness evaluation measured
           - window stays open after a fall; S replays, LEFT/RIGHT picks a generation
        """
        def __init__(self, genomes, labels, record=True):
            """
            Initialize the replay viewer, controls, and recording state.
            
            Parameters:
                genomes (list[np.ndarray]): Genomes available for replay.
                labels (list[str]): Display labels for each genome.
                record (bool): Whether to record replay videos.

            Returns:
                None.
            """
            super().__init__(input_sequence=[])
            self.label.opacity = 0        
            self.genomes = genomes
            self.labels = labels
            self.record = record
            self.x0 = 0.0
            self.progress_m = 0.0
            self.writer = None
            self.recording = False
            self.want_frame = False
            self.tail_left = 0
            self.which = len(genomes) - 1                # start on the final / best genome
            self.running = False
            self.tick = 0
            self.finished = False
            self.msg = ""
            self.info = pyglet.text.Label(
                "", font_size=14, x=10, y=self.window.height - 20,
                anchor_x="left", anchor_y="center")
            self.current_keys = [0, 0, 0, 0]
            self.key_labels = [
                pyglet.text.Label(
                    letter, font_size=36,
                    x=self.window.width // 2 - 90 + i * 60, y=40,
                    anchor_x="center", anchor_y="center")
                for i, letter in enumerate("QWOP")
            ]
            self.score_label = pyglet.text.Label(
                "", font_size=14, x=10, y=self.window.height - 45,
                anchor_x="left", anchor_y="center")
            self.show_idle_message()
            pyglet.clock.schedule_interval(self.update, TICK)

        # --- helpers --------------------------------------------------
        def show_idle_message(self):
            """
            Display replay instructions and the selected genome.

            Parameters:
                None.

            Returns:
                None.
            """
            self.info.text = (f"{self.labels[self.which]}   |   S = start/restart, "
                              f"LEFT/RIGHT = other generation, SPACE = pause, ESC = quit")

        def start(self):
            """
            Reset the physics world and begin replaying the selected genome.

            Parameters:
                None.

            Returns:
                None.
            """
            self.stop_recording()
            self.simulation = simulation_ticks(self.genomes[self.which], render=True)
            self.space, self.character, _ = next(self.simulation)
            self.x0 = progress_x(self.character)      # same distance measure as training
            self.progress_m = 0.0
            self.game_over = False
            self.finished = False
            self.paused = False
            self.qDown = self.wDown = self.oDown = self.pDown = False
            self.tick = 0
            self.current_keys = [0, 0, 0, 0]
            self.running = True
            self.info.text = f"{self.labels[self.which]}   |   running..."
            print(f"Playing {self.labels[self.which]}")
            if self.record:
                self.start_recording()

        # --- MP4 recording ---------------------------------------------
        def start_recording(self):
            """
            Open an MP4 writer for the current replay.

            Parameters:
                None.

            Returns:
                None.
            """
            try:
                import imageio
            except ImportError:
                print("Recording skipped: run  pip install imageio imageio-ffmpeg")
                self.record = False
                return
            os.makedirs(VIDEO_DIR, exist_ok=True)
            name = re.sub(r"[^A-Za-z0-9]+", "_", self.labels[self.which]).strip("_")[:40]
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_path = os.path.join(VIDEO_DIR, f"qwop_{name}_{stamp}.mp4")
            self.writer = imageio.get_writer(
                self.video_path, fps=VIDEO_FPS, codec="libx264",
                quality=8, macro_block_size=1)
            self.recording = True
            self.want_frame = True              # first frame
            print(f"Recording -> {self.video_path}")

        def grab_frame(self):
            """
            Capture the current game window as an RGB image.

            Parameters:
                None.
            
            Returns:
                np.ndarray: RGB frame with even dimensions.
            """
            img = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            raw = img.get_data("RGB", img.width * 3)
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(img.height, img.width, 3)
            frame = frame[::-1]                 # OpenGL is bottom-up (delete if video is upside down)
            h, w = frame.shape[0] // 2 * 2, frame.shape[1] // 2 * 2
            return np.ascontiguousarray(frame[:h, :w])

        def begin_tail(self):
            """
            Keep recording briefly after the run ends.

            Parameters:
                None.

            Returns:
                None.
            """
            if self.recording:
                self.tail_left = int(RECORD_TAIL_SECONDS * VIDEO_FPS)
                pyglet.clock.schedule_interval(self._tail, 1 / VIDEO_FPS)

        def _tail(self, dt):
            """
            Advance the post-run recording countdown.
            
            Parameters:
                dt (float): Time elapsed since the previous callback, in seconds.
            
            Returns:
                None.
            """
            if self.tail_left <= 0:
                pyglet.clock.unschedule(self._tail)
                self.stop_recording()
            else:
                self.want_frame = True
                self.tail_left -= 1

        def stop_recording(self):
            """
            Close the video writer and stop capturing frames.

            Parameters:
                None.

            Returns:
                None.
            """
            pyglet.clock.unschedule(self._tail)
            if self.writer is not None:
                self.writer.close()
                self.writer = None
                print(f"Saved video -> {self.video_path}")
            self.recording = False
            self.want_frame = False

        def end_game(self):
            """
            Stop recording and close the game.

            Parameters:
                None.

            Returns:
                None.
            """
            self.stop_recording()
            super().end_game()

        # --- events ---------------------------------------------------
        def on_key_press(self, symbol, modifiers):
            """
            Handle replay, pause, navigation, and debug keyboard controls.
            
            Parameters:
                symbol (int): Pyglet key symbol.
                modifiers (int): Active keyboard modifier flags.

            Returns:
                None.
            """
            if symbol == key.S:
                self.start()
            elif symbol == key.ESCAPE:
                self.end_game()
            elif symbol == key.SPACE:
                self.paused = not self.paused
            elif symbol == key.LEFT:
                self.which = max(0, self.which - 1)
                self.show_idle_message()
            elif symbol == key.RIGHT:
                self.which = min(len(self.genomes) - 1, self.which + 1)
                self.show_idle_message()
            elif symbol == key.D:
                self.debug_draw = not self.debug_draw

        def on_key_release(self, symbol, modifiers):
            """
            Ignore key-release events during automatic playback.
            
            Parameters:
                symbol (int): Pyglet key symbol.
                modifiers (int): Active keyboard modifier flags.

            Returns:
                None.
            """
            pass                                      

        def hit_ground(self, arbiter, space, data):
            """
            Stop playback when the runner touches the ground.
            
            Parameters:
                arbiter (pymunk.Arbiter): Collision information.
                space (pymunk.Space): Physics space.
                data (dict): Collision handler data.
            
            Returns:
                bool: False to prevent further collision handling.
            """
            if not self.game_over:
                self.game_over = True
                self.running = False
                self.info.text = (f"FELL at {self.progress_m:.1f} m   |   "
                                  f"S = replay, LEFT/RIGHT = other generation")
                print(f"Fell. distance = {self.progress_m:.2f} m")
                self.begin_tail()
            return False

        def on_draw(self):
            """
            Draw the game, status text, and active key indicators.

            Parameters:
                None.

            Returns:
                None.
            """
            super().on_draw()
            self.info.draw()
            self.score_label.text = f"scored distance ({PROGRESS_MEASURE}): {self.progress_m:.1f} m"
            self.score_label.draw()
            # QWOP indicator: bright yellow = key currently pressed by the EA's sequence
            for pressed, label in zip(self.current_keys, self.key_labels):
                label.color = (255, 230, 0, 255) if pressed else (70, 70, 70, 255)
                label.draw()
            if self.recording and self.want_frame:
                self.want_frame = False
                self.writer.append_data(self.grab_frame())

        # --- main loop ------------------------------------------------
        def update(self, dt):
            """
            Advance playback by one simulation tick and check stopping conditions.
            
            Parameters:
                dt (float): Time elapsed since the previous callback, in seconds.

            Returns:
                None.
            """
            if not self.running or self.paused or self.game_over:
                return

            genome = self.genomes[self.which]
            step_idx = min(self.tick // TICKS_PER_STEP, len(genome) - 1)
            self.current_keys = [int(b) for b in genome[step_idx]]
            _, _, frame = next(self.simulation)
            result, done = frame
            self.tick += 1
            self.progress_m = result["distance"]
            if self.recording and self.tick % int(round(1 / (VIDEO_FPS * TICK))) == 0:
                self.want_frame = True

            if done:
                self.finished = result["finished"]
                self.running = False
                self.game_over = True
                if self.finished:
                    message = f"FINISHED {self.progress_m:.2f} m in {result['time']:.2f} s!"
                else:
                    message = f"RUN ENDED ({result['end_reason']}) at {self.progress_m:.2f} m"
                self.info.text = message + "   |   S = replay"
                print(message)
                self.want_frame = self.recording
                self.begin_tail()

        def run(self):
            """
            Start the Pyglet event loop for the replay viewer.

            Parameters:
                None.

            Returns:
                None.
            """
            print("Press S in the game window to start.")
            pyglet.app.run()

    return AutoPlayGame


def load_play_genomes():
    """
    Load saved generation-best and overall-best genomes for replay.

    Parameters:
        None.
    
    Returns:
        tuple[list[np.ndarray], list[str]]: Saved genomes and their display labels.
    """
    genomes, labels = [], []
    if os.path.exists(HISTORY_FILE):
        data = np.load(HISTORY_FILE)
        for i, g in enumerate(data["genomes"]):
            genomes.append(g)
            labels.append(f"generation {i} best (fitness {data['fitness'][i]:.1f})")
        # always put the overall best last so the viewer starts on it
        if os.path.exists(BEST_FILE):
            genomes.append(np.load(BEST_FILE))
            labels.append("BEST EVER")
    elif os.path.exists(BEST_FILE):
        genomes.append(np.load(BEST_FILE))
        labels.append("BEST EVER")
    return genomes, labels


def run_play(record=True):
    """
    Load evolved genomes and launch the visual replay viewer.
    
    Parameters:
        record (bool): Whether to save replay videos.

    Returns:
        None.
    """
    genomes, labels = load_play_genomes()
    if not genomes:
        sys.exit(f"No evolved genome found in {OUT_DIR}/. Run:  python qwop_ea.py train")
    Viewer = make_viewer_class()
    Viewer(genomes, labels, record=record).run()


# =====================================================================
# CLI
# =====================================================================
def main():
    """
    Parse command-line options and run the selected QWOP mode.

    Parameters:
        None.

    Returns:
        None.
    """
    ap = argparse.ArgumentParser(description="QWOP evolutionary algorithm")
    ap.add_argument("mode", choices=["train", "play", "baseline", "test"],
                    help="train = run EA headless | play = watch (press S) | "
                         "baseline = random search | test = check one random genome")
    ap.add_argument("--generations", type=int, default=GENERATIONS)
    ap.add_argument("--pop", type=int, default=POP_SIZE)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-record", action="store_true", help="play: do not save an MP4")
    args = ap.parse_args()

    if args.mode == "train":
        best, _ = run_ea(args.generations, args.pop, args.workers, args.seed)
        res = best[2]
        print(f"\nBest fitness {best[0]:.2f} | distance {res['distance']:.2f} m | "
              f"finished: {res['finished']}")
        print(f"Saved {BEST_FILE}. Now run:  python qwop_ea.py play   (then press S)")
    elif args.mode == "play":
        run_play(record=not args.no_record)
    elif args.mode == "baseline":
        run_baseline(workers=args.workers, seed=args.seed)
    elif args.mode == "test":
        rng = np.random.default_rng(args.seed)
        t0 = time.time()
        fit, res = evaluate(random_genome(rng))
        print(json.dumps(res, indent=2), f"\nfitness={fit:.2f}  ({time.time() - t0:.2f}s)")


if __name__ == "__main__":
    main()
