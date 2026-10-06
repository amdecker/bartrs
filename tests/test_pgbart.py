import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb

from bartrs import PGBART


NUM_TUNE = 50
NUM_DRAWS = 50
NUM_CHAINS = 2
BATCH_SIZE = (0.1, 0.1)
NUM_TREES = 10
NUM_PARTICLES = 5
RANDOM_SEED = 42


def test_bikes():
    bikes = pd.read_csv(pm.get_data("bikes.csv"))
    X = bikes[["hour", "temperature", "humidity", "workingday"]]
    Y = bikes["count"]

    with pm.Model() as model:
        alpha = pm.Exponential("alpha", 1.0)
        mu = pmb.BART("mu", X, np.log(Y), m=NUM_TREES, response="constant")
        y = pm.NegativeBinomial("y", mu=pm.math.exp(mu), alpha=alpha, observed=Y)

        idata = pm.sample(
            tune=NUM_TUNE,
            draws=NUM_DRAWS,
            chains=NUM_CHAINS,
            step=[PGBART([mu], batch=BATCH_SIZE, num_particles=NUM_PARTICLES)],
            random_seed=RANDOM_SEED,
        )


def test_coal():
    coal = np.loadtxt(pm.get_data("coal.csv"))
    years = int(coal.max() - coal.min())
    bins = years // 4
    hist, x_edges = np.histogram(coal, bins=bins)
    x_centers = x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2
    x_data = x_centers[:, None]
    # Express data as the rate number of disaster per year
    y_data = hist

    with pm.Model() as model:
        mu = pmb.BART("mu", X=x_data, Y=np.log(y_data), m=NUM_TREES, response="constant")
        exp_mu = pm.Deterministic("exp_mu", pm.math.exp(mu))
        y_pred = pm.Poisson("y_pred", mu=exp_mu, observed=y_data)

        idata = pm.sample(
            tune=NUM_TUNE,
            draws=NUM_DRAWS,
            chains=NUM_CHAINS,
            step=[PGBART([mu], batch=BATCH_SIZE, num_particles=NUM_PARTICLES)],
            random_seed=RANDOM_SEED,
        )


def _test_model(random_seed):
    """Sample a small BART regression with parallel chains and return the draws of mu."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (100, 1))
    Y = rng.normal(np.sin(3 * X[:, 0]), 0.3)

    with pm.Model() as model:
        mu = pmb.BART("mu", X, Y, m=NUM_TREES, response="constant")
        y = pm.Normal("y", mu=mu, sigma=0.3, observed=Y)

        idata = pm.sample(
            tune=NUM_TUNE,
            draws=NUM_DRAWS,
            chains=NUM_CHAINS,
            cores=NUM_CHAINS,
            step=[PGBART([mu], batch=BATCH_SIZE, num_particles=NUM_PARTICLES)],
            random_seed=random_seed,
        )
    return idata.posterior["mu"]


def test_chains_differ():
    """Ensure chains are not exact copies of each other."""
    draws = _test_model(RANDOM_SEED)
    assert not np.array_equal(draws.sel(chain=0), draws.sel(chain=1))


def test_random_seed_reproduces_chains():
    """Ensure using the same seed produces the same result and different seeds
    produce different results."""
    draws = _test_model(RANDOM_SEED)
    draws_again = _test_model(RANDOM_SEED)
    other_seed = _test_model(RANDOM_SEED + 1)
    assert np.array_equal(draws, draws_again)
    assert not np.array_equal(draws, other_seed)
