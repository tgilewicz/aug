import random
import scipy.stats as stats


def uniform(lower, upper):
    assert type(lower) == type(upper)

    if isinstance(lower, list) or isinstance(lower, tuple):
        params = [uniform(l, u) for l, u in zip(lower, upper)]
        return tuple(params) if isinstance(lower, tuple) else params

    if isinstance(lower, int):
        return random.randint(lower, upper)
    else:
        return random.uniform(lower, upper)


def rand_bool():
    return bool(random.getrandbits(1))


def truncnorm(lower, upper, mu, sigma):
    assert type(lower) == type(upper)

    if isinstance(lower, list) or isinstance(lower, tuple):
        params = [truncnorm(l, u, mu, sigma) for l, u in zip(lower, upper)]
        return tuple(params) if isinstance(lower, tuple) else params

    return stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu,
                           scale=sigma).rvs(1)[0]
