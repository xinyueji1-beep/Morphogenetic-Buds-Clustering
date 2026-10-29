"""One shared, genuine two-moon scale protocol for every comparator."""
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

def gen_scaled_moons(n, seed):
    x, y = make_moons(n_samples=n, noise=0.08, random_state=seed)
    return StandardScaler().fit_transform(x), y, 2
