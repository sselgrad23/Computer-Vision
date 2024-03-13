import numpy as np


def resample(particles, particles_w):
    idx = np.arange(0, particles.shape[0])
    sampled_idx = np.random.choice(idx, particles.shape[0], replace=True, p=particles_w.reshape(-1))
    sampled_w = particles_w[sampled_idx]
    sampled_w /= np.sum(sampled_w)

    return particles[sampled_idx], sampled_w
