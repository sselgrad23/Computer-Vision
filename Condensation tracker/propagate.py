import numpy as np


def propagate(particles, frame_height, frame_width, params):
    num_particles = particles.shape[0]

    if params['model'] == 0:
        A = np.identity(2)
        w = np.array([params['sigma_position'], params['sigma_position']])
    else:
        A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        w = np.array([params['sigma_position'], params['sigma_position'], params['sigma_velocity'],
                     params['sigma_velocity']])

    new_samples = []
    for item in range(num_particles):
        new_sample = A @ particles[item] + w * np.random.randn(2 if params['model'] == 0 else 4)
        new_sample[0] = np.clip(new_sample[0], 0, frame_width)
        new_sample[1] = np.clip(new_sample[1], 0, frame_height)
        new_samples.append(new_sample)

    new_sample_array = np.vstack(new_samples)

    return new_sample_array
