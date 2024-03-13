import numpy as np
from color_histogram import color_histogram
from chi2_cost import chi2_cost


def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist, sigma_observe):
    num_particles = particles.shape[0]
    particles_w = np.zeros((num_particles, 1))

    min_x = np.clip((particles[:, 0] - bbox_width / 2), 0, frame.shape[1]-1-bbox_width)
    min_y = np.clip((particles[:, 1] - bbox_height / 2), 0, frame.shape[0]-1-bbox_height)
    max_x = np.clip((min_x + bbox_width), 0, frame.shape[1]-1)
    max_y = np.clip((min_y + bbox_height), 0, frame.shape[0]-1)

    for i in range(num_particles):
        hist_particle = color_histogram(min_x[i], min_y[i], max_x[i], max_y[i], frame, hist_bin)
        chiCost = chi2_cost(hist, hist_particle)
        particles_w[i] = 1/(np.sqrt(2*np.pi)*sigma_observe) * np.exp(-(chiCost ** 2)/(2 * sigma_observe ** 2))

    particles_w /= np.sum(particles_w)
    return particles_w

