import numpy as np


def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):

    R, _ = np.histogram(frame[int(ymin):int(ymax) + 1, int(xmin):int(xmax) + 1, 0], hist_bin,
                        range=[0, 255], density=True)
    G, _ = np.histogram(frame[int(ymin):int(ymax) + 1, int(xmin):int(xmax) + 1, 1], hist_bin,
                        range=[0, 255], density=True)
    B, _ = np.histogram(frame[int(ymin):int(ymax) + 1, int(xmin):int(xmax) + 1, 2], hist_bin,
                        range=[0, 255], density=True)

    hist = np.vstack((R, G, B))
    hist = hist / np.sum(hist)

    return hist
