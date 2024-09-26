from matplotlib import animation
import numpy as np 
import matplotlib.pyplot as plt
from utils.data_utils import is_numpy_event_data, is_pyg_event_data, pyg2numpy_event_convertor

def animate_events(events, transform):
    
    if is_pyg_event_data(events):
        events = pyg2numpy_event_convertor(events)
    else:
        assert is_numpy_event_data(events), "The input data must be a structured numpy array or a PyG dataset."
    
    frames = transform(events)
    
    fig, ax = plt.subplots()
    if frames.shape[1] == 2:
        rgb = np.zeros((frames.shape[0], 3, *frames.shape[2:]))
        rgb[:, 0, ...] = frames[:, 0, ...] > 0
        rgb[:, 2, ...] = frames[:, 1, ...] > 0
        frames = rgb
    if frames.shape[1] in [1, 2, 3]:
        frames = np.moveaxis(frames, 1, 3)
    im = ax.imshow(frames[0])
    # plt.axis("off")
    # plt.axis("tight")

    def animate(frame):
        im.set_data(frame)
        return [im]

    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=100)
    plt.close(fig)

    return anim
        