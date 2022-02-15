import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

def draw_polygon(ax, points, color):
    p = Polygon(points, facecolor=color)
    ax.add_patch(p)

def plt2rgb():
    fig = plt.gca().figure
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr

def animate(imgs, filename='animation.mp4', _return=True, fps=10, embed=False):
    if isinstance(imgs, dict):
        imgs = imgs['image']
    from moviepy.editor import ImageSequenceClip
    imgs = ImageSequenceClip(imgs, fps=fps)
    imgs.write_videofile(filename, fps=fps)
    if _return:
        from IPython.display import Video
        return Video(filename, embed=embed) 


def plot_trajectoreis(objects, states, width=1.):
    plt.figure(figsize=(5,5))
    ax = plt.gca()
    images = []
    for s in states:
        ax.set_xlim([-width, width])
        ax.set_ylim([-width, width])
        for r, r_s in zip(objects, s):
            r.set_state(r_s[:3])
            r.plot(ax)
        images.append(plt2rgb())
        ax.clear()
    plt.close()
    return images

def plot_contacts(objects, contacts=None, width=1.):
    plt.figure(figsize=(5,5))
    ax = plt.gca()
    ax.set_xlim(-width, width)
    ax.set_ylim(-width, width)
    for i in objects:
        i.plot(ax)

    for i in contacts:
        p = i[0]
        n_c = i[1]
        plt.scatter(p[0], p[1])
        d = np.linalg.norm(n_c)
        n_c = n_c/d * (max(d, 0.05))
        plt.arrow(p[0], p[1], n_c[0], n_c[1])
    plt.show()


"""
The above are visualization code.
Feel free to revise them.
"""
    
class RigidBody:
    """
    We store state and wrench inside the object class.
    Every time the apply_wrench, and step function will modify the two variables in-place. 
    """
    def __init__(self, points, color, mass=1., inertia=None, state=None):
        self.points = np.float32(points)
        self.color = color
        self.mass = mass
        if inertia is None:
            inertia = mass
        self.inertia = inertia

        self.state = state
        self.wrench = np.zeros(3)

    def plot(self, ax):
        draw_polygon(ax, self.get_shape(), self.color)

    def set_state(self, state):
        self.state = np.array(state)

    def get_state(self):
        return self.state

    def object2world(self, loc):
        x, y, theta = self.state[:3]
        c, s = np.cos(theta), np.sin(theta)
        return np.array(loc) @ np.array([[c, -s], [s, c]]).T + np.array([x, y])

    def world2object(self, loc):
        x, y, theta = self.state[:3]
        c, s = np.cos(theta), np.sin(theta)
        return (np.array(loc) - np.array([x, y])) @ np.array([[c, s], [-s, c]]).T


    def get_shape(self):
        """
        TODO: your code here
        return the vertices of the current object in the current state. 
        you should not modify self.points
        """
        return self.points.copy()

    def apply_wrench(self, loc, force):
        """
        TODO: your code here
        apply a force (2d vector) at location (2d) in the world frame

        you should revise self.wrenches
        """

    def step(self, h):
        q, dq = self.state[:3], self.state[3:]
        linear = self.wrench[:2]
        angular = self.wrench[2]
        """
        TODO: your code here
        modify object states according to the total wrenches of the current objects.
        """
        self.wrenches = np.zeros(6) # do not remove this
        return self.state.copy()


def GJK(A, B, plot=False):
    """
    TODO: your code here
    Take two objects as input and compute all contact points.
    """
    return [([0., 0.], [0., 1.])]

    
    
def LCP(objects, contacts, h):
    """ TODO: your code here
    modify objects' wrenches to include contact forces 
    """
    pass