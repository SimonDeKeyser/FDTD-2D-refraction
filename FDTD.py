import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import matplotlib.patches as patches
from PIL import Image


def Step(nx, ny, c, dx, dy, dt, obj):
    global ox, oy, p, nd, sigma_x, sigma_y, npml, sigma_p_x, sigma_p_y, auxiliary
    p_y = (np.append(p, p[:, 0].reshape((nx, 1)), axis=1) - np.append(p[:, -1].reshape((nx, 1)), p, axis=1)) / dy
    p_x = (np.append(p, p[0, :].reshape((1, ny)), axis=0) - np.append(p[-1, :].reshape((1, ny)), p, axis=0)) / dx

    p_y[-1, :] = 0  # No periodic boundaries
    p_x[-1, :] = 0  # No periodic boundaries
    p_y[:, -1] = 0  # No periodic boundaries
    p_x[:, -1] = 0  # No periodic boundaries

    if obj == 'thin':
        p_x[:int(2 * nd) + 1, int(2 * nd) + npml] = 0  # Thin sheet
        p_y[:int(2 * nd) + 1, int(2 * nd) + npml] = 0  # Thin sheet
    elif obj == 'thick':
        p_x[:int(2 * nd) + 1, int(2 * nd) + npml: int(3 * nd) + 1 + npml] = 0  # Thick object
        p_y[:int(2 * nd) + 1, int(2 * nd) + npml: int(3 * nd) + 1 + npml] = 0  # Thick object
    elif obj == 'triangle':
        for i in range(0, int(nd / 2) + 1):
            p_x[0: 4 * i, 2 * nd + i + npml] = 0  # half triangle
            p_y[0: 4 * i, 2 * nd + i + npml] = 0  # half triangle
            p_x[0: 4 * i, 3 * nd - i + npml] = 0  # half triangle
            p_y[0: 4 * i, 3 * nd - i + npml] = 0  # half triangle

    ox = ox * (1 - dt * sigma_x) - dt * p_x
    oy = oy * (1 - dt * sigma_y) - dt * p_y
    ox_x = (ox[1:, :] - ox[:-1, :]) / dx
    oy_y = (oy[:, 1:] - oy[:, :-1]) / dy

    p = p - (c ** 2) * dt * (ox_x + oy_y) - (sigma_p_y + sigma_p_x) * p * dt #+ auxiliary*dt , not needed at the moment
    #auxiliary = - (c ** 2) * dt * (sigma_p_x * oy_y + sigma_p_y * ox_x)
def Simulation(dx, kd, dt, nt, obj, plot=False, save=False):
    """
    This function simulates the scattering of a 2D wave around an infinitely thin sheet and
    returns the recorded pressure in 3 observers.
    Parameters
    ----------
    dx, kd, nt, obj : space discretisation, k*d, #timesteps, object: ['thin',freefield_thin', 'thick', 'freefield_thick','triangle',freefield_triangle]
    Returns
    -------
    bront, [rec1, rec2, rec3] : Source, list of wave recorded at three postions
    """
    global ox, oy, p, nd, sigma_x, sigma_y, npml, sigma_p_x, sigma_p_y, auxiliary
    # INITIALISATION 2D-GRID AND SIMULATION PARAMETERS-------------------------
    c = 340  # geluidssnelheid - speed of sound (wave speed)
    dy = dx
    d = 1  # lengte d
    k = kd / d  # wavenumber
    if obj == 'thin' or obj == 'freefield_thin':
        L = 6 * d  # length of simulation domain
    elif obj == 'thick' or obj == 'freefield_thick' or obj == 'triangle' or obj == 'freefield_triangle':
        L = 7 * d
    else:
        raise ValueError('Choose obj: thin, freefield_thin, thick, freefield_thick,triangle,freefield_triangle')

    npml = 40  # number of PML layers
    nx = npml + int(4 * d / dx)  # number of cells in x direction
    ny = 2 * npml + int(L / dy)  # number of cells in y direction
    nd = int(d / dx)  # number of cells in d length

    # location of source(central) and receivers
    x_bron = int(nd / 10)
    y_bron = npml + nd

    x_recorder1 = int(nd / 2)
    if obj == 'thin' or obj == 'freefield_thin':
        y_recorder1 = y_bron + 2 * nd  # Location receiver 1
    elif obj == 'thick' or obj == 'freefield_thick' or obj == 'triangle' or obj == 'freefield_triangle':
        y_recorder1 = y_bron + 3 * nd  # Location receiver 1

    x_recorder2 = int(nd / 2)
    y_recorder2 = y_recorder1 + nd  # Location receiver 2

    x_recorder3 = int(nd / 2)
    y_recorder3 = y_recorder2 + nd  # Location receiver 2

    # source pulse information
    A = 10
    fc = k * c / (2 * np.pi)
    t0 = 0
    sigma = 5E-4

    # PML implementation
    sigma_max_left = 700  # Max amount of damping left
    sigma_max_right = 100  # Max amount of damping right
    sigma_max_up = 700  # Max amount of damping upward

    hoogte_PML = nx - int(2 * nd) + 1 - 10  # Height from which wave starts damping (numbers of layers)
    breedte_PML_links = y_bron - 10  # How much to the right of left simulation wall will wave start damping (numbers of layers)
    breedte_PML_rechts = ny - y_recorder3 - 10  # How much to the left of right simulation wall will wave start damping (numbers of layers)
    sigma_x = np.zeros((nx + 1, ny))
    sigma_y = np.zeros((nx, ny + 1))
    m = 1  # Power of the PML (3 to 4), if too high, the sigma_max is too small

    sigma_x[-hoogte_PML:, :] = [[sigma_max_up * (i / len(sigma_x[-hoogte_PML:, :])) ** m] * sigma_x.shape[1] for i in
                                range(0, len(sigma_x[:hoogte_PML, :]), 1)]
    sigma_y[:, :breedte_PML_links] = np.array(
        [[sigma_max_left * (i / len(sigma_y[:breedte_PML_links, :])) ** m] * sigma_y.shape[0] for i in
         range(len(sigma_y[:breedte_PML_links, :]), 0, -1)]).transpose()
    sigma_y[:, breedte_PML_rechts:] = np.array(
        [[sigma_max_right * (i / len(sigma_y[breedte_PML_rechts:, :])) ** m] * sigma_y.shape[0] for i in
         range(0, sigma_y.shape[1] - breedte_PML_rechts, 1)]).transpose()

    # initialisation of o and p fields
    ox = np.zeros((nx + 1, ny))
    oy = np.zeros((nx, ny + 1))
    p = np.zeros((nx, ny))

    sigma_p_x = np.zeros((nx, ny))
    sigma_p_y = np.zeros((nx, ny))
    auxiliary = np.zeros((nx, ny))

    sigma_p_x[-hoogte_PML:, :] = [[sigma_max_up * (i / len(sigma_p_x[-hoogte_PML:, :])) ** m] * sigma_p_x.shape[1] for i in
                                range(0, len(sigma_p_x[:hoogte_PML, :]), 1)]
    sigma_p_y[:, :breedte_PML_links] = np.array(
        [[sigma_max_left * (i / len(sigma_p_y[:breedte_PML_links, :])) ** m] * sigma_p_y.shape[0] for i in
         range(len(sigma_p_y[:breedte_PML_links, :]), 0, -1)]).transpose()
    sigma_p_y[:, breedte_PML_rechts:] = np.array(
        [[sigma_max_right * (i / len(sigma_p_y[breedte_PML_rechts:, :])) ** m] * sigma_p_y.shape[0] for i in
         range(0, sigma_p_y.shape[1] - breedte_PML_rechts, 1)]).transpose()

    # initialisation time series receivers
    recorder1 = np.zeros((nt, 1))
    recorder2 = np.zeros((nt, 1))
    recorder3 = np.zeros((nt, 1))

    bront = np.zeros((nt, 1))
    tijdreeks = np.zeros((nt, 1))
    bron = 0

    # TIME ITTERATION----------------------------------------------------
    if plot or save:
        fig, ax = plt.subplots()
        plt.xlabel('x/d')
        plt.ylabel('y/d')
        plt.ylim([1, nx])
        plt.xlim([1, ny])
        ax.set_yticks(np.linspace(0, 4 * nd, 5))
        ax.set_yticklabels(np.arange(5))
        if obj == 'thin' or obj == 'freefield_thin':
            ax.set_xticks(npml + nd * np.arange(7))
            ax.set_xticklabels(np.arange(7))
            movie = []
        elif obj == 'thick' or obj == 'freefield_thick':
            ax.set_xticks(npml + nd * np.arange(8))
            ax.set_xticklabels(np.arange(8))
            movie = []
            if obj == 'freefield_thick':
                rect = patches.Rectangle((int(2 * nd) + npml, 0), nd + dx, 2 * nd + dx, fill=False, ls=':')
            else:
                rect = patches.Rectangle((int(2 * nd) + npml, 0), nd + dx, 2 * nd + dx, facecolor='k')
        elif obj == 'triangle' or obj == 'freefield_triangle':
            ax.set_xticks(npml + nd * np.arange(8))
            ax.set_xticklabels(np.arange(8))
            movie = []
            vert = np.array([(int(2 * nd) + npml, 0), (int(5 * nd / 2) + npml, int(2 * nd)), (int(3 * nd) + npml, 0)])
            if obj == 'freefield_triangle':
                tr = patches.Polygon(vert, closed=True, fill=False, ls=':')
            elif obj == 'triangle':
                tr = patches.Polygon(vert, closed=True, facecolor='k')
    for it in range(0, nt):
        t = it * dt
        print('%d/%d' % (it, nt))

        bron = A * np.sin(2 * np.pi * fc * (t - t0)) * np.exp(-((t - t0) ** 2) / (sigma))  # update source for new time
        bront[it, 0] = bron
        p[x_bron, y_bron] = p[x_bron, y_bron] + bron  # adding source term to propagation
        Step(nx, ny, c, dx, dy, dt, obj)  # propagate over one time step
        if obj == 'thin':
            ox[:int(2 * nd), int(2 * nd) + npml] = 0  # thin sheet
            oy[:int(2 * nd), int(2 * nd) + npml] = 0  # thin sheet
        elif obj == 'thick':
            ox[:int(2 * nd), int(2 * nd) + npml:int(3 * nd) + npml] = 0  # Thick object
            oy[:int(2 * nd), int(2 * nd) + npml:int(3 * nd) + npml] = 0  # Thick object
        elif obj == 'triangle':
            for i in range(0, int(nd / 2) + 1):
                ox[: 4 * i, 2 * nd + i + npml] = 0  # half triangle
                oy[: 4 * i, 2 * nd + i + npml] = 0  # half triangle
                ox[: 4 * i, 3 * nd - i + npml] = 0  # half triangle
                oy[: 4 * i, 3 * nd - i + npml] = 0  # half triangle
        ox[0, :] = 0  # ground
        oy[0, :] = 0  # ground

        recorder1[it] = p[x_recorder1, y_recorder1]  # store p field at receiver locations
        recorder2[it] = p[x_recorder2, y_recorder2]
        recorder3[it] = p[x_recorder3, y_recorder3]

        if plot or save:
            # presenting the p field
            if obj == 'thin' or obj == 'freefield_thin':
                if obj == 'freefield_thin':
                    obj_plot = \
                    ax.plot(np.full(np.arange(int(2 * nd)).shape, int(2 * nd) + npml), np.arange(int(2 * nd)),
                            color='k', ls=':', linewidth=60 * dx)[0]
                else:
                    obj_plot = \
                    ax.plot(np.full(np.arange(int(2 * nd)).shape, int(2 * nd) + npml), np.arange(int(2 * nd)),
                            color='k', linewidth=60 * dx)[0]
            elif obj == 'thick' or obj == 'freefield_thick':
                obj_plot = ax.add_patch(rect)
            elif obj == 'triangle' or obj == 'freefield_triangle':
                obj_plot = ax.add_patch(tr)
            artists = [
                ax.text(0.5, 1.05, '%d/%d' % (it, nt),
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=ax.transAxes),
                ax.imshow(p, vmin=-0.02 * A, vmax=0.02 * A),
                ax.plot(y_bron, x_bron, 'ks', fillstyle="none")[0],
                ax.plot(y_recorder1, x_recorder1, 'ro', fillstyle="none")[0],
                ax.plot(y_recorder2, x_recorder2, 'ro', fillstyle="none")[0],
                ax.plot(y_recorder3, x_recorder3, 'ro', fillstyle="none")[0],
                obj_plot
            ]
            movie.append(artists)

    if plot or save:
        my_anim = ArtistAnimation(fig, movie, interval=50, repeat_delay=1000,
                                  blit=True)
        if save:
            my_anim.save('{}_kd={}.gif'.format(obj, kd), writer='pillow', fps=30)
            plt.close()
        if plot:
            plt.show()
    return bront, [recorder1, recorder2, recorder3]
