import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import matplotlib.patches as patches
from PIL import Image


class FDTD():
    """
    This is an FDTD simulation engine for the scattering of a 2D wave around an object which
    returns the recorded pressure in 3 observers.
    Parameters
    ----------
    dx: space discretisation
    kd: k*d
    nt: #timesteps
    obj: Choose ['thin',freefield_thin', 'thick', 'freefield_thick','triangle',freefield_triangle]
    animation: set True if you eventually want to show or save a 2D animation of the p field.
    Attributes
    -------
    bront, recorder1, recorder2, recorder3 : Source, list of wave recorded at three postions
    Methods
    -------
    plot_recorders: Plot p w.r.t. time for the recorded locations
    """

    def __init__(self, dx, kd, dt, nt, obj, animation=False):
        self.dx, self.kd, self.dt, self.nt = dx, kd, dt, nt
        self.animation = animation
        self.freefield = False
        if obj in ['thin', 'freefield_thin', 'thick', 'freefield_thick', 'triangle', 'freefield_triangle']:
            self.obj = obj
        else:
            raise ValueError('Choose obj: thin, freefield_thin, thick, freefield_thick,triangle,freefield_triangle')

    def params_init(self):
        if 'freefield' in self.obj:
            self.freefield = True
        ## PARAMETERS
        self.c = 340  # geluidssnelheid - speed of sound (wave speed)
        self.dy = self.dx
        self.d = 1  # lengte d
        self.k = self.kd / self.d  # wavenumber
        self.npml = 20  # Extra layers around simulation domain
        self.nx = self.npml + int(4 * self.d / self.dx)  # number of cells in x direction
        self.nd = int(self.d / self.dx)  # number of cells in d length

        ## SOURCE PARAMETERS
        self.A = 10
        self.fc = self.k * self.c / (2 * np.pi)
        self.t0 = 0
        self.sigma = 5E-4

        if self.obj in ['thin']:
            self.L = 6 * self.d  # length of simulation domain
            self.obj_thickness = 0  # thickness object
            self.x_obj = int(2 * self.nd)  # x-length of object
            self.y_obj = int(2 * self.nd) + self.npml  # y coordinate of object

        elif self.obj == 'freefield_thin':
            self.L = 6 * self.d  # length of simulation domain
            self.obj_thickness = 0  # thickness object
            self.x_obj = int(2 * self.nd)  # x-length of object
            self.y_obj = int(2 * self.nd) + self.npml  # y coordinate of object

        else:
            self.L = 7 * self.d
            self.obj_thickness = self.nd
            if self.obj in ['thick', 'freefield_thick']:
                self.x_obj = int(2 * self.nd)  # x-length of object
                self.y_obj = int(2 * self.nd) + self.npml  # y coordinate of object

        self.ny = 2 * self.npml + int(self.L / self.dy)  # number of cells in y direction

    def run_init(self):
        ## SOURCE AND RECEIVER LOCATIONS


        if self.obj == 'freefield_thin':

            xh = 2*self.nd #values for testing
            yh = self.nd

            self.x_bron = int(self.nd / 10) + xh
            self.y_bron = self.nd + yh
            self.x_recorder1 = int(self.nd / 2) + xh
            self.y_recorder1 = self.y_bron + 2 * self.nd   # Location receiver 1
            self.x_recorder2 = int(self.nd / 2) + xh
            self.y_recorder2 = self.y_recorder1 + self.nd  # Location receiver 2
            self.x_recorder3 = int(self.nd / 2) + xh
            self.y_recorder3 = self.y_recorder2 + self.nd  # Location receiver 3

        else:
            self.x_bron = int(self.nd / 10)
            self.y_bron = self.npml + self.nd
            self.x_recorder1 = int(self.nd / 2)
            self.y_recorder1 = self.y_bron + 2 * self.nd + self.obj_thickness  # Location receiver 1
            self.x_recorder2 = int(self.nd / 2)
            self.y_recorder2 = self.y_recorder1 + self.nd  # Location receiver 2
            self.x_recorder3 = int(self.nd / 2)
            self.y_recorder3 = self.y_recorder2 + self.nd  # Location receiver 3

        ## PML

        if self.obj in ['freefield_thin']:
            self.sigma_x = np.zeros((self.nx + 1, self.ny))
            self.sigma_y = np.zeros((self.nx, self.ny + 1))
            self.sigma_p_x = np.zeros((self.nx, self.ny))
            self.sigma_p_y = np.zeros((self.nx, self.ny))
        else:
            sigma_max_left = 700  # Max amount of damping left
            sigma_max_right = 100  # Max amount of damping right
            sigma_max_up = 700  # Max amount of damping upward
            hoogte_PML = self.nx - int(2 * self.nd) + 1 - 10  # Height from which wave starts damping (numbers of layers)
            breedte_PML_links = self.y_bron - 10  # How much to the right of left simulation wall will wave start damping (numbers of layers)
            breedte_PML_rechts = self.ny - self.y_recorder3 - 10  # How much to the left of right simulation wall will wave start damping (numbers of layers)
            self.sigma_x = np.zeros((self.nx + 1, self.ny))
            self.sigma_y = np.zeros((self.nx, self.ny + 1))
            self.sigma_p_x = np.zeros((self.nx, self.ny))
            self.sigma_p_y = np.zeros((self.nx, self.ny))
            m = 1  # Power of the PML (3 to 4), if too high, the sigma_max is too small

            self.sigma_x[-hoogte_PML:, :] = [
                [sigma_max_up * (i / len(self.sigma_x[-hoogte_PML:, :])) ** m] * self.sigma_x.shape[1] for i in
                range(0, len(self.sigma_x[:hoogte_PML, :]), 1)]
            self.sigma_y[:, :breedte_PML_links] = np.array(
                [[sigma_max_left * (i / len(self.sigma_y[:breedte_PML_links, :])) ** m] * self.sigma_y.shape[0] for i in
                 range(len(self.sigma_y[:breedte_PML_links, :]), 0, -1)]).transpose()
            self.sigma_y[:, breedte_PML_rechts:] = np.array(
                [[sigma_max_right * (i / len(self.sigma_y[breedte_PML_rechts:, :])) ** m] * self.sigma_y.shape[0] for i in
                 range(0, self.sigma_y.shape[1] - breedte_PML_rechts, 1)]).transpose()

            self.sigma_p_x[-hoogte_PML:, :] = [
                [sigma_max_up * (i / len(self.sigma_p_x[-hoogte_PML:, :])) ** m] * self.sigma_p_x.shape[1] for i in
                range(0, len(self.sigma_p_x[:hoogte_PML, :]), 1)]
            self.sigma_p_y[:, :breedte_PML_links] = np.array(
                [[sigma_max_left * (i / len(self.sigma_p_y[:breedte_PML_links, :])) ** m] * self.sigma_p_y.shape[0] for i in
                 range(len(self.sigma_p_y[:breedte_PML_links, :]), 0, -1)]).transpose()
            self.sigma_p_y[:, breedte_PML_rechts:] = np.array(
                [[sigma_max_right * (i / len(self.sigma_p_y[breedte_PML_rechts:, :])) ** m] * self.sigma_p_y.shape[0] for i
                 in
                 range(0, self.sigma_p_y.shape[1] - breedte_PML_rechts, 1)]).transpose()




        ## P and O fields
        self.ox = np.zeros((self.nx + 1, self.ny))
        self.oy = np.zeros((self.nx, self.ny + 1))
        self.p = np.zeros((self.nx, self.ny))

        ## Timeseries
        self.recorder1 = np.zeros((self.nt, 1))
        self.recorder2 = np.zeros((self.nt, 1))
        self.recorder3 = np.zeros((self.nt, 1))

        self.bront = np.zeros((self.nt, 1))
        self.tijdreeks = np.zeros((self.nt, 1))
        return self

    def run(self):
        self.params_init()
        self.run_init()
        if self.animation:
            self.animation_init()
        for it in range(0, self.nt):
            t = it * self.dt
            self.tijdreeks[it, 0] = t
            print('%d/%d' % (it, self.nt))

            bron = self.A * np.sin(2 * np.pi * self.fc * (t - self.t0)) * np.exp(
                -((t - self.t0) ** 2) / (self.sigma))  # update source for new time
            self.bront[it, 0] = bron
            self.p[self.x_bron, self.y_bron] += bron  # adding source term to propagation

            self.timestep()  # propagate over one time step

            if not self.freefield:
                self.hard_walls_o()  # implement the hard walls

            self.recorder1[it] = self.p[self.x_recorder1, self.y_recorder1]  # store p field at receiver locations
            self.recorder2[it] = self.p[self.x_recorder2, self.y_recorder2]
            self.recorder3[it] = self.p[self.x_recorder3, self.y_recorder3]

            if self.animation:
                self.animate_2D(it)

        if self.animation:
            my_anim = ArtistAnimation(self.fig, self.movie, interval=50, repeat_delay=1000,
                                      blit=True)
            choice = 3
            while choice not in [0, 1, 2]:
                choice = int(input('__ANIMATION__ Show(0) or Save(1), or Abort(2)?:'))
            if choice == 1:
                path = '{}_kd={}.gif'.format(obj, kd)
                print('Saving animation as: {}'.format(path))
                my_anim.save(path, writer='pillow', fps=30)
                plt.close()
                print('Done')
            elif choice == 0:
                print('Showing animation...')
                plt.show()
            else:
                plt.close()
                pass

        return self

    def animation_init(self):
        self.fig, self.ax = plt.subplots()
        plt.xlabel('x/d')
        plt.ylabel('y/d')
        plt.ylim([1, self.nx])
        plt.xlim([1, self.ny])
        self.ax.set_yticks(np.linspace(0, 4 * self.nd, 5))
        self.ax.set_yticklabels(np.arange(5))
        if self.obj in ['thin', 'freefield_thin']:
            self.ax.set_xticks(self.npml + self.nd * np.arange(7))
            self.ax.set_xticklabels(np.arange(7))
        else:
            self.ax.set_xticks(self.npml + self.nd * np.arange(8))
            self.ax.set_xticklabels(np.arange(8))
            if self.obj == 'freefield_thick':
                self.rect = patches.Rectangle((self.y_obj, 0), self.obj_thickness + self.dx, self.x_obj + self.dx,
                                              fill=False, ls=':')
            elif self.obj == 'thick':
                self.rect = patches.Rectangle((self.y_obj, 0), self.obj_thickness + self.dx, self.x_obj + self.dx,
                                              facecolor='k')
            elif self.obj == 'triangle':
                vert = np.array(
                    [(int(2 * self.nd) + self.npml, 0), (int(5 * self.nd / 2) + self.npml, int(2 * self.nd)),
                     (int(3 * self.nd) + self.npml, 0)])
                self.tr = patches.Polygon(vert, closed=True, facecolor='k')
            elif self.obj == 'freefield_triangle':
                vert = np.array(
                    [(int(2 * self.nd) + self.npml, 0), (int(5 * self.nd / 2) + self.npml, int(2 * self.nd)),
                     (int(3 * self.nd) + self.npml, 0)])
                self.tr = patches.Polygon(vert, closed=True, fill=False, ls=':')
        self.movie = []
        return self

    def animate_2D(self, it):
        if self.obj in ['thin', 'freefield_thin']:
            if self.obj == 'freefield_thin':
                obj_plot = \
                    self.ax.plot(np.full(np.arange(int(2 * self.nd)).shape, self.y_obj), np.arange(self.x_obj),
                                 color='k', ls=':', linewidth=60 * self.dx)[0]
            else:
                obj_plot = \
                    self.ax.plot(np.full(np.arange(int(2 * self.nd)).shape, self.y_obj), np.arange(self.x_obj),
                                 color='k', linewidth=60 * self.dx)[0]
        elif self.obj in ['thick', 'freefield_thick']:
            obj_plot = self.ax.add_patch(self.rect)
        elif self.obj in ['triangle', 'freefield_triangle']:
            obj_plot = self.ax.add_patch(self.tr)
        artists = [
            self.ax.text(0.5, 1.05, '%d/%d' % (it, self.nt),
                         size=plt.rcParams["axes.titlesize"],
                         ha="center", transform=self.ax.transAxes),
            self.ax.imshow(self.p, vmin=-0.02 * self.A, vmax=0.02 * self.A),
            self.ax.plot(self.y_bron, self.x_bron, 'ks', fillstyle="none")[0],
            self.ax.plot(self.y_recorder1, self.x_recorder1, 'ro', fillstyle="none")[0],
            self.ax.plot(self.y_recorder2, self.x_recorder2, 'ro', fillstyle="none")[0],
            self.ax.plot(self.y_recorder3, self.x_recorder3, 'ro', fillstyle="none")[0],
            obj_plot
        ]
        self.movie.append(artists)
        return self

    def plot_recorders(self):
        plt.plot(self.tijdreeks, self.recorder1, label='Recorder 1')
        plt.plot(self.tijdreeks, self.recorder2, label='Recorder 2')
        plt.plot(self.tijdreeks, self.recorder3, label='Recorder 3')
        plt.title('Recorded pressure vs time, {} object case, kd={}'.format(self.obj, self.kd))
        plt.xlabel('Time [s]')
        plt.ylabel('Pressure [N/m^2]')
        plt.legend()
        plt.grid()
        plt.show()

    def hard_walls_o(self):
        if self.obj in ['thin', 'thick']:
            self.ox[:self.x_obj, self.y_obj:self.y_obj + self.obj_thickness] = 0
            self.oy[:self.x_obj, self.y_obj:self.y_obj + self.obj_thickness] = 0


        elif self.obj == 'triangle':
            for i in range(0, int(self.nd / 2) + 1):
                self.ox[: 4 * i, 2 * self.nd + i + self.npml] = 0  # half triangle
                self.oy[: 4 * i, 2 * self.nd + i + self.npml] = 0  # half triangle
                self.ox[: 4 * i, 3 * self.nd - i + self.npml] = 0  # half triangle
                self.oy[: 4 * i, 3 * self.nd - i + self.npml] = 0  # half triangle

        self.ox[0, :] = 0  # ground
        self.oy[0, :] = 0  # ground
        return self

    def hard_walls_derivative(self):
        if self.obj in ['thin', 'thick']:
            self.p_x[:self.x_obj + 1, self.y_obj:self.y_obj + self.obj_thickness + 1] = 0
            self.p_y[:self.x_obj + 1, self.y_obj:self.y_obj + self.obj_thickness + 1] = 0
        elif self.obj == 'triangle':
            for i in range(0, int(self.nd / 2) + 1):
                self.p_x[: 4 * i, 2 * self.nd + i + self.npml] = 0  # half triangle
                self.p_y[: 4 * i, 2 * self.nd + i + self.npml] = 0  # half triangle
                self.p_x[: 4 * i, 3 * self.nd - i + self.npml] = 0  # half triangle
                self.p_y[: 4 * i, 3 * self.nd - i + self.npml] = 0  # half triangle
        return self

    def timestep(self):
        self.p_y = (np.append(self.p, self.p[:, 0].reshape((self.nx, 1)), axis=1) - np.append(
            self.p[:, -1].reshape((self.nx, 1)), self.p, axis=1)) / self.dy
        self.p_x = (np.append(self.p, self.p[0, :].reshape((1, self.ny)), axis=0) - np.append(
            self.p[-1, :].reshape((1, self.ny)), self.p, axis=0)) / self.dx

        self.no_periodic_boundaries()  # implement no periodic boundaries
        if not self.freefield:
            self.hard_walls_derivative()  # implement hard walls for the derivatives

        self.ox = self.ox * (1 - self.dt * self.sigma_x) - self.dt * self.p_x
        self.oy = self.oy * (1 - self.dt * self.sigma_y) - self.dt * self.p_y
        self.ox_x = (self.ox[1:, :] - self.ox[:-1, :]) / self.dx
        self.oy_y = (self.oy[:, 1:] - self.oy[:, :-1]) / self.dy

        self.p = self.p - (self.c ** 2) * self.dt * (self.ox_x + self.oy_y) - (
                    self.sigma_p_y + self.sigma_p_x) * self.p * self.dt
        return self

    def no_periodic_boundaries(self):
        self.p_y[-1, :] = 0  # No periodic boundaries
        self.p_x[-1, :] = 0  # No periodic boundaries
        self.p_y[:, -1] = 0  # No periodic boundaries
        self.p_x[:, -1] = 0  # No periodic boundaries
        return self