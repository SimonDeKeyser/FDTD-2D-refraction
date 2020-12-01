import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import matplotlib.patches as patches
from PIL import Image
import scipy.special as sc
from scipy.special import hankel2

class FDTD():
    """
    This is an FDTD simulation engine for the scattering of a 2D wave around an object which
    returns the recorded pressure in 3 observers.
    Parameters
    ----------
    dx -- space discretisation
    kd -- k*d
    nt -- #timesteps
    obj -- Choose ['thin',freefield_thin', 'thick', 'freefield_thick','triangle',freefield_triangle]
    animation -- set True if you eventually want to show or save a 2D animation of the p field.
    Attributes
    -------
    bront, recorder1, recorder2, recorder3 -- Source, list of wave recorded at three postions
    Methods
    -------
    run() -- run the simulation
    plot_recorders() -- Plot p w.r.t. time for the recorded locations
    plot_source() -- Plot p generated by the source
    TF_ANA(recorder_number,plot=False) -- Get the analytical transferfunction at the given recordernumber(1,2,3)
    """
    def __init__(self, dx, kd, dt, nt, source ,obj, animation=False):
        self.dx, self.kd, self.dt, self.nt = dx, kd, dt, nt
        self.animation = animation
        self.freefield = False
        self.initialized = False
        self.A, self.sigma = source
        if obj in ['thin', 'freefield_thin', 'thick', 'freefield_thick','triangle','freefield_triangle']:
            self.obj = obj
        else:
            raise ValueError('Choose obj: thin, freefield_thin, thick, freefield_thick,triangle,freefield_triangle')
    
    ## INITIALIZATION--------------------------------------------
    def params_init(self):
        self.initialized = True
        if 'freefield' in  self.obj:
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
        self.fc = self.k * self.c / (2 * np.pi)
        self.t0 = self.nt*self.dt/2

        if self.obj in ['thin','freefield_thin']:
            self.L = 6 * self.d  # length of simulation domain
            self.obj_thickness = 0 # thickness object
            self.x_obj = int(2 * self.nd) # x-length of object
            self.y_obj = int(2 * self.nd) + self.npml # y coordinate of object
        else:
            self.L = 7 * self.d
            self.obj_thickness = self.nd
            if self.obj in ['thick','freefield_thick']:
                self.x_obj = int(2 * self.nd) # x-length of object
                self.y_obj = int(2 * self.nd) + self.npml # y coordinate of object

        self.ny = 2 * self.npml + int(self.L / self.dy)  # number of cells in y direction

    def PML_init(self):
        ## PML
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

        self.sigma_x[-hoogte_PML:, :] = [[sigma_max_up * (i / len(self.sigma_x[-hoogte_PML:, :])) ** m] * self.sigma_x.shape[1] for i in
                                    range(0, len(self.sigma_x[:hoogte_PML, :]), 1)]
        self.sigma_y[:, :breedte_PML_links] = np.array(
            [[sigma_max_left * (i / len(self.sigma_y[:breedte_PML_links, :])) ** m] * self.sigma_y.shape[0] for i in
            range(len(self.sigma_y[:breedte_PML_links, :]), 0, -1)]).transpose()
        self.sigma_y[:, breedte_PML_rechts:] = np.array(
            [[sigma_max_right * (i / len(self.sigma_y[breedte_PML_rechts:, :])) ** m] * self.sigma_y.shape[0] for i in
            range(0, self.sigma_y.shape[1] - breedte_PML_rechts, 1)]).transpose()

        self.sigma_p_x[-hoogte_PML:, :] = [[sigma_max_up * (i / len(self.sigma_p_x[-hoogte_PML:, :])) ** m] * self.sigma_p_x.shape[1] for i in
                                    range(0, len(self.sigma_p_x[:hoogte_PML, :]), 1)]
        self.sigma_p_y[:, :breedte_PML_links] = np.array(
            [[sigma_max_left * (i / len(self.sigma_p_y[:breedte_PML_links, :])) ** m] * self.sigma_p_y.shape[0] for i in
            range(len(self.sigma_p_y[:breedte_PML_links, :]), 0, -1)]).transpose()
        self.sigma_p_y[:, breedte_PML_rechts:] = np.array(
            [[sigma_max_right * (i / len(self.sigma_p_y[breedte_PML_rechts:, :])) ** m] * self.sigma_p_y.shape[0] for i in
            range(0, self.sigma_p_y.shape[1] - breedte_PML_rechts, 1)]).transpose()
        return self

    def run_init(self):
        ## SOURCE AND RECEIVER LOCATIONS
        self.y_bron = self.npml + self.nd
        self.x_bron = int(self.nd / 10)
        self.x_recorder1 = int(self.nd / 2)
        self.x_recorder2 = int(self.nd / 2)
        self.x_recorder3 = int(self.nd / 2)
        self.y_recorder1 = self.y_bron + 2 * self.nd  + self.obj_thickness # y location receiver 1
        self.y_recorder2 = self.y_recorder1 + self.nd  # y location receiver 2
        self.y_recorder3 = self.y_recorder2 + self.nd  # y location receiver 3

        if not self.freefield:
            self.PML_init()
        else:
            self.nx *= 4 # enlarge simulation domain for freefield
            self.ny = self.nx
            self.x_bron += int(self.nx/2) # shift source and recorders location to middle
            self.y_bron = int(self.ny/2)
            self.x_recorder1 += int(self.nx/2)
            self.x_recorder2 += int(self.nx/2)
            self.x_recorder3 += int(self.nx/2)
            self.y_recorder1 = self.y_bron + 2 * self.nd  + self.obj_thickness # y location receiver 1
            self.y_recorder2 = self.y_recorder1 + self.nd  # y location receiver 2
            self.y_recorder3 = self.y_recorder2 + self.nd  # y location receiver 3

        print(self.x_recorder1*self.dx,self.y_recorder1*self.dy)
        print(self.x_bron*self.dx,self.y_bron*self.dy)
        ## P and O fields
        self.ox = np.zeros((self.nx + 1, self.ny))
        self.oy = np.zeros((self.nx, self.ny + 1))
        self.p = np.zeros((self.nx, self.ny))

        ## Timeseries
        self.recorder1 = np.zeros(self.nt)
        self.recorder2 = np.zeros(self.nt)
        self.recorder3 = np.zeros(self.nt)

        self.bront = np.zeros(self.nt)
        self.tijdreeks = np.zeros(self.nt)
        return self

    def source(self,t):
        bron = self.A * np.sin(2 * np.pi * self.fc *(t-self.t0)) * np.exp(-((t - self.t0) ** 2) / (self.sigma))
        return bron

    ## RUN SIMULATION--------------------------------------------
    def run(self):
        self.params_init()
        self.run_init()
        if self.animation:
            self.animation_init()
        for it in range(0, self.nt):
            t = it * self.dt
            self.tijdreeks[it] = t
            print('%d/%d' % (it, self.nt))

            prefactor = self.c*self.dt/(self.dx**2)
            bron = self.source(t)
            self.p[self.x_bron, self.y_bron] += bron*prefactor  # adding source term to propagation
            self.bront[it] = bron
            
            if not self.freefield:
                self.timestep() # propagate over one time step
                self.hard_walls_o() # implement the hard walls
            else:
                self.freefield_timestep()

            self.recorder1[it] = self.p[self.x_recorder1, self.y_recorder1]  # store p field at receiver locations
            self.recorder2[it] = self.p[self.x_recorder2, self.y_recorder2]
            self.recorder3[it] = self.p[self.x_recorder3, self.y_recorder3]

            if self.animation:
                self.animate_2D(it)

        if self.animation:
            my_anim = ArtistAnimation(self.fig, self.movie, interval=50, repeat_delay=1000,
                                  blit=True)
            choice = 3
            while choice not in [0,1,2]:
                choice = int(input('__ANIMATION__ Show(0) or Save(1), or Abort(2)?:'))
            if choice == 1:
                path = '{}_kd={}.gif'.format(self.obj, self.kd)
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
        if self.freefield:
            choice = 2
            while choice not in [0,1]:
                choice = int(input('__FREEFIELD__ Save recorders?(0), or Abort(1)?:'))
            if choice == 0:
                recorders = [self.recorder1,self.recorder2,self.recorder3]
                path = '{}_recorders_kd={}.npy'.format(self.obj, self.kd)
                print('Saving as: {}'.format(path))
                np.save(path,recorders)
                path = '{}_source_kd={}.npy'.format(self.obj, self.kd)
                print('Saving as: {}'.format(path))
                np.save(path,self.bront)
                print('Done')
        return self

    def timestep(self):
        self.p_y = (np.append(self.p, self.p[:, 0].reshape((self.nx, 1)), axis=1) - np.append(self.p[:, -1].reshape((self.nx, 1)), self.p, axis=1)) / self.dy
        self.p_x = (np.append(self.p, self.p[0, :].reshape((1, self.ny)), axis=0) - np.append(self.p[-1, :].reshape((1, self.ny)), self.p, axis=0)) / self.dx

        self.no_periodic_boundaries() # implement no periodic boundaries
        self.hard_walls_derivative() # implement hard walls for the derivatives

        self.ox = self.ox * (1 - self.dt * self.sigma_x) - self.dt * self.p_x
        self.oy = self.oy * (1 - self.dt * self.sigma_y) - self.dt * self.p_y
        self.ox_x = (self.ox[1:, :] - self.ox[:-1, :]) / self.dx
        self.oy_y = (self.oy[:, 1:] - self.oy[:, :-1]) / self.dy

        self.p = self.p - (self.c ** 2) * self.dt * (self.ox_x + self.oy_y) - (self.sigma_p_y + self.sigma_p_x) * self.p * self.dt
        return self
    
    def freefield_timestep(self):
        self.p_y = (np.append(self.p, self.p[:, 0].reshape((self.nx, 1)), axis=1) - np.append(self.p[:, -1].reshape((self.nx, 1)), self.p, axis=1)) / self.dy
        self.p_x = (np.append(self.p, self.p[0, :].reshape((1, self.ny)), axis=0) - np.append(self.p[-1, :].reshape((1, self.ny)), self.p, axis=0)) / self.dx

        self.ox -= self.dt * self.p_x
        self.oy -= self.dt * self.p_y

        self.ox_x = (self.ox[1:, :] - self.ox[:-1, :]) / self.dx
        self.oy_y = (self.oy[:, 1:] - self.oy[:, :-1]) / self.dy

        self.p -= (self.c ** 2) * self.dt * (self.ox_x + self.oy_y) 
        return self

    ## WALLS AND BOUNDARIES--------------------------------------
    def no_periodic_boundaries(self):
        self.p_y[-1, :] = 0  # No periodic boundaries
        self.p_x[-1, :] = 0  # No periodic boundaries
        self.p_y[:, -1] = 0  # No periodic boundaries
        self.p_x[:, -1] = 0  # No periodic boundaries
        return self

    def hard_walls_o(self):
        if self.obj in ['thin','thick']:
            self.ox[:self.x_obj, self.y_obj:self.y_obj+self.obj_thickness] = 0 
            self.oy[:self.x_obj, self.y_obj:self.y_obj+self.obj_thickness] = 0  

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
        if self.obj in ['thin','thick']:
            self.p_x[:self.x_obj+1, self.y_obj:self.y_obj+self.obj_thickness+1] = 0 
            self.p_y[:self.x_obj+1, self.y_obj:self.y_obj+self.obj_thickness+1] = 0  
        elif self.obj == 'triangle':
            for i in range(0, int(self.nd / 2) + 1):
                self.p_x[: 4 * i, 2 * self.nd + i + self.npml] = 0  # half triangle
                self.p_y[: 4 * i, 2 * self.nd + i + self.npml] = 0  # half triangle
                self.p_x[: 4 * i, 3 * self.nd - i + self.npml] = 0  # half triangle
                self.p_y[: 4 * i, 3 * self.nd - i + self.npml] = 0  # half triangle
        return self

    ## POSTPROCESSING--------------------------------------------
    def animation_init(self):
        self.fig, self.ax = plt.subplots()
        plt.xlabel('x/d')
        plt.ylabel('y/d')
        plt.ylim([1, self.nx])
        plt.xlim([1, self.ny])
        self.ax.set_yticks(np.linspace(0, 4 * self.nd, 5))
        self.ax.set_yticklabels(np.arange(5))
        if self.obj in ['thin','freefield_thin']:
            self.ax.set_xticks(self.npml + self.nd * np.arange(7))
            self.ax.set_xticklabels(np.arange(7))
        else: 
            self.ax.set_xticks(self.npml + self.nd * np.arange(8))
            self.ax.set_xticklabels(np.arange(8))
            if self.obj == 'freefield_thick':
                self.rect = patches.Rectangle((self.y_obj, 0), self.obj_thickness + self.dx, self.x_obj + self.dx, fill=False, ls=':')
            elif self.obj == 'thick':
                self.rect = patches.Rectangle((self.y_obj, 0), self.obj_thickness + self.dx, self.x_obj + self.dx, facecolor = 'k')
            elif self.obj == 'triangle':
                vert = np.array([(int(2 * self.nd) + self.npml, 0), (int(5 * self.nd / 2) + self.npml, int(2 * self.nd)), (int(3 * self.nd) + self.npml, 0)])
                self.tr = patches.Polygon(vert, closed=True, facecolor='k')
            elif self.obj == 'freefield_triangle':
                vert = np.array([(int(2 * self.nd) + self.npml, 0), (int(5 * self.nd / 2) + self.npml, int(2 * self.nd)), (int(3 * self.nd) + self.npml, 0)])
                self.tr = patches.Polygon(vert, closed=True, fill=False, ls=':')
        self.movie = []
        return self

    def animate_2D(self,it):
        if self.obj in ['thin','freefield_thin']:
            if self.obj == 'freefield_thin':
                obj_plot = \
                self.ax.plot(np.full(np.arange(int(2 * self.nd)).shape, self.y_obj), np.arange(self.x_obj),
                        color='k', ls=':', linewidth=60 * self.dx)[0]
            else:
                obj_plot = \
                self.ax.plot(np.full(np.arange(int(2 * self.nd)).shape, self.y_obj), np.arange(self.x_obj),
                        color='k', linewidth=60 * self.dx)[0]
        elif self.obj in ['thick','freefield_thick']:
            obj_plot = self.ax.add_patch(self.rect)
        elif self.obj in ['triangle','freefield_triangle']:
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

    def plot_recorders(self, recorders = []):
        if not self.initialized and recorders == []:
            raise ValueError('No fields were recorded yet, first run the simulation')
        elif self.initialized and recorders == []:
            recorders = [self.recorder1,self.recorder2,self.recorder3]
            tijd = self.tijdreeks
        else:
            self.params_init()
            tijd = np.arange(recorders.shape[1])*self.dt
        plt.plot(tijd,recorders[0],label='Recorder 1')
        plt.plot(tijd,recorders[1],label='Recorder 2')
        plt.plot(tijd,recorders[2],label='Recorder 3')
        plt.title('Recorded pressure vs time, {} object case, kd={}'.format(self.obj,self.kd))
        plt.xlabel('Time [s]')
        plt.ylabel('Pressure [N/m^2]')
        plt.legend()
        plt.grid()
        plt.show()
    
    def plot_source(self, source = []):
        if not self.initialized and source == []:
            raise ValueError('No fields were recorded yet, first run the simulation')
        elif self.initialized and source == []:
            tijd = self.tijdreeks
            source = self.source(tijd)
        else:
            self.params_init()
            tijd = np.arange(source.shape[0])*self.dt
        plt.plot(tijd,source)
        plt.title('Pressure generated by the source, kd={}'.format(self.kd))
        plt.xlabel('Time [s]')
        plt.ylabel('Pressure [N/m^2]')
        plt.grid()
        plt.show()

    def TF_ANA(self,recorder_number,plot=False):
        if not self.initialized:
            self.params_init()
        if self.obj == 'thin':
            TF = self.TF_ANA_THIN(recorder_number,plot)
            return TF, self.k_vec
        else:
            print('No analytical transferfunction available for given object')

    def TF_ANA_THIN(self,recorder_number,plot):
        self.n = 2
        self.k_vec = np.linspace(0.1,10,self.nt) # kd vector
        theta_R = 2*np.pi - np.arctan([2/3,4/3,2])[recorder_number-1] # angle between recorder and sheet
        theta_R_m = 2*np.pi - np.arctan([2/5,4/5,6/5])[recorder_number-1] # angle between recorder and mirror sheet
        theta_S, theta_S_m = (np.arctan(10/19),np.arctan(10/21)) # angle between source and sheet and mirror source
        a,a_m = (np.sqrt(1.9**2 + 1)*self.d,np.sqrt(2.1**2 + 1)*self.d) # distance from source to top of sheet and mirror source
        b = np.array([np.sqrt(1.5**2 + 1)*self.d,np.sqrt(1.5**2 + 4)*self.d,np.sqrt(1.5**2 + 9)*self.d])[recorder_number-1]  # distance from recorder to top of sheet
        b_m = np.array([np.sqrt(2.5**2 + 1)*self.d,np.sqrt(2.5**2 + 4)*self.d,np.sqrt(2.5**2 + 9)*self.d])[recorder_number-1]  # distance from recorder to top of mirror sheet
        D_up = self.Diff_coeff(theta_S,theta_R,a,b) # source diffraction coefficient 
        D_m_up = self.Diff_coeff(theta_S_m,theta_R,a_m,b) # source reflection diffraction coefficient 
        D_down = self.Diff_coeff(theta_S,theta_R_m,a,b_m) # source diffraction coefficient mirror wedge
        D_m_down = self.Diff_coeff(theta_S_m,theta_R_m,a_m,b_m) # source reflection diffraction coefficient mirror wedge
        TF = self.k_vec*((D_up+D_down)*hankel2(0,self.k_vec*a) + (D_m_up+D_m_down)*hankel2(0,self.k_vec*a_m))/4
        Amp = np.abs(TF)
        Phase = np.unwrap(np.angle(TF)) 
        if plot:
            ax1 = plt.subplot(2,1,1)
            ax1.plot(self.k_vec,Amp)
            ax1.set_ylabel('Amplitude')
            ax1.set_title('Analytical transfer function, thin sheet case')
            ax1.grid()
            ax2 = plt.subplot(2,1,2)
            ax2.plot(self.k_vec,Phase)
            ax2.set_xlabel('kd')
            ax2.set_ylabel('Phase')
            ax2.grid()
            plt.show()
        return TF

    def TF_SIM(self,recorder_number,plot=False, recorders = [], source = []):
        if not self.initialized and recorders == [] and source == []:
            raise ValueError('No fields were recorded yet, first run the simulation')
        elif self.initialized and recorders == [] and source == []:
            recorder = [self.recorder1,self.recorder2,self.recorder3][recorder_number-1]
            time = np.arange(self.nt)*self.dt
            source = self.bront
        else:
            self.params_init()
            time = np.arange(self.nt)*self.dt
            recorder = recorders[recorder_number-1]

        n_samples = 9000
        FFT_source, k_vec = self.fft(source,n_samples)
        FFT_recorder, _ = self.fft(recorder,n_samples)
        TF = FFT_recorder/FFT_source
        return TF, k_vec

    def fft(self,f, n_samples):
        if not self.initialized:
            self.params_init()
        else:
            pass
        k_vec = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(n_samples,d=self.dt))/self.c
        self.bw_min = np.argmin(np.absolute(k_vec - 0.5*self.k))
        self.bw_max = np.argmin(np.absolute(k_vec - 1.5*self.k))
        FFT = np.fft.fftshift(np.fft.fft(f,n_samples))[self.bw_min:self.bw_max]
        return FFT, k_vec[self.bw_min:self.bw_max]

    def plot_time_fft(self,time,source,recorder,k_vec,FFT_source,FFT_recorder):
        ax1 = plt.subplot(2,3,1)
        ax1.plot(time,source)
        ax1.set_ylabel('Pressure [N/m^2]')
        ax1.set_xlabel('Time [s]')
        ax1.set_title('Source (t)')
        ax1.grid()

        ax2 = plt.subplot(2,3,2)
        ax2.plot(k_vec*self.d,np.abs(FFT_source))
        ax2.set_xlabel('kd')
        ax2.set_title('Source amplitude (FFT)')
        ax2.set_ylabel('Amplitude')
        ax2.set_xlim([0.5*self.kd, 1.5*self.kd])
        ax2.grid()

        ax3 = plt.subplot(2,3,3)
        ax3.plot(k_vec*self.d,np.unwrap(np.angle(FFT_source)))
        ax3.set_xlabel('kd')
        ax3.set_ylabel('Phase')
        ax3.set_title('Source phase (FFT)')
        ax3.set_xlim([0.5*self.kd, 1.5*self.kd])
        ax3.grid()

        ax4 = plt.subplot(2,3,4)
        ax4.plot(time,recorder)
        ax4.set_ylabel('Pressure [N/m^2]')
        ax4.set_xlabel('Time [s]')
        ax4.set_title('Recorder (t)')
        ax4.grid()

        ax5 = plt.subplot(2,3,5)
        ax5.plot(k_vec*self.d,np.abs(FFT_recorder))
        ax5.set_xlabel('kd')
        ax5.set_title('Recorder amplitude (FFT)')
        ax5.set_ylabel('Amplitude')
        ax5.set_xlim([0.5*self.kd, 1.5*self.kd])
        ax5.grid()

        ax6 = plt.subplot(2,3,6)
        ax6.plot(k_vec*self.d,np.unwrap(np.angle(FFT_recorder)))
        ax6.set_xlabel('kd')
        ax6.set_ylabel('Phase')
        ax6.set_title('Recorder phase (FFT)')
        ax6.set_xlim([0.5*self.kd, 1.5*self.kd])
        ax6.grid()
        plt.suptitle('Summarising time and fft plots for source and recorder')
        plt.tight_layout()
        plt.show()
        
    def plot_FDTD_ana(self,k_vec,TF_ana,TF_sim):
        Amplratio = np.abs(TF_sim/TF_ana)
        Phasediff = np.unwrap(np.angle(TF_sim)) - np.unwrap(np.angle(TF_ana))
        ax1 = plt.subplot(2,2,1)
        ax1.plot(k_vec*self.d,Amplratio)
        ax1.set_xlabel('kd')
        ax1.set_ylabel('Ratio')
        ax1.set_title('Amplitude ratio')
        ax1.set_xlim([0.5*self.kd, 1.5*self.kd])
        ax1.grid()

        ax2 = plt.subplot(2,2,2)
        ax2.plot(k_vec*self.d,Phasediff)
        ax2.set_xlabel('kd')
        ax2.set_ylabel('Difference')
        ax2.set_xlim([0.5*self.kd, 1.5*self.kd])
        ax2.set_title('Phase difference')
        ax2.grid()

        ax3 = plt.subplot(2,2,3)
        ax3.plot(k_vec*self.d,np.abs(TF_sim),label='FDTD')
        ax3.plot(k_vec*self.d,np.abs(TF_ana),label='Analytical')
        ax3.set_ylabel('Amplitude')
        ax3.set_xlabel('kd')
        ax3.set_title('Comparison of amplitudes')
        ax3.set_xlim([0.5*self.kd, 1.5*self.kd])
        ax3.grid()
        ax3.legend()

        ax4 = plt.subplot(2,2,4)
        ax4.plot(k_vec*self.d,np.unwrap(np.angle(TF_sim)),label='FDTD')
        ax4.plot(k_vec*self.d,np.unwrap(np.angle(TF_ana)),label='Analytical')
        ax4.set_ylabel('Phase')
        ax4.set_xlabel('kd')
        ax4.set_title('Comparison of phases')
        ax4.set_xlim([0.5*self.kd, 1.5*self.kd])
        ax4.grid()
        ax4.legend()

        plt.suptitle('FDTD vs analytical transfer function at recorder: {} case, kd = {}'.format(self.obj,self.kd))
        plt.tight_layout()
        plt.show()
    
    ## USEFUL FUNCTIONS------------------------------------------
    
    def F(self,x): # Fresnel coefficients
        S1,C1 = sc.fresnel(np.inf)
        S2,C2 = sc.fresnel(np.sqrt(2*x/np.pi))
        I = np.sqrt(np.pi/2)*((C1 + 1j*S1) - (C2 + 1j*S2))
        out = -2*1j*np.sqrt(x)*np.exp(-1j*x)*I
        return out

    def cotg(self,x): # Cotangent with x in rads
        return sc.cotdg(np.rad2deg(x))

    def Diff_coeff(self,theta_S,theta_R,a,b):
        a_plus = theta_R + theta_S
        a_min = theta_R - theta_S
        L = a*b/(a+b) 
        N_plus = lambda a: np.round((np.pi+a)/(2*np.pi*self.n)).astype(int) # N+, integer
        N_min = lambda a: np.round(-(np.pi-a)/(2*np.pi*self.n)).astype(int) # N-, integer
        A_plus = lambda a: 2*np.cos((2*self.n*np.pi*N_plus(a)-a)/2)**2 # A+(a)
        A_min = lambda a: 2*np.cos((2*self.n*np.pi*N_min(a)-a)/2)**2 # A-(a)
        C = np.exp(1j*self.k_vec*(a+b)+1j*np.pi/4)/(2*self.n*np.sqrt(L*2*np.pi*self.k_vec)) # prefactor of diffraction coefficient
        D1 = self.cotg((np.pi-a_min)/(2*self.n))*self.F(self.k_vec*L*A_min(a_min)) # 1st term of diffraction coefficient
        D2 = self.cotg((np.pi-a_plus)/(2*self.n))*self.F(self.k_vec*L*A_min(a_plus)) # 2nd term of diffraction coefficient
        D3 = self.cotg((np.pi+a_plus)/(2*self.n))*self.F(self.k_vec*L*A_plus(a_plus))# 3th term of diffraction coefficient
        D4 = self.cotg((np.pi+a_min)/(2*self.n))*self.F(self.k_vec*L*A_plus(a_min)) # 4th term of diffraction coefficient
        D = (D1 + D2 + D3 + D4)*C # D_11 look only at solutions 1 to start
        return D