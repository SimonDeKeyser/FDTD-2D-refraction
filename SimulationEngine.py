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
    def __init__(self, dx, kd, dt, nt, source, kdmin, kdmax ,obj, animation=False):
        self.dx, self.kd, self.dt, self.nt = dx, kd, dt, nt
        self.kdmin, self.kdmax = kdmin, kdmax
        self.animation = animation
        self.freefield = False
        self.runned = False
        self.A, self.sigma = source
        if obj in ['thin', 'freefield_thin', 'thick', 'freefield_thick','triangle','freefield_triangle']:
            self.obj = obj
            self.params_init()
        else:
            raise ValueError('Choose obj: thin, freefield_thin, thick, freefield_thick,triangle,freefield_triangle')
    
    ## INITIALIZATION--------------------------------------------
    def params_init(self):
        if 'freefield' in  self.obj:
            self.freefield = True
        ## PARAMETERS
        self.c = 340  # geluidssnelheid - speed of sound (wave speed)
        self.dy = self.dx
        self.d = 1  # lengte d
        self.k = self.kd / self.d  # wavenumber
        self.npml = 70  # Extra layers around simulation domain
        self.nx = self.npml + int(4 * self.d / self.dx)  # number of cells in x direction
        self.nd = int(self.d / self.dx)  # number of cells in d length

        ## SOURCE PARAMETERS
        self.fc = self.k * self.c / (2 * np.pi)
        self.t0 = self.sigma*1e3

        if self.obj in ['thin','freefield_thin']:
            self.L = 6 * self.d  # length of simulation domain
            self.obj_thickness = 0
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
        hoogte_PML = 1 # Height from which wave starts damping (numbers of layers)
        breedte_PML_links = 1# How much to the right of left simulation wall will wave start damping (numbers of layers)
        breedte_PML_rechts = 1 # How much to the left of right simulation wall will wave start damping (numbers of layers)
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

        ## P and O fields
        self.ox = np.zeros((self.nx + 1, self.ny))
        self.oy = np.zeros((self.nx, self.ny + 1))
        self.p = np.zeros((self.nx, self.ny))

        ## Timeseries
        self.recorder1 = np.zeros(self.nt)
        self.recorder2 = np.zeros(self.nt)
        self.recorder3 = np.zeros(self.nt)

        self.bront = np.zeros(self.nt)
        return self

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

    ## RUN SIMULATION--------------------------------------------
    def source(self,t):
        bron = self.A * np.sin(2 * np.pi * self.fc *(t-self.t0)) * np.exp(-((t - self.t0) ** 2) / (self.sigma))
        return bron

    def run(self):
        self.run_init()
        if self.animation:
            self.animation_init()
        for it in range(0, self.nt):
            t = it * self.dt
            print('%d/%d' % (it, self.nt))

            bron = self.source(t)
            self.bront[it] = bron
            prefactor = self.c*self.dt/(self.dx**2)
            self.p[self.x_bron, self.y_bron] += bron*prefactor  # adding source term to propagation
            
            if not self.freefield:
                self.timestep() # propagate over one time step
            else:
                self.freefield_timestep()

            self.recorder1[it] = self.p[self.x_recorder1, self.y_recorder1]  # store p field at receiver locations
            self.recorder2[it] = self.p[self.x_recorder2, self.y_recorder2]
            self.recorder3[it] = self.p[self.x_recorder3, self.y_recorder3]

            if self.animation:
                self.animate_2D(it)

        self.output()
        self.runned = True
        return self

    def timestep(self):
        self.ox[1:-1,:] -= (self.dt/self.dx) * (self.p[1:,:]-self.p[:-1,:])
        self.oy[:,1:-1] -= (self.dt/self.dx) * (self.p[:,1:]-self.p[:,:-1])

        self.ox_x = (self.ox[1:, :] - self.ox[:-1, :]) / self.dx
        self.oy_y = (self.oy[:, 1:] - self.oy[:, :-1]) / self.dy
        self.hard_walls()
        self.p -= (self.c ** 2) * self.dt * (self.ox_x + self.oy_y)
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

    def save(self):
        recorders = [self.recorder1,self.recorder2,self.recorder3]
        path = '{}_recorders_kd={}.npy'.format(self.obj, self.kd)
        print('Saving as: {}'.format(path))
        np.save(path,recorders)
        path = '{}_source_kd={}.npy'.format(self.obj, self.kd)
        print('Saving as: {}'.format(path))
        np.save(path,self.bront)
        print('Done')

    def load(self):
        try:
            recorders = np.load('{}_recorders_kd={}.npy'.format(self.obj, self.kd)) #loading saved recorders, MAKE SURE YOU USED THE SAME PARAMETERS
            source = np.load('{}_source_kd={}.npy'.format(self.obj, self.kd))
            return recorders,source
        except FileNotFoundError:
            print('ERROR: Run the simulation first and save before loading')
            quit()

    ## WALLS AND BOUNDARIES--------------------------------------
    def hard_walls(self):
        if self.obj == 'thin':
            self.ox_x[:self.x_obj+1, self.y_obj] = 0 
            self.oy_y[:self.x_obj+1, self.y_obj] = 0 
        elif self.obj == 'thick':
            self.oy_y[:self.x_obj+1, self.y_obj:self.y_obj+self.obj_thickness] = 0 
            self.ox_x[:self.x_obj+1, self.y_obj:self.y_obj+self.obj_thickness] = 0 
        elif self.obj == 'triangle':
            for i in range(0, int(self.nd / 2) + 1):
                self.ox_x[: 4 * i+1, 2 * self.nd + i + self.npml] = 0  # half triangle
                self.oy_y[: 4 * i+1, 2 * self.nd + i + self.npml] = 0  # half triangle
                self.ox_x[: 4 * i+1, 3 * self.nd - i + self.npml] = 0  # half triangle
                self.oy_y[: 4 * i+1, 3 * self.nd - i + self.npml] = 0  # half triangle
        return self

    ## POSTPROCESSING--------------------------------------------
    def output(self):
        if self.animation:
            my_anim = ArtistAnimation(self.fig, self.movie, interval=50, repeat_delay=1000,
                                  blit=True)
            choice = 3
            while choice not in ['0','1','A','a']:
                choice = input('__ANIMATION__ Show(0) // Save(1) // Abort(A):')
            if choice == '1':
                path = '{}_kd={}.gif'.format(self.obj, self.kd)
                print('Saving animation as: {}'.format(path))
                my_anim.save(path, writer='pillow', fps=30)
                plt.close()
                print('Done')
            elif choice == '0':
                print('Showing animation...')
                plt.show()
            else:
                plt.close()
                pass
    
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

    def TF_ANA(self,recorder_number):
        if self.obj == 'thin':
            TF = self.TF_thin(recorder_number)
            return TF
        elif self.obj == 'freefield_thin':
            TF = self.TF_freefield_thin(recorder_number)
            return TF
        elif self.obj == 'thick':
            TF = self.TF_thick(recorder_number)
            return TF
        elif self.obj == 'triangle':
            TF = self.TF_triangle(recorder_number)
            return TF
        else:
            print('No analytical transferfunction available (yet) for given object')
            quit()

    def TF_thin(self,recorder_number):
        self.n = 2
        k_vec,_,_= self.k_vec()
        theta_R = 2*np.pi - np.arctan([2/3,4/3,2])[recorder_number-1] # angle between recorder and sheet
        theta_R_m = 2*np.pi - np.arctan([2/5,4/5,6/5])[recorder_number-1] # angle between recorder and mirror sheet
        theta_S, theta_S_m = (np.arctan(10/19),np.arctan(10/21)) # angle between source and sheet and mirror source
        a,a_m = (np.sqrt(1.9**2 + 1)*self.d,np.sqrt(2.1**2 + 1)*self.d) # distance from source to top of sheet and mirror source
        b = np.array([np.sqrt(1.5**2 + 1)*self.d,np.sqrt(1.5**2 + 4)*self.d,np.sqrt(1.5**2 + 9)*self.d])[recorder_number-1]  # distance from recorder to top of sheet
        b_m = np.array([np.sqrt(2.5**2 + 1)*self.d,np.sqrt(2.5**2 + 4)*self.d,np.sqrt(2.5**2 + 9)*self.d])[recorder_number-1]  # distance from recorder to top of mirror sheet
        D_up = self.Diffraction(k_vec,theta_S,theta_R,a,b) # source diffraction 
        D_m_up = self.Diffraction(k_vec,theta_S_m,theta_R,a_m,b) # source reflection diffraction 
        D_down = self.Diffraction(k_vec,theta_S,theta_R_m,a,b_m) # source diffraction at mirror wedge
        D_m_down = self.Diffraction(k_vec,theta_S_m,theta_R_m,a_m,b_m) # source reflection diffraction at mirror wedge
        TF = D_up + D_down + D_m_up + D_m_down
        return TF
    
    def TF_thick(self,recorder_number):
        self.n = 1.5
        k_vec,_,_= self.k_vec()
        theta_R = 2*np.pi - np.arctan([2/3,4/3,2])[recorder_number-1] # angle between recorder and sheet
        theta_R_m = 2*np.pi - np.arctan([2/5,4/5,6/5])[recorder_number-1] # angle between recorder and mirror sheet
        theta_S, theta_S_m = (np.arctan(10/19),np.arctan(10/21)) # angle between source and sheet and mirror source
        a,a_m = (np.sqrt(1.9**2 + 1)*self.d,np.sqrt(2.1**2 + 1)*self.d) # distance from source to top of sheet and mirror source
        b = np.array([np.sqrt(1.5**2 + 1)*self.d,np.sqrt(1.5**2 + 4)*self.d,np.sqrt(1.5**2 + 9)*self.d])[recorder_number-1]  # distance from recorder to top of sheet
        b_m = np.array([np.sqrt(2.5**2 + 1)*self.d,np.sqrt(2.5**2 + 4)*self.d,np.sqrt(2.5**2 + 9)*self.d])[recorder_number-1]  # distance from recorder to top of mirror sheet
        D_up = self.Diffraction(k_vec,theta_S,0,a,self.d)*self.Diffraction(k_vec,0,theta_R,a+self.d,b,order=2) # source diffraction 
        D_m_up = self.Diffraction(k_vec,theta_S_m,0,a_m,self.d)*self.Diffraction(k_vec,0,theta_R,a_m+self.d,b,order=2)  # source reflection diffraction 
        D_down = self.Diffraction(k_vec,theta_S,0,a,self.d)*self.Diffraction(k_vec,0,theta_R_m,a+self.d,b_m,order=2) # source diffraction at mirror wedge
        D_m_down = self.Diffraction(k_vec,theta_S_m,0,a_m,self.d)*self.Diffraction(k_vec,0,theta_R_m,a_m+self.d,b_m,order=2) # source reflection diffraction at mirror wedge
        TF = D_up + D_down + D_m_up + D_m_down
        return TF

    def TF_triangle(self,recorder_number):
        phi = np.arctan(1/4) #half top angle of triangle
        self.n =(2*np.pi - 2*phi)/np.pi
        k_vec,_,_= self.k_vec()
        theta_R = 2*np.pi - phi- np.arctan([3/3,5/3,7/3])[recorder_number-1] # angle between recorder and sheet
        theta_R_m = 2*np.pi - np.arctan([3/5,5/5,7/5])[recorder_number-1] # angle between recorder and mirror sheet
        theta_S, theta_S_m = (np.arctan(15/19)-phi,np.arctan(15/21)-phi) # angle between source and sheet and mirror source
        a,a_m = (np.sqrt(1.9**2 + 1.5**2)*self.d,np.sqrt(2.1**2 + 1.5**2)*self.d) # distance from source to top of sheet and mirror source
        b = np.array([np.sqrt(1.5**2 + 1.5**2)*self.d,np.sqrt(1.5**2 + 2.5**2)*self.d,np.sqrt(1.5**2 + 3.5**2)*self.d])[recorder_number-1]  # distance from recorder to top of sheet
        b_m = np.array([np.sqrt(2.5**2 + 1.5**2)*self.d,np.sqrt(2.5**2 + 2.5**2)*self.d,np.sqrt(2.5**2 + 3.5**2)*self.d])[recorder_number-1]  # distance from recorder to top of mirror sheet
        D_up = self.Diffraction(k_vec,theta_S,theta_R,a,b) # source diffraction 
        D_m_up = self.Diffraction(k_vec,theta_S_m,theta_R,a_m,b) # source reflection diffraction 
        D_down = self.Diffraction(k_vec,theta_S,theta_R_m,a,b_m) # source diffraction at mirror wedge
        D_m_down = self.Diffraction(k_vec,theta_S_m,theta_R_m,a_m,b_m) # source reflection diffraction at mirror wedge
        TF = D_up + D_down + D_m_up + D_m_down
        return TF

    def TF_freefield_thin(self,recorder_number):
        k_vec,_,_ = self.k_vec()
        r = lambda i: np.sqrt((1+i)**2+(2/5)**2)*self.d
        r = r(recorder_number)
        TF = -1j*hankel2(0,k_vec*r)*k_vec/4
        return TF

    def TF_FDTD(self,recorder_number, recorders = [], source = []):
        if not self.runned and recorders == [] and source == []:
            raise ValueError('No fields were recorded yet, first run the simulation')
        elif self.runned and recorders == [] and source == []:
            recorder = [self.recorder1,self.recorder2,self.recorder3][recorder_number-1]
            source = self.bront
        else:
            recorder = recorders[recorder_number-1]

        FFT_source = self.fft(source)
        FFT_recorder = self.fft(recorder)
        TF = FFT_recorder/FFT_source
        return TF

    def fft(self, f):
        n_samples = 9000
        k_vec, bw_min, bw_max = self.k_vec()
        FFT = np.fft.fftshift(np.fft.fft(f,n_samples))[bw_min:bw_max]
        return FFT

    def k_vec(self):
        n_samples = 9000
        k_vec = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(n_samples,d=self.dt))/self.c
        bw_min = np.argmin(np.absolute(k_vec - self.kdmin/self.d))
        bw_max = np.argmin(np.absolute(k_vec - self.kdmax/self.d))
        k_vec = k_vec[bw_min:bw_max]
        return k_vec, bw_min, bw_max

    def time_fft_summary(self, recorder_number, recorders=[], source=[]):
        if not self.runned and recorders == [] and source == []:
            raise ValueError('No fields were recorded yet, first run the simulation')
        elif self.runned and recorders == [] and source == []:
            recorder = [self.recorder1,self.recorder2,self.recorder3][recorder_number-1]
            source = self.bront
        else:
            if self.nt != len(source):
                print('Make sure that the loaded simulation has the same parameters as the FDTD class')
                print('Run the simulation and save or redefine the parameters according to the saved file')
                quit()
            recorder = recorders[recorder_number-1]

        FFT_source = self.fft(source)
        FFT_recorder = self.fft(recorders[0])
        time = np.arange(self.nt)*self.dt
        k_vec,_,_ = self.k_vec()

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
        ax2.grid()

        ax3 = plt.subplot(2,3,3)
        ax3.plot(k_vec*self.d,np.unwrap(np.angle(FFT_source)))
        ax3.set_xlabel('kd')
        ax3.set_ylabel('Phase')
        ax3.set_title('Source phase (FFT)')
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
        ax5.grid()

        ax6 = plt.subplot(2,3,6)
        ax6.plot(k_vec*self.d,np.unwrap(np.angle(FFT_recorder)))
        ax6.set_xlabel('kd')
        ax6.set_ylabel('Phase')
        ax6.set_title('Recorder phase (FFT)')
        ax6.grid()
        plt.suptitle('Summarising time and fft plots for source and recorder')
        plt.tight_layout()
        plt.show()
        
    def FDTD_ana_comparison(self,TF_sim,TF_ana):
        k_vec, _, _ = self.k_vec()
        Amplratio = np.abs(TF_sim/TF_ana)
        Phasediff = np.unwrap(np.angle(TF_sim)) - np.unwrap(np.angle(TF_ana))

        ax1 = plt.subplot(2,2,1)
        ax1.plot(k_vec*self.d,Amplratio)
        ax1.set_xlabel('kd')
        ax1.set_ylabel('Ratio')
        ax1.set_title('Amplitude ratio')
        ax1.grid()

        ax2 = plt.subplot(2,2,2)
        ax2.plot(k_vec*self.d,Phasediff)
        ax2.set_xlabel('kd')
        ax2.set_ylabel('Difference')
        ax2.set_title('Phase difference')
        ax2.grid()

        ax3 = plt.subplot(2,2,3)
        ax3.plot(k_vec*self.d,np.abs(TF_sim),label='FDTD')
        ax3.plot(k_vec*self.d,np.abs(TF_ana),label='Analytical')
        ax3.set_ylabel('Amplitude')
        ax3.set_xlabel('kd')
        ax3.set_title('Comparison of amplitudes')
        ax3.grid()
        ax3.legend()

        ax4 = plt.subplot(2,2,4)
        ax4.plot(k_vec*self.d,np.unwrap(np.angle(TF_sim)),label='FDTD')
        ax4.plot(k_vec*self.d,np.unwrap(np.angle(TF_ana)),label='Analytical')
        ax4.set_ylabel('Phase')
        ax4.set_xlabel('kd')
        ax4.set_title('Comparison of phases')
        ax4.grid()
        ax4.legend()

        plt.suptitle('FDTD vs analytical transfer function at recorder: {} case, kd = {}'.format(self.obj,self.kd))
        plt.tight_layout()
        plt.show()

    ## USEFUL FUNCTIONS------------------------------------------
    def F(self,x): # Fresnel coefficients
        S1,C1 = sc.fresnel(np.inf)
        S2,C2 = sc.fresnel(np.sqrt(2*x/np.pi))
        I = np.sqrt(2/np.pi)*((C1 + 1j*S1) - (C2 + 1j*S2))
        out = -2*1j*np.sqrt(x)*np.exp(-1j*x)*I
        return out

    def cotg(self,x): # Cotangent with x in rads
        return sc.cotdg(np.rad2deg(x))

    def Diffraction(self,k_vec,theta_S,theta_R,a,b,order=1):
        a_plus = theta_R + theta_S
        a_min = theta_R - theta_S
        L = a*b/(a+b) 
        N_plus = lambda a: np.round((np.pi+a)/(2*np.pi*self.n)).astype(int) # N+, integer
        N_min = lambda a: np.round(-(np.pi-a)/(2*np.pi*self.n)).astype(int) # N-, integer
        A_plus = lambda a: 2*np.cos((2*self.n*np.pi*N_plus(a)-a)/2)**2 # A+(a)
        A_min = lambda a: 2*np.cos((2*self.n*np.pi*N_min(a)-a)/2)**2 # A-(a)
        if self.obj== 'thick' and order == 1:
            C2D = -1j*k_vec*hankel2(0,k_vec*a)/4
        elif self.obj== 'thick' and order == 2:
            C2D = -1j*hankel2(0,k_vec*b)/4
        else:
            C2D = -1*k_vec*hankel2(0,k_vec*b)*hankel2(0,k_vec*a)/(16) # Propagation factor 2D
        C = C2D*np.exp(1j*np.pi/4)/(2*self.n*np.sqrt(2*np.pi*k_vec)) # prefactor of diffraction coefficient
        D1 = self.cotg((np.pi-a_min)/(2*self.n))*self.F(k_vec*L*A_min(a_min)) # 1st term of diffraction coefficient
        D2 = self.cotg((np.pi-a_plus)/(2*self.n))*self.F(k_vec*L*A_min(a_plus)) # 2nd term of diffraction coefficient
        D3 = self.cotg((np.pi+a_plus)/(2*self.n))*self.F(k_vec*L*A_plus(a_plus))# 3th term of diffraction coefficient
        D4 = self.cotg((np.pi+a_min)/(2*self.n))*self.F(k_vec*L*A_plus(a_min)) # 4th term of diffraction coefficient
        D = (D1 + D2 + D3 + D4)*C # D_11 look only at solutions 1 to start
        return D