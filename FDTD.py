import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import matplotlib.patches as patches
from PIL import Image

def step_SIT_SIP(nx, ny, c, dx, dy, dt):
    global ox, oy, p
    p_y = (np.append(p, p[:, 0].reshape((nx, 1)), axis=1) - np.append(p[:, -1].reshape((nx, 1)), p, axis=1)) / dy
    p_x = (np.append(p, p[0, :].reshape((1, ny)), axis=0) - np.append(p[-1, :].reshape((1, ny)), p, axis=0)) / dx

    p_y[-1, :] = 0 #No periodic boundaries
    p_x[-1, :] = 0 #No periodic boundaries
    p_y[:, -1] = 0 #No periodic boundaries
    p_x[:, -1] = 0 #No periodic boundaries
    p_x[:int(2 * nd)+1, int(2 * nd): int(3 * nd)+1] = 0 #Thick object
    p_y[:int(2 * nd)+1, int(2 * nd): int(3 * nd)+1] = 0 #Thick object

    ox = ox*(1-dt*sigma_x) - dt * p_x
    oy = oy*(1-dt*sigma_y) - dt * p_y
    ox_x = (ox[1:, :] - ox[:-1, :]) / dx
    oy_y = (oy[:, 1:] - oy[:, :-1]) / dy

    p = p - (c ** 2) * dt * (ox_x + oy_y)

def Sim_thin(dx,kd,plot=False,save=False):
    global ox, oy, p, nd, sigma_x, sigma_y
    # INITIALISATION 2D-GRID AND SIMULATION PARAMETERS-------------------------
    c = 340  # geluidssnelheid - speed of sound (wave speed)
    dx = 0.05  # ruimtelijke discretisatiestap - spatial discretisation step
    dy = dx

    kd = 1  # frequentie-lengte
    d = 1  # lengte d
    k = kd / d  # wavenumber
    L = 6 * d  # length of simulation domain

    nx = int(4*d / dx)  # aantal cellen in x richting - number of cells in x direction
    ny = int(L / dy)  # aantal cellen in y richting - number of cells in y direction
    nd = int(d / dx)  # aantal cellen in lengte d

    CFL = 0.8  # Courant getal - Courant number

    dt = CFL / (c * np.sqrt((1 / dx ** 2) + (1 / dy ** 2)))  # tijdstap - time step

    nt = 6*int(40 // CFL)  #Number of time steps

    # location of source(central) and receivers
    x_bron = int(nd / 10)
    y_bron = nd

    x_recorder1 = int(nd / 2)
    y_recorder1 = y_bron + 2 * nd  # Location receiver 1

    x_recorder2 = int(nd / 2)
    y_recorder2 = y_recorder1 + nd  # Location receiver 2

    x_recorder3 = int(nd / 2)
    y_recorder3 = y_recorder2 + nd  # Location receiver 2

    # source pulse information
    A = 10
    fc = k * c / (2 * np.pi)
    t0 = 2.5E-2
    sigma = 5E-4

    # PML implementation
    sigma_max_left = 2000 #Max amount of damping left
    sigma_max_right = 100 #Max amount of damping right
    sigma_max_up = 2000 #Max amount of damping upward

    hoogte_PML = 40 #Height from which wave starts damping (numbers of layers)
    breedte_PML_links = 10 #How much to the right of left simulation wall will wave start damping (numbers of layers)
    breedte_PML_rechts = 40 #How much to the left of right simulation wall will wave start damping (numbers of layers)
    sigma_x = np.zeros((nx + 1, ny))
    sigma_y = np.zeros((nx, ny + 1))
    m = 1 #Power of the PML (3 to 4), if too high, the sigma_max is too small

    sigma_x[-hoogte_PML:,:] = [[sigma_max_up*(i/len(sigma_x[-hoogte_PML:,:]))**m]*sigma_x.shape[1] for i in range(0,len(sigma_x[:hoogte_PML,:]),1)]
    sigma_y[:,:breedte_PML_links] = np.array([[sigma_max_left*(i/len(sigma_y[:breedte_PML_links,:]))**m]*sigma_y.shape[0] for i in range(len(sigma_y[:breedte_PML_links,:]),0,-1)]).transpose()
    sigma_y[: ,breedte_PML_rechts: ] = np.array([[i]*sigma_y.shape[0] for i in range(0, sigma_y.shape[1]-breedte_PML_rechts, 1)]).transpose()

    # initialisation of o and p fields
    global ox, oy, p
    ox = np.zeros((nx + 1, ny))
    oy = np.zeros((nx, ny + 1))
    p = np.zeros((nx, ny))

    # initialisation time series receivers
    recorder1 = np.zeros((nt, 1))
    recorder2 = np.zeros((nt, 1))
    recorder3 = np.zeros((nt, 1))

    bront = np.zeros((nt, 1))
    tijdreeks = np.zeros((nt, 1))
    bron = 0

    # TIME ITTERATION----------------------------------------------------
    fig, ax = plt.subplots()
    plt.xlabel('x/d')
    plt.ylabel('y/d')
    plt.ylim([1, 4*nd])
    plt.xlim([1, ny + 1])
    ax.set_yticks(np.linspace(0,4*nd,5))
    ax.set_yticklabels(np.arange(5))
    ax.set_xticks(np.linspace(0,ny,7))
    ax.set_xticklabels(np.arange(7))
    movie = []
    for it in range(0, nt):
        t = (it - 1) * dt
        tijdreeks[it, 0] = t
        print('%d/%d' % (it, nt))

        bron = A * np.sin(2 * np.pi * fc * (t - t0)) * np.exp(-((t - t0) ** 2) / (sigma))  # bron updaten bij nieuw tijd - update source for new time

        p[x_bron, y_bron] = p[x_bron, y_bron] + bron  # druk toevoegen bij de drukvergelijking op bronlocatie - adding source term to propagation
        step_SIT_SIP(nx, ny, c, dx, dy, dt)  # propagatie over 1 tijdstap - propagate over one time step
        ox[:int(2 * nd), int(2 * nd)] = 0  # thin sheet
        oy[:int(2 * nd), int(2 * nd)] = 0  # thin sheet
        ox[0, :] = 0  # ground
        oy[0, :] = 0  # ground

        recorder1[it] = p[x_recorder1, y_recorder1]  # store p field at receiver locations
        recorder2[it] = p[x_recorder2, y_recorder2]
        recorder3[it] = p[x_recorder3, y_recorder3]

        # presenting the p field

        # view(0,90)
        # shading interp
        artists = [
            ax.text(0.5, 1.05, '%d/%d' % (it, nt),
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes),
            ax.imshow(p, vmin=-0.02 * A, vmax=0.02 * A),
            ax.plot(y_bron, x_bron, 'ks', fillstyle="none")[0],
            ax.plot(y_recorder1, x_recorder1, 'ro', fillstyle="none")[0],
            ax.plot(y_recorder2, x_recorder2, 'ro', fillstyle="none")[0],
            ax.plot(y_recorder3, x_recorder3, 'ro', fillstyle="none")[0],
            ax.plot(np.full(np.arange(int(2 * nd)).shape, int(2 * nd)), np.arange(int(2 * nd)), color='k',linewidth=60*dx)[0]
        ]
        movie.append(artists)

        # mov(it) = getframe #wegcommentarieren voor simulatie vlugger te laten lopen - if this line is removed simulation runs much faster
    my_anim = ArtistAnimation(fig, movie, interval=50, repeat_delay=1000,
                            blit=True)
    if save:
        my_anim.save('Thin_kd={}.gif'.format(kd),writer='pillow',fps=30)
    if plot:
        plt.show()
    return recorder1,recorder2,recorder3

def Sim_thick(dx,kd,plot=False,save=False):
    global ox, oy, p, nd, sigma_x, sigma_y
    # INITIALISATION 2D-GRID AND SIMULATION PARAMETERS-------------------------
    c = 340  # speed of sound (wave speed)
    dx = 0.05  # spatial discretisation step
    dy = dx

    kd = 1  # frequentie-lengte
    d = 1  # lengte d
    k = kd / d  # wavenumber
    L = 7 * d  # length of simulation domain

    nx = int(4*d / dx)  # aantal cellen in x richting - number of cells in x direction
    ny = int(L / dy)  # aantal cellen in y richting - number of cells in y direction
    nd = int(d / dx)  # aantal cellen in lengte d

    CFL = 0.8  # Courant number

    dt = CFL / (c * np.sqrt((1 / dx ** 2) + (1 / dy ** 2)))  # time step

    nt = 6*int(40 // CFL)  # number of time steps

    # location of source(central) and receivers
    x_bron = int(nd / 10)
    y_bron = nd

    x_recorder1 = int(nd / 2)
    y_recorder1 = y_bron + 3 * nd  # location receiver 1

    x_recorder2 = int(nd / 2)
    y_recorder2 = y_recorder1 + nd  # location receiver 2

    x_recorder3 = int(nd / 2)
    y_recorder3 = y_recorder2 + nd  # location receiver 2

    # source pulse information
    A = 10
    fc = k * c / (2 * np.pi)
    t0 = 2.5E-2
    sigma = 5E-4

    # PML implementation
    sigma_max_left = 1 #Max amount of demping left
    sigma_max_right = 1000 #Max amount of demping right
    sigma_max_up = 2000 #Max amount of demping up

    hoogte_PML = 40 #Height from which wave starts damping (numbers of layers)
    breedte_PML_links = 10 #How much to the right of left simulation wall will wave start damping (numbers of layers)
    breedte_PML_rechts = 40 #How much to the left of right simulation wall will wave start damping (numbers of layers)
    sigma_x = np.zeros((nx + 1, ny))
    sigma_y = np.zeros((nx, ny + 1))
    m = 1 #Power of the PML (3 to 4), if too high, the sigma_max is too small

    sigma_x[-hoogte_PML:,:] = [[sigma_max_up*(i/len(sigma_x[-hoogte_PML:,:]))**m]*sigma_x.shape[1] for i in range(0,len(sigma_x[:hoogte_PML,:]),1)]
    sigma_y[:,:breedte_PML_links] = np.array([[sigma_max_left*(i/len(sigma_y[:breedte_PML_links,:]))**m]*sigma_y.shape[0] for i in range(len(sigma_y[:breedte_PML_links,:]),0,-1)]).transpose()
    sigma_y[: ,breedte_PML_rechts: ] = np.array([[i]*sigma_y.shape[0] for i in range(0, sigma_y.shape[1]-breedte_PML_rechts, 1)]).transpose()

    # initialisation of o and p fields
    ox = np.zeros((nx + 1, ny))
    oy = np.zeros((nx, ny + 1))
    p = np.zeros((nx, ny))

    # initialisation time series receivers
    recorder1 = np.zeros((nt, 1))
    recorder2 = np.zeros((nt, 1))
    recorder3 = np.zeros((nt, 1))

    bront = np.zeros((nt, 1))
    tijdreeks = np.zeros((nt, 1))
    bron = 0

    # TIME ITTERATION----------------------------------------------------
    fig, ax = plt.subplots()
    plt.xlabel('x/d')
    plt.ylabel('y/d')
    plt.ylim([1, 4*nd])
    plt.xlim([1, ny + 1])
    ax.set_yticks(np.linspace(0,4*nd,5))
    ax.set_yticklabels(np.arange(5))
    ax.set_xticks(np.linspace(0,ny,8))
    ax.set_xticklabels(np.arange(8))
    movie = []
    rect = patches.Rectangle((int(2 * nd)-dx,-dx),nd+dx,2*nd+dx,facecolor='k')
    for it in range(0, nt):
        t = (it - 1) * dt
        tijdreeks[it, 0] = t
        print('%d/%d' % (it, nt))

        bron = A * np.sin(2 * np.pi * fc * (t - t0)) * np.exp(-((t - t0) ** 2) / (sigma))  # bron updaten bij nieuw tijd - update source for new time

        p[x_bron, y_bron] = p[x_bron, y_bron] + bron  # druk toevoegen bij de drukvergelijking op bronlocatie - adding source term to propagation
        step_SIT_SIP(nx, ny, c, dx, dy, dt)  # propagatie over 1 tijdstap - propagate over one time step
        ox[:int(2 * nd), int(2 * nd):int(3 * nd)] = 0  # Thick object
        oy[:int(2 * nd), int(2 * nd):int(3 * nd)] = 0  # Thick object

        ox[0, :] = 0  # ground
        oy[0, :] = 0  # ground

        recorder1[it] = p[x_recorder1, y_recorder1]  # druk opnemen op recorders en referentieplaatsen - store p field at receiver locations
        recorder2[it] = p[x_recorder2, y_recorder2]
        recorder3[it] = p[x_recorder3, y_recorder3]

        # presenting the p field

        # view(0,90)
        # shading interp
        artists = [
            ax.text(0.5, 1.05, '%d/%d' % (it, nt),
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes, ),
            ax.imshow(p, vmin=-0.02 * A, vmax=0.02 * A),
            ax.plot(y_bron, x_bron, 'ks', fillstyle="none")[0],
            ax.plot(y_recorder1, x_recorder1, 'ro', fillstyle="none")[0],
            ax.plot(y_recorder2, x_recorder2, 'ro', fillstyle="none")[0],
            ax.plot(y_recorder3, x_recorder3, 'ro', fillstyle="none")[0],
            ax.add_patch(rect),
        ]
        movie.append(artists)

        # mov(it) = getframe #wegcommentarieren voor simulatie vlugger te laten lopen - if this line is removed simulation runs much faster
    my_anim = ArtistAnimation(fig, movie, interval=50, repeat_delay=1000,
                            blit=True)
    if save:
        my_anim.save('Thick_kd={}.gif'.format(kd),writer='pillow',fps=30)
    if plot:
        plt.show()
    return recorder1, recorder2, recorder3
