import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def heney_greenstein_phase_function(g, b):
    """Henyey-Greenstein phase function."""
    return (1 - b**2) / (4 * np.pi * (1 + b**2 + 2 * b * np.cos(g))**(3/2))

def compute_slab_depth(filling_factor, radius):
    cross_sectional_area = np.pi * radius**2
    mean_free_path = 1 / (filling_factor * cross_sectional_area)
    # Set the slab depth to be several times the mean free path
    slab_depth = (10**6) * mean_free_path
    return slab_depth

def direct_mcrt(n_photons, radius, ssa, incidence_angle, b, f, slab_depth):
    r_mu_over_mu_0 = []
    mu_0 = np.cos(np.radians(incidence_angle))
    
    for emission_angle in tqdm(range(-90, 91, 1)):  # Emission angle range from -90 to 90 degrees
        photons_scattered = 0
        mu = np.cos(np.radians(emission_angle))
        
        for _ in range(n_photons):
            x, y, z = 0, 0, 0  # Starting at the origin
            
            # Initial photon direction based on incidence angle
            direction = np.array([np.sin(np.radians(incidence_angle)), 0, np.cos(np.radians(incidence_angle))])
            
            while True:
                # Absorption Probability
                v1 =  np.random.rand()
                if v1 > ssa:
                    break  # Photon absorbed
                 
                # To determine the scattering angle
                while True:
                    g_guess = np.random.uniform(0, np.pi)  # Guess for scattering angle
                    p_g_guess = heney_greenstein_phase_function(g_guess, b)
                    dP_dg = p_g_guess * 2 * np.pi * np.sin(g_guess)  # Probability density function for scattering at angle g_guess
                    
                    v2 = np.random.rand() * dP_dg.max()  # Random variable uniformly distributed between 0 and max_dP_dg
                    if dP_dg > v2:
                        g = g_guess  # Accept g_guess as scattering angle
                        break
                
                # Azimuthal Angle Determination
                psi = 2 * np.pi * np.random.rand()  # Azimuthal angle uniformly distributed between 0 and 2pi
                
                # Update photon direction using scattering angles
                direction[0] = np.sin(g) * np.cos(psi)
                direction[1] = np.sin(g) * np.sin(psi)
                direction[2] = np.cos(g)
                
                # Update photon position
                x += direction[0]
                y += direction[1]
                z += direction[2]
                
                # Apply periodic boundary conditions
                x = x % (2 * radius)
                y = y % (2 * radius)
                z = z % (2 * radius)
                
                # Escape condition: ensure photons do not exceed slab depth
                if z > slab_depth:
                    break  # Photon escaped
                
                # Increase count of scattered photons
                photons_scattered += 1
                    
        reflectance = photons_scattered / n_photons
        r_mu_over_mu_0.append(reflectance * mu / mu_0)
    
    return r_mu_over_mu_0

# Simulation parameters
n_photons = 10**4  # Reduced number of photons for faster computation
radius = 0.01  # Arbitrary unit
ssas = [0.2, 0.5, 0.7]  # Assumed single scattering albedos
incidence_angle = 30  # Incidence angle in degrees
bs = [-0.5,0.5]  # Asymmetry factors
f = 0.2  # Filling factor

# Plot results for each asymmetry factor and SSA
plt.figure(figsize=(10, 6))
for ssa in ssas:
    for b in bs:
        # Run direct MCRT simulation
        r_mu_over_mu_0 = direct_mcrt(n_photons, radius, ssa, incidence_angle, b, f, slab_depth)

        # Plot results
        emission_angles = np.arange(-90, 91, 1)
        plt.plot(emission_angles, r_mu_over_mu_0, label=f'SSA = {ssa}, b = {b}')

plt.xlabel('Emission Angle (degrees)')
plt.ylabel(r'$r\mu / \mu_0$')
plt.title('Direct MCRT for Different SSA and Asymmetry Factors')
plt.grid(True)
plt.legend()
plt.show()
