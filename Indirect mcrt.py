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
    slab_depth = 10**6 * mean_free_path
    return slab_depth

def indirect_mcrt(n_photons, radius, ssa, incidence_angle, b, filling_factor):
    slab_depth = compute_slab_depth(filling_factor, radius)
    r_mu_over_mu_0 = []
    mu_0 = np.cos(np.radians(incidence_angle))
    E_0 = 1.0  # Initial energy of the photon packet

    for emission_angle in tqdm(range(-90, 91, 1)):  # Emission angle range from -90 to 90 degrees
        total_energy = 0.0
        mu = np.cos(np.radians(emission_angle))

        for _ in range(n_photons):
            # Start photon at the surface of the particle
            theta = np.random.rand() * 2 * np.pi
            x, y, z = radius * np.cos(theta), radius * np.sin(theta), 0

            # Initial photon direction based on incidence angle
            direction = np.array([np.sin(np.radians(incidence_angle)), 0, np.cos(np.radians(incidence_angle))])

            energy = E_0  # Initial energy of the photon packet
            h_i = 1  # Initial visibility of the photon

            while energy > 0.001:  # Continue until the energy is negligible
                # Distance to next scattering event
                mean_free_path = 1 / (filling_factor * np.pi * radius**2)
                travel_distance = -mean_free_path * np.log(np.random.rand())

                # Update photon position
                x += travel_distance * direction[0]
                y += travel_distance * direction[1]
                z += travel_distance * direction[2]

                # Apply periodic boundary conditions
                x = x % (2 * radius)
                y = y % (2 * radius)
                z = z % (2 * radius)

                # Check if the photon escapes
                if z > slab_depth:
                    break

                # Absorption Probability
                if np.random.rand() > ssa:
                    break  # Photon absorbed

                # Scattering Angle Determination
                while True:
                    g_guess = np.random.uniform(0, np.pi)  # Guess for scattering angle
                    p_g_guess = heney_greenstein_phase_function(g_guess, b)
                    dP_dg = p_g_guess * 2 * np.pi * np.sin(g_guess)
                    if np.random.rand() < dP_dg:
                        g = g_guess  # Accept g_guess as scattering angle
                        break

                # Azimuthal Angle Determination
                psi = 2 * np.pi * np.random.rand()  # Azimuthal angle uniformly distributed between 0 and 2pi

                # Update photon direction using scattering angles
                direction[0] = np.sin(g) * np.cos(psi)
                direction[1] = np.sin(g) * np.sin(psi)
                direction[2] = np.cos(g)

                # Energy reduction after scattering
                energy *= ssa * heney_greenstein_phase_function(g, b) 

                total_energy += energy

        reflectance = total_energy / (n_photons * E_0)
        r_mu_over_mu_0.append(reflectance * mu / mu_0)

    return r_mu_over_mu_0

########################################################################################################################

# Simulation parameters
n_photons = 2*(10**4)  # Number of photons
radius = 0.01  # Arbitrary unit
ssas = [0.2, 0.5, 0.7]  # Different SSAs
incidence_angle = 30  # Incidence angle in degrees
b = [-0.5,0.5]  # Asymmetry factor
filling_factor = 0.2  # Fixed filling factor

#########################################################################################################################

# Run indirect MCRT simulation for different SSAs and assymetry factor
plt.figure(figsize=(10, 6))
for ssa in ssas:
    for b in bs:
        r_mu_over_mu_0 = indirect_mcrt(n_photons, radius, ssa, incidence_angle, b, filling_factor)
        # Plot results
        emission_angles = np.arange(-90, 91, 1)
        plt.plot(emission_angles, r_mu_over_mu_0, label=f'SSA = {ssa}, b={b}')

plt.xlabel('Emission Angle (degrees)')
plt.ylabel(r'$r\mu / \mu_0$')
plt.title('Indirect MCRT: Reflectance for Different SSAs and assymetry factor (FF = 0.2)')
plt.legend()
plt.grid(True)
plt.show()
