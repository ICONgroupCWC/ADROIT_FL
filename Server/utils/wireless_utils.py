import yaml
import numpy as np


def calculate_avg_bit_rate(num_users):
    with open('utils/wireless_parameters.yml', 'r') as file:
        wireless_data = yaml.safe_load(file)

    area_size = float(wireless_data['area_size'])
    Pt = float(wireless_data['transmission_power'])
    B = float(wireless_data['bandwidth'])
    N0 = float(wireless_data['noise_spectral_density'])


    # Fixed server position (e.g., at the center of the area)
    server_x, server_y = area_size / 2, area_size / 2

    # Monte Carlo simulation to calculate average bit rate
    num_iterations = 1000  # Number of Monte Carlo iterations
    bit_rates = []  # To store bit rates for each iteration

    for _ in range(num_iterations):
        # Randomly distribute users in 100x100 area
        user_x_coords = np.random.uniform(0, area_size, num_users)
        user_y_coords = np.random.uniform(0, area_size, num_users)

        # Calculate distances to the server
        distances = np.sqrt((user_x_coords - server_x) ** 2 + (user_y_coords - server_y) ** 2)
        distances = np.clip(distances, 1, None)  # Avoid zero distance; minimum distance = 1 m

        # Calculate achievable bit rates for all users
        rates = B * np.log2(1 + (Pt / (distances ** 2 * B * N0)))

        # Compute average bit rate across all users in this iteration
        avg_rate = np.mean(rates)
        bit_rates.append(avg_rate)

    # Calculate the final average bit rate over all iterations
    avg_bit_rate = np.mean(bit_rates)
    print(f"Average Bit Rate: {avg_bit_rate:.2f} bps")

    return avg_bit_rate

def calculate_energy_per_round(model_size, avg_bit_rate):

    with open('utils/wireless_parameters.yml', 'r') as file:
        wireless_data = yaml.safe_load(file)
    Pt = float(wireless_data['transmission_power'])
    transmission_time = model_size / avg_bit_rate  # Time to transmit the model
    energy_per_round = Pt * transmission_time  # Energy consumption

    return energy_per_round

def calculate_energy(model_size, avg_bit_rate):
    model_size_in_bits = model_size * 1e6
    with open('utils/wireless_parameters.yml', 'r') as file:
        wireless_data = yaml.safe_load(file)
    Pt = float(wireless_data['transmission_power'])
    transmission_time = model_size_in_bits / avg_bit_rate  # Time to transmit the model
    cumulative_energy = Pt * transmission_time  # Energy consumption

    return cumulative_energy
