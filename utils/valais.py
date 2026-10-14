import numpy as np
from utils.frequencies import dense_around, OFFICIAL_NOTES_FREQUENCIES, SNAPSHOTS_FREQUENCIES, SNAPSHOTS_FMIN, SNAPSHOTS_FMAX


def get_L(M, D1, D2):
    c = M
    b = D2/2 - D1/2
    return np.sqrt(c**2 - b**2)

def compute_mu_glob_h(D, M):
    D1, D2, D3, D4, D5 = D
    M1, M2, M3, M4 = M

    # Real lenght
    L1 = get_L(M1, D1, D2)
    L2 = get_L(M2, D2, D3)
    L3 = get_L(M3, D3, D4)
    L4 = get_L(M4, D4, D5)

    L = L1 + L2 + L3 + L4

    # Fractions of total lenght
    l1 = L1 / L
    l2 = L2 / L
    l3 = L3 / L

    mu_glob_h = np.array([
        l1,
        l2,
        l3,
        D2/2,
        D3/2,
        D4/2
    ])

    return mu_glob_h, L


# # --- 1st case measured ---
# D1 = [] # Diameters
# M1 = [] # Outside lenghts
# mu_glob_h1 = compute_mu_glob_h(D1, M1)

# # --- 2nd case measured ---
# D2 = [] # Diameters
# M2 = [] # Outside lenghts
# mu_glob_h2 = compute_mu_glob_h(D2, M2)

# # --- 3rd case measured ---
# D3 = [] # Diameters
# M3 = [] # Outside lenghts
# mu_glob_h3 = compute_mu_glob_h(D3, M3)

# --- 4th case measured ---
D4 = [13, 32, 57, 88, 103] # Diameters
M4 = [540, 845, 755, 445] # Outside lenghts
mu_glob_h4, L = compute_mu_glob_h(D4, M4)


OFFICIAL_NOTES = [
    {"freq": 92,  "name": "Do"},
    {"freq": 139, "name": "Sol"},
    {"freq": 185, "name": "Do"},
    {"freq": 234, "name": "Mi"},
    {"freq": 278, "name": "Sol"},
    {"freq": 331, "name": "Sib"},
    {"freq": 371, "name": "Do"},
    {"freq": 417, "name": "Ré"},
    {"freq": 468, "name": "Mi"},
    {"freq": 556, "name": "Sol"},
]


measured_freqs1 = [
    185 - 5, # Do
    234 - 30, # Mi
    278, # Sol
    331 - 21, # Sib
    371, # Do
    417, # Re
    468 - 6.5, # Mi
]

# measured_freqs2 = [
#     185, # Do
#     234 - 5, # Mi
#     278, # Sol
#     331 + 0, # Sib
#     371, # Do
#     417 + 5, # Re
#     468, # Mi
# ]
measured_freqs2 = None # Order of measurments not clear

measured_freqs3 = [
    185 - 20, # Do
    234 - 23, # Mi
    278, # Sol
    331 - 20, # Sib
    371 - 5, # Do
    417, # Re
    468 - 20, # Mi
    556 - 16, # Sol
]

measured_freqs4 = [
    185, # Do
    234, # Mi
    278, # Sol
    331 - 5, # Sib
    371 - 0.5, # Do
    417, # Re
    468 - 14, # Mi
]


# Minimum length of a cone
L_MIN = 220 # mm

# Minimum radius of a cone
R_MIN = 6.5 # mm (min radius constructible)

# Maximum radius of a cone
R_MAX = 51.5 # mm (radius of the bell)
