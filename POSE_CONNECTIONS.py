POSE_CONNECTIONS = [

    # arms
    (11, 13), (13, 15),
    (12, 14), (14, 16),

    # shoulders
    (11, 12),

    # torso
    (11, 23),
    (12, 24),
    (23, 24),

    # legs
    (23, 25), (25, 27),
    (24, 26), (26, 28),

    # feet
    (27, 29),   # left ankle -> heel
    (29, 31),   # left heel -> foot index

    (28, 30),   # right ankle -> heel
    (30, 32),   # right heel -> foot index
]