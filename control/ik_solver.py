import math

L1, L2 = 12, 10  # cm


def inverse_kinematics(x, y):
    d = math.hypot(x, y)
    if d > (L1 + L2):
        return None

    cos_a = (L1*L1 + L2*L2 - d*d) / (2*L1*L2)
    a = math.acos(cos_a)

    b = math.atan2(y, x) - math.atan2(
        L2*math.sin(a),
        L1 + L2*math.cos(a)
    )

    return int(math.degrees(b)), int(math.degrees(a))
