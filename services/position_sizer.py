def compute_position_size(prob_up, volatility):

    edge = prob_up - 0.5

    if edge <= 0:
        return 0

    size = edge / (volatility + 1e-6)

    size = min(size, 0.02)

    return round(size, 4)