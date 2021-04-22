load('common.sage')


class MiMCParams(object):
    def __init__(self, field, n_rounds):
        self.field = field
        self.r = 1
        self.c = 1
        self.m = 2
        self.output_size = 1
        self.round_constants = [field(0)] + \
            [generate_round_constant('MiMC', field, i)
             for i in range(n_rounds - 2)] + [field(0)]


# Parameter sets.
S45 = MiMCParams(field=F91, n_rounds=116)
S80 = MiMCParams(field=F161, n_rounds=204)
S128 = MiMCParams(field=F253, n_rounds=320)


def feistel_permutation(inputs, params):
    x_left, x_right = inputs
    for c_i in params.round_constants:
        x_left, x_right = x_right + (x_left + c_i) ** 3, x_left
    return vector([x_left, x_right])


def MiMC(inputs, params):
    return sponge(feistel_permutation, inputs, params)


# Example for checking a collision between the inputs (1, 2) and (3, 4):
check_collision(
    hash_func=MiMC,
    params=S45,
    input1=vector(S45.field, [1, 2]),
    input2=vector(S45.field, [3, 4]))
