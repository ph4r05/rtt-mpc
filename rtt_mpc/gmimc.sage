load('common.sage')


class GMiMCParams(object):
    def __init__(self, field, r, c, num_rounds):
        self.field = field
        self.r = r
        self.c = c
        self.m = r + c
        self.round_constants = [generate_round_constant('GMiMC_erf', field, i)
                                for i in range(num_rounds)]

        self.output_size = c
        assert self.output_size <= r


# Parameter sets.
S45a = GMiMCParams(field=F91, r=2, c=1, num_rounds=121)
S45b = GMiMCParams(field=F91, r=10, c=1, num_rounds=137)

S80a = GMiMCParams(field=F81, r=2, c=2, num_rounds=111)
S80b = GMiMCParams(field=F161, r=2, c=1, num_rounds=210)
S80c = GMiMCParams(field=F161, r=10, c=1, num_rounds=226)

S128a = GMiMCParams(field=F125, r=2, c=2, num_rounds=166)
S128b = GMiMCParams(field=F253, r=2, c=1, num_rounds=326)
S128c = GMiMCParams(field=F125, r=10, c=2, num_rounds=182)
S128d = GMiMCParams(field=F61, r=8, c=4, num_rounds=101)
S128e = GMiMCParams(field=F253, r=10, c=1, num_rounds=342)

S256a = GMiMCParams(field=F125, r=4, c=4, num_rounds=174)
S256b = GMiMCParams(field=F125, r=10, c=4, num_rounds=186)


def erf_feistel_permutation(x, params):
    for c_i in params.round_constants:
        mask = (x[0] + c_i) ** 3
        x = [x_j + mask for x_j in x[1:]] + [x[0]]
    return vector(params.field, x)


def GMiMC_erf(inputs, params):
    return sponge(erf_feistel_permutation, inputs, params)


# Example for checking a collision between the inputs [1, 2] and [3, 4]:
check_collision(
    hash_func=GMiMC_erf,
    params=S45a,
    input1=vector(S45a.field, [1, 2]),
    input2=vector(S45a.field, [3, 4]))
