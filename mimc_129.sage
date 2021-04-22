# coding: utf-8
# https://mimc.iaik.tugraz.at/code/mimc/mimc_129.sage

from sage.rings.polynomial.polynomial_gf2x import GF2X_BuildIrred_list

n = 129

K = GF(2 ** n, "a")
K.inject_variables()

# set_random_seed(0)
# constants = [K.fetch_int(0)]
# for i in range(1, num_rounds):
#    constants.append(K.random_element())
constants = [0x000000000000000000000000000000000, 0x06382246d63a75db1f522fa4e05cc0657,
             0x1081b8ffc1f376f1d8b798232e330ed56, 0x0ce5d22bd8f8d1d3b262296b92f588787,
             0x1a426aa322e6b4da51f9dba2bb404df1c, 0x03c6946a088909908bc4471593028e7fb,
             0x11c2edf7c3929da2a3a9f3b3b1fc8eb53, 0x10b0157c0fa825d7e9965ace273294881,
             0x0af67ac1897ff10c25501933d6c930e94, 0x00bbbc3c4bbd5a54a0e66d9f0f8a8be79,
             0x0cdd6a5be343bec16d6183f7badd6d5db, 0x08b843af944f96e199c9dac93a8af2888,
             0x18ea7bd63b43d8c85512b7669d9927518, 0x16d13f69cff104e013754c2fe96f5cb92,
             0x168aa09f0672c5e465461aecdd6684b6a, 0x1fe480707dca72bfccb1a77cd4edb67a6,
             0x015cee01dacf5c9e66fe6bdfa93189454, 0x16c891bc68270f24998606cfbdcaa5611,
             0x0cf4fd57ad35bf06d242a9fcb4972075b, 0x1170a49db9a62896c1b59840e33ff427d,
             0x0d675bd0083f069681057099a88261932, 0x1e35e81a5f3b88568cd18456936be6721,
             0x078a0ac6891b377b855f7e715144133b8, 0x1664154b3de279c0d7fc6dd5c1ec4ada8,
             0x01860e08d1d0cceb6e0d068d753afb5b8, 0x0ddbc94e9e569546044a616ab8f462fe8,
             0x06a5ae6c7c22904e4baf44848294f82c9, 0x05adf151652a16f0e3546a701953de05b,
             0x1590219323779a7756933e1d43b092865, 0x0f09ea94a42cf3d8182ec81fe19b4a16b,
             0x14a00adfaa82483db455ced5b42588c23, 0x0f9b2c70cabea07f738f6c4c2fdf05271,
             0x177d81b15626b3d9705a84ba498bed335, 0x010e37ad1a843d68c93bd4d12a5ff777e,
             0x1ec5252e4ce05e567c0ed58e392f82df9, 0x0c51762cdb2a41b86fadca23ade46ec3c,
             0x1bd09cbc69b6aa86c79d4b56e06c65dbf, 0x0472cf4df04d10a8764dce39c758ac89f,
             0x0fa41f59a323dffdc05bdca384ba65007, 0x1056526aa50107101eb34d698dbb1507b,
             0x10f3e7e57d9451bb1c36a7db077d623b2, 0x0c843a8a873aa33444962d64b243ca1f1,
             0x1a06a6d9e59a6f17f829845829fcc51a5, 0x078d4f7fa105e1f396b2bdec55d07e96a,
             0x1336a5eb8a15d2ae237e6605b4a4d5e7e, 0x084bd7ddfc3f58851e405eee24b31e0da,
             0x196f9c6845d9abc8b17815e4efe43ea61, 0x17e060968262d38d5b12be87b0ddca0fe,
             0x13e5ca95c7826b284615893fb6b6615f2, 0x11aafe1def56e71fcf9ffd1f535472262,
             0x0538e78611f47e30797cbfa5eb217d9fd, 0x1adb78502384a7a093d4cb5423eb98dfe,
             0x03f564c552c72f1f615660913f31ae19e, 0x05383aec281add2a5e61fc16e0915e9ab,
             0x17b7d2f156b797ba3bcd04f74970a3698, 0x16bcb475655eece3a2f8decd844f65550,
             0x0df93054f75b723ac4e2ce48d00cf37dd, 0x190b65b81ef953c92ad0b5a15c533824c,
             0x026e03d771818acfad02dd38d3d5ad6d7, 0x088946ee4840404bf1fe6ef874751680f,
             0x1a2164a4a31c13d1a0fe4d86b8a5a8f0c, 0x0c7b325e4aecb36f489a24a31277c18ac,
             0x1a1b145f688b87d5e5926bd19d70858f5, 0x166976d9031782c3a733897c19eadf660,
             0x0cb0de4a36207611a580a97d94a99708e, 0x0453b6e0f8fb6b59d38b466b9b4210b4e,
             0x13bbdd7cda3d39a2bc6391929699b1d02, 0x05449506bb6fa430ff999d13ef9187631,
             0x152c672a79fb3a4de06bf22e9f8a6f7dd, 0x09cf98e8db80e7ec38c662cf0bd84dd49,
             0x07c42b47224719b2e7d6416e7aece843d, 0x18d14c8c96531ee939835090c92a79a08,
             0x189ac9a8952dafa06b3fad1abe9cf37a8, 0x0b382a9f685108884c841cfcdd4e7c065,
             0x0263c639fae4bee461bc66be8fed407f7, 0x118bbb5a626f4130a3246bf144ddeba6f,
             0x11c7f739620fc72ba7112461fb96bcef2, 0x1ce202833557d1e76af8a03cf4e1fccf7,
             0x0a474673a25c26e1c18aeab2015adda20, 0x15c9722c814b888297fcc8c2a096a8730,
             0x1de01e75fa74625e8f0d8231a510c88dc, 0x12057179d8d7584fcdfb1c7302988550a]


def mimc_encryption(p, k, num_rounds):
    state = (p + (k + K.fetch_int(constants[0]))) ^ 3
    for i in range(1, num_rounds):
        state = (state + (k + K.fetch_int(constants[i]))) ^ 3
    state = state + k
    return state


# set_random_seed(0)
num_rounds = ceil(n / log(3, 2))
print("Number of rounds:", num_rounds)
k = K.fetch_int(0x42424242424242424242424242424242)
print("Key:", hex(k.integer_representation()))
p = K.random_element()
c = mimc_encryption(p, k, num_rounds)
print("Plaintext:", hex(p.integer_representation()))
print("Ciphertext:", hex(c.integer_representation()))
