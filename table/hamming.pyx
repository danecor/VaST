from libc cimport stdint
ctypedef stdint.uint64_t uint64

cdef uint64 m1  = 0x5555555555555555
cdef uint64 m2  = 0x3333333333333333
cdef uint64 m4  = 0x0f0f0f0f0f0f0f0f
cdef uint64 h01 = 0x0101010101010101
cdef uint64 shift = 1

def hamming_dist(uint64 state1, uint64 state2):
	return chamm(state1, state2)

cdef int chamm(uint64 state1, uint64 state2):
	cdef uint64 x
	x = state1^state2
	x -= (x >> 1) & m1         
	x = (x & m2) + ((x >> 2) & m2)
	x = (x + (x >> 4)) & m4
	return (x * h01) >> 56

def hamming_neighbours(uint64 n, size_t dist, size_t num_bits):
	assert (dist > 0) and (dist < 5)
	results = []
	if dist==1:
		return hamming_neighbours_1(n, num_bits)
	if dist==2:
		return hamming_neighbours_2(n, num_bits)
	elif dist==3:
		return hamming_neighbours_3(n, num_bits)
	elif dist==4:
		return hamming_neighbours_4(n, num_bits)

cdef list hamming_neighbours_1(uint64 n, size_t num_bits):
	cdef size_t i
	results = []
	for i in xrange(num_bits):
		results.append(n^(shift << i))
	return results

cdef list hamming_neighbours_2(uint64 n, size_t num_bits):
	cdef size_t i, j
	results = []
	for i in xrange(num_bits):
		for j in xrange(i+1, num_bits):
			results.append(n^(shift << i)^(shift << j))
	return results

cdef list hamming_neighbours_3(uint64 n, size_t num_bits):
	cdef size_t i, j, k
	results = []
	for i in xrange(num_bits):
		for j in xrange(i+1, num_bits):
			for k in xrange(j+1, num_bits):
				results.append(n^(shift << i)^(shift << j)^(shift << k))
	return results

cdef list hamming_neighbours_4(uint64 n, size_t num_bits):
	cdef size_t i, j, k, l
	results = []
	for i in xrange(num_bits):
		for j in xrange(i+1, num_bits):
			for k in xrange(j+1, num_bits):
				for l in xrange(k+1, num_bits):
					results.append(n^(shift << i)^(shift << j)^(shift << k)^(shift << l))
	return results