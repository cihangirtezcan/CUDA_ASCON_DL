#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <Windows.h>
#include "rdrand.h"
#include <intrin.h>
#include <immintrin.h>
#include <string.h>
#include <ctime>

#define BLOCKS				256
#define THREADS				1024
#define TRIALS				1024*4
//__int64 trial = 1024*4, keys = 100;
__int64 trial = 1, keys = 100;
int rotation = 0;
double PCFreq = 0.0;
__int64 CounterStart = 0;

#define RDRAND_MASK	0x40000000
#define RETRY_LIMIT 10
#ifdef _WIN64
typedef uint64_t _wordlen_t;
#else
typedef uint32_t _wordlen_t;
#endif
#define bit32 unsigned int
#define bit64 unsigned __int64 
bit64 key[2], *key_d, *nonce, *nonce_d, *IV_d, *keyrows_d, *keyrows, *IV;
bit64 state[5] = { 0 }, t[5] = { 0 };
bit64 constants[16] = { 0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b, 0x3c, 0x2d, 0x1e, 0x0f };
int key_choice = 0;
void StartCounter(){
	LARGE_INTEGER li;
	if (!QueryPerformanceFrequency(&li))
		printf("QueryPerformanceFrequency failed!\n");

	PCFreq = double(li.QuadPart) / 1000.0;

	QueryPerformanceCounter(&li);
	CounterStart = li.QuadPart;
}
double GetCounter(){
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return double(li.QuadPart - CounterStart) / PCFreq;
}
void print_state(bit64 state[5]) {	for (int i = 0; i < 5; i++) printf("%016I64x\n", state[i]);}
void add_constant(bit64 state[5], int i, int a) {	state[2] = state[2] ^ constants[12 - a + i];}
void sbox(bit64 x[5]) {
	x[0] ^= x[4]; x[4] ^= x[3]; x[2] ^= x[1];
	t[0] = x[0]; t[1] = x[1]; t[2] = x[2]; t[3] = x[3]; t[4] = x[4];
	t[0] = ~t[0]; t[1] = ~t[1]; t[2] = ~t[2]; t[3] = ~t[3]; t[4] = ~t[4];
	t[0] &= x[1]; t[1] &= x[2]; t[2] &= x[3]; t[3] &= x[4]; t[4] &= x[0];
	x[0] ^= t[1]; x[1] ^= t[2]; x[2] ^= t[3]; x[3] ^= t[4]; x[4] ^= t[0];
	x[1] ^= x[0]; x[0] ^= x[4]; x[3] ^= x[2]; x[2] = ~x[2];
}
bit64 rotate(bit64 x, int l) {
	bit64 temp;
	temp = (x >> l) ^ (x << (64 - l));
	return temp;
}
__device__ bit64 rotater(bit64 x, int l) {
	bit64 temp;
	temp = (x >> l) ^ (x << (64 - l));
	return temp;
}
void linear(bit64 state[5]) {
	bit64 temp0, temp1;
	temp0 = rotate(state[0], 19);
	temp1 = rotate(state[0], 28);
	state[0] ^= temp0 ^ temp1;
	temp0 = rotate(state[1], 61);
	temp1 = rotate(state[1], 39);
	state[1] ^= temp0 ^ temp1;
	temp0 = rotate(state[2], 1);
	temp1 = rotate(state[2], 6);
	state[2] ^= temp0 ^ temp1;
	temp0 = rotate(state[3], 10);
	temp1 = rotate(state[3], 17);
	state[3] ^= temp0 ^ temp1;
	temp0 = rotate(state[4], 7);
	temp1 = rotate(state[4], 41);
	state[4] ^= temp0 ^ temp1;
}
void p(bit64 state[5], int a) {
	for (int i = 0; i < a; i++) {
		add_constant(state, i, a);
		sbox(state);
		linear(state);
	}
}
void initialization(bit64 state[5], bit64 key[2]) {
	p(state, 12);
	state[3] ^= key[0];
	state[4] ^= key[1];
}
void encrypt(bit64 state[5], int length, bit64 plaintext[], bit64 ciphertext[]) {
	ciphertext[0] = plaintext[0] ^ state[0];
	for (int i = 1; i < length; i++) {
		p(state, 6);
		ciphertext[i] = plaintext[i] ^ state[0];
		state[0] = plaintext[i] ^ state[0];
	}
}
void decrypt(bit64 state[5], int length, bit64 plaintext[], bit64 ciphertext[]) {
	ciphertext[0] = plaintext[0] ^ state[0];
	for (int i = 1; i < length; i++) {
		p(state, 6);
		ciphertext[i] = plaintext[i] ^ state[0];
		state[0] = plaintext[i];
	}
}
void main_old() {
	bit64 IV = 0x80400c0600000000, key[2] = { 0xffffffffffffffff, 0x123456789abcdef0 }, nonce[2] = { 0 };
	bit64 plaintext[10] = { 0 }, ciphertext[10];
	state[0] = IV;
	state[1] = key[0];
	state[2] = key[1];
	state[3] = nonce[0];
	state[4] = nonce[1];
	print_state(state); printf("\n");
	//p(state, 12);
	initialization(state, key);
	print_state(state);
	encrypt(state, 10, plaintext, ciphertext); printf("\n");
	for (int i = 0; i < 10; i++) printf("%016I64x\n", ciphertext[i]);
	state[0] = IV;
	state[1] = key[0];
	state[2] = key[1];
	state[3] = nonce[0];
	state[4] = nonce[1]; printf("\n");
	print_state(state); printf("\n");
	//p(state, 12);
	initialization(state, key);
	print_state(state);
	decrypt(state, 10, ciphertext, plaintext); printf("\n");
	for (int i = 0; i < 10; i++) printf("%016I64x\n", plaintext[i]);
}
/*! \brief Queries cpuid to see if rdrand is supported
*
* rdrand support in a CPU is determined by examining the 30th bit of the ecx
* register after calling cpuid.
*
* \return bool of whether or not rdrand is supported
*/
int RdRand_cpuid() {
	int info[4] = { -1, -1, -1, -1 };
	/* Are we on an Intel processor? */
	__cpuid(info, /*feature bits*/0);
	if (memcmp((void *)&info[1], (void *) "Genu", 4) != 0 ||
		memcmp((void *)&info[3], (void *) "ineI", 4) != 0 ||
		memcmp((void *)&info[2], (void *) "ntel", 4) != 0) {
		return 0;
	}
	/* Do we have RDRAND? */
	__cpuid(info, /*feature bits*/1);
	int ecx = info[2];
	if ((ecx & RDRAND_MASK) == RDRAND_MASK)
		return 1;
	else
		return 0;
}
/*! \brief Determines whether or not rdrand is supported by the CPU
*
* This function simply serves as a cache of the result provided by cpuid,
* since calling cpuid is so expensive. The result is stored in a static
* variable to save from calling cpuid on each invocation of rdrand.
*
* \return bool/int of whether or not rdrand is supported
*/
int RdRand_isSupported() {
	static int supported = RDRAND_SUPPORT_UNKNOWN;

	if (supported == RDRAND_SUPPORT_UNKNOWN)
	{
		if (RdRand_cpuid())
			supported = RDRAND_SUPPORTED;
		else
			supported = RDRAND_UNSUPPORTED;
	}

	return (supported == RDRAND_SUPPORTED) ? 1 : 0;
}
int rdrand_16(uint16_t* x, int retry) {
	if (RdRand_isSupported())	{
		if (retry)		{
			for (int i = 0; i < RETRY_LIMIT; i++)			{
				if (_rdrand16_step(x))
					return RDRAND_SUCCESS;
			}
			return RDRAND_NOT_READY;
		}
		else		{
			if (_rdrand16_step(x))				return RDRAND_SUCCESS;
			else				return RDRAND_NOT_READY;
		}
	}
	else	{		return RDRAND_UNSUPPORTED;	}
}
int rdrand_32(uint32_t* x, int retry) {
	if (RdRand_isSupported())	{
		if (retry)		{
			for (int i = 0; i < RETRY_LIMIT; i++)			{
				if (_rdrand32_step(x))
					return RDRAND_SUCCESS;
			}
			return RDRAND_NOT_READY;
		}
		else		{
			if (_rdrand32_step(x))
				return RDRAND_SUCCESS;
			else
				return RDRAND_NOT_READY;
		}
	}
	else	{		return RDRAND_UNSUPPORTED;	}
}
int rdrand_64(uint64_t* x, int retry) {
	if (RdRand_isSupported())	{
		if (retry)		{
			for (int i = 0; i < RETRY_LIMIT; i++)			{
				if (_rdrand64_step(x))
					return RDRAND_SUCCESS;
			}
			return RDRAND_NOT_READY;
		}
		else		{
			if (_rdrand64_step(x))				return RDRAND_SUCCESS;
			else				return RDRAND_NOT_READY;
		}
	}
	else	{		return RDRAND_UNSUPPORTED;	}
}
int rdrand_get_n_64(unsigned int n, uint64_t *dest) {
	int success;
	int count;
	unsigned int i;

	for (i = 0; i<n; i++) 	{
		count = 0;
		do 		{
			success = rdrand_64(dest, 1);
		} while ((success == 0) && (count++ < RETRY_LIMIT));
		if (success != RDRAND_SUCCESS) return success;
		dest = &(dest[1]);
	}
	return RDRAND_SUCCESS;
}
int rdrand_get_n_32(unsigned int n, uint32_t *dest) {
	int success;
	int count;
	unsigned int i;
	for (i = 0; i<n; i++) 	{
		count = 0;
		do 		{
			success = rdrand_32(dest, 1);
		} while ((success == 0) && (count++ < RETRY_LIMIT));
		if (success != RDRAND_SUCCESS) return success;
		dest = &(dest[1]);
	}
	return RDRAND_SUCCESS;
}
int rdrand_get_bytes(unsigned int n, unsigned char *dest) {
	unsigned char *start;
	unsigned char *residualstart;
	_wordlen_t *blockstart;
	_wordlen_t i, temprand;
	unsigned int count;
	unsigned int residual;
	unsigned int startlen;
	unsigned int length;
	int success;

	/* Compute the address of the first 32- or 64- bit aligned block in the destination buffer, depending on whether we are in 32- or 64-bit mode */
	start = dest;
	if (((uint32_t)start % (uint32_t) sizeof(_wordlen_t)) == 0) 	{
		blockstart = (_wordlen_t *)start;
		count = n;
		startlen = 0;
	}
	else 	{
		blockstart = (_wordlen_t *)(((_wordlen_t)start & ~(_wordlen_t)(sizeof(_wordlen_t)-1)) + (_wordlen_t)sizeof(_wordlen_t));
		count = n - (sizeof(_wordlen_t)-(unsigned int)((_wordlen_t)start % sizeof(_wordlen_t)));
		startlen = (unsigned int)((_wordlen_t)blockstart - (_wordlen_t)start);
	}

	/* Compute the number of 32- or 64- bit blocks and the remaining number of bytes */
	residual = count % sizeof(_wordlen_t);
	length = count / sizeof(_wordlen_t);
	if (residual != 0) 	{
		residualstart = (unsigned char *)(blockstart + length);
	}

	/* Get a temporary random number for use in the residuals. Failout if retry fails */
	if (startlen > 0) 	{
#ifdef _WIN64
		if ((success = rdrand_64((uint64_t *)&temprand, 1)) != RDRAND_SUCCESS) return success;
#else
		if ((success = rdrand_32((uint32_t *)&temprand, 1)) != RDRAND_SUCCESS) return success;
#endif
	}

	/* populate the starting misaligned block */
	for (i = 0; i<startlen; i++) 	{
		start[i] = (unsigned char)(temprand & 0xff);
		temprand = temprand >> 8;
	}

	/* populate the central aligned block. Fail out if retry fails */

#ifdef _WIN64
	if ((success = rdrand_get_n_64(length, (uint64_t *)(blockstart))) != RDRAND_SUCCESS) return success;
#else
	if ((success = rdrand_get_n_32(length, (uint32_t *)(blockstart))) != RDRAND_SUCCESS) return success;
#endif
	/* populate the final misaligned block */
	if (residual > 0)
	{
#ifdef _WIN64
		if ((success = rdrand_64((uint64_t *)&temprand, 1)) != RDRAND_SUCCESS) return success;
#else
		if ((success = rdrand_32((uint32_t *)&temprand, 1)) != RDRAND_SUCCESS) return success;
#endif

		for (i = 0; i<residual; i++) 		{
			residualstart[i] = (unsigned char)(temprand & 0xff);
			temprand = temprand >> 8;
		}
	}
	return RDRAND_SUCCESS;
}

__global__ void ASCON6_eprint(bit64 IV[], bit64 key[], bit64 nonce[], __int64 counter[]) {
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	bit64 initial0, initial1, initial2, initial3, initial4;
	bit64 pair0, pair1, pair2, pair3, pair4;
	bit64 t0, t1, t2, t3, t4;

	initial0 = IV[threadIndex];
	initial1 = key[2 * threadIndex];
	initial2 = key[2 * threadIndex + 1];
	initial3 = nonce[2 * threadIndex];
	initial4 = nonce[2 * threadIndex + 1];

	for (int c = 0; c < TRIALS; c++) {
		pair0 = initial0;
		pair1 = initial1;
		pair2 = initial2;
		//	pair3 = initial3 ^ 0x0040000000000000;
		//	pair4 = initial4 ^ 0x0040000000000000;
		pair3 = initial3 ^ 0x8000000000000000;
		pair4 = initial4 ^ 0x8000000000000000;

		for (int i = 0; i < 5; i++) {
			initial0 ^= initial4; initial4 ^= initial3; initial2 ^= initial1;
			t0 = initial0; t1 = initial1; t2 = initial2; t3 = initial3; t4 = initial4;
			t0 = ~t0; t1 = ~t1; t2 = ~t2; t3 = ~t3; t4 = ~t4;
			t0 &= initial1; t1 &= initial2; t2 &= initial3; t3 &= initial4; t4 &= initial0;
			initial0 ^= t1; initial1 ^= t2; initial2 ^= t3; initial3 ^= t4; initial4 ^= t0;
			initial1 ^= initial0; initial0 ^= initial4; initial3 ^= initial2; initial2 = ~initial2;
			// Liner layer //
			t0 = rotater(initial0, 19);
			t1 = rotater(initial0, 28);
			initial0 ^= t0 ^ t1;
			t0 = rotater(initial1, 61);
			t1 = rotater(initial1, 39);
			initial1 ^= t0 ^ t1;
			t0 = rotater(initial2, 1);
			t1 = rotater(initial2, 6);
			initial2 ^= t0 ^ t1;
			t0 = rotater(initial3, 10);
			t1 = rotater(initial3, 17);
			initial3 ^= t0 ^ t1;
			t0 = rotater(initial4, 7);
			t1 = rotater(initial4, 41);
			initial4 ^= t0 ^ t1;
		}
		initial0 ^= initial4; initial4 ^= initial3; initial2 ^= initial1;
		t0 = initial0; t1 = initial1; t2 = initial2; t3 = initial3; t4 = initial4;
		t0 = ~t0; t1 = ~t1; t2 = ~t2; t3 = ~t3; t4 = ~t4;
		t0 &= initial1; t1 &= initial2; t2 &= initial3; t3 &= initial4; t4 &= initial0;
		initial0 ^= t1; initial1 ^= t2; initial2 ^= t3; initial3 ^= t4; initial4 ^= t0;
		initial1 ^= initial0; initial0 ^= initial4; initial3 ^= initial2; initial2 = ~initial2;
		for (int i = 0; i < 5; i++) {
			pair0 ^= pair4; pair4 ^= pair3; pair2 ^= pair1;
			t0 = pair0; t1 = pair1; t2 = pair2; t3 = pair3; t4 = pair4;
			t0 = ~t0; t1 = ~t1; t2 = ~t2; t3 = ~t3; t4 = ~t4;
			t0 &= pair1; t1 &= pair2; t2 &= pair3; t3 &= pair4; t4 &= pair0;
			pair0 ^= t1; pair1 ^= t2; pair2 ^= t3; pair3 ^= t4; pair4 ^= t0;
			pair1 ^= pair0; pair0 ^= pair4; pair3 ^= pair2; pair2 = ~pair2;
			// Liner layer //
			t0 = rotater(pair0, 19);
			t1 = rotater(pair0, 28);
			pair0 ^= t0 ^ t1;
			t0 = rotater(pair1, 61);
			t1 = rotater(pair1, 39);
			pair1 ^= t0 ^ t1;
			t0 = rotater(pair2, 1);
			t1 = rotater(pair2, 6);
			pair2 ^= t0 ^ t1;
			t0 = rotater(pair3, 10);
			t1 = rotater(pair3, 17);
			pair3 ^= t0 ^ t1;
			t0 = rotater(pair4, 7);
			t1 = rotater(pair4, 41);
			pair4 ^= t0 ^ t1;
		}
		pair0 ^= pair4; pair4 ^= pair3; pair2 ^= pair1;
		t0 = pair0; t1 = pair1; t2 = pair2; t3 = pair3; t4 = pair4;
		t0 = ~t0; t1 = ~t1; t2 = ~t2; t3 = ~t3; t4 = ~t4;
		t0 &= pair1; t1 &= pair2; t2 &= pair3; t3 &= pair4; t4 &= pair0;
		pair0 ^= t1; pair1 ^= t2; pair2 ^= t3; pair3 ^= t4; pair4 ^= t0;
		pair1 ^= pair0; pair0 ^= pair4; pair3 ^= pair2; pair2 = ~pair2;
		/*		t1 = 0;
		t0 = initial0 & 0x9224b6d24b6eda49;
		for (int i = 0; i < 64; i++) t1 ^= ((t0 >> i) & 0x1);
		t0 = pair0 & 0x9224b6d24b6eda49;
		for (int i = 0; i < 64; i++) t1 ^= ((t0 >> i) & 0x1);
		if (t1 == 0) counter[threadIndex]++;*/

		//	t0 = (initial0 ^ pair0) & 0x9224b6d24b6eda49;
		//		t0 = (initial0 ^ pair0) & 0x0000000000000200;
//		t0 = (initial0 ^ pair0) & 0x9324496da496ddb4; // Authors 6-round
		t0 = (initial0 ^ pair0) & 0x0200000000000000;
//		t0 = (initial0 ^ pair0) & 0x892db492dbb69264;
//		t0 = (initial4 ^ pair4) & 0x297cc63cdc4b8fec;
		t0 ^= t0 >> 1;
		t0 ^= t0 >> 2;
		t0 = (t0 & 0x1111111111111111UL) * 0x1111111111111111UL;
		t0 = (t0 >> 60) & 1;
		if (t0 == 0) counter[threadIndex]++;
	}
}
__global__ void ASCON12_benchmark(bit64 IV[], bit64 key[], bit64 nonce[], bit64 keystream, __int64 trial) {
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	bit64 initial0, initial1, initial2, initial3, initial4;
	bit64 t0, t1, t2, t3, t4;
	bit64 IV2 = IV[threadIndex];
	bit64 key0 = key[2 * threadIndex];
	bit64 key1 = key[2 * threadIndex]+1;
	bit64 nonce0 = key[2 * threadIndex];
	bit64 nonce1 = key[2 * threadIndex] + 1;
	for (__int64 c = 0; c < trial; c++) {
		initial0 = IV2;
		initial1 = key0;
		initial2 = key1;
		initial3 = nonce0;
		initial4 = nonce1;
#pragma unroll
		for (int i = 0; i < 12; i++) {
			initial0 ^= initial4; initial4 ^= initial3; initial2 ^= initial1;
			t0 = initial0; t1 = initial1; t2 = initial2; t3 = initial3; t4 = initial4;
			t0 = ~t0; t1 = ~t1; t2 = ~t2; t3 = ~t3; t4 = ~t4;
			t0 &= initial1; t1 &= initial2; t2 &= initial3; t3 &= initial4; t4 &= initial0;
			initial0 ^= t1; initial1 ^= t2; initial2 ^= t3; initial3 ^= t4; initial4 ^= t0;
			initial1 ^= initial0; initial0 ^= initial4; initial3 ^= initial2; initial2 = ~initial2;
			// Liner layer //
			t0 = rotater(initial0, 19);
			t1 = rotater(initial0, 28);
			initial0 ^= t0 ^ t1;
			t0 = rotater(initial1, 61);
			t1 = rotater(initial1, 39);
			initial1 ^= t0 ^ t1;
			t0 = rotater(initial2, 1);
			t1 = rotater(initial2, 6);
			initial2 ^= t0 ^ t1;
			t0 = rotater(initial3, 10);
			t1 = rotater(initial3, 17);
			initial3 ^= t0 ^ t1;
			t0 = rotater(initial4, 7);
			t1 = rotater(initial4, 41);
			initial4 ^= t0 ^ t1;
		}
		nonce1++;
		if (initial0 == keystream) printf("Hello world\n");
	}
}
__global__ void ASCON_crypto24(bit64 IV[], bit64 key[], bit64 nonce[], __int64 counter[], __int64 trial) {
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	bit64 initial0, initial1, initial2, initial3, initial4;
	bit64 pair0, pair1, pair2, pair3, pair4;
	bit64 t0, t1, t2, t3, t4;
	initial0 = IV[threadIndex];
	initial1 = key[2 * threadIndex];
	initial2 = key[2 * threadIndex + 1];
	initial3 = nonce[2 * threadIndex];
	initial4 = nonce[2 * threadIndex + 1];
	for (__int64 c = 0; c < trial; c++) {
		pair0 = initial0;
		pair1 = initial1;
		pair2 = initial2;
		pair3 = initial3 ^ 0x8000000000000000;
		pair4 = initial4 ^ 0x8000000000000000;
#pragma unroll
		for (int i = 0; i < 5; i++) {
			initial0 ^= initial4; initial4 ^= initial3; initial2 ^= initial1;
			t0 = initial0; t1 = initial1; t2 = initial2; t3 = initial3; t4 = initial4;
			t0 = ~t0; t1 = ~t1; t2 = ~t2; t3 = ~t3; t4 = ~t4;
			t0 &= initial1; t1 &= initial2; t2 &= initial3; t3 &= initial4; t4 &= initial0;
			initial0 ^= t1; initial1 ^= t2; initial2 ^= t3; initial3 ^= t4; initial4 ^= t0;
			initial1 ^= initial0; initial0 ^= initial4; initial3 ^= initial2; initial2 = ~initial2;
			// Liner layer //
			t0 = rotater(initial0, 19);			t1 = rotater(initial0, 28);			initial0 ^= t0 ^ t1;
			t0 = rotater(initial1, 61);			t1 = rotater(initial1, 39);			initial1 ^= t0 ^ t1;
			t0 = rotater(initial2, 1);			t1 = rotater(initial2, 6);			initial2 ^= t0 ^ t1;
			t0 = rotater(initial3, 10);			t1 = rotater(initial3, 17);			initial3 ^= t0 ^ t1;
			t0 = rotater(initial4, 7);			t1 = rotater(initial4, 41);			initial4 ^= t0 ^ t1;
		}
		initial0 ^= initial4; initial4 ^= initial3; initial2 ^= initial1;
		t0 = initial0; t1 = initial1; t2 = initial2; t3 = initial3; t4 = initial4;
		t0 = ~t0; t1 = ~t1; t2 = ~t2; t3 = ~t3; t4 = ~t4;
		t0 &= initial1; t1 &= initial2; t2 &= initial3; t3 &= initial4; t4 &= initial0;
		initial0 ^= t1; initial1 ^= t2; initial2 ^= t3; initial3 ^= t4; initial4 ^= t0;
		initial1 ^= initial0; initial0 ^= initial4; initial3 ^= initial2; initial2 = ~initial2;
#pragma unroll
		for (int i = 0; i < 5; i++) {
			pair0 ^= pair4; pair4 ^= pair3; pair2 ^= pair1;
			t0 = pair0; t1 = pair1; t2 = pair2; t3 = pair3; t4 = pair4;
			t0 = ~t0; t1 = ~t1; t2 = ~t2; t3 = ~t3; t4 = ~t4;
			t0 &= pair1; t1 &= pair2; t2 &= pair3; t3 &= pair4; t4 &= pair0;
			pair0 ^= t1; pair1 ^= t2; pair2 ^= t3; pair3 ^= t4; pair4 ^= t0;
			pair1 ^= pair0; pair0 ^= pair4; pair3 ^= pair2; pair2 = ~pair2;
			// Liner layer //
			t0 = rotater(pair0, 19);			t1 = rotater(pair0, 28);			pair0 ^= t0 ^ t1;
			t0 = rotater(pair1, 61);			t1 = rotater(pair1, 39);			pair1 ^= t0 ^ t1;
			t0 = rotater(pair2, 1);				t1 = rotater(pair2, 6);			pair2 ^= t0 ^ t1;
			t0 = rotater(pair3, 10);			t1 = rotater(pair3, 17);			pair3 ^= t0 ^ t1;
			t0 = rotater(pair4, 7);				t1 = rotater(pair4, 41);			pair4 ^= t0 ^ t1;
		}
		pair0 ^= pair4; pair4 ^= pair3; pair2 ^= pair1;
		t0 = pair0; t1 = pair1; t2 = pair2; t3 = pair3; t4 = pair4;
		t0 = ~t0; t1 = ~t1; t2 = ~t2; t3 = ~t3; t4 = ~t4;
		t0 &= pair1; t1 &= pair2; t2 &= pair3; t3 &= pair4; t4 &= pair0;
		pair0 ^= t1; pair1 ^= t2; pair2 ^= t3; pair3 ^= t4; pair4 ^= t0;
		pair1 ^= pair0; pair0 ^= pair4; pair3 ^= pair2; pair2 = ~pair2;
		t0 = (initial0 ^ pair0) & 0x0200000000000000;
		t0 ^= t0 >> 1;
		t0 ^= t0 >> 2;
		t0 = (t0 & 0x1111111111111111UL) * 0x1111111111111111UL;
		t0 = (t0 >> 60) & 1;
		if (t0 == 0) counter[threadIndex]++;
	}
}
void ASCON_benchmark() {
	printf("Trial = 2^18 +  ");
	scanf_s("%d", &trial);
	trial = (__int64)1 << trial;
	float milliseconds = 0;
	cudaMalloc((void **)&key_d, 2 * sizeof(bit64));
	cudaMalloc((void **)&nonce_d, BLOCKS * THREADS * 2 * sizeof(bit64));
	cudaMalloc((void **)&keyrows_d, BLOCKS * THREADS * 2 * sizeof(bit64));
	cudaMalloc((void **)&IV_d, BLOCKS * THREADS * sizeof(bit64));
	rdrand_64(key, 0);
	rdrand_64(key + 1, 0);
	cudaEvent_t start, stop;
		for (int j = 0; j < THREADS * BLOCKS * 2; j++) { rdrand_64(nonce + j, 0); rdrand_64(keyrows + j, 0); }
		for (int j = 0; j < THREADS * BLOCKS; j++) { rdrand_64(IV + j, 0); }
		cudaMemcpy(nonce_d, nonce, BLOCKS * THREADS * 2 * sizeof(bit64), cudaMemcpyHostToDevice);
		cudaMemcpy(keyrows_d, keyrows, BLOCKS * THREADS * 2 * sizeof(bit64), cudaMemcpyHostToDevice);
		cudaMemcpy(IV_d, IV, BLOCKS * THREADS * sizeof(bit64), cudaMemcpyHostToDevice);
		StartCounter();
		cudaDeviceSynchronize(); clock_t beginTime = clock();
		cudaEventCreate(&start);	cudaEventCreate(&stop);	cudaEventRecord(start);
		ASCON12_benchmark << <BLOCKS, THREADS >> >(IV_d, keyrows_d, nonce_d, 0x0123456789abcdef, trial);
		cudaEventRecord(stop);	cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);	printf("Time elapsed: %f milliseconds ", milliseconds);	printf("Time of kernel: %lf\n", GetCounter());	
}
void ASCON55_crypto24() {
	FILE* fp;	fopen_s(&fp, "crypto24.txt", "w");
	__int64 *counter_d, total_counter = 0, bias, average_bias = 0, experiment;
	__int64 *counter;
	printf("Trial = 2^18 +  ");
	scanf_s("%d", &trial);
	trial = (__int64)1 << trial;
	experiment = trial*THREADS*BLOCKS;
	float milliseconds = 0;
	total_counter = 0;
	cudaMalloc((void **)&key_d, 2 * sizeof(bit64));
	cudaMalloc((void **)&nonce_d, BLOCKS * THREADS * 2 * sizeof(bit64));
	cudaMalloc((void **)&keyrows_d, BLOCKS * THREADS * 2 * sizeof(bit64));
	cudaMalloc((void **)&IV_d, BLOCKS * THREADS * sizeof(bit64));
	cudaMalloc((void **)&counter_d, BLOCKS * THREADS * sizeof(bit64));
	rdrand_64(key, 0);	rdrand_64(key + 1, 0);
	
	counter = (__int64*)calloc(BLOCKS * THREADS, sizeof(bit64));
	cudaEvent_t start, stop;
	for (int j = 0; j < THREADS * BLOCKS * 2; j++) { rdrand_64(nonce + j, 0); rdrand_64(keyrows + j, 0); }
	for (int j = 0; j < THREADS * BLOCKS; j++) { rdrand_64(IV + j, 0); }
	cudaMemcpy(counter_d, counter, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyHostToDevice);
	cudaMemcpy(nonce_d, nonce, BLOCKS * THREADS * 2 * sizeof(bit64), cudaMemcpyHostToDevice);
	cudaMemcpy(keyrows_d, keyrows, BLOCKS * THREADS * 2 * sizeof(bit64), cudaMemcpyHostToDevice);
	cudaMemcpy(IV_d, IV, BLOCKS * THREADS * sizeof(bit64), cudaMemcpyHostToDevice);
	StartCounter();
	cudaDeviceSynchronize(); clock_t beginTime = clock();
	cudaEventCreate(&start);	cudaEventCreate(&stop);	cudaEventRecord(start);
	ASCON_crypto24 << <BLOCKS, THREADS >> >(IV_d, keyrows_d, nonce_d, counter_d, trial);
	cudaEventRecord(stop);	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);	printf("Time elapsed: %f milliseconds ", milliseconds);	printf("Time of kernel: %lf\n", GetCounter());

	cudaMemcpy(counter, counter_d, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyDeviceToHost);
	for (int i = 0; i < BLOCKS*THREADS; i++) total_counter += counter[i];
	bias = (experiment) / 2 - total_counter;
//	printf("Total counter: %I64d Bias: %I64d\n", total_counter, bias);
	printf("\nTotal counter: %I64d\nDifference from the Expected Value: %I64d\nBias: 2^-%lf (For an experiment with 2^%lf data)\n", total_counter, bias, ((log(BLOCKS) + log(THREADS) + log(trial)) / log(2)) - (log(abs(bias)) / log(2)), (log(BLOCKS) + log(THREADS) + log(trial)) / log(2));
	fprintf(fp, "Total counter: %I64d Bias: %I64d\n", total_counter, bias);
	cudaFree(key_d); cudaFree(nonce_d); cudaFree(counter_d);
	fclose(fp);
}
int main(void) {
	cudaSetDevice(0);
	__int64 *counter_d, total_counter=0, bias, average_bias=0, experiment;

	nonce = (bit64*)calloc(BLOCKS * THREADS * 2, sizeof(bit64));
	keyrows = (bit64*)calloc(BLOCKS * THREADS * 2, sizeof(bit64));
	IV = (bit64*)calloc(BLOCKS * THREADS, sizeof(bit64));
	int choice = 0, key_choice = 0;
	printf(
		"(1) CUDA ASCON 6-round DL Check of CRYPTO'24 Paper (eprint 2024-871)\n"
		"(2) CUDA ASCON 5.5-round DL Verification of CRYPTO'24 Paper  (eprint 2024-871)\n"
		"(3) CUDA ASCON 12-round Initialization Benchmark\n"
		"Choice: "
		);
	scanf_s("%d", &choice);
	if (choice == 1) {
		FILE* fp;
		fopen_s(&fp, "eprint6r.txt", "w");
		printf("Trial = 2^30 +  ");
		scanf_s("%d", &trial);
		//trial = pow(2, trial);
		trial = (__int64)1 << trial;
		experiment = TRIALS*trial*THREADS*BLOCKS;
		for (int m = 0; m < 1; m++) {
			//			__int64 counter[BLOCKS * THREADS] = { 0 };
			__int64 *counter;
			counter = (__int64*)calloc(BLOCKS * THREADS, sizeof(bit64));
			total_counter = 0;
			cudaMalloc((void **)&key_d, 2 * sizeof(bit64));
			cudaMalloc((void **)&nonce_d, BLOCKS * THREADS * 2 * sizeof(bit64));
			cudaMalloc((void **)&keyrows_d, BLOCKS * THREADS * 2 * sizeof(bit64));
			cudaMalloc((void **)&IV_d, BLOCKS * THREADS * sizeof(bit64));
			cudaMalloc((void **)&counter_d, BLOCKS * THREADS * sizeof(bit64));
			rdrand_64(key, 0);
			rdrand_64(key + 1, 0);
			//		printf("%I64x %I64x\n", key[0], key[1]);
			StartCounter();
			cudaMemcpy(counter_d, counter, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyHostToDevice);
			for (int i = 0; i < trial; i++) {
				for (int j = 0; j < THREADS * BLOCKS * 2; j++) { rdrand_64(nonce + j, 0); rdrand_64(keyrows + j, 0); }
				for (int j = 0; j < THREADS * BLOCKS; j++) { rdrand_64(IV + j, 0); }
				cudaMemcpy(counter, counter_d, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyDeviceToHost);
				cudaMemcpy(nonce_d, nonce, BLOCKS * THREADS * 2 * sizeof(bit64), cudaMemcpyHostToDevice);
				cudaMemcpy(keyrows_d, keyrows, BLOCKS * THREADS * 2 * sizeof(bit64), cudaMemcpyHostToDevice);
				cudaMemcpy(IV_d, IV, BLOCKS * THREADS * sizeof(bit64), cudaMemcpyHostToDevice);
				cudaMemcpy(counter_d, counter, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyHostToDevice);
				ASCON6_eprint << <BLOCKS, THREADS >> >(IV_d, keyrows_d, nonce_d, counter_d);
			}
			cudaMemcpy(counter, counter_d, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyDeviceToHost);
			printf("Time of kernel: %lf\n", GetCounter());
			for (int i = 0; i < BLOCKS*THREADS; i++) total_counter += counter[i];
			bias = (experiment) / 2 - total_counter;
//			printf("%03d: Total counter: %I64d Bias: %I64d\n", m, total_counter, bias);
			printf("\nTotal counter: %I64d\nDifference from the Expected Value: %I64d\nBias: 2^-%lf (For an experiment with 2^%lf data)\n", total_counter, bias, ((log(BLOCKS) + log(THREADS) + log(TRIALS) + log(trial)) / log(2)) - (log(abs(bias)) / log(2)), (log(BLOCKS) + log(THREADS) + log(trial) + log(TRIALS)) / log(2));
			fprintf(fp,"%03d: Total counter: %I64d Bias: %I64d\n", m, total_counter, bias);
			cudaFree(key_d); cudaFree(nonce_d); cudaFree(counter_d);
			fclose(fp);
		}
	}
	else if (choice == 2)	ASCON55_crypto24();
	else if (choice == 3) ASCON_benchmark();
	free(nonce); free(keyrows); free(IV);
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}

