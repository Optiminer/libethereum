// Kernels for pipelined hashing by Optiminer.
// Please send ETH donations to 72afb263c4402ce70b17ce36c991ef3674e47f35

typedef union {
    ulong ulongs[128 / sizeof(ulong)];  // 16
    uint uints[128 / sizeof(uint)];     // 32
} hash128_t; // 64 bytes

typedef union {
    ulong ulongs[64 / sizeof(ulong)];  // 8
    uint uints[64 / sizeof(uint)];     // 16
} hash64_t; // 64 bytes

typedef union {
    ulong ulongs[32 / sizeof(ulong)];  // 4
    uint uints[32 / sizeof(uint)];     // 8
} hash32_t; // 32 bytes

#define UINT2_ROTATE

ulong rot64(ulong x, uint n) {
#ifdef UINT2_ROTATE
    uint2 a = as_uint2(x);
    if (n < 32) {
        a = (uint2)(a.x << n | a.y >> (32-n), a.y << n | a.x >> (32-n));
    } else {
        a = (uint2)(a.y << (n-32) | a.x >> (64-n), a.x << (n-32) | a.y >> (64-n));
    }
    return as_ulong(a);
#else 
    return rotate(x, (ulong)n);
#endif
}

__constant ulong const Keccak_f1600_RC[24] = {
        (ulong)(0x0000000000000001),
        (ulong)(0x0000000000008082), 
        (ulong)(0x800000000000808A),
        (ulong)(0x8000000080008000),
        (ulong)(0x000000000000808B),
        (ulong)(0x0000000080000001),
        (ulong)(0x8000000080008081),
        (ulong)(0x8000000000008009),
        (ulong)(0x000000000000008A),
        (ulong)(0x0000000000000088),
        (ulong)(0x0000000080008009),
        (ulong)(0x000000008000000A),
        (ulong)(0x000000008000808B),
        (ulong)(0x800000000000008B),
        (ulong)(0x8000000000008089),
        (ulong)(0x8000000000008003),
        (ulong)(0x8000000000008002),
        (ulong)(0x8000000000000080),
        (ulong)(0x000000000000800A),
        (ulong)(0x800000008000000A),
        (ulong)(0x8000000080008081),
        (ulong)(0x8000000000008080),
        (ulong)(0x0000000080000001),
        (ulong)(0x8000000080008008), 
};


void keccak_f1600_round(ulong* a, uint r, uint out_size) {
    ulong b[25];
    ulong t;

    // Theta
    b[0] = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20];
    b[1] = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21];
    b[2] = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22];
    b[3] = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23];
    b[4] = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24];
    t = b[4] ^ rot64(b[1], 1);
    a[0] ^= t;
    a[5] ^= t;
    a[10] ^= t;
    a[15] ^= t;
    a[20] ^= t;
    t = b[0] ^ rot64(b[2], 1);
    a[1] ^= t;
    a[6] ^= t;
    a[11] ^= t;
    a[16] ^= t;
    a[21] ^= t;
    t = b[1] ^ rot64(b[3], 1);
    a[2] ^= t;
    a[7] ^= t;
    a[12] ^= t;
    a[17] ^= t;
    a[22] ^= t;
    t = b[2] ^ rot64(b[4], 1);
    a[3] ^= t;
    a[8] ^= t;
    a[13] ^= t;
    a[18] ^= t;
    a[23] ^= t;
    t = b[3] ^ rot64(b[0], 1);
    a[4] ^= t;
    a[9] ^= t;
    a[14] ^= t;
    a[19] ^= t;
    a[24] ^= t;

    // Rho Pi
    b[0] = a[0];
    b[10] = rot64(a[1], 1);
    b[7] = rot64(a[10], 3);
    b[11] = rot64(a[7], 6);
    b[17] = rot64(a[11], 10);
    b[18] = rot64(a[17], 15);
    b[3] = rot64(a[18], 21);
    b[5] = rot64(a[3], 28);
    b[16] = rot64(a[5], 36);
    b[8] = rot64(a[16], 45);
    b[21] = rot64(a[8], 55);
    b[24] = rot64(a[21], 2);
    b[4] = rot64(a[24], 14);
    b[15] = rot64(a[4], 27);
    b[23] = rot64(a[15], 41);
    b[19] = rot64(a[23], 56);
    b[13] = rot64(a[19], 8);
    b[12] = rot64(a[13], 25);
    b[2] = rot64(a[12], 43);
    b[20] = rot64(a[2], 62);
    b[14] = rot64(a[20], 18);
    b[22] = rot64(a[14], 39);
    b[9] = rot64(a[22], 61);
    b[6] = rot64(a[9], 20);
    b[1] = rot64(a[6], 44);

    // Chi
    a[0] = bitselect(b[0] ^ b[2], b[0], b[1]);
    a[1] = bitselect(b[1] ^ b[3], b[1], b[2]);
    a[2] = bitselect(b[2] ^ b[4], b[2], b[3]);
    a[3] = bitselect(b[3] ^ b[0], b[3], b[4]);
    if (out_size > 4) {
        a[4] = bitselect(b[4] ^ b[1], b[4], b[0]);
        a[5] = bitselect(b[5] ^ b[7], b[5], b[6]);
        a[6] = bitselect(b[6] ^ b[8], b[6], b[7]);
        a[7] = bitselect(b[7] ^ b[9], b[7], b[8]);
        a[8] = bitselect(b[8] ^ b[5], b[8], b[9]);
        if (out_size > 8) {
            a[9] = bitselect(b[9] ^ b[6], b[9], b[5]);
            a[10] = bitselect(b[10] ^ b[12], b[10], b[11]);
            a[11] = bitselect(b[11] ^ b[13], b[11], b[12]);
            a[12] = bitselect(b[12] ^ b[14], b[12], b[13]);
            a[13] = bitselect(b[13] ^ b[10], b[13], b[14]);
            a[14] = bitselect(b[14] ^ b[11], b[14], b[10]);
            a[15] = bitselect(b[15] ^ b[17], b[15], b[16]);
            a[16] = bitselect(b[16] ^ b[18], b[16], b[17]);
            a[17] = bitselect(b[17] ^ b[19], b[17], b[18]);
            a[18] = bitselect(b[18] ^ b[15], b[18], b[19]);
            a[19] = bitselect(b[19] ^ b[16], b[19], b[15]);
            a[20] = bitselect(b[20] ^ b[22], b[20], b[21]);
            a[21] = bitselect(b[21] ^ b[23], b[21], b[22]);
            a[22] = bitselect(b[22] ^ b[24], b[22], b[23]);
            a[23] = bitselect(b[23] ^ b[20], b[23], b[24]);
            a[24] = bitselect(b[24] ^ b[21], b[24], b[20]);
        }
    }

    // Iota
    a[0] ^= Keccak_f1600_RC[r];
}

void keccak_f1600_no_absorb(ulong* a, uint in_size, uint out_size) {
    for (uint i = in_size; i != 25; ++i) {
        a[i] = 0;
    }
    a[in_size] ^= 0x0000000000000001UL;
    a[24-out_size*2] ^= 0x8000000000000000UL;

    for(int r = 0; r<23; ++r) {
      keccak_f1600_round(a, r, 25);
    };

    // final round optimised for digest size
    keccak_f1600_round(a, 23, out_size);
}

#define copy(dst, src, count) for (uint i = 0; i != count; ++i) { (dst)[i] = (src)[i]; }

__kernel void hash_init(
        __constant hash32_t const* header,
        ulong start_nonce,
        __global hash64_t* init)
{
    ulong nonce = start_nonce + get_global_id(0);

    // sha3_512(header .. nonce)
    ulong state[25];
    copy(state, header->ulongs, 4);
    state[4] = nonce;
    keccak_f1600_no_absorb(state, 5 /* in_size */, 8 /* out_size */);
    
    copy(init[get_global_id(0)].ulongs, state, 8);
}

__kernel void hash_final(
        ulong target,
        __global hash64_t const* hash_in,
        __global hash32_t const* mix_in,
        __global uint* output
) {
    uint item = get_global_id(0);
    ulong state[25];

    // keccak_256(keccak_512(header..nonce) .. mix);
    copy(state, hash_in[item].ulongs, 8);
    copy(state + 8, mix_in[item].ulongs, 4);
    keccak_f1600_no_absorb(state, 12 /* in_size */, 4 /* out_size */);

    if (as_ulong(as_uchar8(state[0]).s76543210) < target)
    {
        uint slot = atomic_inc(&output[0]) + 1;
        if (slot <= MAX_OUTPUTS) {
            output[slot] = item;
        }
    }
}

__kernel void finish_and_init_next_hash(
        ulong target,
        __global hash64_t* hash_in_out,
        __global hash32_t const* mix_in,
        __constant hash32_t const* header,
        ulong start_nonce,
        __global uint* output
) {
    hash_final(target, hash_in_out, mix_in, output);  
    hash_init(header, start_nonce, hash_in_out);
}

static const uint FNV_PRIME = 0x01000193;

static uint fnv(const uint x,const uint y)
{
    return x * FNV_PRIME ^ y;
}

__attribute__((reqd_work_group_size(32 * SUBGROUPS, 1, 1)))
__kernel void compute_hash_simple(
    __global hash128_t* const g_dag,
    __global hash64_t* const in,
    __global hash32_t* out
)
{ 
    
  uint item = get_global_id(0) / 32;
  uint subgroup = get_local_id(0) / 32;
  
  // Threads work in group of 32.
  uint t = get_local_id(0) % 32;
  
  __global const hash64_t* init = &in[item];
  uint mix_val = (*init).uints[t % 16];
  uint init0 = (*init).uints[0];
  
  __local uint pi[SUBGROUPS];
  for(uint a = 0; a < 64; ++a) {
    if (a % 32 == t) {
      pi[subgroup] = fnv(init0 ^ a, mix_val) % DAG_SIZE;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint p = pi[subgroup];
    mix_val = fnv(mix_val, g_dag[p].uints[t]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // reduce to output
  __local uint mixed[32*SUBGROUPS];
  mixed[get_local_id(0)] = mix_val;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t < 8) {
    __global hash32_t* fnv_mix = &out[item]; 
    uint i = t * 4 + 32 * subgroup;
    (*fnv_mix).uints[t] = fnv(fnv(fnv(mixed[i], mixed[i+1]), mixed[i+2]), mixed[i+3]);
  }
}


