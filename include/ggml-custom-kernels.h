#pragma once

#include "ggml.h"

#include <stdint.h>

// Tagging convention that lets GPU backends recognize and natively execute
// selected GGML_OP_CUSTOM ops instead of splitting the graph to run the CPU
// callback: the op's userdata must point to a struct whose first member is a
// ggml_custom_kernel_hdr carrying the magic below, and the op itself must be
// explicitly tagged with ggml_custom_kernel_tag(). Backends that do not know the
// convention (or the specific kind) are unaffected: the op keeps its CPU
// callback and runs there as before.

#define GGML_CUSTOM_KERNEL_MAGIC 0x474d434bu // "GMCK"

enum ggml_custom_kernel_kind {
    GGML_CUSTOM_KERNEL_ERF           = 1, // dst = erf(src0); contiguous f32, same nelements
    GGML_CUSTOM_KERNEL_REDUCE_MAX    = 2, // dst[r] = max over src0 row ne0; dst ne0 = nrows(src0)
    GGML_CUSTOM_KERNEL_REDUCE_MIN    = 3,
    GGML_CUSTOM_KERNEL_REDUCE_SUM    = 4,
    GGML_CUSTOM_KERNEL_MSDEFORM_ATTN = 5, // fused multi-scale deformable attention (see below)
};

struct ggml_custom_kernel_hdr {
    uint32_t magic;
    uint32_t kind;
};

#define GGML_CUSTOM_KERNEL_OP_PARAM_MAGIC (GGML_MAX_OP_PARAMS / sizeof(int32_t) - 2)
#define GGML_CUSTOM_KERNEL_OP_PARAM_KIND  (GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1)

static inline void ggml_custom_kernel_tag(struct ggml_tensor * op, enum ggml_custom_kernel_kind kind) {
    op->op_params[GGML_CUSTOM_KERNEL_OP_PARAM_MAGIC] = (int32_t) GGML_CUSTOM_KERNEL_MAGIC;
    op->op_params[GGML_CUSTOM_KERNEL_OP_PARAM_KIND]  = (int32_t) kind;
}

// Fused multi-scale deformable attention (RT-DETR decoder), batch=1.
// All srcs contiguous f32 (ne0-fastest):
//   src[0] value   ne=[head_dim, heads, S, 1]
//   src[1] offsets ne=[heads*levels*points*2, Q, 1, 1]
//   src[2] attn    ne=[levels*points, heads, Q, 1] (post-softmax)
//   src[3] refpts  ne=[4, 1, Q, 1] (cx, cy, w, h)
// dst ne=[Q, head_dim, heads, 1].
struct ggml_custom_kernel_msdeform_attn {
    struct ggml_custom_kernel_hdr hdr;
    int32_t heads, levels, points, head_dim;
    int32_t hl[8], wl[8], level_off[8];
};
