/*
 * ucx_dp_sync.c — One-shot AllReduce for vLLM DP metadata over UCX/RDMA
 *
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Replaces Gloo TCP AllReduce (~100ms P99) with UCX tag-matching over
 * InfiniBand RDMA (~0.1ms P99) for the per-iteration DP metadata sync.
 *
 * Build:
 *   gcc -shared -fPIC -O2 -o _ucx_dp_sync.so ucx_dp_sync.c -lucp -lucs
 *
 * The allreduce is one-shot: every rank sends its full tensor to every
 * other rank, then locally sums. For 256 bytes x 15 peers this is 3.8KB
 * total — well within RDMA eager threshold, so all sends complete in a
 * single round with no rendezvous handshake.
 */

#include <ucp/api/ucp.h>
#include <ucs/type/status.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

#define MAX_RANKS 256

typedef struct {
    ucp_context_h ctx;
    ucp_worker_h  worker;
    ucp_ep_h     *eps;          /* [world_size], NULL for self */
    int            rank;
    int            world_size;
    uint64_t       round;       /* monotonic counter for tag uniqueness */
    uint8_t      **recv_bufs;   /* [world_size] pre-allocated */
    uint8_t       *send_staging;
    size_t         max_bytes;
} ucx_dp_state_t;

/* ---- request completion ---- */

static void req_init(void *request) {
    *(int *)request = 0;
}

static void send_cb(void *request, ucs_status_t status, void *user_data) {
    *(int *)request = 1;
}

static void recv_cb(void *request, ucs_status_t status,
                    const ucp_tag_recv_info_t *info, void *user_data) {
    *(int *)request = 1;
}

/* ---- public API ---- */

int ucx_dp_init(int rank, int world_size, size_t max_bytes,
                void **state_out, void **addr_out, size_t *addr_len_out) {
    ucs_status_t st;

    if (world_size > MAX_RANKS) return -1;

    ucx_dp_state_t *s = (ucx_dp_state_t *)calloc(1, sizeof(*s));
    if (!s) return -1;
    s->rank       = rank;
    s->world_size = world_size;
    s->max_bytes  = max_bytes;

    /* context */
    ucp_config_t *config;
    st = ucp_config_read(NULL, NULL, &config);
    if (st != UCS_OK) goto fail_s;

    ucp_params_t params;
    memset(&params, 0, sizeof(params));
    params.field_mask   = UCP_PARAM_FIELD_FEATURES
                        | UCP_PARAM_FIELD_REQUEST_SIZE
                        | UCP_PARAM_FIELD_REQUEST_INIT;
    params.features     = UCP_FEATURE_TAG;
    params.request_size = sizeof(int);
    params.request_init = req_init;

    st = ucp_init(&params, config, &s->ctx);
    ucp_config_release(config);
    if (st != UCS_OK) goto fail_s;

    /* worker */
    ucp_worker_params_t wp;
    memset(&wp, 0, sizeof(wp));
    wp.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    wp.thread_mode = UCS_THREAD_MODE_SINGLE;

    st = ucp_worker_create(s->ctx, &wp, &s->worker);
    if (st != UCS_OK) goto fail_ctx;

    /* worker address */
    ucp_address_t *addr;
    size_t addr_len;
    st = ucp_worker_get_address(s->worker, &addr, &addr_len);
    if (st != UCS_OK) goto fail_worker;

    /* buffers */
    s->eps          = (ucp_ep_h *)calloc(world_size, sizeof(ucp_ep_h));
    s->recv_bufs    = (uint8_t **)calloc(world_size, sizeof(uint8_t *));
    s->send_staging = (uint8_t *)malloc(max_bytes);
    for (int i = 0; i < world_size; i++) {
        if (i != rank)
            s->recv_bufs[i] = (uint8_t *)malloc(max_bytes);
    }

    *state_out    = s;
    *addr_out     = addr;          /* caller copies, then calls release */
    *addr_len_out = addr_len;
    return 0;

fail_worker: ucp_worker_destroy(s->worker);
fail_ctx:    ucp_cleanup(s->ctx);
fail_s:      free(s);
    return -1;
}

void ucx_dp_release_address(void *state, void *addr) {
    ucx_dp_state_t *s = (ucx_dp_state_t *)state;
    ucp_worker_release_address(s->worker, (ucp_address_t *)addr);
}

int ucx_dp_connect(void *state, int peer, const void *addr, size_t len) {
    ucx_dp_state_t *s = (ucx_dp_state_t *)state;

    ucp_ep_params_t ep;
    memset(&ep, 0, sizeof(ep));
    ep.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep.address    = (const ucp_address_t *)addr;

    ucs_status_t st = ucp_ep_create(s->worker, &ep, &s->eps[peer]);
    return (st == UCS_OK) ? 0 : -1;
}

int ucx_dp_allreduce_inplace(void *state, void *buf, size_t nbytes) {
    ucx_dp_state_t *s = (ucx_dp_state_t *)state;
    if (nbytes > s->max_bytes) return -1;

    uint64_t round = s->round++;
    int ws = s->world_size;

    memcpy(s->send_staging, buf, nbytes);

    ucs_status_ptr_t recv_reqs[MAX_RANKS];
    ucs_status_ptr_t send_reqs[MAX_RANKS];

    /* post all receives */
    for (int i = 0; i < ws; i++) {
        if (i == s->rank) { recv_reqs[i] = NULL; continue; }

        ucp_request_param_t p;
        memset(&p, 0, sizeof(p));
        p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK
                       | UCP_OP_ATTR_FIELD_FLAGS;
        p.cb.recv      = recv_cb;
        p.flags        = UCP_OP_ATTR_FLAG_NO_IMM_CMPL;

        ucp_tag_t tag = (round << 16) | (uint64_t)i;
        recv_reqs[i]  = ucp_tag_recv_nbx(s->worker, s->recv_bufs[i],
                                          nbytes, tag, ~(ucp_tag_t)0, &p);
    }

    /* post all sends */
    ucp_tag_t my_tag = (round << 16) | (uint64_t)s->rank;
    for (int i = 0; i < ws; i++) {
        if (i == s->rank) { send_reqs[i] = NULL; continue; }

        ucp_request_param_t p;
        memset(&p, 0, sizeof(p));
        p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK;
        p.cb.send      = send_cb;

        send_reqs[i] = ucp_tag_send_nbx(s->eps[i], s->send_staging,
                                          nbytes, my_tag, &p);
    }

    /* progress until every request completes */
    for (;;) {
        ucp_worker_progress(s->worker);
        int done = 1;
        for (int i = 0; i < ws; i++) {
            if (i == s->rank) continue;
            if (recv_reqs[i] && !UCS_PTR_IS_ERR(recv_reqs[i])) {
                if (ucp_request_check_status(recv_reqs[i]) == UCS_INPROGRESS) {
                    done = 0; break;
                }
            }
            if (send_reqs[i] && !UCS_PTR_IS_ERR(send_reqs[i])) {
                if (ucp_request_check_status(send_reqs[i]) == UCS_INPROGRESS) {
                    done = 0; break;
                }
            }
        }
        if (done) break;
    }

    /* free requests */
    for (int i = 0; i < ws; i++) {
        if (i == s->rank) continue;
        if (recv_reqs[i] && !UCS_PTR_IS_ERR(recv_reqs[i]))
            ucp_request_free(recv_reqs[i]);
        if (send_reqs[i] && !UCS_PTR_IS_ERR(send_reqs[i]))
            ucp_request_free(send_reqs[i]);
    }

    /*
     * Memory fence: ensure all recv buffer writes from UCX are
     * visible before we read them for the reduction. On ARM
     * (aarch64) the store from the transport thread and our read
     * may not be ordered without this.
     */
    __sync_synchronize();

    /* local reduce: buf = local + sum(received) */
    int32_t *out   = (int32_t *)buf;
    int32_t *local = (int32_t *)s->send_staging;
    int count      = (int)(nbytes / sizeof(int32_t));
    memcpy(out, local, nbytes);
    for (int i = 0; i < ws; i++) {
        if (i == s->rank) continue;
        int32_t *peer = (int32_t *)s->recv_bufs[i];
        for (int j = 0; j < count; j++)
            out[j] += peer[j];
    }

    return 0;
}

void ucx_dp_finalize(void *state) {
    ucx_dp_state_t *s = (ucx_dp_state_t *)state;
    if (!s) return;

    /* flush pending endpoint ops */
    for (int i = 0; i < 64; i++)
        ucp_worker_progress(s->worker);

    if (s->worker) ucp_worker_destroy(s->worker);
    if (s->ctx)    ucp_cleanup(s->ctx);

    for (int i = 0; i < s->world_size; i++) {
        if (s->recv_bufs && s->recv_bufs[i]) free(s->recv_bufs[i]);
    }
    free(s->eps);
    free(s->recv_bufs);
    free(s->send_staging);
    free(s);
}
