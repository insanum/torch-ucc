/**
 * * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * *
 * * See file LICENSE for terms.
 * */

#include "torch_ucx_coll.hpp"

namespace c10d {

static inline int get_recv_peer(int group_rank, int group_size,
                                int step, bool is_reverse)
{
    if (is_reverse) {
        return (group_rank - 1 - step + group_size) % group_size;
    } else {
        return (group_rank + 1 + step) % group_size;
    }
}

static inline int get_send_peer(int group_rank, int group_size,
                                int step, bool is_reverse)
{
    if (is_reverse) {
        return (group_rank + 1 + step) % group_size;
    } else {
        return (group_rank - 1 - step + group_size) % group_size;
    }
}

static inline void torch_ucx_memcpy(void *dst, torch_ucx_memtype_t dst_mtype,
                                    void *src, torch_ucx_memtype_t src_mtype,
                                    size_t size, torch_ucx_coll_comm_t *comm)
{
    if ((src_mtype == TORCH_UCX_HOST) && (dst_mtype == TORCH_UCX_HOST)) {
        memcpy(dst, src, size);
        return;
    }

#ifdef USE_CUDA
    cudaMemcpyKind mk;
    if (comm->stream == 0) {
        cudaStreamCreateWithFlags(&comm->stream, cudaStreamNonBlocking);
    }
    if ((src_mtype == TORCH_UCX_CUDA) && (dst_mtype == TORCH_UCX_CUDA)) {
        mk = cudaMemcpyDeviceToDevice;
    } else if ((src_mtype == TORCH_UCX_CUDA) && (dst_mtype == TORCH_UCX_HOST)) {
        mk = cudaMemcpyDeviceToHost;
    } else if ((src_mtype == TORCH_UCX_HOST) && (dst_mtype == TORCH_UCX_CUDA)) {
        mk = cudaMemcpyHostToDevice;
    }
    cudaMemcpyAsync(dst, src, size, mk, comm->stream);
#else
    fprintf(stderr, "TorchUCC: CUDA buffers are used but plugin CUDA support is disabled");
#endif 
}

static inline void sync_stream(torch_ucx_memtype_t dst_mtype,
                               torch_ucx_memtype_t src_mtype,
                               torch_ucx_coll_comm_t *comm)
{
#ifdef USE_CUDA
    if ((src_mtype == TORCH_UCX_CUDA) || (dst_mtype == TORCH_UCX_CUDA)) {
        cudaStreamSynchronize(comm->stream);
    }
#endif
}

torch_ucc_status_t torch_ucx_alltoall_progress(torch_ucx_coll_request_t *request)
{
    torch_ucx_comm_t  *p2p_comm  = request->comm->p2p_comm;
    int               group_size = p2p_comm->size;
    int               group_rank = p2p_comm->rank;
    size_t            data_size  = request->len;
    ptrdiff_t         sbuf       = (ptrdiff_t)request->src_buffer;
    ptrdiff_t         rbuf       = (ptrdiff_t)request->dst_buffer;
    bool              reverse    = request->comm->config.reverse;
    int               max_polls  = request->comm->config.max_polls;
    int               chunk      = request->comm->config.chunk;
    uint32_t          tag        = request->tag;

    int total_reqs, n_polls, released_slot;
    torch_ucx_status_t st;


    if ((chunk > group_size - 1) || (chunk <= 0)) {
        total_reqs = group_size - 1;
    } else {
        total_reqs = chunk;
    }
    
    n_polls = 0;
    while ((n_polls++ < max_polls) &&
           ((request->n_sreqs != group_size - 1) || (request->n_rreqs != group_size - 1))) {
        if (request->n_rreqs < group_size - 1) {
            st = torch_ucx_req_test(p2p_comm, request->reqs, total_reqs,
                                    &released_slot, 1, 1);
            if (st == TORCH_UCX_OK) {
                int peer = get_recv_peer(group_rank, group_size, 
                                         request->n_rreqs, reverse);
                torch_ucx_recv_nb(p2p_comm, (void*)(rbuf + peer * data_size),
                                  data_size, peer, tag,
                                  &request->reqs[released_slot],
                                  TORCH_UCX_COLL_TAG);
                request->n_rreqs++;
                n_polls = 0;
            }
        }
        if (request->n_sreqs < group_size - 1) {
            st = torch_ucx_req_test(p2p_comm, request->reqs + total_reqs,
                                    total_reqs, &released_slot, 1, 1);
            if (st == TORCH_UCX_OK) {
                int peer = get_send_peer(group_rank, group_size,
                                         request->n_sreqs, reverse);
                torch_ucx_send_nb(p2p_comm, (void*)(sbuf + peer * data_size),
                                  data_size, peer, tag,
                                  &request->reqs[released_slot + total_reqs],
                                  TORCH_UCX_COLL_TAG);
                request->n_sreqs++;
                n_polls = 0;
            }
        }
    }

    if ((request->n_sreqs != group_size - 1) || (request->n_rreqs != group_size - 1)) {
        return TORCH_UCC_OK;
    }

    st = torch_ucx_req_test(p2p_comm, request->reqs, 2*(total_reqs), NULL,
                            max_polls, 2*(total_reqs));
    if (st == TORCH_UCX_INPROGRESS) {
        return TORCH_UCC_OK;
    }
    sync_stream(request->dst_buf_mtype, request->src_buf_mtype, request->comm);
    delete[] request->reqs;
    request->status = TORCH_UCC_OK;

    return TORCH_UCC_OK;
}


torch_ucx_status_t torch_ucx_alltoall_start(torch_ucx_coll_comm_t *comm,
                                            torch_ucx_coll_request_t *request)
{
    torch_ucx_comm_t  *p2p_comm  = comm->p2p_comm;
    int               group_size = p2p_comm->size;
    int               group_rank = p2p_comm->rank;
    size_t            data_size  = request->len;
    ptrdiff_t         sbuf       = (ptrdiff_t)request->src_buffer;
    ptrdiff_t         rbuf       = (ptrdiff_t)request->dst_buffer;
    bool              reverse    = comm->config.reverse;
    uint32_t          tag        = comm->last_tag;
    int total_reqs;

    if ((comm->config.chunk > group_size - 1) || (comm->config.chunk <= 0)) {
        total_reqs = group_size - 1;
    } else {
        total_reqs = comm->config.chunk;
    }
    request->reqs = new torch_ucx_request_t*[2*total_reqs];
    torch_ucx_memcpy((void*)(rbuf+data_size*group_rank), request->dst_buf_mtype,
                     (void*)(sbuf+data_size*group_rank), request->src_buf_mtype,
                     data_size, comm);
    for (int step = 0; step < total_reqs; step++) {
        int peer = get_recv_peer(group_rank, group_size, step, reverse);
        torch_ucx_recv_nb(p2p_comm, (void*)(rbuf + peer * data_size), data_size,
                          peer, tag, &request->reqs[step],
                          TORCH_UCX_COLL_TAG);
        peer = get_send_peer(group_rank, group_size, step, reverse);
        torch_ucx_send_nb(p2p_comm, (void*)(sbuf + peer * data_size), data_size,
                          peer, tag, &request->reqs[step + total_reqs],
                          TORCH_UCX_COLL_TAG);
    }
    request->tag      = tag;
    request->comm     = comm;
    request->n_rreqs  = total_reqs;
    request->n_sreqs  = total_reqs;
    request->status   = TORCH_UCC_INPROGRESS;
    request->progress = torch_ucx_alltoall_progress;

    comm->last_tag++;
    return TORCH_UCX_OK;
}

torch_ucc_status_t torch_ucx_alltoallv_progress(torch_ucx_coll_request_t *request)
{
    torch_ucx_comm_t  *p2p_comm  = request->comm->p2p_comm;
    int               group_size = p2p_comm->size;
    int               group_rank = p2p_comm->rank;
    ptrdiff_t         sbuf       = (ptrdiff_t)request->src_buffer;
    ptrdiff_t         rbuf       = (ptrdiff_t)request->dst_buffer;
    bool              reverse    = request->comm->config.reverse;
    int               max_polls  = request->comm->config.max_polls;
    int               chunk      = request->comm->config.chunk;
    uint32_t          tag        = request->tag;
    int send_size, recv_size;
    int send_displ, recv_displ;
    int total_reqs, n_polls, released_slot;
    torch_ucx_status_t st;


    if ((chunk > group_size - 1) || (chunk <= 0)) {
        total_reqs = group_size - 1;
    } else {
        total_reqs = chunk;
    }
    
    n_polls = 0;
    while ((n_polls++ < max_polls) &&
           ((request->n_sreqs != group_size - 1) || (request->n_rreqs != group_size - 1))) {
        if (request->n_rreqs < group_size - 1) {
            st = torch_ucx_req_test(p2p_comm, request->reqs, total_reqs,
                                    &released_slot, 1, 1);
            if (st == TORCH_UCX_OK) {
                int peer = get_recv_peer(group_rank, group_size, 
                                         request->n_rreqs, reverse);
                recv_size = request->recv_lengths[peer];
                recv_displ = request->recv_offsets[peer];
                torch_ucx_recv_nb(p2p_comm, (void*)(rbuf + recv_displ),
                                  recv_size, peer, tag,
                                  &request->reqs[released_slot],
                                  TORCH_UCX_COLL_TAG);
                request->n_rreqs++;
                n_polls = 0;
            }
        }
        if (request->n_sreqs < group_size - 1) {
            st = torch_ucx_req_test(p2p_comm, request->reqs + total_reqs,
                                    total_reqs, &released_slot, 1, 1);
            if (st == TORCH_UCX_OK) {
                int peer = get_send_peer(group_rank, group_size,
                                         request->n_sreqs, reverse);
                send_size = request->send_lengths[peer];
                send_displ = request->send_offsets[peer];
                torch_ucx_send_nb(p2p_comm, (void*)(sbuf + send_displ),
                                  send_size, peer, tag,
                                  &request->reqs[released_slot + total_reqs],
                                  TORCH_UCX_COLL_TAG);
                request->n_sreqs++;
                n_polls = 0;
            }
        }
    }

    if ((request->n_sreqs != group_size - 1) || (request->n_rreqs != group_size - 1)) {
        return TORCH_UCC_OK;
    }

    st = torch_ucx_req_test(p2p_comm, request->reqs, 2*total_reqs, NULL,
                            max_polls, 2*total_reqs);
    if (st == TORCH_UCX_INPROGRESS) {
        return TORCH_UCC_OK;
    }
    sync_stream(request->dst_buf_mtype, request->src_buf_mtype, request->comm);
    delete[] request->reqs;
    request->status = TORCH_UCC_OK;

    return TORCH_UCC_OK;
}

torch_ucx_status_t torch_ucx_alltoallv_start(torch_ucx_coll_comm_t *comm,
                                             torch_ucx_coll_request_t *request)
{
    torch_ucx_comm_t  *p2p_comm  = comm->p2p_comm;
    int               group_size = p2p_comm->size;
    int               group_rank = p2p_comm->rank;
    ptrdiff_t         sbuf       = (ptrdiff_t)request->src_buffer;
    ptrdiff_t         rbuf       = (ptrdiff_t)request->dst_buffer;
    bool              reverse    = comm->config.reverse;
    uint32_t          tag        = comm->last_tag;
    int total_reqs;
    int send_size, recv_size;
    int send_displ, recv_displ;

    if ((comm->config.chunk > group_size - 1) || (comm->config.chunk <= 0)) {
        total_reqs = group_size - 1;
    } else {
        total_reqs = comm->config.chunk;
    }
    request->reqs = new torch_ucx_request_t*[2*total_reqs];

    send_size = request->send_lengths[group_rank];
    recv_size = request->recv_lengths[group_rank];
    send_displ = request->send_offsets[group_rank];
    recv_displ = request->recv_offsets[group_rank];

    torch_ucx_memcpy((void*)(rbuf+recv_displ), request->dst_buf_mtype,
                     (void*)(sbuf+send_displ), request->src_buf_mtype,
                     send_size, comm);
    for (int step = 0; step < total_reqs; step++) {
        int peer = get_recv_peer(group_rank, group_size, step, reverse);
        recv_size = request->recv_lengths[peer];
        recv_displ = request->recv_offsets[peer];

        torch_ucx_recv_nb(p2p_comm, (void*)(rbuf + recv_displ), recv_size,
                          peer, tag, &request->reqs[step],
                          TORCH_UCX_COLL_TAG);
        peer = get_send_peer(group_rank, group_size, step, reverse);
        send_size = request->send_lengths[peer];
        send_displ = request->send_offsets[peer];
        torch_ucx_send_nb(p2p_comm, (void*)(sbuf + send_displ), send_size,
                          peer, tag, &request->reqs[step + total_reqs],
                          TORCH_UCX_COLL_TAG);
    }
    request->tag      = tag;
    request->comm     = comm;
    request->n_rreqs  = total_reqs;
    request->n_sreqs  = total_reqs;
    request->status   = TORCH_UCC_INPROGRESS;
    request->progress = torch_ucx_alltoallv_progress;

    comm->last_tag++;
    return TORCH_UCX_OK;
}

torch_ucc_status_t torch_ucx_alltoall(void *coll_comm,
                                      void *send_buffer, torch_ucx_memtype_t send_mtype,
                                      void *recv_buffer, torch_ucx_memtype_t recv_mtype,
                                      size_t len, torch_ucc_coll_request_t **request)
{
    torch_ucx_coll_comm_t  *comm = (torch_ucx_coll_comm_t*)coll_comm;
    torch_ucx_comm_t  *p2p_comm  = comm->p2p_comm;
    int               group_size = p2p_comm->size;
    int               group_rank = p2p_comm->rank;
    ptrdiff_t         sbuf       = (ptrdiff_t)send_buffer;
    ptrdiff_t         rbuf       = (ptrdiff_t)recv_buffer;
    size_t            data_size  = len;
    bool              reverse    = comm->config.reverse;
    uint32_t          tag        = comm->last_tag;
    torch_ucx_coll_request_t *req;
    int total_reqs;

    req = new torch_ucx_coll_request_t;
    req->len           = len;
    req->src_buffer    = send_buffer;
    req->dst_buffer    = recv_buffer;
    req->src_buf_mtype = send_mtype;
    req->dst_buf_mtype = recv_mtype;

    if ((comm->config.chunk > group_size - 1) || (comm->config.chunk <= 0)) {
        total_reqs = group_size - 1;
    } else {
        total_reqs = comm->config.chunk;
    }
    req->reqs = new torch_ucx_request_t*[2*total_reqs];
    torch_ucx_memcpy((void*)(rbuf+data_size*group_rank), recv_mtype,
                     (void*)(sbuf+data_size*group_rank), send_mtype,
                     data_size, comm);
    for (int step = 0; step < total_reqs; step++) {
        int peer = get_recv_peer(group_rank, group_size, step, reverse);
        torch_ucx_recv_nb(p2p_comm, (void*)(rbuf + peer * data_size), data_size,
                          peer, tag, &req->reqs[step],
                          TORCH_UCX_COLL_TAG);
        peer = get_send_peer(group_rank, group_size, step, reverse);
        torch_ucx_send_nb(p2p_comm, (void*)(sbuf + peer * data_size), data_size,
                          peer, tag, &req->reqs[step + total_reqs],
                          TORCH_UCX_COLL_TAG);
    }
    req->tag      = tag;
    req->comm     = comm;
    req->n_rreqs  = total_reqs;
    req->n_sreqs  = total_reqs;
    req->status   = TORCH_UCC_INPROGRESS;
    req->progress = torch_ucx_alltoall_progress;

    comm->last_tag++;
    *request = (torch_ucc_coll_request_t*)req;

    return TORCH_UCC_OK;
}

}
