/*
 * Copyright (c) Broadcom Inc. 2020-2020. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <torch_bnxt_co.hpp>
#include <c10d/Utils.hpp>
#include <map>

namespace c10d {

struct torch_bnxt_co_request_t {
    torch_ucc_coll_request_t super;
    torch_bnxt_co_comm_t* comm;
    torch_ucc_status_t status;

    bnxt_co_coll_type_t coll_type;
    bnxt_co_dt_t datatype;
    int src_rank; /* source rank for broadcast */
    uint8_t *src_buf;
    uint32_t src_buf_len;
    uint8_t *dst_buf;
    uint32_t dst_buf_len;
    size_t elem_size;

    uint64_t tag;
    uint8_t *resp_msg;
    size_t resp_msg_len;

    at::Tensor flat_tensor;
};

std::map<at::ScalarType, bnxt_co_dt_t> bnxt_co_type_map = {
    { at::kChar,   BNXT_CO_DT_INT8 },
    { at::kShort,  BNXT_CO_DT_INT16 },
    { at::kInt,    BNXT_CO_DT_INT32 },
    { at::kLong,   BNXT_CO_DT_INT64 },
    { at::kByte,   BNXT_CO_DT_UINT8 },
    { at::kHalf,   BNXT_CO_DT_FLOAT16 },
    { at::kFloat,  BNXT_CO_DT_FLOAT32 },
    { at::kDouble, BNXT_CO_DT_FLOAT64 },
};

torch_ucc_status_t torch_bnxt_co_init(
    torch_ucx_comm_t* p2p_comm,
    torch_ucc_coll_comm_t** comm)
{
    torch_bnxt_co_comm_t* bnxt_co_comm;
    char* env;

    bnxt_co_comm = new torch_bnxt_co_comm_t;
    bnxt_co_comm->p2p_comm = p2p_comm;

    env = std::getenv("BNXT_CO_APP_ADDR");
    if (env)
        bnxt_co_comm->config.app_addr = env;
    else
        throw std::runtime_error("ProcessGroupUCC init failed "
                                 "(missing BNXT_CO_APP_ADDR)");

    env = std::getenv("BNXT_CO_APP_PORT");
    if (env)
        bnxt_co_comm->config.app_port = std::atoi(env);
    else
        throw std::runtime_error("ProcessGroupUCC init failed "
                                 "(missing BNXT_CO_APP_PORT)");

    env = std::getenv("BNXT_CO_MASTER_ADDR");
    if (env)
        bnxt_co_comm->config.master_addr = env;
    else
        throw std::runtime_error("ProcessGroupUCC init failed "
                                 "(missing BNXT_CO_MASTER_ADDR)");

    env = std::getenv("BNXT_CO_MASTER_PORT");
    if (env)
        bnxt_co_comm->config.master_port = std::atoi(env);
    else
        throw std::runtime_error("ProcessGroupUCC init failed "
                                 "(missing BNXT_CO_MASTER_PORT)");

    env = std::getenv("BNXT_CO_LOGS");
    if (env)
        bnxt_co_comm->config.logs = std::atoi(env);
    else
        bnxt_co_comm->config.logs = BNXT_CO_LOGS_DEFAULT;

    if (bnxt_co_init(p2p_comm->rank, p2p_comm->size,
                     bnxt_co_comm->config.app_addr,
                     bnxt_co_comm->config.app_port,
                     bnxt_co_comm->config.master_addr,
                     bnxt_co_comm->config.master_port,
                     bnxt_co_comm->config.logs,
                     &bnxt_co_comm->bnxt_co_ctx) != BNXT_CO_OK) {
        return TORCH_UCC_ERROR;
    }

    *comm = (torch_ucc_coll_comm_t*)bnxt_co_comm;

    return TORCH_UCC_OK;
}

torch_ucc_status_t torch_bnxt_co_allgather(
    torch_ucc_coll_comm_t* coll_comm,
    at::Tensor& input_tensor,
    std::vector<at::Tensor>& output_tensors,
    torch_ucc_coll_request_t** request)
{
    torch_bnxt_co_comm_t* bnxt_co_comm = (torch_bnxt_co_comm_t*)coll_comm;
    torch_bnxt_co_request_t* coll_req;

    coll_req = new torch_bnxt_co_request_t;

    std::vector<at::Tensor> input_tensors = { input_tensor };
    torch_ucc_coll_request_init(coll_comm,
                                (torch_ucc_coll_request_t*)coll_req,
                                &input_tensors, &output_tensors);

    coll_req->comm = bnxt_co_comm;
    coll_req->status = TORCH_UCC_OPERATION_INITIALIZED;
    coll_req->coll_type = BNXT_CO_ALLGATHER;
    coll_req->datatype = bnxt_co_type_map.at(input_tensor.scalar_type());
    coll_req->src_buf = (uint8_t *)input_tensor.data_ptr();
    coll_req->src_buf_len = (input_tensor.element_size() *
                             input_tensor.numel());
    coll_req->flat_tensor = newLikeFlat(output_tensors);
    coll_req->dst_buf = (uint8_t *)coll_req->flat_tensor.data_ptr();
    coll_req->dst_buf_len = (coll_req->flat_tensor.element_size() *
                             coll_req->flat_tensor.numel());
    coll_req->elem_size = (input_tensor.element_size() *
                           input_tensor.numel());

    *request = (torch_ucc_coll_request_t*)coll_req;

    return TORCH_UCC_OK;
}

torch_ucc_status_t torch_bnxt_co_alltoall(
    torch_ucc_coll_comm_t* coll_comm,
    at::Tensor& input_tensor,
    at::Tensor& output_tensor,
    torch_ucc_coll_request_t** request)
{
    torch_bnxt_co_comm_t* bnxt_co_comm = (torch_bnxt_co_comm_t*)coll_comm;
    torch_bnxt_co_request_t* coll_req;

    coll_req = new torch_bnxt_co_request_t;

    std::vector<at::Tensor> input_tensors = { input_tensor };
    std::vector<at::Tensor> output_tensors = { output_tensor };
    torch_ucc_coll_request_init(coll_comm,
                                (torch_ucc_coll_request_t*)coll_req,
                                &input_tensors, &output_tensors);

    coll_req->comm = bnxt_co_comm;
    coll_req->status = TORCH_UCC_OPERATION_INITIALIZED;
    coll_req->coll_type = BNXT_CO_ALLTOALL;
    coll_req->datatype = bnxt_co_type_map.at(input_tensor.scalar_type());
    coll_req->src_buf = (uint8_t *)input_tensor.data_ptr();
    coll_req->src_buf_len = (input_tensor.element_size() *
                             input_tensor.numel());
    coll_req->dst_buf = (uint8_t *)output_tensor.data_ptr();
    coll_req->dst_buf_len = (output_tensor.element_size() *
                             output_tensor.numel());
    coll_req->elem_size = ((input_tensor.element_size() *
                            input_tensor.numel()) /
                           bnxt_co_comm->p2p_comm->size);

    *request = (torch_ucc_coll_request_t*)coll_req;

    return TORCH_UCC_OK;
}

torch_ucc_status_t torch_bnxt_co_alltoallv(
    torch_ucc_coll_comm_t* coll_comm,
    at::Tensor& input_tensor,
    uint32_t* send_lengths,
    uint32_t* send_offsets,
    at::Tensor& output_tensor,
    uint32_t* recv_lengths,
    uint32_t* recv_offsets,
    torch_ucc_coll_request_t** request)
{
    return TORCH_UCC_OK;
}

torch_ucc_status_t torch_bnxt_co_allreduce(
    torch_ucc_coll_comm_t* coll_comm,
    at::Tensor& tensor,
    const AllreduceOptions& opts,
    torch_ucc_coll_request_t** request)
{
    return TORCH_UCC_OK;
}

torch_ucc_status_t torch_bnxt_co_barrier(
    torch_ucc_coll_comm_t* coll_comm,
    torch_ucc_coll_request_t** request)
{
    torch_bnxt_co_comm_t* bnxt_co_comm = (torch_bnxt_co_comm_t*)coll_comm;
    torch_bnxt_co_request_t* coll_req;

    coll_req = new torch_bnxt_co_request_t;

    torch_ucc_coll_request_init(coll_comm,
                                (torch_ucc_coll_request_t*)coll_req,
                                nullptr, nullptr);

    coll_req->comm = bnxt_co_comm;
    coll_req->status = TORCH_UCC_OPERATION_INITIALIZED;
    coll_req->coll_type = BNXT_CO_BARRIER;

    *request = (torch_ucc_coll_request_t*)coll_req;

    return TORCH_UCC_OK;
}

torch_ucc_status_t torch_bnxt_co_broadcast(
    torch_ucc_coll_comm_t* coll_comm,
    at::Tensor& tensor,
    int root,
    torch_ucc_coll_request_t** request)
{
    torch_bnxt_co_comm_t* bnxt_co_comm = (torch_bnxt_co_comm_t*)coll_comm;
    torch_bnxt_co_request_t* coll_req;

    coll_req = new torch_bnxt_co_request_t;

    std::vector<at::Tensor> tensors = { tensor };
    torch_ucc_coll_request_init(coll_comm,
                                (torch_ucc_coll_request_t*)coll_req,
                                &tensors, nullptr);

    coll_req->comm = bnxt_co_comm;
    coll_req->status = TORCH_UCC_OPERATION_INITIALIZED;
    coll_req->coll_type = BNXT_CO_BROADCAST;
    coll_req->datatype = bnxt_co_type_map.at(tensor.scalar_type());
    coll_req->src_rank = root;
    coll_req->src_buf = (uint8_t *)tensor.data_ptr();
    coll_req->src_buf_len = (tensor.element_size() * tensor.numel());
    coll_req->dst_buf = coll_req->src_buf;
    coll_req->dst_buf_len = coll_req->src_buf_len;
    coll_req->elem_size = ((tensor.element_size() * tensor.numel()) /
                           bnxt_co_comm->p2p_comm->size);

    *request = (torch_ucc_coll_request_t*)coll_req;

    return TORCH_UCC_OK;
}

static torch_ucc_status_t torch_bnxt_co_coll_cmd(torch_bnxt_co_request_t* req)
{
    switch (req->coll_type) {
    case BNXT_CO_BARRIER:
        return (bnxt_co_barrier_cmd(req->comm->bnxt_co_ctx,
                                    &req->tag) == BNXT_CO_OK)
                   ? TORCH_UCC_OK : TORCH_UCC_ERROR;

    case BNXT_CO_BROADCAST:
        return (bnxt_co_broadcast_cmd(req->comm->bnxt_co_ctx,
                                      req->datatype,
                                      req->src_rank,
                                      req->src_buf,
                                      req->src_buf_len,
                                      &req->tag) == BNXT_CO_OK)
                   ? TORCH_UCC_OK : TORCH_UCC_ERROR;

    case BNXT_CO_ALLREDUCE:
        return TORCH_UCC_ERROR;

    case BNXT_CO_ALLTOALL:
        return (bnxt_co_alltoall_cmd(req->comm->bnxt_co_ctx,
                                     req->datatype,
                                     req->src_buf,
                                     req->src_buf_len,
                                     &req->tag) == BNXT_CO_OK)
                   ? TORCH_UCC_OK : TORCH_UCC_ERROR;

    case BNXT_CO_ALLTOALLV:
        return TORCH_UCC_ERROR;

    case BNXT_CO_ALLGATHER:
        return (bnxt_co_allgather_cmd(req->comm->bnxt_co_ctx,
                                      req->datatype,
                                      req->src_buf,
                                      req->src_buf_len,
                                      &req->tag) == BNXT_CO_OK)
                   ? TORCH_UCC_OK : TORCH_UCC_ERROR;

    default:
        return TORCH_UCC_ERROR;
    }
}

static torch_ucc_status_t torch_bnxt_co_coll_resp(torch_bnxt_co_request_t* req)
{
    switch (req->coll_type) {
    case BNXT_CO_BARRIER:
        return (bnxt_co_barrier_resp(req->comm,
                                     req->resp_msg,
                                     req->resp_msg_len) == BNXT_CO_OK)
                   ? TORCH_UCC_OK : TORCH_UCC_ERROR;

    case BNXT_CO_BROADCAST:
        /* copy the result into the output buffer */
        return (bnxt_co_broadcast_resp(req->comm,
                                       req->datatype,
                                       req->src_rank,
                                       req->resp_msg,
                                       req->resp_msg_len,
                                       req->dst_buf,
                                       req->dst_buf_len) == BNXT_CO_OK)
                   ? TORCH_UCC_OK : TORCH_UCC_ERROR;

    case BNXT_CO_ALLREDUCE:
        return TORCH_UCC_ERROR;

    case BNXT_CO_ALLTOALL:
        /* copy the result into the output buffer */
        return (bnxt_co_alltoall_resp(req->comm,
                                      req->datatype,
                                      req->resp_msg,
                                      req->resp_msg_len,
                                      req->dst_buf,
                                      req->dst_buf_len) == BNXT_CO_OK)
                   ? TORCH_UCC_OK : TORCH_UCC_ERROR;

    case BNXT_CO_ALLTOALLV:
        return TORCH_UCC_ERROR;

    case BNXT_CO_ALLGATHER:
        if (bnxt_co_allgather_resp(req->comm,
                                   req->datatype,
                                   req->resp_msg,
                                   req->resp_msg_len,
                                   req->dst_buf,
                                   req->dst_buf_len) != BNXT_CO_OK) {
            return TORCH_UCC_ERROR;
        }

        /* copy the flat data back to the output tensors */
        {
            std::vector<at::Tensor>& output_vec = req->super.dst;
            for (int i = 0; i < req->comm->p2p_comm->size; i++)
                output_vec[i].copy_(req->flat_tensor[i]);
        }

        return TORCH_UCC_OK;

    default:
        return TORCH_UCC_ERROR;
    }
}

torch_ucc_status_t torch_bnxt_co_progress(torch_ucc_coll_request_t* request)
{
    torch_bnxt_co_request_t* req = (torch_bnxt_co_request_t*)request;
    int rc;

    if (req->status == TORCH_UCC_OPERATION_INITIALIZED) {
#ifdef USE_CUDA
        /*
         * For cuda tensors we need first to make sure that all operations
         * submitted to stream are done, thas is captured by the tensor_ready
         * event. For cpu tensors tensor_ready.query() always return true.
         */
        if (!request->tensor_ready.query()) {
            return TORCH_UCC_OK;
        }
#endif
        torch_bnxt_co_coll_cmd(req);
        req->status = TORCH_UCC_INPROGRESS;
    }

    if (req->status == TORCH_UCC_INPROGRESS) {
        bnxt_co_progress(req->comm->bnxt_co_ctx);
        rc = bnxt_co_probe(req->comm->bnxt_co_ctx,
                           &req->tag,
                           &req->resp_msg,
                           &req->resp_msg_len);
        if (rc == -1) {
            fprintf(stderr,
                    "ERROR: Failed to probe the cmd channel for tag (%lu)\n",
                    req->tag);
            req->status = TORCH_UCC_ERROR;
            return TORCH_UCC_ERROR;
        } else if (rc == 0) {
            /* no response available, still waiting */
            return TORCH_UCC_INPROGRESS;
        } else {
            /* the response was received, process it */
            req->status = torch_bnxt_co_coll_resp(req);
            return req->status;
        }
    }

    return TORCH_UCC_OK;
}

torch_ucc_status_t torch_bnxt_co_test(
    torch_ucc_coll_request_t* request)
{
    torch_bnxt_co_request_t* req = (torch_bnxt_co_request_t*)request;

    if ((req->status == TORCH_UCC_OPERATION_INITIALIZED) ||
        (req->status == TORCH_UCC_INPROGRESS))
        return TORCH_UCC_INPROGRESS;
    else
        return req->status;
}

torch_ucc_status_t torch_bnxt_co_free(
    torch_ucc_coll_request_t* request)
{
    torch_bnxt_co_request_t* req = (torch_bnxt_co_request_t*)request;
    free(req);
    return TORCH_UCC_OK;
}

torch_ucc_status_t torch_bnxt_co_close(torch_ucc_coll_comm_t* comm)
{
    torch_bnxt_co_comm_t* bnxt_co_comm = (torch_bnxt_co_comm_t*)comm;

    bnxt_co_finish(bnxt_co_comm->bnxt_co_ctx, true);
    delete bnxt_co_comm;

    return TORCH_UCC_OK;
}

torch_ucc_coll_ops_t bnxt_co_coll_ops{ torch_bnxt_co_init,
                                       torch_bnxt_co_allgather,
                                       torch_bnxt_co_alltoall,
                                       torch_bnxt_co_alltoallv,
                                       torch_bnxt_co_allreduce,
                                       torch_bnxt_co_barrier,
                                       torch_bnxt_co_broadcast,
                                       torch_bnxt_co_progress,
                                       torch_bnxt_co_test,
                                       torch_bnxt_co_free,
                                       torch_bnxt_co_close };

} // namespace c10d

