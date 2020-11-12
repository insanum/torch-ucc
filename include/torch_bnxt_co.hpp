/*
 * Copyright (c) Broadcom Inc. 2020-2020. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#pragma once

#include <api/xccl.h>
#include <torch_ucc_ops.hpp>
#include <torch_ucc_sendrecv.hpp>

#include <bnxt_co.h>

namespace c10d {

struct torch_bnxt_co_comm_t {
    torch_ucc_coll_comm_t super;
    torch_ucx_comm_t* p2p_comm;
    bnxt_co_ctx_h bnxt_co_ctx;

    struct {
        /*
         * host_proc_rank   = OMPI_COMM_WORLD_RANK (rank)
         * world_size       = OMPI_COMM_WORLD_SIZE (size)
         * bnxt_co_app_addr = BNXT_CO_APP_ADDR
         * bnxt_co_app_port = BNXT_CO_APP_PORT
         * master_addr      = BNXT_CO_MASTER_ADDR
         * master_port      = BNXT_CO_MASTER_PORT
         * logs             = BNXT_CO_LOGS
         */
        char *app_addr;
        uint16_t app_port;
        char *master_addr;
        uint16_t master_port;
#define BNXT_CO_LOGS_DEFAULT -1
        uint32_t logs;
    } config;
};

} // namespace c10d

