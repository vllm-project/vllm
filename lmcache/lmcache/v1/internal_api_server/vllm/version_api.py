# SPDX-License-Identifier: Apache-2.0
# Standard

# Third Party
from fastapi import APIRouter

# First Party
from lmcache import utils

router = APIRouter()


@router.get("/lmc_version")
async def get_lmc_version():
    return utils.VERSION


@router.get("/commit_id")
async def get_commit_id():
    return utils.COMMIT_ID


@router.get("/version")
async def get_version():
    return utils.get_version()
