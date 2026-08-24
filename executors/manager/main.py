import os
import json
import asyncio
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

class CodeRequest(BaseModel):
    code: str

class CodeResponse(BaseModel):
    outputs: list[Any]

class AllocationResponse(BaseModel):
    container_id: int

class ContainerConfig:
    def __init__(self):
        config_path = os.getenv("CONTAINER_CONFIG_PATH", "container_config.json")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def get_url(self, container_id: int) -> str:
        container_key = str(container_id)
        if container_key not in self.config:
            raise HTTPException(status_code=404, detail=f"Container {container_id} not found")
        
        return self.config[container_key]['url']
    
    def get_type(self, container_id: int) -> str:
        container_key = str(container_id)
        if container_key not in self.config:
            raise HTTPException(status_code=404, detail=f"Container {container_id} not found")
        
        return self.config[container_key]['type']
    
    def get_containers_by_type(self, container_type: Optional[str] = None) -> list[int]:
        """Get list of container IDs of specific type"""
        containers = []
        for container_id_str in self.config.keys():
            container_id = int(container_id_str)
            if container_type is None or self.get_type(container_id) == container_type:
                containers.append(container_id)
        return containers
    
    @property
    def urls(self):
        """Backward compatibility property"""
        return self.config

config = ContainerConfig()

# Type-based container queues
available_containers_by_type = {}
allocated_containers = set()

# Initialize queues per container type
for container_id_str in config.config.keys():
    container_id = int(container_id_str)
    container_type = config.get_type(container_id)
    
    if container_type not in available_containers_by_type:
        available_containers_by_type[container_type] = asyncio.Queue()
    
    available_containers_by_type[container_type].put_nowait(container_id)

print("🔧 Initialized container queues:")
for container_type, queue in available_containers_by_type.items():
    print(f"   {container_type}: {queue.qsize()} containers")

http_limits = httpx.Limits(max_connections=1000, max_keepalive_connections=1000)
http_client = httpx.AsyncClient(timeout=18000.0, limits=http_limits)
ready_check_client = httpx.AsyncClient(timeout=15.0, limits=http_limits)

async def make_request(client: httpx.AsyncClient, method: str, container_id: int, endpoint: str, **kwargs):
    container_url = config.get_url(container_id)
    url = f"{container_url}/{endpoint}"
    
    try:
        response = await client.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail=f"Container {container_id} is not available")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Container error: {e.response.text}")

app = FastAPI(title="Executor Manager")

async def _wait_for_container(request: Request, queue: asyncio.Queue, allocation_timeout: float) -> Optional[int]:
    """
    Wait for a container from *queue*, but abort immediately if the HTTP client
    disconnects.  Returns the container_id, or None if the client disconnected.
    Raises asyncio.TimeoutError if allocation_timeout is exceeded.

    Uses a single persistent get_task so that no orphaned queue.get() futures
    are created across poll iterations.  The old asyncio.shield(queue.get())
    pattern recreated a new coroutine every second, causing containers
    dequeued by stale futures to disappear from both available and allocated
    sets (available=0, allocated=0 with containers unaccounted for).
    """
    start = asyncio.get_event_loop().time()
    # One persistent get task — reused across all poll iterations.
    get_task = asyncio.ensure_future(queue.get())
    container_id = None
    try:
        while True:
            if await request.is_disconnected():
                return None
            elapsed = asyncio.get_event_loop().time() - start
            if elapsed >= allocation_timeout:
                raise asyncio.TimeoutError()
            # Wait up to 1 s for the get to complete, then re-check disconnect/timeout.
            # asyncio.wait does NOT cancel get_task on timeout — it stays alive for
            # the next iteration, unlike wait_for which would cancel it.
            done, _ = await asyncio.wait([get_task], timeout=1.0)
            if get_task in done:
                container_id = get_task.result()
                return container_id
            # else: poll timed out, loop to re-check disconnect + allocation_timeout
    finally:
        if not get_task.done():
            get_task.cancel()
            try:
                await get_task
            except (asyncio.CancelledError, Exception):
                pass
        elif container_id is None and not get_task.cancelled():
            # Task finished with a result but we're exiting via an exception path
            # before returning it — put the container back to avoid leaking it.
            try:
                leaked_id = get_task.result()
                queue.put_nowait(leaked_id)
            except Exception:
                pass

async def _recover_stuck_containers(container_type: str) -> int:
    """Health-check every allocated container of *container_type*.

    Containers that fail the health check are assumed to be zombie allocations
    (the client process died without calling /deallocate).  They are restarted
    and returned to the available queue so waiting callers can proceed.

    Returns the number of containers successfully recovered.
    """
    recovered = 0
    for container_id in list(allocated_containers):
        if config.get_type(container_id) != container_type:
            continue
        # Guard against a concurrent recovery already handling this container.
        if container_id not in allocated_containers:
            continue
        try:
            await ready_check_client.get(f"{config.get_url(container_id)}/health")
            # Healthy → leave it alone; its owner is still alive.
        except Exception:
            # Unresponsive → treat as a zombie allocation: restart and re-queue.
            print(f"🔄 Recovering stuck container {container_id} (failed health check)")
            try:
                await ready_check_client.post(f"{config.get_url(container_id)}/restart")
            except Exception as restart_err:
                print(f"⚠️  Could not restart container {container_id}: {restart_err}")
            allocated_containers.discard(container_id)
            available_containers_by_type[container_type].put_nowait(container_id)
            recovered += 1
    return recovered


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/allocate", response_model=AllocationResponse)
async def allocate_container(request: Request, container_type: Optional[str] = None):
    allocation_timeout = float(os.getenv("ALLOCATION_TIMEOUT", "3600"))

    # Build ordered list of queues to try
    if container_type is None:
        queues = list(available_containers_by_type.items())
    else:
        if container_type not in available_containers_by_type:
            raise HTTPException(status_code=400, detail=f"Unknown container type: {container_type}")
        queues = [(container_type, available_containers_by_type[container_type])]

    # Fast path: grab immediately if something is available
    for type_name, queue in queues:
        if queue.qsize() > 0:
            try:
                container_id = await asyncio.wait_for(queue.get(), timeout=0.1)
                allocated_containers.add(container_id)
                return AllocationResponse(container_id=container_id)
            except asyncio.TimeoutError:
                continue

    # Slow path: wait on the first matching queue, aborting if client disconnects
    type_name, queue = queues[0]
    try:
        container_id = await _wait_for_container(request, queue, allocation_timeout)
    except asyncio.TimeoutError:
        # Recover: health-check all allocated containers of this type and restart
        # any that are unresponsive (zombie allocations from crashed processes).
        # Recovered containers are re-queued so this caller — and other waiters —
        # can pick them up.
        recovered = await _recover_stuck_containers(type_name)
        if recovered:
            print(f"♻️  Recovered {recovered} stuck container(s) of type '{type_name}' — retrying allocation")
            try:
                container_id = await asyncio.wait_for(queue.get(), timeout=2.0)
                allocated_containers.add(container_id)
                return AllocationResponse(container_id=container_id)
            except asyncio.TimeoutError:
                pass  # Another waiter grabbed it first; fall through to 503
        raise HTTPException(
            status_code=503,
            detail=f"No container of type '{type_name}' became available within {allocation_timeout}s — "
                   f"all {len(config.get_containers_by_type(type_name))} containers are busy and healthy",
        )
    if container_id is None:
        raise HTTPException(status_code=499, detail="Client disconnected during allocation")
    allocated_containers.add(container_id)
    return AllocationResponse(container_id=container_id)

@app.post("/deallocate/{container_id}")
async def deallocate_container(container_id: int):
    if container_id not in allocated_containers:
        raise HTTPException(status_code=400, detail=f"Container {container_id} is not allocated")
    
    await make_request(http_client, "POST", container_id, "restart")
    allocated_containers.discard(container_id)
    
    # Put container back in the appropriate type queue
    container_type = config.get_type(container_id)
    available_containers_by_type[container_type].put_nowait(container_id)
    
    return {"status": "deallocated", "container_id": container_id}

@app.post("/restart")
async def restart_all_allocated_containers():
    """Restart and deallocate all currently allocated containers"""
    if not allocated_containers:
        return {"status": "no_containers_to_restart", "restarted_containers": []}
    
    restarted_containers = []
    failed_containers = []
    
    # Create a copy of allocated_containers to iterate over
    containers_to_restart = list(allocated_containers)
    
    for container_id in containers_to_restart:
        try:
            # Restart the container
            await make_request(http_client, "POST", container_id, "restart")
            
            # Remove from allocated set
            allocated_containers.discard(container_id)
            
            # Put container back in the appropriate type queue
            container_type = config.get_type(container_id)
            available_containers_by_type[container_type].put_nowait(container_id)
            
            restarted_containers.append(container_id)
        except Exception as e:
            # If restart fails, still remove from allocated set to avoid stuck containers
            allocated_containers.discard(container_id)
            failed_containers.append({"container_id": container_id, "error": str(e)})
    
    return {
        "status": "restart_completed",
        "restarted_containers": restarted_containers,
        "failed_containers": failed_containers,
        "total_restarted": len(restarted_containers),
        "total_failed": len(failed_containers)
    }

@app.post("/session/{container_id}/execute")
async def execute_code(container_id: int, request: CodeRequest):
    if container_id not in allocated_containers:
        raise HTTPException(status_code=400, detail=f"Container {container_id} is not allocated or invalid")
    return await make_request(http_client, "POST", container_id, "execute", json={"code": request.code})

@app.get("/session/{container_id}/health")
async def container_health(container_id: int):
    return await make_request(ready_check_client, "GET", container_id, "health")

@app.get("/session/{container_id}/ready")
async def container_ready(container_id: int):
    return await make_request(ready_check_client, "GET", container_id, "ready")

@app.post("/session/{container_id}/restart")
async def restart_container(container_id: int):
    return await make_request(http_client, "POST", container_id, "restart")

@app.get("/session/{container_id}/restart-status")
async def restart_status(container_id: int):
    return await make_request(ready_check_client, "GET", container_id, "restart-status")

@app.api_route("/session/{container_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_to_container(container_id: int, path: str, request: Request):
    kwargs = {}
    
    if request.query_params:
        kwargs['params'] = request.query_params
    
    if request.method in ["POST", "PUT", "PATCH"]:
        body = await request.body()
        if body:
            kwargs['content'] = body
    
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ["host", "content-length"]}
    if headers:
        kwargs['headers'] = headers
    
    return await make_request(http_client, request.method, container_id, path, **kwargs)

@app.get("/status")
async def get_status():
    available_by_type = {}
    total_available = 0
    for container_type, queue in available_containers_by_type.items():
        count = queue.qsize()
        available_by_type[container_type] = count
        total_available += count
    
    return {
        "available_containers": total_available,
        "available_by_type": available_by_type,
        "allocated_containers": len(allocated_containers),
        "total_containers": len(config.config),
        "allocated_container_ids": list(allocated_containers)
    }

@app.post("/reset")
async def reset_all():
    """
    Hard-reset manager state without restarting the process.

    Creates fresh asyncio queues for each container type (abandoning any
    stale queue.get() waiters that survived a dead client process), clears
    the allocated set, restarts every executor container, then re-populates
    the queues.  Stale waiters that were blocked on the old queues will
    eventually hit the ALLOCATION_TIMEOUT and return 503 — they will never
    receive a container from the new queues.
    """
    global available_containers_by_type, allocated_containers

    # Replace queues so stale asyncio waiters are abandoned on old queue objects
    new_queues: dict[str, asyncio.Queue] = {t: asyncio.Queue() for t in available_containers_by_type}
    available_containers_by_type = new_queues
    allocated_containers = set()

    restarted, failed = [], []
    for container_id in list(config.config.keys()):
        container_id = int(container_id)
        try:
            await make_request(ready_check_client, "POST", container_id, "restart")
            restarted.append(container_id)
        except Exception as e:
            failed.append({"container_id": container_id, "error": str(e)})
        container_type = config.get_type(container_id)
        available_containers_by_type[container_type].put_nowait(container_id)

    return {
        "status": "reset_completed",
        "restarted": restarted,
        "failed": failed,
        "available_containers": sum(q.qsize() for q in available_containers_by_type.values()),
    }

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()
    await ready_check_client.aclose()

async def _health_watchdog():
    """Background task: periodically health-check all allocated containers.

    If an allocated container fails the (deep) health check, restart it
    and return it to the available queue. This catches containers that
    died mid-run (kernel crash, OOM, stuck threads) — the existing
    _recover_stuck_containers only fires on allocation timeout, which
    may never happen if other containers are available.
    """
    interval = float(os.getenv("HEALTH_CHECK_INTERVAL", "120"))
    while True:
        await asyncio.sleep(interval)
        for container_id in list(allocated_containers):
            try:
                resp = await ready_check_client.get(
                    f"{config.get_url(container_id)}/health"
                )
                data = resp.json()
                if data.get("status") == "ok":
                    continue
                # Unhealthy but HTTP server alive — try restart
                reason = data.get("reason", "unknown")
                print(f"🔄 Watchdog: container {container_id} unhealthy ({reason}), restarting...")
                try:
                    await ready_check_client.post(
                        f"{config.get_url(container_id)}/restart"
                    )
                except Exception as restart_err:
                    print(f"⚠️  Watchdog: could not restart container {container_id}: {restart_err}")
            except Exception:
                # HTTP server itself is dead — can't restart via API.
                # Return to pool so next allocation attempt triggers recovery.
                print(f"🔄 Watchdog: container {container_id} unresponsive (HTTP dead), returning to pool")
                allocated_containers.discard(container_id)
                container_type = config.get_type(container_id)
                available_containers_by_type[container_type].put_nowait(container_id)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(_health_watchdog())

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port) 