import os
import threading
import time
from jupyter_client import KernelManager
from output_cleaning import clean_jupyter_output

class KernelExecutor:
    def __init__(self, timeout: int = 18000):
        self.km = None
        self.kc = None
        self.code_folder = "/code"
        self._lock = threading.Lock()
        self._restarting = False
        self._restart_thread = None
        self.timeout = timeout
    
    def start(self):
        """
        Start the kernel. Uses locking to prevent conflicts with other threads.

        Forces TCP transport for ZeroMQ sockets and starts the kernel process
        with its own mount namespace (via unshare --mount) so that bind mounts
        for /data and /submission only affect the kernel process, not the
        executor (uvicorn + KernelManager + ZeroMQ). This prevents ZeroMQ
        corruption that occurs when the executor itself runs inside an
        isolated mount namespace.
        """
        with self._lock:
            self.km = KernelManager()
            # Force TCP transport — IPC sockets are fragile under namespace isolation
            self.km.transport = "tcp"
            # Start the kernel in its own mount namespace. The kernel_cmd
            # wraps ipykernel in unshare --mount + bind mounts so only the
            # kernel process sees the isolated /data and /submission.
            # The executor process (this one) stays in the parent namespace,
            # keeping ZeroMQ sockets healthy.
            import sys, shlex
            data_dir = os.environ.get("SANDBOX_DATA_DIR", "/data")
            submission_dir = os.environ.get("SANDBOX_SUBMISSION_DIR", "/submission")
            bind_data_src = os.environ.get("BIND_DATA_SRC", "")
            bind_submission_src = os.environ.get("BIND_SUBMISSION_SRC", "")
            kernel_cmd = ["unshare", "--mount", "bash", "-c"]
            inner_cmd = ""
            if bind_data_src:
                inner_cmd += f"mount --bind {shlex.quote(bind_data_src)} {shlex.quote(data_dir)} 2>/dev/null || true; "
            if bind_submission_src:
                inner_cmd += f"mount --bind {shlex.quote(bind_submission_src)} {shlex.quote(submission_dir)} 2>/dev/null || true; "
            inner_cmd += f"exec {sys.executable} -m ipykernel_launcher -f $1"
            kernel_cmd.append(inner_cmd)
            self.km.kernel_cmd = kernel_cmd + ["{connection_file}"]
            self.km.start_kernel()
            self.kc = self.km.blocking_client()
            
            if self.code_folder and os.path.exists(self.code_folder):
                self._setup_code_folder()
    
    def _setup_code_folder(self):
        """This is an helper function to load custom code into the kernel."""
        if not os.path.exists(self.code_folder):
            return
        
        if not self.kc:
            return
        
        # Wait for kernel to be ready before loading files
        try:
            self.kc.wait_for_ready(timeout=10)
        except Exception as e:
            print(f"Kernel not ready for code loading: {e}")
            return
        
        loaded_files = []
        
        for item in os.listdir(self.code_folder):
            if item.endswith('.py') and not item.startswith('_'):
                file_path = os.path.join(self.code_folder, item)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    
                    msg_id = self.kc.execute(file_content, silent=False)
                    
                    self.kc.wait_for_ready(timeout=5)
                    
                    loaded_files.append(item)
                    print(f"Loaded {item} successfully")
                    
                except Exception as e:
                    print(f"Could not load {item}: {e}")
        
        print(f"Code folder setup completed. Loaded files: {loaded_files}")
    
    def _shutdown_kernel_with_timeout(self, km, timeout: float = 15.0):
        """Shut down km gracefully; force-kill with SIGKILL if it doesn't exit in time."""
        done = threading.Event()

        def _try_shutdown():
            try:
                km.shutdown_kernel(now=False)
            except Exception:
                pass
            finally:
                done.set()

        t = threading.Thread(target=_try_shutdown, daemon=True)
        t.start()
        if not done.wait(timeout=timeout):
            print(f"DEBUG: shutdown_kernel timed out after {timeout}s, force-killing kernel")
            try:
                km.kill_kernel()
            except Exception as e:
                print(f"DEBUG: kill_kernel failed: {e}")

    def stop(self):
        # Clear refs under the lock so no new execute() can start, then shut
        # down the kernel *outside* the lock so the lock is never held during
        # a potentially blocking shutdown_kernel() call.
        with self._lock:
            kc, km = self.kc, self.km
            self.kc = None
            self.km = None

        if kc:
            try:
                kc.stop_channels()
            except Exception:
                pass

        if km:
            self._shutdown_kernel_with_timeout(km)
    
    def restart(self):
        """Non-blocking restart of the kernel"""
        with self._lock:
            if self._restarting:
                return {"status": "already_restarting"}
            
            self._restarting = True
            self._restart_thread = threading.Thread(target=self._restart_worker)
            self._restart_thread.daemon = True
            self._restart_thread.start()
            
            return {"status": "restarting"}
    
    def _restart_worker(self):
        """Worker thread for restarting the kernel"""
        try:
            print("Restarting kernel...")
            self.stop()
            time.sleep(0.5)  # Brief pause to ensure cleanup
            self.start()
            print("Kernel restart completed")
        except Exception as e:
            print(f"Error during restart: {e}")
        finally:
            with self._lock:
                self._restarting = False
    
    @property
    def is_restarting(self):
        """Check if kernel is currently restarting"""
        with self._lock:
            return self._restarting
    
    def execute(self, code: str):
        """
        Execute code in the kernel.
        Args:
            code: The code to execute.
        Returns:
            A list of outputs.
        """
        verbose = os.getenv("VERBOSE_JUPYTER_LOG", "0") == "1"
        if verbose:
            print(f"DEBUG: Executing code with timeout={self.timeout}s: {code[:50]}...")
        
        # Wait for any in-progress restart to finish.  Cap the wait so that a
        # stuck _restart_worker (e.g. force-killing a zombie kernel) never
        # blocks this call indefinitely.
        restart_deadline = time.time() + self.timeout + 30
        while self.is_restarting:
            if time.time() > restart_deadline:
                return [{"type": "error", "name": "KernelError",
                         "value": "Kernel restart timed out; container may be unresponsive",
                         "traceback": []}]
            time.sleep(0.1)

        # Check if kernel is available
        if not self.kc:
            return [{"type": "error", "name": "KernelError", "value": "Kernel not available", "traceback": []}]
        
        if not self.km:
            return [{"type": "error", "name": "KernelError", "value": "Kernel not available", "traceback": []}]
        
        # Auto-restart on dead kernel (stale socket, crashed process, etc.)
        # Retry up to 2 times: initial attempt + 1 restart retry.
        for attempt in range(2):
            try:
                self.kc.wait_for_ready(timeout=10)
                break
            except Exception as e:
                if attempt == 0:
                    print(f"Kernel not ready: {e} — auto-restarting kernel and retrying...")
                    self.restart()
                    # Wait for restart to complete (reuse the deadline pattern)
                    restart_deadline = time.time() + self.timeout + 30
                    while self.is_restarting:
                        if time.time() > restart_deadline:
                            return [{"type": "error", "name": "KernelError",
                                     "value": "Kernel restart timed out; container may be unresponsive",
                                     "traceback": []}]
                        time.sleep(0.1)
                    if not self.kc or not self.km:
                        return [{"type": "error", "name": "KernelError", "value": "Kernel not available after restart", "traceback": []}]
                    continue
                print(f"Kernel not ready after restart: {e}")
                return [{"type": "error", "name": "KernelError", "value": "Kernel not ready", "traceback": []}]
       
        with self._lock:
            msg_id = self.kc.execute(code, silent=False)
            outputs = []

            # Track timeout state using mutable object
            timeout_state = {"occurred": False}
            timer = None
            start_time = None
            execution_started = False

            # Set up timeout handler — only called once the execution timer is armed
            def timeout_handler():
                timeout_state["occurred"] = True
                try:
                    self.km.interrupt_kernel()
                    print(f"DEBUG: Execution timed out after {self.timeout} seconds, scheduling kernel restart")
                    def delayed_restart():
                        time.sleep(1)  # Give interrupt time to take effect
                        print("DEBUG: Restarting kernel after timeout")
                        self.restart()
                    restart_thread = threading.Thread(target=delayed_restart)
                    restart_thread.daemon = True
                    restart_thread.start()
                except Exception as e:
                    print(f"DEBUG: Failed to interrupt kernel: {e}")

            # submit_time guards the pre-execution wait so we don't queue forever
            # if the kernel never picks up our request.
            submit_time = time.time()

            try:
                while not timeout_state["occurred"]:
                    # Use the execution clock once the kernel signals busy; otherwise
                    # use the submit clock so queued tasks don't burn execution budget.
                    if execution_started:
                        elapsed = time.time() - start_time
                    else:
                        elapsed = time.time() - submit_time

                    if elapsed >= self.timeout:
                        timeout_state["occurred"] = True
                        break

                    try:
                        msg = self.kc.get_iopub_msg(timeout=1)

                        if msg['parent_header'].get('msg_id') == msg_id:
                            msg_type = msg['msg_type']
                            content = msg['content']
                            if verbose:
                                print(f"DEBUG: Received message type: {msg_type}")

                            if msg_type == 'status':
                                execution_state = content['execution_state']
                                if execution_state == 'busy' and not execution_started:
                                    # Kernel has started executing our code — arm the timer NOW
                                    execution_started = True
                                    start_time = time.time()
                                    timer = threading.Timer(self.timeout, timeout_handler)
                                    timer.start()
                                elif execution_state == 'idle':
                                    print("DEBUG: Execution completed normally")
                                    break
                            elif msg_type == 'execute_result':
                                outputs.append({
                                    'type': 'result',
                                    'data': content['data']
                                })
                            elif msg_type == 'stream':
                                outputs.append({
                                    'type': 'stream',
                                    'name': content['name'],
                                    'text': content['text']
                                })
                            elif msg_type == 'error':
                                outputs.append({
                                    'type': 'error',
                                    'name': content['ename'],
                                    'value': content['evalue'],
                                    'traceback': content['traceback']
                                })

                    except Exception as e:
                        # Ignore frequent poll timeouts to avoid log spam; emit only when verbose
                        if verbose:
                            print(f"DEBUG: Exception in message handling: {e}")
                        continue
            finally:
                if timer is not None:
                    timer.cancel()

            # Check if we hit timeout
            timed_out = timeout_state["occurred"]
            if not timed_out and execution_started:
                timed_out = (time.time() - start_time) >= self.timeout

            if verbose:
                if execution_started:
                    print(f"DEBUG: Elapsed execution time: {time.time() - start_time:.2f}s, Timed out: {timed_out}")
                else:
                    print(f"DEBUG: Kernel never became busy. Timed out: {timed_out}")
                print(f"DEBUG: Number of outputs collected: {len(outputs)}")

            # If we hit timeout, add timeout error
            if timed_out:
                outputs.append({
                    'type': 'error',
                    'name': 'TimeoutError',
                    'value': f'Execution timed out after {self.timeout} seconds',
                    'traceback': []
                })
        
        return clean_jupyter_output(outputs) 
