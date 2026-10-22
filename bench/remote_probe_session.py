"""Temporarily replace an authorized vLLM service, probe, and restore it.

Run on the GPU host. Captured environment stays in a mode-0600 remote file.
The existing service is restored in finally, even if startup or the probe fails.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def healthy(port):
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status == 200
    except Exception:
        return False


def wait_health(port, process=None, timeout=360):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if healthy(port):
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Server exited with {process.returncode}")
        time.sleep(2)
    raise RuntimeError(f"Server port {port} did not become healthy")


def finish_process(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)


def main(args):
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=False)
    os.chmod(work, 0o700)
    proc = Path(f"/proc/{args.original_pid}")
    command = [s.decode() for s in (proc / "cmdline").read_bytes().split(b"\0") if s]
    environment = dict(s.decode().split("=", 1) for s in (proc / "environ").read_bytes().split(b"\0") if s)
    state = dict(command=command, environment=environment, cwd=os.readlink(proc / "cwd"))
    if "vllm" not in " ".join(command) or "--max-num-seqs" not in command:
        raise RuntimeError("Unexpected original server command")
    state_path = work / "original_state.json"
    state_path.write_text(json.dumps(state))
    os.chmod(state_path, 0o600)
    restore_script = work / "restore_original.py"
    restore_script.write_text("import json,os\n" + f"s=json.load(open({str(state_path)!r}))\n" +
                              "os.chdir(s['cwd'])\nos.execvpe(s['command'][0],s['command'],s['environment'])\n")
    experiment = command.copy()
    experiment[experiment.index("--port") + 1] = str(args.port)
    experiment[experiment.index("--max-num-seqs") + 1] = str(args.concurrency)
    if healthy(args.port):
        raise RuntimeError("Experiment port already in use")
    # Confirm idle before pausing the original service.
    metrics = urllib.request.urlopen("http://127.0.0.1:8001/metrics").read().decode()
    load = [float(l.rsplit(" ", 1)[1]) for l in metrics.splitlines()
            if l.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))]
    if any(load):
        raise RuntimeError("Original service has active work; no changes made")
    (work / "experiment_command.json").write_text(json.dumps(experiment, indent=2))
    own = None
    interrupted = False
    try:
        interrupted = True
        print("Stopping idle original service", flush=True)
        os.kill(args.original_pid, signal.SIGTERM)
        deadline = time.monotonic() + 15
        while proc.exists() and time.monotonic() < deadline:
            time.sleep(1)
        if proc.exists():
            print("Original API did not finish shutdown; terminating it after grace period", flush=True)
            os.kill(args.original_pid, signal.SIGKILL)
            time.sleep(3)
        if healthy(8001):
            raise RuntimeError("Original service automatically restarted")
        time.sleep(3)
        print("Starting independent concurrent server on", args.port, flush=True)
        with (work / "experiment_server.log").open("w") as log:
            own = subprocess.Popen(experiment, env=environment, cwd=state["cwd"],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        wait_health(args.port, own)
        print("Concurrent server healthy; running probe", flush=True)
        subprocess.run([sys.executable, args.probe, "--url", f"http://127.0.0.1:{args.port}",
                        "--out", str(work / "results"), "--repeats", str(args.repeats)], check=True)
    finally:
        print("Stopping experiment server", flush=True)
        finish_process(own)
        if interrupted:
            time.sleep(3)
            if not healthy(8001):
                # Launch the original command in tmux so it survives this supervisor.
                session = "flowmesh-vllm-9b-gptq-131072"
                existing = subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0
                if existing:
                    session += "-restored"
                shell_command = f"exec {sys.executable} {restore_script} >> {work / 'restored_server.log'} 2>&1"
                subprocess.run(["tmux", "new-session", "-d", "-s", session, shell_command], check=True)
            wait_health(8001)
            processes = subprocess.check_output(["ps", "-eo", "args"], text=True).splitlines()
            restored = [line for line in processes if line == " ".join(command)]
            status = dict(healthy=healthy(8001), original_command_matched=bool(restored),
                          original_port=8001, experiment_port_closed=not healthy(args.port))
            (work / "restoration.json").write_text(json.dumps(status, indent=2))
            print("RESTORATION", json.dumps(status), flush=True)
            if not status["original_command_matched"]:
                raise RuntimeError("Health restored but original command not verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-pid", required=True, type=int)
    parser.add_argument("--work", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    main(parser.parse_args())
