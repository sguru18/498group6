"""Docker-based sandbox for executing SWE-bench tasks.

Provides the DockerSandbox context manager, which pulls the official SWE-bench
Docker image for a given task, starts a container, checks out the base commit,
and exposes bash/file I/O helpers for the agent loop.

Image naming convention (official SWE-bench images):
    swebench/sweb.eval.x86_64.<instance_id>:latest
where instance_id is the task_id, e.g. 'django__django-11099'.

Example:
    django__django-11099  ->  swebench/sweb.eval.x86_64.django__django-11099:latest
"""

import base64
import io
import os
import tarfile
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import docker as _docker_type


def _instance_to_image(instance_id: str) -> str:
    """Convert a SWE-bench instance ID to the official Docker image name.

    The Docker Hub images use '_1776_' as a separator instead of '__'.

    Args:
        instance_id: Task ID, e.g. 'django__django-11099'.

    Returns:
        Full Docker image tag, e.g.
        'swebench/sweb.eval.x86_64.django_1776_django-11099:latest'.
    """
    image_id = instance_id.replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{image_id}:latest"


def _get_docker_client() -> Any:
    """Return a Docker SDK client, raising a clear error if Docker is not running."""
    try:
        import docker
    except ImportError as e:
        raise ImportError(
            "docker package required: pip install docker"
        ) from e
    try:
        client = docker.from_env()
        client.ping()  # verify daemon is reachable
        return client
    except Exception as exc:
        raise RuntimeError(
            "Docker daemon is not running or not reachable. "
            "Please start Docker Desktop and try again."
        ) from exc


class DockerSandbox:
    """Context manager that sets up a Docker container with a SWE-bench task repo.

    Usage::

        with DockerSandbox(task, timeout=120) as sandbox:
            output = sandbox.run("ls /testbed")
            sandbox.write_file("/testbed/patch.py", "# fix")
            content = sandbox.read_file("/testbed/patch.py")

    The container is stopped and removed on exit.
    """

    WORKDIR: str = "/testbed"

    def __init__(self, task: dict, timeout: int = 60) -> None:
        """Initialise the sandbox with a task dict and optional command timeout.

        Args:
            task: Task dict from swebench_tasks.load_lite_tasks() / get_task().
                  Must contain 'task_id', 'repo', and 'base_commit'.
            timeout: Maximum seconds to allow for any single bash command.
        """
        self._task: dict = task
        self.timeout: int = timeout
        self._container: Any = None  # docker.models.containers.Container

        task_id: str = task["task_id"]
        # Docker container names must match [a-zA-Z0-9][a-zA-Z0-9_.-]
        self._container_name: str = f"cs498_{task_id}".replace("/", "_")
        self._image: str = _instance_to_image(task["task_id"])

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "DockerSandbox":
        """Pull image, start container, and checkout base commit.

        Returns:
            self, so callers can use ``with DockerSandbox(...) as sb:``.
        """
        client = _get_docker_client()

        # Remove any stale container with the same name
        try:
            old = client.containers.get(self._container_name)
            old.stop(timeout=5)
            old.remove(force=True)
        except Exception:
            pass

        print(f"[DockerSandbox] Pulling image {self._image} …")
        try:
            import subprocess as _sp
            result = _sp.run(
                ["docker", "pull", "--platform", "linux/amd64", self._image],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip())
        except Exception as exc:
            raise RuntimeError(
                f"Failed to pull SWE-bench image '{self._image}': {exc}\n"
                "Make sure Docker Hub is reachable and the image exists."
            ) from exc

        print(f"[DockerSandbox] Starting container {self._container_name} …")
        self._container = client.containers.run(
            self._image,
            command="sleep infinity",  # keep alive
            name=self._container_name,
            detach=True,
            working_dir=self.WORKDIR,
            platform="linux/amd64",  # SWE-bench images are x86_64 only
            mem_limit="4g",
        )

        # Checkout the base commit
        base_commit: str = self._task["base_commit"]
        print(f"[DockerSandbox] Checking out {base_commit} …")
        out = self.run(f"git checkout {base_commit} 2>&1")
        print(f"[DockerSandbox] git checkout output: {out[:200]}")
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop and remove the container."""
        if self._container is not None:
            try:
                self._container.stop(timeout=10)
            except Exception:
                pass
            try:
                self._container.remove(force=True)
            except Exception:
                pass
            self._container = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, command: str) -> str:
        """Run a bash command in the container and return stdout + stderr.

        The command is executed inside ``/bin/bash -c``. Output is truncated
        to 4000 characters.

        Args:
            command: Bash command string to execute.

        Returns:
            Combined stdout/stderr as a string (up to 4000 chars).
        """
        if self._container is None:
            raise RuntimeError("Sandbox is not active (use as context manager).")

        result_holder: list = []
        error_holder: list = []

        def _exec() -> None:
            try:
                exit_code, output = self._container.exec_run(
                    ["/bin/bash", "-c", command],
                    workdir=self.WORKDIR,
                    demux=False,
                )
                text = output.decode("utf-8", errors="replace") if output else ""
                result_holder.append((exit_code, text))
            except Exception as exc:
                error_holder.append(exc)

        t = threading.Thread(target=_exec, daemon=True)
        t.start()
        t.join(timeout=self.timeout)

        if error_holder:
            return f"[SANDBOX ERROR]: {error_holder[0]}"
        if not result_holder:
            return f"[TIMEOUT: command exceeded {self.timeout}s]"

        _exit_code, text = result_holder[0]
        # Truncate to 4000 chars
        if len(text) > 4000:
            text = text[:3970] + "\n…[truncated]"
        return text

    def write_file(self, path: str, content: str) -> None:
        """Write content to a file path inside the container.

        Uses ``docker cp``-style tar streaming to avoid shell-quoting issues.

        Args:
            path: Absolute path inside the container, e.g. '/testbed/fix.py'.
            content: File content as a string (UTF-8).
        """
        if self._container is None:
            raise RuntimeError("Sandbox is not active.")

        encoded: bytes = content.encode("utf-8")
        filename: str = os.path.basename(path)
        dirpath: str = os.path.dirname(path) or "/"

        # Ensure parent directory exists
        self.run(f"mkdir -p {dirpath}")

        # Build an in-memory tar archive containing the single file
        tarstream = io.BytesIO()
        with tarfile.open(fileobj=tarstream, mode="w") as tar:
            info = tarfile.TarInfo(name=filename)
            info.size = len(encoded)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(encoded))
        tarstream.seek(0)

        self._container.put_archive(dirpath, tarstream)

    def read_file(self, path: str) -> str:
        """Read a file from the container.

        Args:
            path: Absolute path inside the container.

        Returns:
            File contents as a string, or empty string if not found.
        """
        if self._container is None:
            raise RuntimeError("Sandbox is not active.")
        try:
            bits, _stat = self._container.get_archive(path)
            tarstream = io.BytesIO(b"".join(bits))
            with tarfile.open(fileobj=tarstream) as tar:
                members = tar.getmembers()
                if not members:
                    return ""
                f = tar.extractfile(members[0])
                if f is None:
                    return ""
                return f.read().decode("utf-8", errors="replace")
        except Exception:
            return ""
