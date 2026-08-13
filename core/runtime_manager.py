from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from PySide6.QtCore import QStandardPaths

from core.config_manager import ConfigManager
from core.version import APP_ID


CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class RuntimeManager:
    """Discover and diagnose external Python and Ultralytics runtimes."""

    def __init__(self, config: ConfigManager) -> None:
        self.config = config

    @staticmethod
    def application_mode() -> str:
        return "Frozen / Portable" if getattr(sys, "frozen", False) else "Source"

    @staticmethod
    def managed_runtime_folder() -> Path:
        base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
        if not base:
            base = str(Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_ID)
        base_path = Path(base)
        if base_path.name.casefold() != APP_ID.casefold():
            base_path /= APP_ID
        return base_path / "runtime" / ".venv"

    @classmethod
    def managed_python(cls) -> Path:
        return cls.managed_runtime_folder() / "Scripts" / "python.exe"

    @classmethod
    def managed_yolo(cls) -> Path:
        return cls.managed_runtime_folder() / "Scripts" / "yolo.exe"

    @staticmethod
    def _resolve_program(value: str) -> str | None:
        candidate = value.strip().strip('"')
        if not candidate:
            return None
        path = Path(candidate).expanduser()
        if path.is_file():
            try:
                return str(path.resolve())
            except OSError:
                return str(path)
        return shutil.which(candidate)

    @staticmethod
    def _run(command: list[str], timeout: int = 12) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=CREATE_NO_WINDOW,
            check=False,
        )

    def _python_candidates(self) -> Iterable[tuple[str, list[str], str]]:
        configured = str(self.config.get("python_executable", "")).strip()
        if configured:
            resolved = self._resolve_program(configured)
            if resolved:
                yield resolved, [], "configured"

        managed = self.managed_python()
        if managed.is_file():
            yield str(managed), [], "managed runtime"

        py = shutil.which("py")
        if py:
            yield py, ["-3"], "Windows py launcher"

        python = shutil.which("python")
        if python:
            yield python, [], "PATH"

        python3 = shutil.which("python3")
        if python3:
            yield python3, [], "PATH"

    def discover_python(self, validate: bool = True) -> dict[str, Any]:
        seen: set[tuple[str, tuple[str, ...]]] = set()
        failures: list[str] = []
        for program, prefix_args, source in self._python_candidates():
            key = (program.lower(), tuple(prefix_args))
            if key in seen:
                continue
            seen.add(key)
            result: dict[str, Any] = {
                "available": True,
                "program": program,
                "prefix_args": prefix_args,
                "source": source,
                "version": "",
                "error": "",
                "command": subprocess.list2cmdline([program, *prefix_args]),
            }
            if not validate:
                return result
            try:
                completed = self._run([program, *prefix_args, "--version"], timeout=8)
                output = (completed.stdout or completed.stderr).strip()
                if completed.returncode == 0 and output:
                    result["version"] = output.splitlines()[-1]
                    return result
                failures.append(f"{source}: {output or 'version check failed'}")
            except (OSError, subprocess.SubprocessError) as exc:
                failures.append(f"{source}: {exc}")
        return {
            "available": False,
            "program": "",
            "prefix_args": [],
            "source": "not found",
            "version": "",
            "error": "; ".join(failures),
            "command": "",
        }

    def _yolo_candidates(self) -> Iterable[tuple[str, str]]:
        configured = str(self.config.get("yolo_command", "")).strip()
        if configured:
            resolved = self._resolve_program(configured)
            if resolved:
                yield resolved, "configured"

        managed = self.managed_yolo()
        if managed.is_file():
            yield str(managed), "managed runtime"

        path_yolo = shutil.which("yolo") or shutil.which("yolo.exe")
        if path_yolo:
            yield path_yolo, "PATH"

    def discover_yolo(self, validate: bool = True) -> dict[str, Any]:
        seen: set[str] = set()
        failures: list[str] = []
        for program, source in self._yolo_candidates():
            key = program.lower()
            if key in seen:
                continue
            seen.add(key)
            result: dict[str, Any] = {
                "available": True,
                "program": program,
                "source": source,
                "version": "",
                "error": "",
            }
            if not validate:
                return result
            try:
                completed = self._run([program, "version"], timeout=15)
                output = (completed.stdout or completed.stderr).strip()
                if completed.returncode == 0:
                    result["version"] = output.splitlines()[-1] if output else "Available"
                    return result
                failures.append(f"{source}: {output or 'version check failed'}")
            except (OSError, subprocess.SubprocessError) as exc:
                failures.append(f"{source}: {exc}")
        return {
            "available": False,
            "program": "",
            "source": "not found",
            "version": "",
            "error": "; ".join(failures),
        }

    def resolve_yolo_command(self) -> str | None:
        return str(self.discover_yolo(validate=False).get("program") or "") or None

    def yolo_command_for_preview(self) -> str:
        resolved = self.resolve_yolo_command()
        if resolved:
            return resolved
        configured = str(self.config.get("yolo_command", "")).strip()
        return configured or "yolo"

    def package_diagnostics(self, python: dict[str, Any]) -> dict[str, Any]:
        empty = {
            "ultralytics_available": False,
            "ultralytics_version": "Not found",
            "torch_available": False,
            "torch_version": "Not found",
            "cuda_available": False,
            "torch_cuda_version": "Not available",
            "gpu_count": 0,
            "gpu_name": "Not available",
            "package_error": "Python runtime was not found.",
        }
        if not python.get("available"):
            return empty

        script = (
            "import json\n"
            "r={'ultralytics_available':False,'ultralytics_version':'Not found',"
            "'torch_available':False,'torch_version':'Not found','cuda_available':False,"
            "'torch_cuda_version':'Not available','gpu_count':0,'gpu_name':'Not available','errors':[]}\n"
            "try:\n import ultralytics; r['ultralytics_available']=True; r['ultralytics_version']=str(ultralytics.__version__)\n"
            "except Exception as e: r['errors'].append('ultralytics: '+str(e))\n"
            "try:\n import torch; r['torch_available']=True; r['torch_version']=str(torch.__version__); "
            "r['cuda_available']=bool(torch.cuda.is_available()); r['torch_cuda_version']=str(torch.version.cuda or 'Not available'); "
            "r['gpu_count']=int(torch.cuda.device_count()) if r['cuda_available'] else 0; "
            "r['gpu_name']=str(torch.cuda.get_device_name(0)) if r['gpu_count'] else 'Not available'\n"
            "except Exception as e: r['errors'].append('torch: '+str(e))\n"
            "print('__YOLO_TRAINER_DIAGNOSTICS__'+json.dumps(r))"
        )
        command = [str(python["program"]), *python.get("prefix_args", []), "-c", script]
        try:
            completed = self._run(command, timeout=30)
            marker = "__YOLO_TRAINER_DIAGNOSTICS__"
            payload = next(
                (line.split(marker, 1)[1] for line in reversed(completed.stdout.splitlines()) if marker in line),
                "",
            )
            if not payload:
                empty["package_error"] = (completed.stderr or completed.stdout or "Package diagnostics failed.").strip()
                return empty
            values = json.loads(payload)
            values["package_error"] = "; ".join(values.pop("errors", []))
            return values
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            empty["package_error"] = str(exc)
            return empty

    def run_diagnostics(self) -> dict[str, Any]:
        python = self.discover_python(validate=True)
        yolo = self.discover_yolo(validate=True)
        packages = self.package_diagnostics(python)
        result: dict[str, Any] = {
            "application_mode": self.application_mode(),
            "python_executable": python.get("command") or "Not found",
            "python_version": python.get("version") or "Not found",
            "python_source": python.get("source") or "not found",
            "python_available": bool(python.get("available")),
            "python_error": python.get("error", ""),
            "yolo_command": yolo.get("program") or "Not found",
            "yolo_available": bool(yolo.get("available")),
            "yolo_source": yolo.get("source") or "not found",
            "yolo_version": yolo.get("version") or "Not found",
            "yolo_error": yolo.get("error", ""),
            "managed_runtime_folder": str(self.managed_runtime_folder()),
            **packages,
        }
        if not result["yolo_available"]:
            result["status"] = "Missing"
            result["readiness"] = "YOLO runtime was not found. Configure an existing environment or create a managed environment."
        elif result["torch_available"] and result["cuda_available"]:
            result["status"] = "Ready"
            result["readiness"] = "Ready for GPU"
        elif result["torch_available"]:
            result["status"] = "Partial"
            result["readiness"] = "YOLO is available, but CUDA is not available. Training and inference may run significantly slower on CPU."
        else:
            result["status"] = "Partial"
            result["readiness"] = "YOLO is available, but PyTorch diagnostics are unavailable for the selected Python environment."
        return result

    def save_managed_runtime(self) -> dict[str, str]:
        values = {
            "python_executable": str(self.managed_python()),
            "yolo_command": str(self.managed_yolo()),
        }
        self.config.save(values)
        return values
