"""Command line interface for Sideris Desktop.

    sideris                          open the desktop window
    sideris setup --model qwen2.5-1.5b --yes
    sideris chat --level high
    sideris serve
    sideris service install|uninstall|run|status
    sideris train start --dataset data.jsonl --hours 2
    sideris train status
    sideris status

Every optional action still respects the permission switches.  Passing
``--yes`` to setup is an explicit grant of the model-download permission
for that run; it does not turn on the background service or training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __app_name__, __release_name__, __version__
from .config import (
    SettingsStore,
    default_data_dir,
    open_path,
    paths,
    platform_label,
    total_ram_gb,
)
from .core import LEVELS, Assistant, backend_report, level_info
from .downloads import human_bytes
from .installer import Setup, activate_runtime, quick_status
from .models import catalogue, get_spec, recommend_key
from .server import probe_server, run_server
from .service import (
    ServiceError,
    install_login_item,
    install_service,
    login_item_status,
    remove_login_item,
    run_service,
    service_status,
    uninstall_service,
)
from .training import (
    TrainingError,
    TrainingManager,
    TrainingNotReadyError,
    TrainingPermissionError,
    example_dataset,
    latest_checkpoint,
    prepare_dataset,
)

LEVEL_CHOICES = list(LEVELS)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _print(message: str = "") -> None:
    """Print to stdout when one exists.

    The packaged Windows/macOS builds are windowed, so ``sys.stdout`` can
    be ``None``.  CLI output is a courtesy there; it must never crash a
    background service or a helper process.
    """
    stream = sys.stdout
    if stream is None:
        return
    try:
        print(message, file=stream, flush=True)
    except Exception:
        pass


def _store(root=None) -> SettingsStore:
    return SettingsStore(root)


def _permission_hint(name: str) -> str:
    return (
        f"Permission '{name}' is off. Turn it on in Settings, or run "
        f"'sideris setup --model qwen2.5-1.5b --yes"
        + (" --background" if name == "background_service" else "")
        + (" --training" if name == "training" else "")
        + (" --start-at-login" if name == "start_at_login" else "")
        + "'."
    )


def _require_model(store: SettingsStore):
    from .models import ModelManager

    settings = store.load()
    manager = ModelManager(store.root)
    selected = manager.select(settings.model)
    if not selected:
        _print(
            "No local model is installed yet.\n"
            "Run:  sideris setup --model qwen2.5-1.5b --yes\n"
            "The model is downloaded once and then runs offline."
        )
        return None, None
    report = backend_report()
    if not report["available"]:
        _print(
            "The model is on disk but no inference runtime is installed.\n"
            "Run:  sideris setup --model <model> --yes"
        )
        return None, None
    assistant = Assistant(settings, manager, root=store.root)
    return assistant, selected


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------
def cmd_setup(args) -> int:
    store = _store()
    settings = store.load()
    model = args.model or settings.model
    try:
        get_spec(model)
    except KeyError as exc:
        _print(f"Unknown model: {exc}")
        _print("Available:")
        for spec in catalogue():
            _print(f"  {spec.key:<16} {spec.size_label:>8}  {spec.ram_label:<9} {spec.blurb}")
        return 2

    consent = bool(args.yes)
    if not consent:
        interactive = sys.stdin.isatty()
        if not interactive:
            _print("Refusing to download without consent. Re-run with --yes to allow it.")
            return 2
        _print(
            f"{__app_name__} can download {get_spec(model).display_name} "
            f"({get_spec(model).size_label}) from Hugging Face to "
            f"{store.paths.models}."
        )
        answer = input("Allow the download and continue setup? [y/N] ").strip().lower()
        consent = answer in {"y", "yes"}
        if not consent:
            _print("Nothing was downloaded or installed.")
            return 1

    settings.model = model
    # Consent is additive: running setup without --yes must never revoke a
    # download permission the user already granted elsewhere.
    if consent:
        settings.permissions.model_download = True
    if args.background:
        settings.permissions.background_service = True
    if args.training:
        settings.permissions.training = True
    if args.start_at_login:
        settings.permissions.start_at_login = True
    store.save(settings)

    def log(line: str) -> None:
        _print(line)

    def progress(percent: float, message: str) -> None:
        if sys.stdout.isatty():
            sys.stdout.write(f"\r[{percent:5.1f}%] {message[:70]:<70}")
            sys.stdout.flush()
            if percent >= 100:
                sys.stdout.write("\n")

    _print(f"{__release_name__} {__version__}  -  setup")
    _print(f"Data directory: {store.root}")
    _print(f"Model: {get_spec(model).display_name} ({get_spec(model).size_label})")
    result = Setup(
        store.root, settings_store=store, on_log=log, on_progress=progress
    ).run(model=model, settings=settings)
    _print("")
    for item in result["results"]:
        _print(f"  {item['key']:<12} {item['status']:<12} {item['detail']}")
    _print("")
    if result["ok"]:
        _print(
            "Setup complete. Run 'sideris' to open the desktop app, or "
            "'sideris chat' to talk in the terminal."
        )
        return 0
    _print("Setup finished with problems. Fix the step above and run setup again;")
    _print("completed steps and partial downloads are kept and will resume.")
    return 1


# --------------------------------------------------------------------------
# chat
# --------------------------------------------------------------------------
def cmd_chat(args) -> int:
    store = _store()
    settings = store.load()
    level = args.level or settings.thinking_level
    if level not in LEVELS:
        _print(f"Unknown level {level!r}; choose from {', '.join(LEVELS)}.")
        return 2
    if args.model:
        settings.model = args.model
        store.save(settings)
    assistant, selected = _require_model(store)
    if not assistant:
        return 2
    if args.message:
        messages = [{"role": "user", "content": " ".join(args.message)}]
        reply = assistant.chat(
            messages,
            level=level,
            on_token=lambda piece: (sys.stdout.write(piece), sys.stdout.flush()),
        )
        if reply.error:
            print(f"\nError: {reply.error}", file=sys.stderr)
            return 1
        print()
        return 0

    _print(f"{__release_name__} {__version__} - local chat")
    _print(f"Model: {selected[0]}   Level: {level}   Backend: {backend_report()['preferred']}")
    _print("Commands: /new  /level low|medium|high|max  /exit")
    _print("")
    history: list[dict] = []
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            _print("")
            return 0
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            return 0
        if prompt == "/new":
            history.clear()
            _print("(new conversation)")
            continue
        if prompt.startswith("/level"):
            parts = prompt.split()
            if len(parts) == 2 and parts[1] in LEVELS:
                level = parts[1]
                _print(f"(level: {level} - {level_info(level)['label']})")
            else:
                _print("Usage: /level low|medium|high|max")
            continue
        history.append({"role": "user", "content": prompt})

        def on_status(text: str) -> None:
            if level in {"high", "max"} and sys.stderr.isatty():
                sys.stderr.write(f"\r… {text:<50}")
                sys.stderr.flush()

        print("sideris> ", end="", flush=True)
        reply = assistant.chat(
            history,
            level=level,
            on_token=lambda piece: (sys.stdout.write(piece), sys.stdout.flush()),
            on_status=on_status,
        )
        if level in {"high", "max"} and sys.stderr.isatty():
            sys.stderr.write("\r" + " " * 60 + "\r")
            sys.stderr.flush()
        if reply.error:
            _print(f"\nError: {reply.error}")
            history.pop()
            continue
        print("\n")
        history.append({"role": "assistant", "content": reply.text})


# --------------------------------------------------------------------------
# serve / service
# --------------------------------------------------------------------------
def cmd_serve(args) -> int:
    store = _store()
    settings = store.load()
    port = args.port or settings.port
    runtime_info = probe_server(port)
    if runtime_info:
        _print(f"A {__app_name__} service is already running on http://127.0.0.1:{port}")
        return 0
    try:
        return run_server(
            store.root,
            port=port,
            announce=not args.quiet,
        )
    except SystemExit as exc:
        _print(str(exc))
        return 1


def cmd_service(args) -> int:
    store = _store()
    action = args.action
    if action == "run":
        return run_service(port=args.port, quiet=args.quiet)
    if action == "install":
        if not store.load().permissions.background_service:
            _print(_permission_hint("background_service"))
            return 3
        try:
            state = install_service(permission=True)
        except (PermissionError, ServiceError, OSError) as exc:
            _print(f"Could not install the background service: {exc}")
            return 1
        _print(f"Background service installed ({state.get('mode')}).")
        _print("It will start when you sign in and serve http://127.0.0.1:8765")
        return 0
    if action == "uninstall":
        result = uninstall_service(permission=True)
        _print("Background service removed.")
        if result.get("errors"):
            for error in result["errors"]:
                _print(f"  warning: {error}")
        return 0
    if action == "status":
        info = service_status()
        _print(f"Installed: {info['installed']} ({info.get('mode') or 'not installed'})")
        _print(f"Running:   {info['running']}")
        _print(f"Heartbeat: {info['last_heartbeat'] or 'never'}")
        _print(f"At login:  {info['start_at_login']}")
        return 0
    _print(f"Unknown action {action}")
    return 2


def cmd_login(args) -> int:
    store = _store()
    if args.action == "enable":
        if not store.load().permissions.start_at_login:
            _print(_permission_hint("start_at_login"))
            return 3
        try:
            install_login_item(permission=True)
            _print("Sideris will open at login.")
            return 0
        except (PermissionError, ServiceError, OSError) as exc:
            _print(f"Could not enable start at login: {exc}")
            return 1
    if args.action == "disable":
        remove_login_item()
        _print("Start at login disabled.")
        return 0
    _print(f"Start at login: {login_item_status()}")
    return 0


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def cmd_train(args) -> int:
    store = _store()
    manager = TrainingManager(store.root, store)
    action = args.action
    if action == "start":
        if not store.load().permissions.training:
            _print(_permission_hint("training"))
            return 3
        dataset = args.dataset
        if not dataset:
            _print("--dataset is required (a .jsonl file).")
            return 2
        try:
            prepare_dataset(dataset)
        except TrainingError as exc:
            _print(str(exc))
            return 2
        if not manager.runtime_ready():
            _print(
                "The training runtime is not installed. Run:\n"
                "  sideris setup --training --yes\n"
                "This installs torch, transformers, peft, datasets and accelerate "
                "into a private virtual environment (several hundred MB)."
            )
            return 3
        try:
            state = manager.start(
                dataset,
                hours=args.hours,
                epochs=args.epochs,
                permission=True,
            )
        except TrainingPermissionError as exc:
            _print(str(exc))
            return 3
        except TrainingNotReadyError as exc:
            _print(str(exc))
            return 3
        except TrainingError as exc:
            _print(f"Could not start training: {exc}")
            return 1
        _print(f"Training started (run {state.run_id}, pid {state.pid}).")
        if state.resumed_from:
            _print(f"Resumed from: {state.resumed_from}")
        _print(f"Log: {state.log_file}")
        _print("Check progress with: sideris train status")
        return 0
    if action == "stop":
        # Stopping is always allowed; revoking a permission must not trap a
        # run that is already using the CPU.
        try:
            state = manager.stop()
        except TrainingError as exc:
            _print(str(exc))
            return 1
        _print(f"Stop requested. The worker will save a checkpoint and exit ({state.status}).")
        return 0
    if action == "status":
        info = manager.status()
        state = info["state"]
        checkpoint = info.get("latest_checkpoint")
        if args.json:
            _print(json.dumps(info, indent=2))
            return 0
        _print(f"Status:     {state['status']}")
        if state.get("run_id"):
            _print(f"Run:        {state['run_id']}  (pid {state.get('pid')})")
        _print(f"Dataset:    {state.get('dataset') or '-'}")
        _print(f"Base model: {state.get('base_model')}")
        if state.get("max_steps"):
            _print(f"Progress:   step {state.get('step')}/{state.get('max_steps')}")
        if state.get("loss") is not None:
            _print(f"Loss:       {state.get('loss'):.4f}")
        _print(f"Checkpoint: {checkpoint or '-'}")
        if state.get("resumed_from"):
            _print(f"Resumed:    {state['resumed_from']}")
        if state.get("error"):
            _print(f"Note:       {state['error']}")
        _print(f"Runtime:    {'ready' if info.get('runtime_ready') else 'not installed'}")
        return 0
    _print(f"Unknown action {action}")
    return 2


# --------------------------------------------------------------------------
# status / models / gui
# --------------------------------------------------------------------------
def cmd_status(args) -> int:
    store = _store()
    settings = store.load()
    quick = quick_status()
    service = service_status()
    training = TrainingManager(store.root, store).status()
    payload = {
        "app": __app_name__,
        "release": __release_name__,
        "version": __version__,
        "platform": platform_label(),
        "data_dir": str(store.root),
        "ram_gb": total_ram_gb(),
        "model": {
            "configured": settings.model,
            "installed": quick["installed_models"],
            "ready": quick["ready"],
            "suggested": recommend_key(),
        },
        "backend": backend_report(),
        "permissions": settings.permissions.to_dict(),
        "setup": {"ready": quick["ready"], "steps": quick["steps"]},
        "service": service,
        "training": {
            "status": training["state"]["status"],
            "checkpoint": training.get("latest_checkpoint"),
            "runtime_ready": training.get("runtime_ready"),
        },
        "server": probe_server(settings.port),
    }
    if args.json:
        _print(json.dumps(payload, indent=2, default=str))
        return 0
    _print(f"{__release_name__} {__version__}")
    _print(f"Platform:    {platform_label()}  ({total_ram_gb()} GB RAM)")
    _print(f"Data:        {store.root}")
    _print(f"Model:       {settings.model} ({'installed' if quick['ready'] else 'not installed'})")
    installed = ", ".join(m["name"] for m in quick["installed_models"]) or "none"
    _print(f"Installed:   {installed}")
    backend = backend_report()
    _print(f"Backend:     {backend['preferred'] or 'not installed'}")
    _print(f"Permissions: {settings.permissions.summary()}")
    _print(f"Service:     {'running' if service['running'] else ('installed' if service['installed'] else 'not installed')}")
    _print(f"Training:    {training['state']['status']}  checkpoint: {training.get('latest_checkpoint') or '-'}")
    _print(f"Setup:       {'ready' if quick['ready'] else 'incomplete - run setup'}")
    if not quick["ready"]:
        _print("")
        _print("Next step:   sideris setup --model qwen2.5-1.5b --yes")
    return 0


def cmd_models(args) -> int:
    store = _store()
    settings = store.load()
    from .models import ModelManager

    manager = ModelManager(store.root)
    if args.action == "list":
        _print(f"Models in {manager.paths.models}:")
        for spec in catalogue():
            mark = "*" if spec.key == settings.model else " "
            state = "installed" if manager.is_complete(spec.key) else "not installed"
            _print(
                f" {mark} {spec.key:<16} {spec.size_label:>8}  {spec.ram_label:<10} "
                f"{state:<14} {spec.recommended_for}"
            )
        _print("")
        _print("Download with: sideris models download <key>")
        return 0
    if args.action == "download":
        if not settings.permissions.model_download:
            _print(_permission_hint("model_download"))
            return 3
        try:
            spec = get_spec(args.key or settings.model)
        except KeyError as exc:
            _print(str(exc))
            return 2
        _print(f"Downloading {spec.display_name} ({spec.size_label}); resumable.")
        manager.allow_download = True
        try:
            result = manager.download(
                spec.key,
                permission=True,
                progress=lambda p: _progress_line(p),
            )
        except Exception as exc:
            _print(f"\nDownload failed: {exc}")
            return 1
        _print(f"\nInstalled at {result.path} ({human_bytes(result.bytes_written)}).")
        return 0
    if args.action == "remove":
        key = args.key or settings.model
        if manager.remove(key):
            _print(f"Removed {key}.")
            return 0
        _print(f"{key} is not installed.")
        return 1
    _print(f"Unknown action {args.action}")
    return 2


def _progress_line(progress) -> None:
    if not sys.stdout.isatty():
        return
    percent = progress.percent
    if percent is None:
        sys.stdout.write(f"\r{human_bytes(progress.downloaded)}")
    else:
        bar_width = 24
        filled = int(bar_width * percent / 100)
        bar = "#" * filled + "-" * (bar_width - filled)
        sys.stdout.write(
            f"\r[{bar}] {percent:5.1f}%  {human_bytes(progress.downloaded)}"
            f" / {human_bytes(progress.total)}"
        )
    sys.stdout.flush()


def cmd_open(args) -> int:
    store = _store()
    target = {
        "data": store.root,
        "models": store.paths.models,
        "chats": store.paths.chats,
        "workspace": store.paths.workspace,
        "logs": store.paths.logs,
        "checkpoints": store.paths.checkpoints,
    }.get(args.what, store.root)
    target.mkdir(parents=True, exist_ok=True)
    open_path(target)
    _print(str(target))
    return 0


def cmd_example_dataset(args) -> int:
    store = _store()
    store.paths.datasets.mkdir(parents=True, exist_ok=True)
    target = Path(args.path) if args.path else store.paths.datasets / "example.jsonl"
    target.write_text(example_dataset(), encoding="utf-8")
    _print(f"Wrote {target}")
    _print(
        "Format: one JSON object per line, either {\"text\": ...} or "
        "{\"prompt\": ..., \"response\": ...}."
    )
    return 0


def cmd_gui(args) -> int:
    from .app import run_gui

    return run_gui()


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sideris",
        description=(
            f"{__release_name__} {__version__} - a private AI assistant that "
            "runs on your own computer."
        ),
    )
    parser.add_argument("--version", action="version", version=f"{__app_name__} {__version__}")
    sub = parser.add_subparsers(dest="command")

    setup = sub.add_parser("setup", help="install the model and optional runtimes")
    setup.add_argument("--model", help="catalogue key, e.g. qwen2.5-1.5b")
    setup.add_argument("--yes", action="store_true", help="allow the model download without prompting")
    setup.add_argument("--background", action="store_true", help="grant permission and install the background service")
    setup.add_argument("--training", action="store_true", help="grant permission and install the fine-tuning runtime")
    setup.add_argument("--start-at-login", action="store_true", help="grant permission and open Sideris at login")
    setup.set_defaults(func=cmd_setup)

    chat = sub.add_parser("chat", help="chat in the terminal")
    chat.add_argument("message", nargs="*", help="ask one question and exit")
    chat.add_argument("--level", choices=LEVEL_CHOICES, help="thinking level")
    chat.add_argument("--model", help="switch to another catalogue model")
    chat.set_defaults(func=cmd_chat)

    serve = sub.add_parser("serve", help="run the local HTTP service")
    serve.add_argument("--port", type=int)
    serve.add_argument("--quiet", action="store_true")
    serve.set_defaults(func=cmd_serve)

    service = sub.add_parser("service", help="manage the background service")
    service.add_argument("action", choices=["run", "install", "uninstall", "status"])
    service.add_argument("--port", type=int)
    service.add_argument("--quiet", action="store_true")
    service.set_defaults(func=cmd_service)

    login = sub.add_parser("login", help="manage start at login")
    login.add_argument("action", choices=["enable", "disable", "status"])
    login.set_defaults(func=cmd_login)

    train = sub.add_parser("train", help="fine-tune the local model on your data")
    train.add_argument("action", choices=["start", "stop", "status"])
    train.add_argument("--dataset", help="path to a .jsonl dataset")
    train.add_argument("--hours", type=float, default=2.0, help="time budget in hours")
    train.add_argument("--epochs", type=float, default=1.0)
    train.add_argument("--json", action="store_true")
    train.set_defaults(func=cmd_train)

    status = sub.add_parser("status", help="show local status")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    models = sub.add_parser("models", help="list, download or remove local models")
    models.add_argument("action", choices=["list", "download", "remove"])
    models.add_argument("key", nargs="?")
    models.set_defaults(func=cmd_models)

    open_cmd = sub.add_parser("open", help="open a Sideris folder")
    open_cmd.add_argument("what", nargs="?", default="data",
                          choices=["data", "models", "chats", "workspace", "logs", "checkpoints"])
    open_cmd.set_defaults(func=cmd_open)

    dataset = sub.add_parser("example-dataset", help="write an example training dataset")
    dataset.add_argument("path", nargs="?")
    dataset.set_defaults(func=cmd_example_dataset)

    gui = sub.add_parser("gui", help="open the desktop window")
    gui.set_defaults(func=cmd_gui)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        return cmd_gui(args)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        _print("")
        return 130
    except BrokenPipeError:
        return 0
    except Exception as exc:
        _print(f"Error: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
