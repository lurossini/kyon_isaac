from __future__ import annotations

import json
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional, Union

# pip install wandb
import wandb


@dataclass
class Checkpoint:
    """Represents a checkpoint saved with the run."""
    id: str                     # unique identifier inside this helper
    name: str                   # path within run files or artifact
    source: Literal["file", "artifact"]
    artifact: Optional[str] = None  # e.g., "model:latest" or "my-artifact:v3"
    size: Optional[int] = None      # bytes if known


class IsaacLabWandbInspector:
    """
    Helper for extracting repo info, diffs, and checkpoints from a W&B run.

    Works with a live SDK Run or a public API Run, e.g.:
        api = wandb.Api()
        run = api.run("entity/project/run_id")
        insp = IsaacLabWandbInspector(run)

    Exposes:
      - repo_remote            → str | None
      - sha1                   → str | None
      - get_diff_text()        → str | None
      - list_checkpoints()     → List[Checkpoint]
      - get_last_checkpoint()  → Checkpoint | None
      - download_checkpoint()  → Path
    """

    # Common names W&B uses for the captured git diff
    _DIFF_CANDIDATES = (
        "diff.patch", "diff.txt", "git.diff", "code.diff", "patch.diff"
    )
    _METADATA_FILENAME = "wandb-metadata.json"

    def __init__(self, run: Any):
        self.run = run
        self._files_cache: Optional[List[Any]] = None
        self._meta_cache: Optional[dict] = None
        self._diff_cache: Optional[str] = None
        self._ckpt_cache: Optional[List[Checkpoint]] = None

    # ---------- Public API ----------

    @property
    def repo_remote(self) -> Optional[str]:
        """Git remote URL of the training repo, if recorded."""
        meta = self._load_metadata()
        if meta:
            git = meta.get("git") or {}
            # Newer W&B metadata stores 'remote', some setups store 'repo'
            return git.get("remote") or git.get("repo")
        # Fallbacks if metadata missing
        return getattr(self.run, "git_remote_url", None)

    @property
    def sha1(self) -> Optional[str]:
        """Git commit SHA1 for the run."""
        meta = self._load_metadata()
        if meta:
            git = meta.get("git") or {}
            commit = git.get("commit")
            if commit:
                return commit
        # Fallback to public API attribute if available
        return getattr(self.run, "commit", None)

    def get_diff_text(self) -> Optional[str]:
        """Return the captured git diff text, if present."""
        if self._diff_cache is not None:
            return self._diff_cache

        # Prefer well-known diff filenames; otherwise any file containing 'diff'
        files = self._list_run_files()
        by_name = {f.name: f for f in files}
        target = None
        for candidate in self._DIFF_CANDIDATES:
            if candidate in by_name:
                target = by_name[candidate]
                break
        if target is None:
            for f in files:
                name = f.name.lower()
                if "diff" in name and (name.endswith(".patch") or name.endswith(".txt") or name.endswith(".diff")):
                    target = f
                    break
        if target is None:
            self._diff_cache = None
            return None

        # Download to a temp file and read
        with tempfile.TemporaryDirectory() as td:
            downloaded = target.download(root=td)  # type: ignore
            try:
                # download() may return a file object or a path string
                if hasattr(downloaded, 'read'):
                    # It's a file object
                    self._diff_cache = downloaded.read()
                    if isinstance(self._diff_cache, bytes):
                        self._diff_cache = self._diff_cache.decode("utf-8", errors="replace")
                    downloaded.close()
                else:
                    # It's a path
                    local_path = Path(downloaded)
                    try:
                        self._diff_cache = local_path.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        # As a fallback, open in binary then decode
                        self._diff_cache = local_path.read_bytes().decode("utf-8", errors="replace")
            except Exception:
                self._diff_cache = None
        return self._diff_cache

    def list_checkpoints(self, refresh: bool = False) -> List[Checkpoint]:
        """
        Return all .pt checkpoint files saved either directly under the run
        (run files) or inside any artifacts logged by this run.
        """
        if self._ckpt_cache is not None and not refresh:
            return list(self._ckpt_cache)

        ckpts: List[Checkpoint] = []
        # 1) Direct run files (*.pt)
        for f in self._list_run_files():
            if f.name.endswith(".pt"):
                ckpts.append(
                    Checkpoint(
                        id=f"file::{f.name}",
                        name=f.name,
                        source="file",
                        artifact=None,
                        size=getattr(f, "size", None),
                    )
                )

        # 2) Logged artifacts (*.pt inside the artifact)
        try:
            for art in self._iter_logged_artifacts():
                art_id = f"{art.name}:{art.version}"
                # artifact.manifest is stable across API versions
                entries = getattr(art, "manifest", None)
                # Newer public API: manifest.entries is a dict of path->entry
                if entries and hasattr(entries, "entries"):
                    for relpath, _entry in entries.entries.items():  # type: ignore
                        if relpath.endswith(".pt"):
                            ckpts.append(
                                Checkpoint(
                                    id=f"artifact::{art_id}::{relpath}",
                                    name=relpath,
                                    source="artifact",
                                    artifact=art_id,
                                    size=None,  # size available after download in most cases
                                )
                            )
                else:
                    # Fallback: artifact.files()
                    try:
                        for af in art.files():  # type: ignore
                            if af.name.endswith(".pt"):
                                ckpts.append(
                                    Checkpoint(
                                        id=f"artifact::{art_id}::{af.name}",
                                        name=af.name,
                                        source="artifact",
                                        artifact=art_id,
                                        size=getattr(af, "size", None),
                                    )
                                )
                    except Exception:
                        pass
        except Exception:
            # If artifacts are unavailable, we silently continue with run files only.
            pass

        self._ckpt_cache = ckpts
        return list(ckpts)

    def get_last_checkpoint(self) -> Optional[Checkpoint]:
        """
        Return the checkpoint with the highest iteration number.
        Assumes checkpoint names follow the pattern 'model_<ITERATION>.pt'.
        Returns None if no checkpoints are found or if no checkpoint matches the pattern.
        """
        import re
        
        ckpts = self.list_checkpoints()
        if not ckpts:
            return None
        
        # Pattern to match model_<number>.pt
        pattern = re.compile(r'model_(\d+)\.pt$')
        
        max_iteration = -1
        last_ckpt = None
        
        for ckpt in ckpts:
            # Extract just the filename from the path
            filename = Path(ckpt.name).name
            match = pattern.search(filename)
            if match:
                iteration = int(match.group(1))
                if iteration > max_iteration:
                    max_iteration = iteration
                    last_ckpt = ckpt
        
        return last_ckpt

    def download_checkpoint(
        self,
        which: Union[int, str],
        dest: Union[str, Path] = "checkpoints",
        exist_ok: bool = True,
    ) -> Path:
        """
        Download a checkpoint identified by:
          - integer index from list_checkpoints(), or
          - exact 'id' from list_checkpoints(), or
          - file name suffix match (last resort; downloads the first match).

        Returns the local filesystem path to the downloaded .pt.
        """
        ckpts = self.list_checkpoints()
        if isinstance(which, int):
            if which < 0 or which >= len(ckpts):
                raise IndexError(f"Checkpoint index out of range [0, {len(ckpts)-1}]")
            target = ckpts[which]
        else:
            # exact id match
            target = next((c for c in ckpts if c.id == which), None)
            if target is None:
                # try exact name match
                target = next((c for c in ckpts if c.name == which), None)
            if target is None:
                # try suffix match
                matches = [c for c in ckpts if c.name.endswith(str(which))]
                if len(matches) == 1:
                    target = matches[0]
                elif len(matches) > 1:
                    ids = "\n  - ".join(m.id for m in matches)
                    raise ValueError(
                        f"Ambiguous checkpoint name; multiple matches:\n  - {ids}\n"
                        "Specify an exact id from list_checkpoints()."
                    )
        if target is None:
            raise ValueError(
                f"Checkpoint '{which}' not found. Use list_checkpoints() to see options."
            )

        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)

        if target.source == "file":
            # Run file download
            f = self.run.file(target.name)
            downloaded = f.download(root=str(dest_path))  # type: ignore
            # Handle both file object and path string returns
            if hasattr(downloaded, 'name'):
                # It's a file object, get its name/path
                local = Path(downloaded.name)
                downloaded.close()
            else:
                # It's a path string
                local = Path(downloaded)
            # Normalize to <dest>/<basename>
            normalized = dest_path / Path(target.name).name
            if exist_ok or not normalized.exists():
                local.rename(normalized) if local != normalized else None
            return normalized

        # Artifact file download
        assert target.source == "artifact"
        assert target.artifact is not None

        # Locate the artifact object
        artifact_name, artifact_version = target.artifact.split(":")
        art = self._find_logged_artifact(artifact_name, artifact_version)
        if art is None:
            raise RuntimeError(f"Artifact {target.artifact} not available on this run.")

        # Use get_path to pull a single file from the artifact
        apath = art.get_path(target.name)  # type: ignore
        downloaded = apath.download(root=str(dest_path))  # type: ignore
        # Handle both file object and path string returns
        if hasattr(downloaded, 'name'):
            # It's a file object, get its name/path
            local_file = Path(downloaded.name)
            downloaded.close()
        else:
            # It's a path string
            local_file = Path(downloaded)

        normalized = dest_path / Path(target.name).name
        if exist_ok or not normalized.exists():
            local_file.rename(normalized) if local_file != normalized else None
        return normalized

    # ---------- Internals ----------

    def _list_run_files(self) -> List[Any]:
        if self._files_cache is None:
            # Convert to list once to avoid repeated API paging
            self._files_cache = list(self.run.files())  # type: ignore
        return self._files_cache

    def _load_metadata(self) -> Optional[dict]:
        if self._meta_cache is not None:
            return self._meta_cache

        # Try the canonical metadata file first
        meta_file = next((f for f in self._list_run_files()
                          if f.name == self._METADATA_FILENAME), None)
        if meta_file is None:
            self._meta_cache = {}
            return self._meta_cache

        with tempfile.TemporaryDirectory() as td:
            downloaded = meta_file.download(root=td)  # type: ignore
            try:
                # download() may return a file object or a path string
                if hasattr(downloaded, 'read'):
                    # It's a file object
                    self._meta_cache = json.load(downloaded)
                    downloaded.close()
                else:
                    # It's a path
                    local = Path(downloaded)
                    self._meta_cache = json.loads(local.read_text(encoding="utf-8"))
            except Exception:
                self._meta_cache = {}
        return self._meta_cache

    def _iter_logged_artifacts(self) -> Iterable[Any]:
        # Public API Run exposes logged_artifacts(); SDK Run may not.
        if hasattr(self.run, "logged_artifacts"):
            try:
                return list(self.run.logged_artifacts())  # type: ignore
            except Exception:
                return []
        return []

    def _find_logged_artifact(self, name: str, version: str) -> Optional[Any]:
        for art in self._iter_logged_artifacts():
            if getattr(art, "name", None) == name and getattr(art, "version", None) == version:
                return art
        return None



# # Using the public API (works for runs you have access to)
# api = wandb.Api()
# run = api.run("arturo-laurenzi-istituto-italiano-di-tecnologia/isaaclab/8xjaku4i")

# insp = IsaacLabWandbInspector(run)
# print("Remote:", insp.repo_remote)
# print("SHA1  :", insp.sha1)
# print("Diff  :\n", (insp.get_diff_text() or "")[:200], "...")
# print("Checkpoints num:", len(insp.list_checkpoints()))
# print("Last Checkpoint:", insp.get_last_checkpoint())

# path = insp.download_checkpoint(insp.get_last_checkpoint().name, dest="downloaded_checkpoints")
# print("Downloaded last checkpoint to:", path)