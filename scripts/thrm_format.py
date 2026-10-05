# scripts/thrm_format.py
#
# Single source of truth for the THRM prompt format. Imported by both the
# v5 data builder and thrm_demo.py so training and inference never drift.
#
#   <|system|>
#   <global_policy>..</global_policy>
#   <directory_policy>..</directory_policy>
#   <|user|>
#   {issue}
#
#   Contents of {path}:
#   {code}
#   <|assistant|>
#   {unified diff}

DEFAULT_GLOBAL_POLICY = "Standard Policy"
DEFAULT_DIRECTORY_POLICY = "Local Logic"
ELLIPSIS = "..."


def format_context(files):
    """files: list of (path, code_text) pairs."""
    return "\n\n".join(f"Contents of {path}:\n{code.rstrip()}" for path, code in files)


def build_prompt(issue, files,
                 global_policy=DEFAULT_GLOBAL_POLICY,
                 directory_policy=DEFAULT_DIRECTORY_POLICY):
    user = issue.strip()
    if files:
        user += "\n\n" + format_context(files)
    return (
        "<|system|>\n"
        f"<global_policy>{global_policy}</global_policy>\n"
        f"<directory_policy>{directory_policy}</directory_policy>\n"
        "<|user|>\n"
        f"{user}\n"
        "<|assistant|>\n"
    )
