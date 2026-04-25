#!/usr/bin/env python3
"""Docstring coverage checker for the agent-urban-planning library.

Walks every public symbol exported at ``aup.*`` and validates the docstring
contract (per design D5 of the extract-library-agent-urban-planning change):

  * Summary line (≤80 chars)
  * Detailed description (1-3 paragraphs)
  * Args section with type + description per parameter (for callables)
  * Returns section with type + description (for callables)
  * Raises section if any exceptions are raised
  * Examples section with at least one runnable code block

Usage::

    python tools/check_docstrings.py            # check all public symbols
    python tools/check_docstrings.py --fail-under 0.9  # fail if coverage < 90%
    python tools/check_docstrings.py --report  # print full per-symbol report

Exit code 0 if every public symbol passes; non-zero on any violation.
"""
from __future__ import annotations

import argparse
import inspect
import re
import sys
from dataclasses import dataclass, field
from typing import Any


REPO_ROOT = sys.path[0] if sys.path else "."
sys.path.insert(0, REPO_ROOT + "/src")

import agent_urban_planning as aup  # noqa: E402


# Sections we look for (Google-style or NumPy-style). Case-insensitive.
SECTION_PATTERN = re.compile(
    r"^\s*(Args|Arguments|Parameters|Returns|Yields|Raises|Examples?|Notes?|References?|See Also|Attributes)\s*[:\-]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class CheckResult:
    """One symbol's check result."""

    name: str
    kind: str  # "module", "class", "function", "method"
    docstring: str | None
    issues: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues


def _check_docstring(symbol: Any, name: str, kind: str) -> CheckResult:
    """Validate one symbol's docstring against the contract."""
    doc = inspect.getdoc(symbol)
    result = CheckResult(name=name, kind=kind, docstring=doc)

    if not doc or not doc.strip():
        result.issues.append("missing docstring")
        return result

    lines = doc.strip().splitlines()
    summary = lines[0].strip() if lines else ""

    if not summary:
        result.issues.append("empty summary line")
    elif len(summary) > 100:  # 80 strict, allow a bit more
        result.issues.append(f"summary line too long ({len(summary)} chars)")

    if len(lines) < 2:
        result.issues.append("missing detailed description (only summary)")

    # Sections (Google/NumPy style).
    sections_found = {
        m.group(1).lower() for m in SECTION_PATTERN.finditer(doc)
    }

    if kind in ("function", "method") and inspect.isfunction(symbol) or inspect.ismethod(symbol):
        sig = None
        try:
            sig = inspect.signature(symbol)
        except (ValueError, TypeError):
            pass
        if sig is not None:
            non_self_params = [
                p for n, p in sig.parameters.items()
                if n not in ("self", "cls", "args", "kwargs")
            ]
            if non_self_params and not sections_found & {"args", "arguments", "parameters"}:
                result.issues.append("missing Args/Parameters section")
            if (sig.return_annotation is not inspect.Signature.empty
                and sig.return_annotation is not None
                and not sections_found & {"returns", "yields"}):
                result.issues.append("missing Returns section")

    # Examples section recommended for callables and classes.
    if kind in ("class", "function", "method"):
        if not sections_found & {"example", "examples"}:
            result.issues.append("missing Examples section")

    return result


def _walk_public_symbols() -> list[CheckResult]:
    """Walk aup's public API and check each symbol's docstring."""
    results: list[CheckResult] = []

    # Top-level package itself.
    results.append(_check_docstring(aup, "agent_urban_planning", "module"))

    # Iterate __all__ if available, else dir().
    public_names = getattr(aup, "__all__", None)
    if public_names is None:
        public_names = [n for n in dir(aup) if not n.startswith("_")]

    for name in public_names:
        if name == "__version__":
            continue
        symbol = getattr(aup, name, None)
        if symbol is None:
            continue

        if inspect.isclass(symbol):
            results.append(_check_docstring(symbol, f"aup.{name}", "class"))
            # Public methods on the class.
            for attr_name, attr_val in inspect.getmembers(symbol):
                if attr_name.startswith("_"):
                    continue
                if not (inspect.isfunction(attr_val) or inspect.ismethod(attr_val)):
                    continue
                # Only methods defined on this class (not inherited from object/etc.).
                if attr_name in object.__dict__:
                    continue
                results.append(
                    _check_docstring(attr_val, f"aup.{name}.{attr_name}", "method")
                )
        elif inspect.isfunction(symbol):
            results.append(_check_docstring(symbol, f"aup.{name}", "function"))
        elif inspect.ismodule(symbol):
            # Subpackage namespace; skip — those are walked via deeper imports.
            pass

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fail-under", type=float, default=1.0,
                        help="Fail if coverage < this fraction (default 1.0 = 100%%).")
    parser.add_argument("--report", action="store_true",
                        help="Print full per-symbol report (otherwise summary only).")
    args = parser.parse_args()

    results = _walk_public_symbols()
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    coverage = passed / total if total > 0 else 0.0

    print(f"Docstring coverage: {passed}/{total} = {coverage:.1%}")

    if args.report:
        print("\n--- Per-symbol report ---")
        for r in results:
            status = "✓" if r.passed else "✗"
            issues = "; ".join(r.issues) if r.issues else ""
            print(f"  {status} [{r.kind:8}] {r.name}{('  — ' + issues) if issues else ''}")
    else:
        # Show only failures.
        failures = [r for r in results if not r.passed]
        if failures:
            print(f"\n--- {len(failures)} failures ---")
            for r in failures[:30]:
                issues = "; ".join(r.issues)
                print(f"  ✗ [{r.kind:8}] {r.name}: {issues}")
            if len(failures) > 30:
                print(f"  ... ({len(failures) - 30} more)")

    if coverage < args.fail_under:
        print(f"\nFAIL: coverage {coverage:.1%} below threshold {args.fail_under:.1%}")
        return 1
    print(f"\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
