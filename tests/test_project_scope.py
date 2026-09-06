"""Tests for directory-scoped session profiles (``session.scope = project``).

What these are really guarding: the blast radius of a switch. The whole point
of a project profile is that switching account in one directory does NOT move
another directory, and does NOT move the default login. A regression here does
not look broken — both terminals keep working, they just quietly share an
account again — so the assertions are mostly about what did NOT change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap.exceptions import SessionError
from claude_swap.session import (
    PROJECT_MARKER,
    SessionManager,
    find_project_profile,
    project_scope_dir,
    project_session_dir,
    project_slug,
    read_project_marker,
    session_scope,
    write_project_marker,
)
from claude_swap.settings import set_setting


# ── slug: the key everything else hangs off ────────────────────────────────
class TestProjectSlug:
    def test_is_stable_for_the_same_directory(self, tmp_path: Path):
        assert project_slug(tmp_path) == project_slug(tmp_path)

    def test_survives_how_the_path_was_typed(self, tmp_path: Path):
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        assert project_slug(nested) == project_slug(f"{nested}/../b")

    def test_same_basename_in_different_places_stays_distinct(self, tmp_path):
        one = tmp_path / "one" / "api"
        two = tmp_path / "two" / "api"
        one.mkdir(parents=True)
        two.mkdir(parents=True)
        assert project_slug(one) != project_slug(two), (
            "two projects called 'api' must not share one credential store"
        )

    def test_is_filesystem_safe(self, tmp_path: Path):
        awkward = tmp_path / "My Project (v2)!"
        awkward.mkdir()
        slug = project_slug(awkward)
        assert set(slug) <= set("abcdefghijklmnopqrstuvwxyz0123456789-_")

    def test_profile_lands_under_sessions(self, tmp_path: Path):
        d = project_session_dir(tmp_path / "backup", tmp_path)
        assert d.parent == tmp_path / "backup" / "sessions"
        assert d.name.startswith("proj-")


# ── the marker: how a directory finds its profile again ───────────────────
class TestProjectMarker:
    def test_round_trips(self, tmp_path: Path):
        profile = tmp_path / "profile"
        write_project_marker(profile, tmp_path, "2", "a@example.com", "org-1")
        marker = read_project_marker(profile)
        assert marker["accountNum"] == "2"
        assert marker["email"] == "a@example.com"
        assert marker["organizationUuid"] == "org-1"

    def test_corrupt_marker_reads_as_absent(self, tmp_path: Path):
        profile = tmp_path / "profile"
        profile.mkdir()
        (profile / PROJECT_MARKER).write_text("{not json", encoding="utf-8")
        assert read_project_marker(profile) is None

    def test_found_from_a_subdirectory(self, tmp_path: Path):
        backup, project = tmp_path / "backup", tmp_path / "repo"
        (project / "src" / "deep").mkdir(parents=True)
        profile = project_session_dir(backup, project)
        write_project_marker(profile, project, "2", "a@example.com", "org-1")
        assert find_project_profile(backup, project / "src" / "deep") == profile

    def test_absent_when_no_profile_exists(self, tmp_path: Path):
        (tmp_path / "repo").mkdir()
        assert find_project_profile(tmp_path / "backup", tmp_path / "repo") is None


# ── scope: opt-in, and never fatal ────────────────────────────────────────
class TestSessionScope:
    def test_defaults_to_account(self, tmp_path: Path):
        assert session_scope(tmp_path) == "account"

    def test_reads_project_when_configured(self, tmp_path: Path):
        set_setting(tmp_path, "session.scope", "project")
        assert session_scope(tmp_path) == "project"

    def test_unreadable_settings_degrade_rather_than_raise(self):
        assert session_scope(object()) == "account"  # type: ignore[arg-type]

    def test_account_scope_ignores_an_existing_profile(self, tmp_path: Path):
        backup, project = tmp_path / "backup", tmp_path / "repo"
        project.mkdir()
        write_project_marker(
            project_session_dir(backup, project), project, "2", "a@x.com", "o"
        )
        assert project_scope_dir(backup, project) is None, (
            "a profile on disk must not scope switches until scope is opted in"
        )

    def test_project_scope_finds_it(self, tmp_path: Path):
        backup, project = tmp_path / "backup", tmp_path / "repo"
        backup.mkdir()
        project.mkdir()
        set_setting(backup, "session.scope", "project")
        profile = project_session_dir(backup, project)
        write_project_marker(profile, project, "2", "a@x.com", "o")
        assert project_scope_dir(backup, project) == profile


# ── the switch itself: what moves, and what must not ──────────────────────
ORG = "org-uuid"


def _creds(token: str) -> str:
    return json.dumps(
        {
            "claudeAiOauth": {
                "accessToken": token,
                "refreshToken": f"refresh-{token}",
                "expiresAt": 9999999999999,
            }
        }
    )


def _config(email: str) -> str:
    return json.dumps(
        {
            "oauthAccount": {
                "emailAddress": email,
                "accountUuid": f"uuid-{email}",
                "organizationUuid": ORG,
            },
            "theme": "light",
        }
    )


@pytest.fixture(autouse=True)
def no_real_claude_or_network(monkeypatch):
    """Nothing in this module wants the real `claude` or the token endpoint.

    Profile setup probes `claude auth status --json` against the new profile
    and refreshes the account's token first; both are faked here so a test
    exercises the routing, not this machine's login state or the network.
    The probe answers from what the profile actually holds.
    """
    from types import SimpleNamespace

    from claude_swap import session as session_mod
    from claude_swap.switcher import ClaudeAccountSwitcher

    def fake_probe(cmd, env=None, **kwargs):
        config_dir = Path(env["CLAUDE_CONFIG_DIR"])
        try:
            acct = json.loads((config_dir / ".claude.json").read_text())["oauthAccount"]
        except (OSError, KeyError, ValueError):
            acct = None
        if acct and (config_dir / ".credentials.json").exists():
            payload = {
                "loggedIn": True,
                "authMethod": "claude.ai",
                "email": acct["emailAddress"],
                "orgId": acct["organizationUuid"],
            }
        else:
            payload = {"loggedIn": False, "authMethod": "none"}
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(session_mod.subprocess, "run", fake_probe)
    monkeypatch.setattr(
        ClaudeAccountSwitcher,
        "consume_backup_grant",
        lambda self, num, email, creds: SimpleNamespace(
            error=None, credentials=None, stashed=False
        ),
    )


@pytest.fixture
def macos_platform(monkeypatch):
    from claude_swap.models import Platform

    monkeypatch.setattr(Platform, "detect", classmethod(lambda cls: Platform.MACOS))


@pytest.fixture
def switcher(temp_home: Path, macos_platform):
    """Two fully backed-up accounts, so either can be switched onto."""
    from claude_swap.switcher import ClaudeAccountSwitcher

    sw = ClaudeAccountSwitcher(debug=True)
    sw._setup_directories()
    sw._write_json(
        sw.sequence_file,
        {
            "activeAccountNumber": 1,
            "lastUpdated": "2024-01-01T00:00:00Z",
            "sequence": [1, 2],
            "accounts": {
                "1": {
                    "email": "one@example.com",
                    "uuid": "uuid-1",
                    "organizationUuid": ORG,
                    "organizationName": "Org",
                    "added": "2024-01-01T00:00:00Z",
                },
                "2": {
                    "email": "two@example.com",
                    "uuid": "uuid-2",
                    "organizationUuid": ORG,
                    "organizationName": "Org",
                    "added": "2024-01-02T00:00:00Z",
                },
            },
        },
    )
    for num, email in (("1", "one@example.com"), ("2", "two@example.com")):
        sw._write_account_credentials(num, email, _creds(f"token-{num}"))
        sw._write_account_config(num, email, _config(email))
    return sw


@pytest.fixture
def project(tmp_path: Path) -> Path:
    d = tmp_path / "potra"
    d.mkdir()
    return d


def _start_profile(switcher, project: Path, account: str, email: str) -> Path:
    """Stand in for a `cswap run` that already happened in this directory."""
    manager = SessionManager(switcher)
    profile = project_session_dir(switcher.backup_dir, project)
    manager._bootstrap(profile, account, email, ORG)
    write_project_marker(profile, project, account, email, ORG)
    return profile


class TestSwitchProject:
    def test_refuses_when_the_directory_has_no_profile(self, switcher, project):
        with pytest.raises(SessionError, match="No project profile"):
            SessionManager(switcher).switch_project(project, "2")

    def test_repoints_the_profile_in_place(self, switcher, project):
        profile = _start_profile(switcher, project, "1", "one@example.com")
        before = profile.read_bytes() if profile.is_file() else None

        session_dir, num, email = SessionManager(switcher).switch_project(
            project, "2"
        )

        assert session_dir == profile, "the profile directory must not move"
        assert (num, email) == ("2", "two@example.com")
        assert read_project_marker(profile)["email"] == "two@example.com"
        creds = json.loads((profile / ".credentials.json").read_text())
        assert creds["claudeAiOauth"]["accessToken"] == "token-2"
        assert before is None

    def test_leaves_the_default_login_alone(self, switcher, project):
        """The whole feature in one assertion."""
        _start_profile(switcher, project, "1", "one@example.com")
        before = switcher._get_current_account()

        SessionManager(switcher).switch_project(project, "2")

        assert switcher._get_current_account() == before, (
            "a project switch moved the machine-wide default login"
        )

    def test_leaves_other_projects_alone(self, switcher, tmp_path: Path):
        potra = tmp_path / "potra"
        hyphenbox = tmp_path / "hyphenbox"
        potra.mkdir()
        hyphenbox.mkdir()
        _start_profile(switcher, potra, "1", "one@example.com")
        other = _start_profile(switcher, hyphenbox, "2", "two@example.com")
        other_creds_before = (other / ".credentials.json").read_text()

        SessionManager(switcher).switch_project(potra, "2")

        assert (other / ".credentials.json").read_text() == other_creds_before, (
            "switching potra rewrote hyphenbox's credentials"
        )
        assert read_project_marker(other)["email"] == "two@example.com"

    def test_switching_to_the_account_already_held_is_a_no_op(
        self, switcher, project
    ):
        profile = _start_profile(switcher, project, "2", "two@example.com")
        before = (profile / ".credentials.json").read_text()

        with patch.object(SessionManager, "_bootstrap") as bootstrap:
            _, num, _ = SessionManager(switcher).switch_project(project, "2")

        bootstrap.assert_not_called()
        assert num == "2"
        assert (profile / ".credentials.json").read_text() == before

    def test_a_subdirectory_switches_the_project(self, switcher, project):
        profile = _start_profile(switcher, project, "1", "one@example.com")
        deep = project / "src" / "deep"
        deep.mkdir(parents=True)

        session_dir, _, email = SessionManager(switcher).switch_project(deep, "2")

        assert session_dir == profile
        assert email == "two@example.com"


class TestRotateProject:
    def test_moves_to_the_next_account_in_sequence(self, switcher, project):
        _start_profile(switcher, project, "1", "one@example.com")
        _, num, email = SessionManager(switcher).rotate_project(project)
        assert (num, email) == ("2", "two@example.com")

    def test_wraps_around(self, switcher, project):
        _start_profile(switcher, project, "2", "two@example.com")
        _, num, _ = SessionManager(switcher).rotate_project(project)
        assert num == "1"

    def test_skips_disabled_accounts(self, switcher, project):
        switcher.set_account_disabled("2", True)
        _start_profile(switcher, project, "1", "one@example.com")
        with pytest.raises(SessionError, match="already on it"):
            SessionManager(switcher).rotate_project(project)

    def test_refuses_without_a_profile(self, switcher, project):
        with pytest.raises(SessionError, match="No project profile"):
            SessionManager(switcher).rotate_project(project)


# ── routing: the CLI and the dashboard must agree on scope ────────────────
class TestSwitchRouting:
    def test_cli_switch_is_scoped_when_a_project_profile_governs_the_cwd(
        self, switcher, project, monkeypatch, capsys
    ):
        from claude_swap import cli

        set_setting(switcher.backup_dir, "session.scope", "project")
        _start_profile(switcher, project, "1", "one@example.com")
        monkeypatch.chdir(project)

        with patch("claude_swap.cli.ClaudeAccountSwitcher", return_value=switcher), \
             patch.object(type(switcher), "switch_to") as global_switch, \
             patch("os.geteuid", return_value=1000, create=True), \
             patch.object(sys, "argv", ["cswap", "switch", "2"]):
            cli.main()

        global_switch.assert_not_called()
        assert read_project_marker(
            project_session_dir(switcher.backup_dir, project)
        )["email"] == "two@example.com"

    def test_cli_switch_stays_global_without_a_project_profile(
        self, switcher, project, monkeypatch
    ):
        from claude_swap import cli

        set_setting(switcher.backup_dir, "session.scope", "project")
        monkeypatch.chdir(project)

        with patch("claude_swap.cli.ClaudeAccountSwitcher", return_value=switcher), \
             patch.object(type(switcher), "switch_to") as global_switch, \
             patch("os.geteuid", return_value=1000, create=True), \
             patch.object(sys, "argv", ["cswap", "switch", "2"]):
            cli.main()

        global_switch.assert_called_once()


class TestDashboardRouting:
    """The dashboard must make the same scope decision the CLI does.

    Two entry points to one switch is exactly how a "switch this project"
    feature quietly becomes machine-wide again for whoever uses the other one.
    """

    def _app(self, switcher):
        from claude_swap.tui.app import CswapApp

        return CswapApp(switcher)

    def test_scoped_switch_never_touches_the_default_login(
        self, switcher, project, monkeypatch
    ):
        set_setting(switcher.backup_dir, "session.scope", "project")
        _start_profile(switcher, project, "1", "one@example.com")
        monkeypatch.chdir(project)
        app = self._app(switcher)

        assert app.project_scope == project_session_dir(
            switcher.backup_dir, project
        )
        payload = app._scoped_switch_payload(
            SessionManager(switcher), str(project), "2"
        )
        assert payload["scope"] == "project"
        assert payload["email"] == "two@example.com"
        assert "potra" in payload["message"]

    def test_no_scope_without_the_setting(self, switcher, project, monkeypatch):
        _start_profile(switcher, project, "1", "one@example.com")
        monkeypatch.chdir(project)
        assert self._app(switcher).project_scope is None


class TestScopedDisplay:
    """A scoped switch that the display contradicts reads as a failed switch.

    The snapshot's ``is_active`` means "is the default login" — which a
    project switch never moves. Left alone, the list keeps marking the old
    account after a successful switch, which is precisely how this feature
    looks broken while working.
    """

    def _app(self, switcher):
        from claude_swap.tui.app import CswapApp

        return CswapApp(switcher)

    def _snapshot(self, active: str):
        from claude_swap.models import AccountSnapshot, AccountsSnapshot, UsageEntry

        accounts = tuple(
            AccountSnapshot(
                number=num,
                email=email,
                org_name="Org",
                org_uuid=ORG,
                is_active=(num == active),
                kind="oauth",
                switchable=True,
                usage=UsageEntry(),
            )
            for num, email in (("1", "one@example.com"), ("2", "two@example.com"))
        )
        return AccountsSnapshot(
            active_number=active, accounts=accounts, taken_at=0.0
        )

    def test_active_follows_the_project_not_the_default_login(
        self, switcher, project, monkeypatch
    ):
        set_setting(switcher.backup_dir, "session.scope", "project")
        _start_profile(switcher, project, "2", "two@example.com")
        monkeypatch.chdir(project)

        # The default login is still account 1; the project holds account 2.
        reprojected = self._app(switcher)._reproject(self._snapshot("1"))

        assert reprojected.active_number == "2", (
            "the list must mark the account this directory holds"
        )
        assert [a.number for a in reprojected.accounts if a.is_active] == ["2"]

    def test_untouched_without_project_scope(self, switcher, project, monkeypatch):
        _start_profile(switcher, project, "2", "two@example.com")
        monkeypatch.chdir(project)
        snap = self._snapshot("1")
        assert self._app(switcher)._reproject(snap) is snap


def test_marker_reads_from_a_string_path(tmp_path: Path):
    """The signature accepts str; a TypeError here is a crash, not a None."""
    write_project_marker(tmp_path, tmp_path, "1", "a@example.com", "org")
    assert read_project_marker(str(tmp_path))["email"] == "a@example.com"


class TestStatusReportsTheDirectory:
    """`status` is the verification command; under project scope it must
    answer "which account is THIS directory on", not "what is the default
    login" — the two differ by design, and only one of them is the question."""

    def test_names_the_directory_and_its_account(
        self, switcher, project, monkeypatch, capsys
    ):
        set_setting(switcher.backup_dir, "session.scope", "project")
        _start_profile(switcher, project, "2", "two@example.com")
        monkeypatch.chdir(project)

        switcher.status()
        out = capsys.readouterr().out

        assert "This directory:" in out
        assert "potra" in out
        assert "two@example.com" in out

    def test_silent_without_project_scope(
        self, switcher, project, monkeypatch, capsys
    ):
        _start_profile(switcher, project, "2", "two@example.com")
        monkeypatch.chdir(project)

        switcher.status()

        assert "This directory:" not in capsys.readouterr().out


# ── select = set this directory's account; `claude` = start on it ─────────
class TestSetProjectAccount:
    def test_first_visit_creates_the_profile(self, switcher, project):
        session_dir, num, email, created = SessionManager(
            switcher
        ).set_project_account(project, "2")

        assert created is True
        assert session_dir == project_session_dir(switcher.backup_dir, project)
        assert (num, email) == ("2", "two@example.com")
        assert read_project_marker(session_dir)["email"] == "two@example.com"
        assert (session_dir / ".credentials.json").is_file()

    def test_later_visits_repoint_without_recreating(self, switcher, project):
        manager = SessionManager(switcher)
        first, *_ = manager.set_project_account(project, "1")
        second, num, _, created = manager.set_project_account(project, "2")

        assert created is False
        assert second == first, "the profile must not move between accounts"
        assert num == "2"

    def test_never_touches_the_default_login(self, switcher, project):
        before = switcher._get_current_account()
        SessionManager(switcher).set_project_account(project, "2")
        assert switcher._get_current_account() == before


class TestRunResolvesTheDirectorysAccount:
    """`cswap run` with no account — what the `claude` shell function calls —
    must land on the account the dashboard set for this directory."""

    def _run(self, argv, monkeypatch, calls, switcher):
        from claude_swap import cli

        class FakeManager:
            def __init__(self, sw):
                pass

            def run(self, identifier, claude_args, share=True,
                    share_history=False, require_session=False, project=None):
                calls.append(("run", identifier, claude_args, project))

            def exec_default(self, claude_args):
                calls.append(("exec_default", claude_args))

        with patch("claude_swap.session.SessionManager", FakeManager), \
             patch("claude_swap.cli.ClaudeAccountSwitcher", return_value=switcher), \
             patch("os.geteuid", return_value=1000, create=True), \
             patch.object(sys, "argv", ["cswap", "run", *argv]):
            cli.main()

    def test_lands_on_the_projects_account(self, switcher, project, monkeypatch):
        set_setting(switcher.backup_dir, "session.scope", "project")
        _start_profile(switcher, project, "2", "two@example.com")
        monkeypatch.chdir(project)
        calls = []

        self._run(["--transparent", "--", "--resume"], monkeypatch, calls, switcher)

        assert calls == [("run", "2", ["--resume"], str(project))]

    def test_a_subdirectory_lands_on_the_same_profile(
        self, switcher, project, monkeypatch
    ):
        set_setting(switcher.backup_dir, "session.scope", "project")
        _start_profile(switcher, project, "2", "two@example.com")
        deep = project / "src"
        deep.mkdir()
        monkeypatch.chdir(deep)
        calls = []

        self._run(["--transparent", "--"], monkeypatch, calls, switcher)

        assert calls[0][3] == str(project), "must resolve to the root's profile"

    def test_transparent_outside_any_project_is_plain_claude(
        self, switcher, project, monkeypatch, capsys
    ):
        set_setting(switcher.backup_dir, "session.scope", "project")
        monkeypatch.chdir(project)
        calls = []

        self._run(["--transparent", "--", "--version"], monkeypatch, calls, switcher)

        assert calls == [("exec_default", ["--version"])]
        assert "No account mapped" not in capsys.readouterr().out, (
            "a bare `claude` outside a project must not chatter"
        )


class TestShellInit:
    def _out(self, shell, capsys):
        from claude_swap import cli

        with patch.object(sys, "argv", ["cswap", "shell-init", shell]):
            cli.main()
        return capsys.readouterr().out

    def test_zsh_defers_entirely_to_cswap_run(self, capsys, temp_home):
        out = self._out("zsh", capsys)
        assert "claude() {" in out
        assert "cswap run --transparent -- \"$@\"" in out
        assert 'command claude "$@"' in out, "must degrade to plain claude"

    def test_fish_has_the_same_shape(self, capsys, temp_home):
        out = self._out("fish", capsys)
        assert "function claude" in out
        assert "cswap run --transparent -- $argv" in out

    def test_the_function_carries_no_policy(self, capsys, temp_home):
        """Every routing decision belongs to cswap, so the shell function
        must not know about scopes, profiles, or CLAUDE_CONFIG_DIR."""
        out = self._out("zsh", capsys)
        for word in ("CLAUDE_CONFIG_DIR", "sessions/", "proj-", ".json"):
            assert word not in out


class TestDashboardSelectSetsTheDirectory:
    def _app(self, switcher):
        from claude_swap.tui.app import CswapApp

        return CswapApp(switcher)

    def test_first_visit_sets_rather_than_switching_globally(
        self, switcher, project, monkeypatch
    ):
        set_setting(switcher.backup_dir, "session.scope", "project")
        monkeypatch.chdir(project)
        app = self._app(switcher)

        with patch.object(type(switcher), "switch_to") as global_switch, \
             patch.object(type(app), "_start_action") as start_action:
            app.do_switch("2")

        global_switch.assert_not_called()
        start_action.assert_called_once()
        assert start_action.call_args[0][0].startswith("Set potra to account 2")

    def test_payload_says_to_run_claude_when_nothing_runs(
        self, switcher, project
    ):
        from claude_swap.tui.app import CswapApp

        payload = CswapApp._set_project_payload(
            SessionManager(switcher), str(project), "2"
        )
        assert payload["scope"] == "project"
        assert payload["created"] is True
        assert "run `claude` in potra" in payload["message"]

    def test_payload_says_the_session_follows_when_one_runs(
        self, switcher, project
    ):
        from claude_swap.tui.app import CswapApp

        _start_profile(switcher, project, "1", "one@example.com")
        with patch("claude_swap.session.profile_is_quiescent", return_value=False):
            payload = CswapApp._set_project_payload(
                SessionManager(switcher), str(project), "2"
            )
        assert "running session follows" in payload["message"]

    def test_scope_off_still_switches_the_default_login(
        self, switcher, project, monkeypatch
    ):
        monkeypatch.chdir(project)
        app = self._app(switcher)
        with patch.object(type(app), "_start_action") as start_action:
            app.do_switch("2")
        assert "Switch to account 2" in start_action.call_args[0][0]


# ── one account, one live place; and never destroy an uncaptured generation ─
import os as _os

from claude_swap.session import live_session_cwds, profiles_holding


def _make_live(session_dir: Path, cwd: str, pid: int | None = None) -> None:
    """A live claude under a profile, in ``cwd`` (own PID is always alive)."""
    pid = pid or _os.getpid()
    d = session_dir / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{pid}.json").write_text(json.dumps({"pid": pid, "cwd": cwd}))


def _rotated(token: str, expires: int) -> str:
    """A NEWER generation of a family: different refresh token, later expiry."""
    return json.dumps(
        {
            "claudeAiOauth": {
                "accessToken": f"{token}-rotated",
                "refreshToken": f"refresh-{token}-rotated",
                "expiresAt": expires,
            }
        }
    )


class TestProfilesHolding:
    def test_finds_the_account_dir_and_every_project_on_the_account(
        self, switcher, tmp_path: Path
    ):
        potra, hyphenbox, other = (tmp_path / n for n in ("potra", "hyphenbox", "other"))
        for d in (potra, hyphenbox, other):
            d.mkdir()
        _start_profile(switcher, potra, "1", "one@example.com")
        _start_profile(switcher, hyphenbox, "1", "one@example.com")
        _start_profile(switcher, other, "2", "two@example.com")

        held = profiles_holding(switcher.backup_dir, "1", "one@example.com", ORG)

        assert project_session_dir(switcher.backup_dir, potra) in held
        assert project_session_dir(switcher.backup_dir, hyphenbox) in held
        assert project_session_dir(switcher.backup_dir, other) not in held
        assert held[0].name == "1-one_example.com", "account dir first"


class TestCaptureBeforeDestroy:
    """The reported bug: switch a directory away from an account whose token
    rotated inside the profile, then back — and the account is logged out,
    because the switch away deleted the only fresh copy."""

    def test_switch_away_advances_the_outgoing_backup(self, switcher, project):
        profile = _start_profile(switcher, project, "1", "one@example.com")
        # claude refreshed inside the profile: profile is ahead of the backup
        newer = _rotated("token-1", 9999999999999 + 1)
        (profile / ".credentials.json").write_text(newer)

        SessionManager(switcher).switch_project(project, "2")

        backup = switcher._read_account_credentials("1", "one@example.com")
        assert json.loads(backup) == json.loads(newer), (
            "the rotated generation must reach the backup before the keychain "
            "entry that held it is deleted"
        )

    def test_and_switching_back_seeds_the_rotated_generation(
        self, switcher, project
    ):
        profile = _start_profile(switcher, project, "1", "one@example.com")
        newer = _rotated("token-1", 9999999999999 + 1)
        (profile / ".credentials.json").write_text(newer)
        manager = SessionManager(switcher)

        manager.switch_project(project, "2")
        manager.switch_project(project, "1")

        seeded = json.loads((profile / ".credentials.json").read_text())
        assert seeded["claudeAiOauth"]["refreshToken"] == "refresh-token-1-rotated"

    def test_an_older_profile_never_regresses_the_backup(self, switcher, project):
        """A backup that moved on (re-login elsewhere) must not be dragged
        back to the profile's older generation."""
        profile = _start_profile(switcher, project, "1", "one@example.com")
        advanced = _rotated("token-1", 9999999999999 + 5)
        switcher._store._write_account_credentials("1", "one@example.com", advanced)
        assert (profile / ".credentials.json").is_file()  # older, still there

        SessionManager(switcher).switch_project(project, "2")

        backup = switcher._read_account_credentials("1", "one@example.com")
        assert json.loads(backup) == json.loads(advanced)


class TestOneLivePlace:
    def test_refuses_an_account_live_in_another_project(
        self, switcher, tmp_path: Path
    ):
        potra, hyphenbox = tmp_path / "potra", tmp_path / "hyphenbox"
        potra.mkdir(); hyphenbox.mkdir()
        other = _start_profile(switcher, hyphenbox, "2", "two@example.com")
        _make_live(other, str(hyphenbox))

        with pytest.raises(SessionError) as e:
            SessionManager(switcher).set_project_account(potra, "2")

        assert "hyphenbox" in str(e.value)
        assert "one place at a time" in str(e.value)

    def test_refuses_the_default_login_while_something_runs_on_it(
        self, switcher, project, monkeypatch
    ):
        from claude_swap import session as session_mod

        monkeypatch.setattr(
            type(switcher), "_get_current_account",
            lambda self: ("two@example.com", ORG),
        )
        default_dir = session_mod.get_default_global_config_path().parent
        _make_live(default_dir, "/somewhere/unbound")

        with pytest.raises(SessionError) as e:
            SessionManager(switcher).set_project_account(project, "2")

        assert "/somewhere/unbound (default login)" in str(e.value)

    def test_the_default_login_is_fine_when_nothing_runs_on_it(
        self, switcher, project, monkeypatch
    ):
        monkeypatch.setattr(
            type(switcher), "_get_current_account",
            lambda self: ("two@example.com", ORG),
        )
        _, num, _, _ = SessionManager(switcher).set_project_account(project, "2")
        assert num == "2"

    def test_a_quiescent_profile_elsewhere_is_no_obstacle(
        self, switcher, tmp_path: Path
    ):
        potra, hyphenbox = tmp_path / "potra", tmp_path / "hyphenbox"
        potra.mkdir(); hyphenbox.mkdir()
        _start_profile(switcher, hyphenbox, "2", "two@example.com")  # not live

        _, num, _, _ = SessionManager(switcher).set_project_account(potra, "2")
        assert num == "2"

    def test_the_directory_itself_is_never_its_own_obstacle(
        self, switcher, project
    ):
        profile = _start_profile(switcher, project, "1", "one@example.com")
        _make_live(profile, str(project))
        _, num, _ = SessionManager(switcher).switch_project(project, "2")
        assert num == "2"


class TestBackupChangesReachProjectProfiles:
    """`_post_backup_write` is the chokepoint for 'the backup moved under a
    profile'. It only ever looked at the account-keyed dir."""

    def test_a_quiescent_project_profile_is_invalidated(self, switcher, project):
        profile = _start_profile(switcher, project, "1", "one@example.com")
        assert (profile / ".credentials.json").is_file()

        switcher._write_account_credentials(
            "1", "one@example.com", _rotated("token-1", 9999999999999 + 1)
        )

        assert not (profile / ".credentials.json").exists(), (
            "a superseded project profile must re-bootstrap, not keep serving "
            "the dead generation"
        )

    def test_a_live_project_profile_is_marked_stale_not_gutted(
        self, switcher, project
    ):
        from claude_swap.session import is_session_stale

        profile = _start_profile(switcher, project, "1", "one@example.com")
        _make_live(profile, str(project))

        switcher._write_account_credentials(
            "1", "one@example.com", _rotated("token-1", 9999999999999 + 1)
        )

        assert (profile / ".credentials.json").is_file(), "never under a live claude"
        assert is_session_stale(profile)


def test_live_session_cwds_lists_where_an_account_runs(tmp_path: Path):
    _make_live(tmp_path, "/a"); _make_live(tmp_path, "/b", pid=_os.getpid())
    assert live_session_cwds(tmp_path) == ["/b"]  # same pid, last record wins


def test_reselecting_a_stale_profile_reseeds_it(switcher, project):
    """The 'already on it' shortcut must not keep a superseded generation."""
    from claude_swap.session import is_session_stale, mark_session_stale

    profile = _start_profile(switcher, project, "1", "one@example.com")
    advanced = _rotated("token-1", 9999999999999 + 5)
    switcher._store._write_account_credentials("1", "one@example.com", advanced)
    mark_session_stale(profile)

    SessionManager(switcher).switch_project(project, "1")

    seeded = json.loads((profile / ".credentials.json").read_text())
    assert seeded == json.loads(advanced)
    assert not is_session_stale(profile)


class TestHistoryIsSharedForDirectories:
    """`--resume` in a directory must list the conversations you already had.
    A profile with its own history folder is a fresh start, which is the
    opposite of 'as if nothing happened'."""

    def test_first_visit_links_history_into_the_profile(self, switcher, project):
        from claude_swap.session import HISTORY_ITEMS

        session_dir, *_ = SessionManager(switcher).set_project_account(project, "2")

        for name in HISTORY_ITEMS:
            assert (session_dir / name).is_symlink(), f"{name} must be shared"

    def test_run_in_a_directory_shares_history_by_default(
        self, switcher, project, monkeypatch
    ):
        from claude_swap import cli

        set_setting(switcher.backup_dir, "session.scope", "project")
        _start_profile(switcher, project, "2", "two@example.com")
        monkeypatch.chdir(project)
        calls = []

        class FakeManager:
            def __init__(self, sw): pass
            def run(self, identifier, claude_args, share=True,
                    share_history=False, require_session=False, project=None):
                calls.append(share_history)

        with patch("claude_swap.session.SessionManager", FakeManager), \
             patch("claude_swap.cli.ClaudeAccountSwitcher", return_value=switcher), \
             patch("os.geteuid", return_value=1000, create=True), \
             patch.object(sys, "argv", ["cswap", "run", "--transparent", "--"]):
            cli.main()
        assert calls == [True]

    def test_but_no_share_history_still_opts_out(
        self, switcher, project, monkeypatch
    ):
        from claude_swap import cli

        set_setting(switcher.backup_dir, "session.scope", "project")
        _start_profile(switcher, project, "2", "two@example.com")
        monkeypatch.chdir(project)
        calls = []

        class FakeManager:
            def __init__(self, sw): pass
            def run(self, identifier, claude_args, share=True,
                    share_history=False, require_session=False, project=None):
                calls.append(share_history)

        with patch("claude_swap.session.SessionManager", FakeManager), \
             patch("claude_swap.cli.ClaudeAccountSwitcher", return_value=switcher), \
             patch("os.geteuid", return_value=1000, create=True), \
             patch.object(sys, "argv", ["cswap", "run", "--no-share-history", "--"]):
            cli.main()
        assert calls == [False]

    def test_account_profiles_keep_the_opt_in(self, switcher, project, monkeypatch):
        from claude_swap import cli

        monkeypatch.chdir(project)
        calls = []

        class FakeManager:
            def __init__(self, sw): pass
            def run(self, identifier, claude_args, share=True,
                    share_history=False, require_session=False, project=None):
                calls.append(share_history)

        with patch("claude_swap.session.SessionManager", FakeManager), \
             patch("claude_swap.cli.ClaudeAccountSwitcher", return_value=switcher), \
             patch("os.geteuid", return_value=1000, create=True), \
             patch.object(sys, "argv", ["cswap", "run", "2", "--"]):
            cli.main()
        assert calls == [False]
