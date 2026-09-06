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


class TestDashboardLaunchesWhenNothingIsRunning:
    """`cswap` alone must be enough.

    Selecting an account in a directory with no live session used to fall
    through to the machine-wide switch (when no profile existed) or re-point a
    profile nothing was running under — both of which look, from the terminal,
    exactly like nothing happened. The dashboard has to launch instead.
    """

    def _app(self, switcher):
        from claude_swap.tui.app import CswapApp

        return CswapApp(switcher)

    def test_scope_is_project_without_any_profile(
        self, switcher, project, monkeypatch
    ):
        set_setting(switcher.backup_dir, "session.scope", "project")
        monkeypatch.chdir(project)
        app = self._app(switcher)
        assert app.scope_is_project is True
        assert app.project_scope is None, "nothing to re-point on a first visit"

    def test_first_visit_queues_a_launch_instead_of_a_global_switch(
        self, switcher, project, monkeypatch
    ):
        set_setting(switcher.backup_dir, "session.scope", "project")
        monkeypatch.chdir(project)
        app = self._app(switcher)

        with patch.object(type(switcher), "switch_to") as global_switch, \
             patch.object(type(app), "exit") as exit_, \
             patch.object(type(app), "_start_action") as start_action:
            app.do_switch("2")

        global_switch.assert_not_called()
        start_action.assert_not_called()
        assert app.pending_launch == ("2", str(project))
        exit_.assert_called_once()

    def test_confirming_queues_the_launch_and_exits(
        self, switcher, project, monkeypatch
    ):
        set_setting(switcher.backup_dir, "session.scope", "project")
        monkeypatch.chdir(project)
        app = self._app(switcher)

        with patch.object(type(app), "exit") as exit_:
            app._queue_launch(str(project), "2")

        assert app.pending_launch == ("2", str(project))
        exit_.assert_called_once()

    def test_a_quiescent_profile_launches_rather_than_repointing(
        self, switcher, project, monkeypatch
    ):
        """Re-pointing a profile nothing runs under would look like a no-op."""
        set_setting(switcher.backup_dir, "session.scope", "project")
        _start_profile(switcher, project, "1", "one@example.com")
        monkeypatch.chdir(project)
        app = self._app(switcher)

        with patch.object(type(app), "exit") as exit_, \
             patch.object(type(app), "_start_action") as start_action:
            app.do_switch("2")

        start_action.assert_not_called()
        assert app.pending_launch == ("2", str(project))
        exit_.assert_called_once()

    def test_a_live_profile_repoints_in_place(
        self, switcher, project, monkeypatch
    ):
        """The live case is the feature: switch under the running session."""
        set_setting(switcher.backup_dir, "session.scope", "project")
        _start_profile(switcher, project, "1", "one@example.com")
        monkeypatch.chdir(project)
        app = self._app(switcher)

        with patch("claude_swap.session.profile_is_quiescent", return_value=False), \
             patch.object(type(app), "exit") as exit_, \
             patch.object(type(app), "_start_action") as start_action:
            app.do_switch("2")

        exit_.assert_not_called()
        assert app.pending_launch is None
        start_action.assert_called_once()

    def test_scope_off_still_switches_the_default_login(
        self, switcher, project, monkeypatch
    ):
        monkeypatch.chdir(project)
        app = self._app(switcher)

        with patch.object(type(app), "_start_action") as start_action, \
             patch.object(type(app), "exit") as exit_:
            app.do_switch("2")

        exit_.assert_not_called()
        start_action.assert_called_once()
        assert "Switch to account 2" in start_action.call_args[0][0]


def test_tui_run_performs_a_queued_launch(switcher, project, monkeypatch):
    """The exec cannot happen inside Textual; run() must do it on the way out."""
    from claude_swap import tui

    calls = []

    class FakeApp:
        return_code = 0
        pending_launch = ("2", str(project))

        def __init__(self, *a, **k):
            pass

        def run(self):
            calls.append("app.run")

    class FakeManager:
        def __init__(self, sw):
            pass

        def run(self, number, args, project=None):
            calls.append(("launch", number, project))

    monkeypatch.setattr("claude_swap.tui.app.CswapApp", FakeApp)
    monkeypatch.setattr("claude_swap.session.SessionManager", FakeManager)

    tui.run(switcher)

    assert calls == ["app.run", ("launch", "2", str(project))]
