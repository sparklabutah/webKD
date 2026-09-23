"""Shared, entirely synthetic test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def isolated_teacher(monkeypatch):
    monkeypatch.setenv("WEBKD_CORRECTOR_MODEL", "mock-teacher")


@pytest.fixture
def tmp(tmp_path):
    return str(tmp_path)
