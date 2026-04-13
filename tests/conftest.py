from __future__ import annotations

import pytest
from mistralai import Mistral

from mistral4cli.local_mistral import (
    DEFAULT_MODEL_ID,
    LocalMistralConfig,
    build_client,
    get_health,
    list_models,
)


@pytest.fixture(scope="session")
def local_config() -> LocalMistralConfig:
    return LocalMistralConfig.from_env()


@pytest.fixture(scope="session")
def local_health(local_config: LocalMistralConfig) -> dict[str, object]:
    health = get_health(local_config.server_url)
    assert health.get("status") == "ok"
    return health


@pytest.fixture(scope="session")
def local_models(local_config: LocalMistralConfig) -> dict[str, object]:
    models = list_models(local_config.server_url)
    data = models.get("data")
    assert isinstance(data, list)
    assert any(
        isinstance(item, dict) and item.get("id") == DEFAULT_MODEL_ID for item in data
    )
    return models


@pytest.fixture(scope="session")
def local_client(
    local_health: dict[str, object], local_config: LocalMistralConfig
) -> Mistral:
    del local_health
    return build_client(local_config)
