"""Tests for the configuration system."""

from __future__ import annotations

import tempfile

import pytest

from mistralcli.config import (
    AppConfig,
    BackendKind,
    ContextConfig,
    ConversationsConfig,
    GenerationConfig,
    LocalBackendConfig,
    LoggingConfig,
    MCPConfig,
    RegistryConfig,
    RemoteBackendConfig,
    ResumePolicy,
    UIConfig,
)


class TestAppConfigDefaults:
    """Test default configuration values."""

    def test_default_config_has_expected_values(self):
        """Default config should have sensible defaults."""

        config = AppConfig.default()

        assert config.version == "3.4"
        assert config.backend == BackendKind.LOCAL

        # Local backend
        assert isinstance(config.local, LocalBackendConfig)
        assert config.local.api_key == "local-test"
        assert (
            config.local.model_id == "unsloth/Mistral-Small-4-119B-2603-GGUF:UD-Q5_K_XL"
        )
        assert config.local.server_url == "http://127.0.0.1:8080"
        assert config.local.timeout_ms == 300_000

        # Generation
        assert isinstance(config.generation, GenerationConfig)
        assert config.generation.temperature == 0.3
        assert config.generation.top_p == 0.95
        assert config.generation.prompt_mode == "reasoning"
        assert config.generation.max_tokens is None

        # Remote backend
        assert isinstance(config.remote, RemoteBackendConfig)
        assert config.remote.model_id == "mistral-small-latest"
        assert config.remote.timeout_ms == 300_000

        # Conversations
        assert isinstance(config.conversations, ConversationsConfig)
        assert config.conversations.enabled is False
        assert config.conversations.store is True
        assert config.conversations.resume_policy == ResumePolicy.LAST

        # Context
        assert isinstance(config.context, ContextConfig)
        assert config.context.auto_compact is True
        assert config.context.threshold == 0.9
        assert config.context.reserve_tokens == 8_192
        assert config.context.local_window_tokens == 262_144
        assert config.context.remote_window_tokens == 256_000
        assert config.context.keep_recent_turns == 6
        assert config.context.summary_max_tokens == 2_048

        # Logging
        assert isinstance(config.logging, LoggingConfig)
        assert config.logging.debug_enabled is True
        assert config.logging.retention_days == 2
        assert config.logging.file_name == "mistralcli.log"

        # MCP
        assert isinstance(config.mcp, MCPConfig)
        assert config.mcp.enabled is True

        # UI
        assert isinstance(config.ui, UIConfig)
        assert config.ui.stream_enabled is True
        assert config.ui.show_reasoning is True
        assert config.ui.show_thinking is True
        assert config.ui.max_tool_rounds == 20

        # Registry
        assert isinstance(config.registry, RegistryConfig)
        assert config.registry.conversation_index_path is None

    def test_default_config_system_prompt(self):
        """Default system prompt should be set."""
        from mistralcli.config import AppConfig

        config = AppConfig.default()
        assert "You are a helpful assistant" in config.ui.system_prompt


class TestConfigValidation:
    """Test configuration validation."""

    def test_invalid_temperature_too_high(self):
        """Temperature must be between 0 and 1."""
        from mistralcli.config import GenerationConfig

        with pytest.raises(ValueError, match="temperature must be between 0 and 1"):
            GenerationConfig(temperature=1.5)

    def test_invalid_temperature_too_low(self):
        """Temperature must be between 0 and 1."""
        from mistralcli.config import GenerationConfig

        with pytest.raises(ValueError, match="temperature must be between 0 and 1"):
            GenerationConfig(temperature=-0.1)

    def test_invalid_top_p(self):
        """Top-p must be between 0 and 1."""
        from mistralcli.config import GenerationConfig

        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            GenerationConfig(top_p=1.5)

    def test_invalid_timeout_negative(self):
        """Timeout must be positive."""
        from mistralcli.config import LocalBackendConfig

        with pytest.raises(ValueError, match="timeout_ms must be positive"):
            LocalBackendConfig(timeout_ms=-1)

    def test_invalid_timeout_zero(self):
        """Timeout must be positive."""
        from mistralcli.config import LocalBackendConfig

        with pytest.raises(ValueError, match="timeout_ms must be positive"):
            LocalBackendConfig(timeout_ms=0)

    def test_invalid_resume_policy(self):
        """Resume policy must be a valid enum value."""
        with pytest.raises(ValueError, match="resume_policy must be a ResumePolicy"):
            ConversationsConfig(resume_policy="invalid")  # type: ignore[arg-type]

    def test_invalid_retention_days(self):
        """Retention days must be at least 1."""
        from mistralcli.config import LoggingConfig

        with pytest.raises(ValueError, match="retention_days must be at least 1"):
            LoggingConfig(retention_days=0)

    def test_invalid_max_tool_rounds(self):
        """Max tool rounds must be at least 1."""
        from mistralcli.config import UIConfig

        with pytest.raises(ValueError, match="max_tool_rounds must be at least 1"):
            UIConfig(max_tool_rounds=0)

    def test_invalid_context_reserve_tokens(self):
        """Reserve tokens must be non-negative."""
        from mistralcli.config import ContextConfig

        with pytest.raises(ValueError, match="reserve_tokens must be non-negative"):
            ContextConfig(reserve_tokens=-1)

    def test_invalid_local_window_tokens(self):
        """Local window tokens must be at least 1024."""
        from mistralcli.config import ContextConfig

        with pytest.raises(
            ValueError, match="local_window_tokens must be at least 1024"
        ):
            ContextConfig(local_window_tokens=512)

    def test_threshold_normalization(self):
        """Threshold should be normalized to 0.1-0.99 range."""
        from mistralcli.config import ContextConfig

        # Test threshold > 1 gets normalized
        config = ContextConfig(threshold=0.95)
        normalized = config.normalized()
        assert normalized.threshold == 0.95

        # Test threshold as percentage
        config = ContextConfig(threshold=90)  # 90%
        normalized = config.normalized()
        assert normalized.threshold == 0.9


class TestYAMLConfigLoading:
    """Test YAML configuration file loading."""

    def test_load_yaml_config(self):
        """YAML config files should load correctly."""
        pytest.importorskip("yaml")

        from mistralcli.config import AppConfig

        yaml_content = """
version: "3.4"
backend: local
local:
  model_id: "test-model"
generation:
  temperature: 0.5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = AppConfig.load(f.name)

        assert config.version == "3.4"
        assert config.local.model_id == "test-model"
        assert config.generation.temperature == 0.5

    def test_load_yaml_with_all_sections(self):
        """YAML config with all sections should load correctly."""
        pytest.importorskip("yaml")

        from mistralcli.config import AppConfig

        yaml_content = """
version: "3.4"
backend: remote
local:
  api_key: "custom-key"
  model_id: "custom-model"
  server_url: "http://custom:8080"
  timeout_ms: 60000
generation:
  temperature: 0.7
  top_p: 0.9
  prompt_mode: null
  max_tokens: 100
remote:
  model_id: "mistral-medium-3.5"
  timeout_ms: 120000
conversations:
  enabled: true
  store: false
  resume_policy: "prompt"
context:
  auto_compact: false
  threshold: 0.8
  reserve_tokens: 4096
  local_window_tokens: 131072
  remote_window_tokens: 128000
  keep_recent_turns: 4
  summary_max_tokens: 1024
ui:
  stream_enabled: false
  show_reasoning: false
  show_thinking: false
  max_tool_rounds: 10
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = AppConfig.load(f.name)

        assert config.backend.value == "remote"
        assert config.local.api_key == "custom-key"
        assert config.generation.temperature == 0.7
        assert config.conversations.enabled is True
        assert config.conversations.resume_policy.value == "prompt"
        assert config.context.auto_compact is False
        assert config.ui.stream_enabled is False


class TestJSONConfigLoading:
    """Test JSON configuration file loading."""

    def test_load_json_config(self):
        """JSON config files should load correctly."""
        from mistralcli.config import AppConfig

        json_content = '{"version": "3.4", "local": {"model_id": "json-model"}}'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_content)
            f.flush()
            config = AppConfig.load(f.name)

        assert config.version == "3.4"
        assert config.local.model_id == "json-model"

    def test_load_json_with_nested_objects(self):
        """JSON config with nested objects should load correctly."""
        from mistralcli.config import AppConfig

        json_content = """{
  "version": "3.4",
  "local": {
    "api_key": "json-key",
    "model_id": "json-model",
    "server_url": "http://json:8080",
    "timeout_ms": 90000
  },
  "generation": {
    "temperature": 0.4,
    "top_p": 0.85,
    "prompt_mode": "reasoning",
    "max_tokens": null
  }
}"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_content)
            f.flush()
            config = AppConfig.load(f.name)

        assert config.local.api_key == "json-key"
        assert config.local.timeout_ms == 90000
        assert config.generation.temperature == 0.4
        assert config.generation.max_tokens is None


class TestConfigSaving:
    """Test configuration file saving."""

    def test_save_yaml_config(self):
        """YAML config should be saved correctly."""
        yaml = pytest.importorskip("yaml")

        from mistralcli.config import AppConfig

        config = AppConfig.default()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config.save(f.name)

        # Verify the file was created and is valid YAML
        with open(f.name) as rf:
            saved_data = yaml.safe_load(rf)

        assert saved_data["version"] == "3.4"
        assert "local" in saved_data
        assert "generation" in saved_data


class TestConfigPathDiscovery:
    """Test configuration file path discovery."""

    def test_explicit_path(self, tmp_path):
        """Explicit path should be used when provided."""
        from mistralcli.config import AppConfig

        config_path = tmp_path / "custom.yaml"
        config_path.write_text("version: '3.4'\nlocal:\n  model_id: 'test'\n")

        config = AppConfig.load(str(config_path))
        assert config.local.model_id == "test"

    def test_explicit_path_not_found(self, tmp_path):
        """FileNotFoundError should be raised for non-existent explicit path."""
        from mistralcli.config import AppConfig

        with pytest.raises(FileNotFoundError):
            AppConfig.load(str(tmp_path / "nonexistent.yaml"))


class TestConfigMerging:
    """Test configuration merging with precedence."""

    def test_cli_args_override_file(self):
        """CLI arguments should override config file values."""
        pytest.importorskip("yaml")
        import argparse

        from mistralcli.cli_config import _merge_config_with_args
        from mistralcli.config import AppConfig

        # Create a config file
        yaml_content = """
version: "3.4"
local:
  model_id: "file-model"
  server_url: "http://file:8080"
generation:
  temperature: 0.5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            app_config = AppConfig.load(f.name)

        # Test 1: CLI arg is None, so file value should be used
        args1 = argparse.Namespace(
            api_key=None,
            model=None,
            server_url=None,
            timeout_ms=None,
            temperature=None,
            top_p=None,
            max_tokens=None,
            prompt_mode=None,
            conversations=None,
            conversation_store=None,
            conversation_resume=None,
            auto_compact=None,
            compact_threshold=None,
            context_reserve_tokens=None,
            context_local_window_tokens=None,
            context_remote_window_tokens=None,
            context_keep_turns=None,
            context_summary_max_tokens=None,
            system_prompt=None,
        )

        config1, generation1, system_prompt1, conversations1, context1 = (
            _merge_config_with_args(args1, app_config)
        )

        # Config file value should be used since CLI arg is None
        assert config1.model_id == "file-model"

        # Test 2: CLI arg overrides file value
        args2 = argparse.Namespace(
            api_key=None,
            model="cli-model",
            server_url=None,
            timeout_ms=None,
            temperature=None,
            top_p=None,
            max_tokens=None,
            prompt_mode=None,
            conversations=None,
            conversation_store=None,
            conversation_resume=None,
            auto_compact=None,
            compact_threshold=None,
            context_reserve_tokens=None,
            context_local_window_tokens=None,
            context_remote_window_tokens=None,
            context_keep_turns=None,
            context_summary_max_tokens=None,
            system_prompt=None,
        )

        config2, generation2, system_prompt2, conversations2, context2 = (
            _merge_config_with_args(args2, app_config)
        )
        assert config2.model_id == "cli-model"

    def test_cli_args_override_when_not_none(self):
        """CLI arguments should override when they are not None."""
        pytest.importorskip("yaml")
        import argparse

        from mistralcli.cli_config import _merge_config_with_args
        from mistralcli.config import AppConfig

        # Create a config file
        yaml_content = """
version: "3.4"
local:
  model_id: "file-model"
  timeout_ms: 100000
generation:
  temperature: 0.5
  top_p: 0.95
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            app_config = AppConfig.load(f.name)

        # Create args with overrides
        args = argparse.Namespace(
            api_key=None,
            model="cli-model",
            server_url=None,
            timeout_ms=200000,
            temperature=0.7,
            top_p=None,
            max_tokens=None,
            prompt_mode=None,
            conversations=None,
            conversation_store=None,
            conversation_resume=None,
            auto_compact=None,
            compact_threshold=None,
            context_reserve_tokens=None,
            context_local_window_tokens=None,
            context_remote_window_tokens=None,
            context_keep_turns=None,
            context_summary_max_tokens=None,
            system_prompt=None,
        )

        config, generation, system_prompt, conversations, context = (
            _merge_config_with_args(args, app_config)
        )

        assert config.model_id == "cli-model"
        assert config.timeout_ms == 200000
        assert generation.temperature == 0.7
        # top_p should come from file since CLI arg is None
        assert generation.top_p == 0.95

    def test_env_var_overrides_file(self, monkeypatch):
        """Environment variables should override config file values."""
        pytest.importorskip("yaml")
        import argparse

        from mistralcli.cli_config import _merge_config_with_args
        from mistralcli.config import AppConfig

        yaml_content = """
version: "3.4"
local:
  model_id: "file-model"
generation:
  temperature: 0.5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            app_config = AppConfig.load(f.name)

        monkeypatch.setenv("MISTRAL_LOCAL_MODEL_ID", "env-model")
        monkeypatch.setenv("MISTRAL_LOCAL_TEMPERATURE", "0.8")

        args = argparse.Namespace(
            api_key=None,
            model=None,
            server_url=None,
            timeout_ms=None,
            temperature=None,
            top_p=None,
            max_tokens=None,
            prompt_mode=None,
            conversations=None,
            conversation_store=None,
            conversation_resume=None,
            auto_compact=None,
            compact_threshold=None,
            context_reserve_tokens=None,
            context_local_window_tokens=None,
            context_remote_window_tokens=None,
            context_keep_turns=None,
            context_summary_max_tokens=None,
            system_prompt=None,
        )

        config, generation, system_prompt, conversations, context = (
            _merge_config_with_args(args, app_config)
        )

        # Env var should override file value
        assert config.model_id == "env-model"
        assert generation.temperature == 0.8


class TestToDict:
    """Test configuration serialization to dictionary."""

    def test_to_dict_contains_all_sections(self):
        """to_dict should include all configuration sections."""
        from mistralcli.config import AppConfig

        config = AppConfig.default()
        data = config.to_dict()

        assert "version" in data
        assert "backend" in data
        assert "local" in data
        assert "generation" in data
        assert "remote" in data
        assert "conversations" in data
        assert "context" in data
        assert "logging" in data
        assert "mcp" in data
        assert "ui" in data
        assert "registry" in data

    def test_to_dict_nested_objects(self):
        """Nested objects should be properly serialized."""
        from mistralcli.config import AppConfig

        config = AppConfig.default()
        data = config.to_dict()

        assert isinstance(data["local"], dict)
        assert isinstance(data["generation"], dict)
        assert isinstance(data["context"], dict)

    def test_to_dict_handles_paths_and_enums(self):
        """Path objects and Enums should be converted to strings."""
        from mistralcli.config import AppConfig

        config = AppConfig.default()
        data = config.to_dict()

        # Check that backend (an Enum) is converted to string
        assert isinstance(data["backend"], str)
        assert data["backend"] == "local"

        # Check that nested Enums are converted
        assert isinstance(data["conversations"]["resume_policy"], str)


class TestFromEnv:
    """Test loading configuration from environment variables."""

    def test_default_creates_valid_config(self):
        """default should create a valid configuration."""
        from mistralcli.config import AppConfig

        config = AppConfig.default()

        assert config.version == "3.4"
        assert config.backend.value == "local"
        assert config.local is not None
        assert config.generation is not None
        assert config.remote is not None
        assert config.conversations is not None
        assert config.context is not None
        assert config.logging is not None
        assert config.mcp is not None
        assert config.ui is not None
        assert config.registry is not None

    def test_local_backend_from_env(self):
        """LocalBackendConfig.from_env should work."""
        from mistralcli.config import LocalBackendConfig

        config = LocalBackendConfig.from_env()
        assert config.api_key == "local-test"
        assert config.model_id == "unsloth/Mistral-Small-4-119B-2603-GGUF:UD-Q5_K_XL"
        assert config.server_url == "http://127.0.0.1:8080"
        assert config.timeout_ms == 300_000

    def test_generation_from_env(self):
        """GenerationConfig.from_env should work."""
        from mistralcli.config import GenerationConfig

        config = GenerationConfig.from_env()
        assert config.temperature == 0.3
        assert config.top_p == 0.95


class TestGenerateConfigCLI:
    """Test --generate-config CLI flag."""

    def test_generate_config_to_stdout(self):
        """--generate-config - should print config to stdout."""
        import io

        from tests.cli_support import FakeClient, FakeStdin, main

        output = io.StringIO()
        exit_code = main(
            ["--generate-config", "-"],
            stdin=FakeStdin(""),
            stdout=output,
            client_factory=lambda _config: FakeClient(),
        )

        assert exit_code == 0
        text = output.getvalue()
        assert "version" in text
        assert "local" in text
        assert "generation" in text

    def test_generate_config_to_file(self, tmp_path):
        """--generate-config PATH should save config to the given path."""
        import io

        from tests.cli_support import FakeClient, FakeStdin, main

        config_path = tmp_path / "out.yaml"
        output = io.StringIO()
        exit_code = main(
            ["--generate-config", str(config_path)],
            stdin=FakeStdin(""),
            stdout=output,
            client_factory=lambda _config: FakeClient(),
        )

        assert exit_code == 0
        assert config_path.exists()
        content = config_path.read_text()
        assert "version" in content
        assert "local" in content

    def test_generate_config_no_arg_uses_default(self, tmp_path, monkeypatch):
        """--generate-config without arg should save to default location."""
        import io

        from tests.cli_support import FakeClient, FakeStdin, main

        # Redirect default config dir to tmp_path
        monkeypatch.setattr("mistralcli.config.DEFAULT_CONFIG_DIR", tmp_path)
        output = io.StringIO()
        exit_code = main(
            ["--generate-config"],
            stdin=FakeStdin(""),
            stdout=output,
            client_factory=lambda _config: FakeClient(),
        )

        assert exit_code == 0
        assert (tmp_path / "config.yaml").exists()


class TestConfigPathCLI:
    """Test --config-path CLI flag."""

    def test_config_path_loads_file(self, tmp_path):
        """--config-path should load the specified config file."""
        pytest.importorskip("yaml")
        import io

        from tests.cli_support import FakeClient, FakeStdin, main

        config_file = tmp_path / "custom.yaml"
        config_file.write_text("version: '3.4'\nlocal:\n  model_id: custom-model\n")

        output = io.StringIO()
        exit_code = main(
            ["--config-path", str(config_file), "--print-defaults"],
            stdin=FakeStdin(""),
            stdout=output,
            client_factory=lambda _config: FakeClient(),
        )

        assert exit_code == 0
        assert "custom-model" in output.getvalue()

    def test_config_path_nonexistent_still_works(self, tmp_path):
        """--config-path with nonexistent file should not crash."""
        import io

        from tests.cli_support import FakeClient, FakeStdin, main

        output = io.StringIO()
        exit_code = main(
            ["--config-path", str(tmp_path / "nope.yaml"), "--print-defaults"],
            stdin=FakeStdin(""),
            stdout=output,
            client_factory=lambda _config: FakeClient(),
        )

        # Should still work with defaults
        assert exit_code == 0


class TestConfigSearchOrder:
    """Test configuration file search order discovery."""

    def test_env_var_path_takes_priority(self, tmp_path, monkeypatch):
        """$MISTRAL_CONFIG_PATH should be found before standard locations."""
        pytest.importorskip("yaml")
        from mistralcli.config import AppConfig

        env_config = tmp_path / "env.yaml"
        env_config.write_text("version: '3.4'\nlocal:\n  model_id: env-path-model\n")

        monkeypatch.setenv("MISTRAL_CONFIG_PATH", str(env_config))
        config = AppConfig.load()
        assert config.local.model_id == "env-path-model"

    def test_xdg_config_found(self, tmp_path, monkeypatch):
        """~/.config/mistralcli/config.yaml should be found."""
        pytest.importorskip("yaml")
        from mistralcli.config import AppConfig

        # Create a config at the default location (mocked)
        fake_default = tmp_path / "config.yaml"
        fake_default.write_text("version: '3.4'\nlocal:\n  model_id: xdg-model\n")

        monkeypatch.delenv("MISTRAL_CONFIG_PATH", raising=False)
        monkeypatch.setattr("mistralcli.config.DEFAULT_CONFIG_FILE", fake_default)
        config = AppConfig.load()
        assert config.local.model_id == "xdg-model"

    def test_no_config_returns_defaults(self, tmp_path, monkeypatch):
        """When no config file exists, defaults should be used."""
        from mistralcli.config import AppConfig

        monkeypatch.delenv("MISTRAL_CONFIG_PATH", raising=False)
        # Point all search locations to nonexistent paths
        monkeypatch.setattr(
            "mistralcli.config.DEFAULT_CONFIG_FILE", tmp_path / "nope.yaml"
        )
        monkeypatch.chdir(tmp_path)
        config = AppConfig.load()
        assert (
            config.local.model_id == "unsloth/Mistral-Small-4-119B-2603-GGUF:UD-Q5_K_XL"
        )

    def test_json_format_discovered(self, tmp_path, monkeypatch):
        """JSON config files should be discovered."""
        from mistralcli.config import AppConfig

        json_config = tmp_path / ".mistralcli.json"
        json_config.write_text(
            '{"version": "3.4", "local": {"model_id": "json-discovered"}}'
        )

        monkeypatch.delenv("MISTRAL_CONFIG_PATH", raising=False)
        monkeypatch.setattr(
            "mistralcli.config.DEFAULT_CONFIG_FILE", tmp_path / "nope.yaml"
        )
        # Patch home to tmp_path so ~/.mistralcli.json is found
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        config = AppConfig.load()
        assert config.local.model_id == "json-discovered"


class TestEnvVarOverridesConfigFile:
    """Test that environment variables override config file values."""

    def test_context_env_overrides_file(self, monkeypatch):
        """Context env vars should override config file context values."""
        pytest.importorskip("yaml")
        import argparse

        from mistralcli.cli_config import _merge_config_with_args
        from mistralcli.config import AppConfig

        yaml_content = """
version: "3.4"
context:
  auto_compact: true
  threshold: 0.8
  reserve_tokens: 4096
  keep_recent_turns: 4
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            app_config = AppConfig.load(f.name)

        monkeypatch.setenv("MISTRAL_CONTEXT_THRESHOLD", "0.95")
        monkeypatch.setenv("MISTRAL_CONTEXT_KEEP_RECENT_TURNS", "10")

        args = argparse.Namespace(
            api_key=None,
            model=None,
            server_url=None,
            timeout_ms=None,
            temperature=None,
            top_p=None,
            max_tokens=None,
            prompt_mode=None,
            conversations=None,
            conversation_store=None,
            conversation_resume=None,
            auto_compact=None,
            compact_threshold=None,
            context_reserve_tokens=None,
            context_local_window_tokens=None,
            context_remote_window_tokens=None,
            context_keep_turns=None,
            context_summary_max_tokens=None,
            system_prompt=None,
        )

        _, _, _, _, context = _merge_config_with_args(args, app_config)

        # Env vars override file
        assert context.threshold == 0.95
        assert context.keep_recent_turns == 10
        # File value used when env var not set
        assert context.reserve_tokens == 4096

    def test_conversations_env_overrides_file(self, monkeypatch):
        """Conversations env vars should override config file values."""
        pytest.importorskip("yaml")
        import argparse

        from mistralcli.cli_config import _merge_config_with_args
        from mistralcli.config import AppConfig

        yaml_content = """
version: "3.4"
conversations:
  enabled: false
  store: true
  resume_policy: "last"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            app_config = AppConfig.load(f.name)

        monkeypatch.setenv("MISTRAL_CONVERSATIONS", "true")
        monkeypatch.setenv("MISTRAL_CONVERSATION_RESUME", "new")

        args = argparse.Namespace(
            api_key=None,
            model=None,
            server_url=None,
            timeout_ms=None,
            temperature=None,
            top_p=None,
            max_tokens=None,
            prompt_mode=None,
            conversations=None,
            conversation_store=None,
            conversation_resume=None,
            auto_compact=None,
            compact_threshold=None,
            context_reserve_tokens=None,
            context_local_window_tokens=None,
            context_remote_window_tokens=None,
            context_keep_turns=None,
            context_summary_max_tokens=None,
            system_prompt=None,
        )

        _, _, _, conversations, _ = _merge_config_with_args(args, app_config)

        # Env vars override file
        assert conversations.enabled is True
        assert conversations.resume_policy == "new"
        # File value used when env var not set
        assert conversations.store is True


class TestFullPrecedenceChain:
    """Test CLI args > env vars > config file > defaults."""

    def test_cli_overrides_env_overrides_file(self, monkeypatch):
        """Full precedence: CLI > env > file > defaults."""
        pytest.importorskip("yaml")
        import argparse

        from mistralcli.cli_config import _merge_config_with_args
        from mistralcli.config import AppConfig

        yaml_content = """
version: "3.4"
local:
  model_id: "file-model"
  server_url: "http://file:8080"
  timeout_ms: 100000
generation:
  temperature: 0.5
  top_p: 0.8
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            app_config = AppConfig.load(f.name)

        # Env var overrides file for model_id and temperature
        monkeypatch.setenv("MISTRAL_LOCAL_MODEL_ID", "env-model")
        monkeypatch.setenv("MISTRAL_LOCAL_TEMPERATURE", "0.7")

        # CLI overrides env for model_id
        args = argparse.Namespace(
            api_key=None,
            model="cli-model",  # CLI override
            server_url=None,
            timeout_ms=None,
            temperature=None,  # No CLI override, env wins
            top_p=None,  # No CLI or env override, file wins
            max_tokens=None,
            prompt_mode=None,
            conversations=None,
            conversation_store=None,
            conversation_resume=None,
            auto_compact=None,
            compact_threshold=None,
            context_reserve_tokens=None,
            context_local_window_tokens=None,
            context_remote_window_tokens=None,
            context_keep_turns=None,
            context_summary_max_tokens=None,
            system_prompt=None,
        )

        config, generation, _, _, _ = _merge_config_with_args(args, app_config)

        # CLI wins over env
        assert config.model_id == "cli-model"
        # Env wins over file
        assert generation.temperature == 0.7
        # File wins over default (default is 0.95)
        assert generation.top_p == 0.8
        # File value used when no env or CLI override
        assert config.server_url == "http://file:8080"
        assert config.timeout_ms == 100000
