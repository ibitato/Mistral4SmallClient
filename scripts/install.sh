#!/usr/bin/env bash
# Install mistralcli through uv tool install.
# Usage:
#   ./scripts/install.sh
#   ./scripts/install.sh /path/to/mistralcli-<version>-py3-none-any.whl
#   ./scripts/install.sh https://.../mistralcli-<version>-py3-none-any.whl
#   ./scripts/install.sh v3.2.1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
HOME_DIR="$(cd ~ && pwd)"
BIN_DIR="${HOME_DIR}/.local/bin"
LEGACY_INSTALL_DIR="${HOME_DIR}/.local/mistralcli"
LEGACY_WRAPPER="${BIN_DIR}/mistralcli"
GITHUB_REPO="ibitato/MistralClient"
UV_INSTALL_URL="https://astral.sh/uv/install.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

die() {
    log_error "$1"
    exit 1
}

ensure_not_root() {
    if [ "$(id -u)" -eq 0 ]; then
        die "Do not run this script as root. Run it as a regular user."
    fi
}

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return
    fi
    log_info "uv not found. Installing uv..."
    curl -fsSL "$UV_INSTALL_URL" | sh
    if [ -x "${HOME_DIR}/.local/bin/uv" ]; then
        echo "${HOME_DIR}/.local/bin/uv"
        return
    fi
    die "Failed to install uv. Install it manually from ${UV_INSTALL_URL}."
}

latest_local_wheel() {
    find "${REPO_ROOT}/dist" -maxdepth 1 -type f -name 'mistralcli-*.whl' | sort | tail -n 1
}

resolve_package_spec() {
    local input="${1:-}"
    if [ -z "$input" ]; then
        local wheel
        wheel="$(latest_local_wheel || true)"
        if [ -n "$wheel" ]; then
            echo "$wheel"
            return
        fi
        die "No local wheel found in ${REPO_ROOT}/dist. Pass a wheel path, release URL, or tag like v3.0.0."
    fi

    if [ -f "$input" ]; then
        echo "$input"
        return
    fi

    case "$input" in
        http://*|https://*)
            echo "$input"
            return
            ;;
        v*)
            local version="${input#v}"
            echo "https://github.com/${GITHUB_REPO}/releases/download/${input}/mistralcli-${version}-py3-none-any.whl"
            return
            ;;
        *.whl)
            echo "$input"
            return
            ;;
    esac

    die "Unsupported install target: ${input}. Pass a wheel path, release URL, or tag like v3.0.0."
}

cleanup_legacy_install() {
    local uv_cmd="$1"

    if "$uv_cmd" tool uninstall mistralcli >/dev/null 2>&1; then
        log_info "Removed existing uv tool install for mistralcli."
    fi

    if [ -f "$LEGACY_WRAPPER" ] && grep -q "${LEGACY_INSTALL_DIR}/bin/python" "$LEGACY_WRAPPER"; then
        rm -f "$LEGACY_WRAPPER"
        log_info "Removed legacy wrapper ${LEGACY_WRAPPER}."
    fi

    if [ -d "$LEGACY_INSTALL_DIR" ]; then
        rm -rf "$LEGACY_INSTALL_DIR"
        log_info "Removed legacy virtual environment ${LEGACY_INSTALL_DIR}."
    fi
}

verify_install() {
    local tool_path="${BIN_DIR}/mistralcli"
    if [ ! -x "$tool_path" ]; then
        die "uv reported success but ${tool_path} was not created."
    fi

    "$tool_path" --version
    "$tool_path" --print-defaults >/dev/null

    case ":$PATH:" in
        *":${BIN_DIR}:"*) ;;
        *)
            log_warn "${BIN_DIR} is not on PATH in this shell."
            log_warn "Run ${tool_path} directly or add ${BIN_DIR} to PATH."
            ;;
    esac
}

main() {
    ensure_not_root
    mkdir -p "$BIN_DIR"

    local uv_cmd
    uv_cmd="$(ensure_uv)"

    local package_spec
    package_spec="$(resolve_package_spec "${1:-}")"

    log_info "Installing mistralcli from ${package_spec}"
    cleanup_legacy_install "$uv_cmd"
    "$uv_cmd" tool install --force "$package_spec"
    verify_install
    log_info "mistralcli is now managed through uv."
}

main "$@"
