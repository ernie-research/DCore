#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_NAME="dcore"
# Venv lives at $HOME/.venvs/$PROJECT_NAME (override via UV_ENV_PATH for shared $HOME).

# --- CUDA variant: accept argument or auto-detect ---
# Usage: bash build_env.sh [cu12|cu13]
CUDA_VARIANT="${1:-}"

if [[ -z "$CUDA_VARIANT" ]]; then
    # Auto-detect based on CUDA toolkit major version
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_MAJOR=$(nvcc --version | grep -oP 'release \K[0-9]+')
    elif [[ -f /usr/local/cuda/version.json ]]; then
        CUDA_MAJOR=$(python3 -c "import json; print(json.load(open('/usr/local/cuda/version.json'))['cuda']['version'].split('.')[0])")
    else
        echo "Cannot detect CUDA version. Specify cu12 or cu13 as argument."
        exit 1
    fi

    if [[ "$CUDA_MAJOR" -ge 13 ]]; then
        CUDA_VARIANT="cu13"
    else
        CUDA_VARIANT="cu12"
    fi
fi

case "$CUDA_VARIANT" in
    cu12)
        export TORCH_CUDA_ARCH_LIST="9.0"
        export CMAKE_CUDA_ARCHITECTURES="90"
        ;;
    cu13)
        export TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0"
        export CMAKE_CUDA_ARCHITECTURES="90;100;120"
        ;;
    *)
        echo "Unknown CUDA variant: $CUDA_VARIANT (expected cu12 or cu13)"
        exit 1
        ;;
esac

echo "Building for CUDA variant: $CUDA_VARIANT"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "  CMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES"

export MAX_JOBS=64

# (If your network requires a proxy, export http_proxy/https_proxy in your shell first.)

# The official static.rust-lang.org is very slow (a full toolchain can take hours);
# the Tsinghua mirror runs at ~50MB/s.
export RUSTUP_DIST_SERVER="https://mirrors.tuna.tsinghua.edu.cn/rustup"
export RUSTUP_UPDATE_ROOT="https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup"

UV_ENV_PATH="$HOME/.venvs/$PROJECT_NAME"
UV_CACHE_PATH="$HOME/.cache/uv"
BASHRC="$HOME/.bashrc"

BEGIN_MARKER="# >>> SpecForge uv environment >>>"
END_MARKER="# <<< SpecForge uv environment <<<"

# Remove any previous config block to avoid duplicate entries.
if [[ -f "$BASHRC" ]]; then
    sed -i \
        "/^${BEGIN_MARKER}$/,/^${END_MARKER}$/d" \
        "$BASHRC"
fi

cat >>"$BASHRC" <<EOF
# >>> SpecForge uv environment >>>
export UV_PROJECT_ENVIRONMENT="\$HOME/.venvs/${PROJECT_NAME}"
export UV_CACHE_DIR="\$HOME/.cache/uv"
export PATH="\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH"
export UV_LINK_MODE="copy"
# <<< SpecForge uv environment <<<
EOF

# Writing to .bashrc does not affect the parent shell, but these exports apply to
# this script and its child processes.
export UV_PROJECT_ENVIRONMENT="$UV_ENV_PATH"
export UV_CACHE_DIR="$UV_CACHE_PATH"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

mkdir -p "$UV_PROJECT_ENVIRONMENT" "$UV_CACHE_DIR"

printf 'UV_PROJECT_ENVIRONMENT=%s\n' "$UV_PROJECT_ENVIRONMENT"
printf 'UV_CACHE_DIR=%s\n' "$UV_CACHE_DIR"

INSTALL_SCRIPT="third_party/sglang/scripts/ci/utils/install_rust_protoc.sh"

if [[ -f "$HOME/.cargo/env" ]]; then
    source "$HOME/.cargo/env"
fi

RUST_OFFLINE_TARBALL="cache/wheels/rust-1.97.1-x86_64-unknown-linux-gnu.tar.gz"

# Check if rust actually works (rustup shim may exist but toolchain is broken)
RUST_OK=false
if command -v rustc >/dev/null 2>&1 && rustc --version >/dev/null 2>&1; then
    RUST_OK=true
fi

if [[ "$RUST_OK" == "false" ]]; then
    if [[ -f "$RUST_OFFLINE_TARBALL" ]]; then
        echo "Installing Rust from offline tarball: $RUST_OFFLINE_TARBALL"
        tar xzf "$RUST_OFFLINE_TARBALL" -C /tmp
        /tmp/rust-1.97.1-x86_64-unknown-linux-gnu/install.sh --prefix="$HOME/.cargo" --disable-ldconfig
        rm -rf /tmp/rust-1.97.1-x86_64-unknown-linux-gnu
    else
        echo "Rust is missing; installing via $INSTALL_SCRIPT..."
        bash "$INSTALL_SCRIPT"
    fi
fi

if ! command -v protoc >/dev/null 2>&1; then
    echo "protoc is missing; installing build tools..."
    bash "$INSTALL_SCRIPT"
fi

# The install script runs in a subshell, so PATH must be re-set in the current shell.
export PATH="$HOME/.cargo/bin:$PATH"

# The image ships a rustup shim without a usable toolchain, and install_rustup.sh only
# checks that the shim exists before declaring it "installed" and skipping, leaving rustc
# permanently unusable. Force-install a toolchain here and make it the default.
if command -v rustup >/dev/null 2>&1 && ! rustc --version >/dev/null 2>&1; then
    rustup toolchain install stable --force
    rustup default stable
fi

# The official static.crates.io source is very slow; the USTC mirror runs at ~900KB/s.
# sgl-kernel has many Rust dependencies, so without a mirror the build stalls on
# downloading crates.
CARGO_CONFIG="$HOME/.cargo/config.toml"
if [[ -f "$CARGO_CONFIG" ]] && grep -q 'mirrors.ustc.edu.cn' "$CARGO_CONFIG"; then
    echo "cargo mirror already configured: $CARGO_CONFIG"
elif [[ -f "$CARGO_CONFIG" ]] && grep -q '^\[source\.' "$CARGO_CONFIG"; then
    # Another source is already configured; appending would create duplicate keys and
    # make cargo error out, so leave it for manual handling.
    echo "WARNING: $CARGO_CONFIG already defines [source.*]; skipping crates.io mirror setup."
else
    mkdir -p "$HOME/.cargo"
    cat >>"$CARGO_CONFIG" <<'EOF'
[source.crates-io]
replace-with = "ustc"

[source.ustc]
registry = "sparse+https://mirrors.ustc.edu.cn/crates.io-index/"
EOF
    echo "cargo crates.io mirror written to $CARGO_CONFIG"
fi

command -v rustc >/dev/null 2>&1
command -v cargo >/dev/null 2>&1
command -v protoc >/dev/null 2>&1

rustc --version
cargo --version
protoc --version


# sgl-kernel's CMake uses FetchContent to download GitHub source tarballs on the fly
# (cutlass/flashinfer/sgl-attn, etc.). uv builds in a fresh /tmp directory every time,
# so downloads are never reused and an occasional proxy drop fails the whole build.
# Here we pre-download with retries, verify sha256, extract into hash-named cache
# directories, and point CMake at the local sources via FETCHCONTENT_SOURCE_DIR_<NAME>.
# When sglang bumps a dependency hash, it is re-fetched automatically.
SGL_KERNEL_DIR="third_party/sglang/sgl-kernel"
SGL_KERNEL_DEPS_DIR="$PWD/cache/sgl-kernel-deps"

prefetch_sgl_kernel_deps() {
    local name url sha parent subdir dir tarball
    local -A src_dirs=()
    local -a cmake_args=()
    mkdir -p "$SGL_KERNEL_DEPS_DIR"
    # Each line: name url sha256 [parent subdir]; packages with a parent are extracted
    # into a subdirectory of the parent's source tree.
    while read -r name url sha parent subdir; do
        if [[ -n "$parent" ]]; then
            dir="${src_dirs[$parent]}/$subdir"
        else
            dir="$SGL_KERNEL_DEPS_DIR/$name-${sha:0:12}"
        fi
        src_dirs[$name]="$dir"
        if [[ ! -f "$dir/.prefetch-ok" ]]; then
            tarball="$SGL_KERNEL_DEPS_DIR/$name-${sha:0:12}.tar.gz"
            echo "Prefetching $name <- $url"
            curl -fsSL --retry 10 --retry-all-errors --retry-delay 3 --connect-timeout 30 \
                -o "$tarball" "$url"
            echo "$sha  $tarball" | sha256sum -c --quiet -
            rm -rf "$dir"
            mkdir -p "$dir"
            tar xzf "$tarball" -C "$dir" --strip-components=1
            rm -f "$tarball"
            touch "$dir/.prefetch-ok"
        fi
        cmake_args+=("-DFETCHCONTENT_SOURCE_DIR_${name^^}=$dir")
    done < <(python3 - "$SGL_KERNEL_DIR" <<'PY'
import pathlib, re, sys
root = pathlib.Path(sys.argv[1])
pat = re.compile(
    r"FetchContent_Declare\(\s*(\S+)\s+URL\s+(\S+)\s+URL_HASH\s+SHA256=([0-9a-f]{64})"
    r"(?:\s+SOURCE_DIR\s+\$\{(\S+)_SOURCE_DIR\}/(\S+))?\s*\)"
)
for path in [root / "CMakeLists.txt", *sorted((root / "cmake").glob("*.cmake"))]:
    for m in pat.finditer(path.read_text()):
        url = m[2].replace("${GITHUB_ARTIFACTORY}", "github.com")
        print(" ".join(x for x in (m[1], url, m[3], m[4], m[5]) if x))
PY
    )
    if [[ ${#cmake_args[@]} -eq 0 ]]; then
        echo "ERROR: no FetchContent_Declare parsed from $SGL_KERNEL_DIR" >&2
        return 1
    fi
    # scikit-build-core forwards CMAKE_ARGS (space-separated) to cmake configure.
    export CMAKE_ARGS="${CMAKE_ARGS:+$CMAKE_ARGS }${cmake_args[*]}"
    echo "sgl-kernel deps ready (${#cmake_args[@]} packages) under $SGL_KERNEL_DEPS_DIR"
}

prefetch_sgl_kernel_deps

UV_SYNC_ARGS=(--frozen --python 3.12.13 -vv)

if [[ "${REBUILD_SGLANG_KERNEL:-0}" == "1" ]]; then
    UV_SYNC_ARGS+=(
        --refresh-package sglang-kernel
        --reinstall-package sglang-kernel
    )
fi

RUST_LOG=info uv sync "${UV_SYNC_ARGS[@]}" --extra "$CUDA_VARIANT"