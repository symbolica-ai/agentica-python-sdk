#!/bin/bash
# Get version from git with tag scrubbing options
#
# Usage: get-version.sh <mode> <package-lang> [suffix]
#
# Modes:
#   prev: 0.2.0-prev210-[githash] (typescript) or 0.2.0.prev210.[githash] (python)
#   dev: 0.2.0-dev210 (typescript) or 0.2.0.dev210 (python)
#   prod: 0.2.0-[suffix] (suffix provided as third argument)
#
# Package Languages:
#   typescript: Uses hyphens as separators (npm semver compatible)
#   python: Uses dots as separators (Python packaging compatible)

set -e

MODE="${1:-}"
SUFFIX=""
PACKAGE_LANG=""

# Parse arguments based on mode
if [ "$1" = "prod" ]; then
    MODE="$1"
    PACKAGE_LANG="$2"
    SUFFIX="$3"
else
    MODE="$1"
    PACKAGE_LANG="$2"
fi

if [ -z "$MODE" ]; then
    echo "Error: mode argument required (prev, dev, or prod)" >&2
    exit 1
fi

if [ "$MODE" != "prev" ] && [ "$MODE" != "dev" ] && [ "$MODE" != "prod" ]; then
    echo "Error: mode must be prev, dev, or prod; NOT $MODE" >&2
    exit 1
fi

if [ -z "$PACKAGE_LANG" ]; then
    echo "Error: package-lang argument required (python or typescript)" >&2
    exit 1
fi

if [ "$PACKAGE_LANG" != "python" ] && [ "$PACKAGE_LANG" != "typescript" ]; then
    echo "Error: package-lang must be python or typescript; NOT $PACKAGE_LANG" >&2
    exit 1
fi

if [ "$MODE" = "prod" ] && [ -z "$SUFFIX" ]; then
    echo "Error: prod mode requires a release suffix argument" >&2
    exit 1
fi

GIT_DESCRIBE=$(git describe --long --dirty 2>/dev/null || echo "")

if [ -z "$GIT_DESCRIBE" ]; then
    echo "0.0.0-unknown"
    exit 0
fi

# Parse and reconstruct from: (ex) v0.2.0-rc1-209-g9edc7e2f1-dirty
# Tag may include suffix like -rc1, -alpha, -beta, etc.
if [[ $GIT_DESCRIBE =~ ^v?([0-9]+\.[0-9]+\.[0-9]+)(-[a-zA-Z0-9]+)?-([0-9]+)-g([a-f0-9]+)(-dirty)?$ ]]; then
    BASE_VERSION="${BASH_REMATCH[1]}"
    TAG_SUFFIX="${BASH_REMATCH[2]}"  # Capture tag's suffix (e.g., -rc, -alpha)
    COMMIT_COUNT="${BASH_REMATCH[3]}"
    GIT_HASH="${BASH_REMATCH[4]}"
    DIRTY="${BASH_REMATCH[5]}"

    # Set separator based on package language
    if [ "$PACKAGE_LANG" = "python" ]; then
        SEP="."
    else
        SEP="-"
    fi

    if [ "$COMMIT_COUNT" = "0" ] && [ -z "$DIRTY" ]; then
        # On exact tag: use tag's suffix if present, otherwise construct based on mode
        if [ -n "$TAG_SUFFIX" ]; then
            echo "${BASE_VERSION}${TAG_SUFFIX}"
        elif [ "$MODE" = "prev" ]; then
            echo "${BASE_VERSION}${SEP}dev${COMMIT_COUNT}+${GIT_HASH}"
        elif [ "$MODE" = "dev" ]; then
            echo "${BASE_VERSION}${SEP}dev${COMMIT_COUNT}"
        elif [ "$MODE" = "prod" ]; then
            if [ "$SUFFIX" = "release" ]; then
                echo "${BASE_VERSION}"
            else
                echo "${BASE_VERSION}-${SUFFIX}"
            fi
        else
            echo "$BASE_VERSION"
        fi
    else
        # Not on exact tag: increment patch version to indicate development after the release
        # This matches setuptools_scm's "guess-next-dev" behavior
        IFS='.' read -r MAJOR MINOR PATCH <<< "$BASE_VERSION"
        NEXT_VERSION="${MAJOR}.${MINOR}.$((PATCH + 1))"
        
        if [ "$MODE" = "prev" ]; then
            # Distinguish between preview and dev versions by appending local id
            # This format conforms to all supported language package specs
            echo "${NEXT_VERSION}${SEP}dev${COMMIT_COUNT}+${GIT_HASH}"
        elif [ "$MODE" = "dev" ]; then
            echo "${NEXT_VERSION}${SEP}dev${COMMIT_COUNT}"
        elif [ "$MODE" = "prod" ]; then
            if [ "$SUFFIX" = "release" ]; then
                echo "${NEXT_VERSION}"
            else
                echo "${NEXT_VERSION}-${SUFFIX}"
            fi
        fi
    fi
else
    echo "0.0.0-unknown"
fi

