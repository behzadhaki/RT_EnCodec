#!/bin/bash
set -e
DEST_DIR="lib/max-sdk"
CHECK_FILE="$DEST_DIR/source/max-sdk-base/c74support/c74_max.h"
if [ -f "$CHECK_FILE" ]; then
    echo "✅ Max SDK already exists at $DEST_DIR"
else
    if [ -d "$DEST_DIR" ]; then
        echo "⚠️  Removing incomplete Max SDK at $DEST_DIR ..."
        rm -rf "$DEST_DIR"
    fi
    echo "⬇️ Cloning Max SDK with submodules..."
    git clone --recurse-submodules https://github.com/Cycling74/max-sdk.git "$DEST_DIR"
    echo "✅ Max SDK cloned successfully"
fi
