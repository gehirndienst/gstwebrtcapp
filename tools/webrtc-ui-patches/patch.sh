#!/bin/bash
usage() {
    echo "Usage: $0 <original_index.html> <modified_index1.html> [<modified_index2.html> ...]"
    exit 1
}
if [ "$#" -lt 2 ]; then
    usage
fi

ORIGINAL_FILE="$1"
shift
WORKING_FILE="index.html"
cp "$ORIGINAL_FILE" "$WORKING_FILE"

# backup
cp "$WORKING_FILE" "${WORKING_FILE}.bak"

# apply patches
PATCH_NUMBER=1
while (( "$#" )); do
    MODIFIED_FILE="$1"
    PATCH_FILE="patch${PATCH_NUMBER}.diff"

    if [ ! -f "$ORIGINAL_FILE" ]; then
        echo "Original file $ORIGINAL_FILE does not exist or is not a regular file"
        exit 1
    fi
    if [ ! -f "$MODIFIED_FILE" ]; then
        echo "Modified file $MODIFIED_FILE does not exist or is not a regular file"
        exit 1
    fi

    diff -u "$ORIGINAL_FILE" "$MODIFIED_FILE" > "$PATCH_FILE"
    if [ $? -gt 1 ]; then
        echo "Trouble to create $PATCH_FILE"
        mv "${WORKING_FILE}.bak" "$WORKING_FILE"  # restore backup
        exit 1
    fi

    patch "$WORKING_FILE" < "$PATCH_FILE"
    PATCH_APPLY_STATUS=$?
    if [ $PATCH_APPLY_STATUS -ne 0 ] && [ $PATCH_APPLY_STATUS -ne 1 ]; then
        echo "Trouble to apply patch${PATCH_NUMBER}"
        mv "${WORKING_FILE}.bak" "$WORKING_FILE"
        exit 1
    fi

    # check for conflict markers
    if grep -q '<<<<<<<' "$WORKING_FILE"; then
        echo "Conflict markers detected after applying patch${PATCH_NUMBER}. Please resolve conflicts in ${WORKING_FILE}."
        mv "${WORKING_FILE}.bak" "$WORKING_FILE"
        exit 1
    fi

    shift
    PATCH_NUMBER=$((PATCH_NUMBER + 1))
done

# if ok, replace the original
mv "$WORKING_FILE" "$ORIGINAL_FILE"
if [ $? -ne 0 ]; then
    echo "Failed to replace $ORIGINAL_FILE with patched file."
    mv "${WORKING_FILE}.bak" "$WORKING_FILE"
    exit 1
fi

# cleanup
rm "${WORKING_FILE}.bak"
for (( i=1; i<$PATCH_NUMBER; i++ )); do
    rm "patch${i}.diff"
done

echo "All patches for WebRTC UI are successfully applied!"
